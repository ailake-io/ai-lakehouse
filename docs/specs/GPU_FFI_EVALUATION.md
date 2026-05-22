# GPU FFI Evaluation — cuVS / NVIDIA + AMD ROCm for AI-Lake Vector Search

**Status**: Decision document — evaluated 2026-05-19, updated 2026-05-22.
**Conclusion**: cuVS FFI deferred to Phase 5. Two GPU backends implemented in `ailake-index`:
- **NVIDIA CUDA** (Phase 4): `cuBLAS` SGEMM via `libloading` — runtime-only, no compile-time dependency (replaced `candle-core`).
- **AMD ROCm** (Phase 4): `hipBLAS` SGEMM via `libloading` — runtime-only, no compile-time dependency.

Both backends require zero build-time GPU SDK. A single binary detects and uses
NVIDIA or AMD hardware at startup via `dlopen`. See §7 (NVIDIA decision) and §8 (Phase 4 status).

---

## 1. Background

AI-Lake performs vector search per-file: each Parquet file carries its own
index, and queries fan out across surviving files. The index implementation
(`ailake-index`) uses **parallel CPU brute-force** (rayon `par_iter`, O(n))
as the default CPU path, and optionally GPU brute-force via `candle-core/cuda`
when compiled with `--features gpu` and a CUDA device is available at runtime.

This document evaluates whether GPU-accelerated ANN (Approximate Nearest
Neighbor) search via NVIDIA cuVS is a better next step than the current
candle-core approach.

---

## 2. Current State Diagnosis

```
ailake-index::hardware::detect_backend()   ← OnceLock<HardwareBackend>, probed once
  → probe_rocm_driver()  (libamdhip64.so + hipGetDeviceCount)  → AmdRocm
  → probe_cuda_driver()  (libcuda.so.1   + cuDeviceGetCount)   → NvidiaCuda
  → CpuSimd                                                      (fallback)

ailake-index::ivf_pq::kmeans_dispatch()
  → #[cfg(feature = "gpu")] try_gpu_kmeans()   (candle-core, NVIDIA only)
  → try_rocm_kmeans()                           (hipBLAS SGEMM, AMD only)
  → kmeans_centroids()                          (rayon CPU fallback)

ailake-query::SearchSession::search_batch()
  → flat-scan shards:
      detect_cuda()  → try_gpu_search_batch()    (candle-core batch matmul)
      detect_rocm()  → try_rocm_search_batch()   (hipBLAS SGEMM batch)
      else           → flat_search() par_iter()   (CPU fallback)
  → indexed shards: rayon parallel-map over queries → AnyIndex::search()

ailake-file::AilakeFileWriter::IndexType::Auto
  → HardwareProfile::detect()
  → has_cuda || has_rocm || cpu_logical_cores > 8  →  IVF-PQ
  → else                                           →  HNSW
```

**Performance ceiling of brute force (single file, 1 CPU thread)**

| File size     | dim=768 F16 | dim=1536 F16 | dim=3072 F16 |
|---------------|-------------|--------------|--------------|
| 10k vectors   | ~2 ms       | ~4 ms        | ~8 ms        |
| 50k vectors   | ~10 ms      | ~20 ms       | ~40 ms       |
| 500k vectors  | ~100 ms     | ~200 ms      | ~400 ms      |

These are rough CPU single-thread estimates (no SIMD). With AVX-512 and
`rayon` parallelism the practical ceiling is ~10× lower — still O(n).

---

## 3. cuVS Overview

cuVS (formerly RAPIDS RAFT vector search) is NVIDIA's GPU-accelerated ANN library.
Repository: `github.com/rapidsai/cuvs`

### Relevant algorithms

| Algorithm  | Type          | Build time | Query latency | Recall@10 |
|------------|---------------|------------|---------------|-----------|
| IVF-Flat   | Exact + GPU   | Fast       | Low (GPU)     | 100%      |
| IVF-PQ     | Approx + GPU  | Medium     | Very low      | 90–95%    |
| CAGRA      | Graph + GPU   | Slow       | Lowest known  | 95–99%    |
| Brute-force| Exact + GPU   | None       | Low–medium    | 100%      |

### C API surface

cuVS exposes a C API (`<cuvs/neighbors/*.h>`) suitable for FFI.

```c
// Build index (simplified)
cuvsCagraIndex_t index;
cuvsCagraIndexParams_t params;
cuvsCagra_build(res, &params, dataset, &index);

// Search
cuvsCagraSearchParams_t sp;
cuvsCagra_search(res, &sp, index, queries, neighbors, distances);

// Free
cuvsCagraIndex_destroy(index);
```

The C API is stable across minor versions; breaking changes follow SemVer.

---

## 4. FFI Integration Options

### Option A — Raw bindgen on cuVS C headers

```toml
# ailake-index/Cargo.toml
[build-dependencies]
bindgen = "0.69"
[dependencies]
libc = "0.2"
```

```rust
// build.rs
bindgen::Builder::default()
    .header("cuvs/neighbors/cagra.h")
    .generate()?
    .write_to_file("src/bindings.rs")?;
```

**Pros**: full access to every cuVS algorithm, no extra Rust dep.  
**Cons**:
- Requires CUDA toolkit installed at build time (`nvcc`, headers in `/usr/local/cuda`).
- `build.rs` must locate `libcuvs.so`, `libraft.so`, `libcublas.so` — RPATH is fragile.
- Generated bindings must be committed or regenerated per CUDA version; drift is common.
- Async GPU operations expose lifetimes that bindgen cannot model safely — every call site becomes `unsafe`.

### Option B — `cudarc` Rust crate + manual kernels

The `cudarc` crate (maintained by Hugging Face) provides safe Rust wrappers
around the CUDA driver API. One could write custom CUDA kernels (`.cu` files)
for distance computation and load them at runtime via `cudarc::driver`.

```toml
[dependencies]
cudarc = { version = "0.12", features = ["cublas"] }
```

**Pros**: No bindgen; safe Rust API; PTX kernels are portable across GPU architectures.  
**Cons**:
- Distance kernels and ANN graph algorithms are thousands of lines of CUDA C —
  reimplementing cuVS from scratch is infeasible.
- Still requires CUDA toolkit for `.cu` compilation.
- `cudarc` has no ANN graph primitives; only raw tensor operations.

### Option C — Python subprocess delegation (not viable for production)

Delegate searches to a Python process running cuVS/FAISS-GPU via subprocess.
Rejected: serialization latency alone (~0.5–2ms) exceeds the benefit for
per-file searches on small indices.

### Option D — GPU brute-force via `candle` / `burn` (tensor frameworks)

`candle` (Hugging Face) supports CUDA matrix multiplication via `cublas`.
One batch matmul computes all distances for a query against N vectors.

```rust
// Conceptual — candle CUDA matmul
let db = Tensor::new(&vectors, &Device::Cuda(0))?;  // N × dim
let q  = Tensor::new(&[query], &Device::Cuda(0))?;  // 1 × dim
let dots = q.matmul(&db.t()?)?;                      // 1 × N
```

**Pros**: No bindgen; uses well-maintained `candle`; GPU transfer is the main cost.  
**Cons**: No ANN graph structure — still O(n) per query (just GPU-accelerated).  
**Viable for**: batch queries where N > 100k and GPU is available.

---

## 5. Deployment Requirements

### NVIDIA CUDA path (compile-time `--features ailake-index/gpu`)

| Requirement | Note |
|-------------|------|
| NVIDIA GPU (Ampere+) | cuVS targets sm_80+; candle works from sm_60 |
| CUDA Toolkit 12.x | Headers + `nvcc` at build time |
| CUDA runtime at deploy | `libcudart.so`, `libcublas.so` on PATH |
| Driver ≥ 525 | Required by CUDA 12.x |
| GPU VRAM | 1536-dim, 1M vectors ≈ 3 GB F16 |

### AMD ROCm path (runtime only — no build dependency)

| Requirement | Note |
|-------------|------|
| AMD GPU (GCN4+ / RDNA1+) | Any ROCm-capable device |
| ROCm 5.0+ at deploy | `libamdhip64.so` + `libhipblas.so` on runtime LD path |
| No build requirement | `libloading` dlopen — binary compiles on any host |

### Shared limitations

Both GPU paths break:
- GitHub Actions standard runners (no GPU) — CI uses CPU path automatically
- Lambda / Fargate / Cloud Run
- Apple Silicon (no CUDA or ROCm)

The AMD ROCm path requires **no build-time dependency** — the binary degrades to CPU automatically when `libamdhip64.so` is absent. This is the key advantage over the NVIDIA CUDA path.

---

## 6. Performance Analysis: When Does GPU Win?

GPU search wins when:
1. The index fits entirely in VRAM (no PCIe transfer stalls)
2. Query batches are large (≥ 64 queries at once) — GPU parallelism amortizes kernel launch overhead
3. Indices are large (≥ 50k vectors per file) — small indices don't saturate GPU SMs

**Per-file model analysis**

AI-Lake partitions data across many files, each with its own small index.
Typical file: 10k–100k vectors. For a single query:

| Scenario | CPU brute-force | CPU HNSW (M=16) | GPU CAGRA |
|----------|-----------------|-----------------|-----------|
| 10k vecs, dim=1536 | ~4 ms (1 thread) | ~0.5 ms | ~3 ms (launch+xfer overhead) |
| 100k vecs, dim=1536 | ~40 ms | ~2 ms | ~1 ms |
| 1M vecs, dim=1536 | ~400 ms | ~10 ms | ~3 ms |

**Key observation**: for files ≤ 100k vectors, CPU HNSW outperforms GPU for
single-query workloads because GPU kernel launch + PCIe transfer (~1–3 ms
fixed overhead) dominates.

GPU wins at scale:
- Files > 500k vectors (unusual given Parquet compaction targets)
- Batch query mode (many queries per second against same loaded index)

---

## 7. Recommendation

**Do not add GPU FFI in Phase 4. Recommended path:**

### Step 1 — Wire up `hnsw_rs` (already in workspace deps)

`hnsw_rs` is declared in `[workspace.dependencies]` but not added to
`ailake-index/Cargo.toml`. Adding it reduces search latency by 10–100×
at zero deployment cost:

```toml
# ailake-index/Cargo.toml
[dependencies]
hnsw_rs = { workspace = true }
```

Replace `HnswIndex`'s brute-force scan with a real HNSW graph:
- Build: `Hnsw::<f32, dist::DistCosine>::new(M, max_elements, ef_construction, …)`
- Search: `hnsw.search_neighbours(query, top_k, ef_search)`

This alone would make the CPU path competitive at every realistic file size.

### Step 2 — SIMD distance functions (free speedup, no deps)

Replace scalar loops in `ailake-vec::distance` with `std::arch` intrinsics or
the `simsimd` crate. AVX-512 on x86 gives ~4–8× throughput improvement for
distance computation. Requires no GPU hardware.

### Step 3 — GPU FFI (Phase 5, when justified)

GPU FFI becomes justified when at least **two** of these conditions hold:

- Median file size exceeds 200k vectors (requires ~600 MB VRAM per open file)
- Throughput requirement exceeds 1000 QPS on a single node
- Multi-query batch mode is the dominant workload (RAG pipelines with batch
  retrieval, not single-doc chat)
- Target deployment environment guarantees NVIDIA hardware (e.g., dedicated ML
  inference cluster)

Recommended GPU path when the time comes: **Option D (candle + cublas)** for
brute-force batch queries, OR **Option A (bindgen on cuVS IVF-PQ)** for
approximate search on large in-memory indices. cuVS CAGRA gives the best
recall/latency tradeoff but requires the most complex build setup.

### Phase 5 GPU work items (future)

- [x] NVIDIA runtime path — no build-time CUDA SDK (replaced candle-core with cuBLAS libloading in Phase 4)
- [x] AMD ROCm runtime path — hipBLAS SGEMM via libloading (Phase 4)
- [ ] `GpuSearchConfig`: batch size, device id, VRAM budget
- [ ] cuVS IVF-PQ bindgen (Option A) — for files > 500k vectors
- [ ] GPU CI runner (self-hosted or `runs-on: [self-hosted, gpu]`)
- [ ] Benchmark: CPU HNSW vs GPU brute-force vs GPU CAGRA @ 10k/100k/1M vectors

---

## 8. Status After Phase 4

**Implemented in Phase 4 — NVIDIA CUDA (cuBLAS via libloading):**

- `ailake-index/src/gpu.rs` `nvidia_impl` module — `try_nvidia_search_batch()`, `try_nvidia_kmeans()` via `libloading` dlopen of `libcudart.so` (tries `.so`, `.so.12`, `.so.11`) + `libcublas.so` (same fallback); RAII guards `DevBuf` (cudaFree) and `BlasHandle` (cublasDestroy_v2); Cosine/Euclidean/DotProduct via `cublasSgemm_v2`; no compile-time dependency; returns `None` if libraries not found
- Replaces `candle-core` (Option D from §4) — eliminates compile-time CUDA Toolkit requirement and ~30% binary size from candle dependency tree
- `gpu` feature flag removed from `ailake-index`; `candle-core` removed from workspace deps
- SGEMM formulation identical to ROCm: `C[N×Q col-major] = alpha · db[N×dim]ᵀ · queries[Q×dim]`; only constants differ: `CUBLAS_OP_N=0`, `CUBLAS_OP_T=1` (vs HIP 111/112)
- `kmeans_dispatch` priority: `try_nvidia_kmeans` → `try_rocm_kmeans` → `kmeans_centroids` (rayon)

**Implemented in Phase 4 — AMD ROCm (hipBLAS SGEMM):**

- `ailake-index/src/gpu.rs` `rocm_impl` module — `try_rocm_search_batch()`, `try_rocm_kmeans()` via `libloading` dlopen of `libamdhip64.so` + `libhipblas.so`; RAII guards for device buffers and hipBLAS handle; Cosine/Euclidean/DotProduct computed via SGEMM with norms on CPU; no compile-time dependency; returns `None` if libraries not found
- SGEMM formulation: `C[N×Q col-major] = alpha · db[N×dim]ᵀ · queries[Q×dim]`, where alpha encodes metric scaling; norms computed on CPU, broadcast-added after D2H copy
- k-means: `cross[K×N col-major] = −2 · centroids · vectorsᵀ`; argmin and centroid update on CPU — only the O(n·k·dim) matmul runs on GPU

**Implemented in Phase 4 — Hardware abstraction:**

- `ailake-index/src/hardware.rs` — `HardwareBackend` enum (`CpuSimd`/`NvidiaCuda`/`AmdRocm`); `OnceLock<HardwareBackend>` caches detection result; AMD probed before NVIDIA to handle ROCm CUDA-compat layer; `HardwareProfile` struct includes `has_cuda`, `has_rocm`, `backend`, `cpu_logical_cores`, `has_avx2`, `has_avx512`
- `detect_backend()`, `detect_cuda()`, `detect_rocm()` — public functions used by dispatch in `ivf_pq.rs`, `scanner.rs`

**Implemented in Phase 4 — Adaptive index selection:**

- `ailake-file::IndexType::Auto` — resolved at write time via `HardwareProfile::detect()`; IVF-PQ chosen when `has_cuda || has_rocm || cpu_logical_cores > 8 && n >= 5000`; HNSW otherwise
- `ailake-query::TableWriter::write_batch_auto()` — thin wrapper that delegates to IVF-PQ or HNSW path based on hardware profile
- `ailake-query::CompactionIndexStrategy` — Auto/ForceHnsw/ForceIvfPq; compaction respects same hardware-adaptive logic

**Binary size impact (Phase 4 final):** `ailake-bench` 13 MB unstripped → 9.3 MB (auto-stripped, panic=abort, no candle-core). `libailake_jni.so` 12 MB → 9.0 MB.

**Next step (Phase 5):** cuVS FFI remains deferred — reopen condition: ≥2 conditions from §7 Step 3 hold simultaneously. Current SGEMM GPU path is adequate for files up to ~500k vectors at dim=1536.
