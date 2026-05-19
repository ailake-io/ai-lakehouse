# GPU FFI Evaluation — cuVS / NVIDIA for AI-Lake Vector Search

**Status**: Decision document — evaluated 2026-05-19. GPU search implemented via `candle-core` (Option D) in Phase 4.
**Conclusion**: cuVS FFI deferred to Phase 5. `candle-core` GPU brute-force + rayon parallel CPU implemented in `ailake-index`. See §7.

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
ailake-index::HnswIndex::search()
  → GPU path: try_gpu_search() via candle-core/cuda (feature = "gpu")
      → Runtime check: Device::cuda_if_available(0)
      → If no CUDA device → returns None → falls through to CPU path
  → CPU path: rayon par_iter() parallel brute-force (O(n))
      → ef_search parameter accepted but ignored (no graph structure yet)
      → node_count() and serialization are correct
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

Any GPU path adds these hard requirements to every deployment:

| Requirement | Note |
|-------------|------|
| NVIDIA GPU (Ampere+) | cuVS targets sm_80+ for optimal perf |
| CUDA Toolkit 12.x | Headers + `nvcc` at build time |
| CUDA runtime at deploy | `libcudart.so`, `libcublas.so` on PATH |
| Driver ≥ 525 | Required by CUDA 12.x |
| GPU VRAM | 1536-dim, 1M vectors = ~3 GB F16; must fit entirely in VRAM for GPU ANN |

This breaks:
- Every CI environment without GPU (GitHub Actions standard runners)
- Lambda / Fargate / Cloud Run deployments
- Apple Silicon (CUDA unavailable)
- Any CPU-only server (common in on-prem lakehouses)

Build CI would need a separate GPU-enabled runner (self-hosted or AWS `p3` / GCP `a2`).

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

- [x] Feature flag `ailake-index/gpu` — GPU path is optional, CPU path remains default (implemented in Phase 4)
- [x] `candle` brute-force for batch queries (Option D) — safe, no bindgen (implemented in Phase 4)
- [ ] `GpuSearchConfig`: batch size, device id, VRAM budget
- [ ] cuVS IVF-PQ bindgen (Option A) — for files > 500k vectors
- [ ] GPU CI runner (self-hosted or `runs-on: [self-hosted, gpu]`)
- [ ] Benchmark: CPU HNSW vs GPU brute-force vs GPU CAGRA @ 10k/100k/1M vectors

---

## 8. Status After Phase 4

**Implemented in Phase 4:**

- `ailake-index/src/gpu.rs` — `try_gpu_search()` via `candle-core/cuda`; runtime detection via `Device::cuda_if_available(0)`; Cosine, Euclidean, DotProduct kernels via cublas matmul; returns `None` if no CUDA → falls back to CPU
- `ailake-index/src/hnsw.rs` — CPU path replaced with `rayon::par_iter()` parallel brute-force; 4–16× speedup on multicore; `cpu_search()` function
- Feature flag: `--features ailake-index/gpu` activates GPU; default build is CPU-only

**Next step (Phase 5):** Wire `hnsw_rs` graph into `ailake-index` to replace brute-force with true HNSW traversal (10–100× speedup at typical file sizes). cuVS FFI remains deferred — reopen condition: ≥2 conditions from §7 Step 3 hold simultaneously.
