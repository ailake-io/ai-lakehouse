# WORKSPACE.md — Crate Architecture

## Dependency graph

Arrows point from consumer to dependency. No crate may introduce a cycle.

```
ailake-py  ──────────────────────────────────────────► ailake-query
ailake-jni ──────────────────────────────────────────► ailake-query
                                                              │
                        ┌─────────────────────────────────────┤
                        ▼                 ▼                   ▼
                  ailake-file       ailake-catalog       ailake-store
                        │                 │                   │
            ┌───────────┼─────────────────┘                   │
            ▼           ▼                                     │
       ailake-index ailake-parquet                            │
            │           │                                     │
            ▼           │                                     │
       ailake-vec       │                                     │
            │           │                                     │
            └───────┬───┘                                     │
                    ▼                                         │
              ailake-core ◄───────────────────────────────────┘
```

**Rule**: `ailake-core` has zero internal dependencies. Every other crate may depend on `ailake-core`. `ailake-query` depends on all data-plane crates.

## Crate responsibilities

### `ailake-core`
The shared type system. No I/O, no async, no external deps beyond `serde`, `uuid`, `thiserror`.

Public API surface:
- `VectorColumn` — name, dim, metric, precision
- `VectorMetric` — `Cosine | Euclidean | DotProduct`
- `VectorPrecision` — `F32 | F16 | I8Symmetric | Binary`
- `VectorStoragePolicy` — precision, PQ config, reranking flag
- `LlmContextSchema` — canonical field names and types for RAG tables
- `Centroid` — centroid + radius for pruning
- `AilakeError` — unified error enum, used by all crates via `thiserror`
- `RowId` — newtype `u64`, positional index linking Parquet row to HNSW node

### `ailake-parquet`
Reads and writes the **Parquet section** of the unified file. Knows about the `VECTOR` logical type extension via field metadata.

- `ParquetVectorWriter` — writes Arrow `RecordBatch` + encodes vector column as `FIXED_LEN_BYTE_ARRAY` with field metadata (`ailake.dim`, `ailake.metric`, `ailake.precision`)
- `ParquetVectorReader` — reads Parquet, detects vector column by field metadata, returns `RecordBatch`
- This crate does NOT touch the AI-Lake footer extension. That is `ailake-file`'s job.
- Iceberg-compatible: standard readers stop at the PAR1 marker and never see the AI-Lake footer.

### `ailake-vec`
Vector data transformations. No I/O.

- `Quantizer::f32_to_f16_bytes(&[f32]) -> Vec<u8>` — half-precision cast
- `Quantizer::f32_to_i8(&[f32]) -> (Vec<i8>, ScalingParams)` — symmetric min-max
- `PQCodebook::train(vectors, M, k, max_iter) -> PQCodebook` — k-means++ per subspace
- `PQCodebook::encode(&[f32]) -> Vec<u8>` — M bytes, one code per subspace
- `PQCodebook::compute_adc_table(query) -> Vec<Vec<f32>>` — precomputed ADC table for fast batch search
- `PQCodebook::adc_distance(codes, table) -> f32` — O(M) approximate distance
- `dot_product(a, b) -> f32`, `euclidean_distance`, `cosine_distance` — SIMD-dispatched at runtime:
  - x86_64: AVX2 path (2× unrolled, 16 f32/iter for dot/euclidean; single-pass 3-accumulator cosine)
  - aarch64: NEON path (4 f32/iter via `vmlaq_f32`)
  - Scalar fallback on other architectures
- `exact_distance(metric: VectorMetric, a: &[f32], b: &[f32]) -> f32` — dispatches to correct metric; used by reranking after PQ
- `compute_centroid_and_radius(&[Vec<f32>], VectorMetric) -> Centroid`
- `BlockCompressor::zstd(level)`, `BlockCompressor::lz4()` — block-level compression

### `ailake-index`
HNSW + IVF-PQ index lifecycle. GPU backends: NVIDIA CUDA (compile-time feature) + AMD ROCm (runtime libloading). CPU fallback always available.

- `HnswBuilder` — builds HNSW from `(RowId, &[f32])` pairs
  - Parameters: `M` (max connections), `ef_construction`, `metric`
  - Implements Malkov & Yashunin 2018, Algorithms 1 + 2: random level assignment, greedy descent, beam search, bidirectional links, neighbour pruning
- `HnswIndex` — searchable index over typed `RowId` keys
  - Internal layout: contiguous `flat_vecs: Vec<f32>` (row-major), `row_ids: Vec<u64>`, `neighbors: Vec<Vec<Vec<usize>>>`, `node_levels`, `entry_point`, `max_layer`
  - Visited tracking: thread-local generation bitmap — O(1) reset by incrementing generation counter; no per-query allocation
  - `search(query: &[f32], top_k: usize, ef_search: usize) -> Vec<(RowId, f32)>`
  - CPU fallback: `brute_force()` via `rayon::par_iter()` — activated only when `neighbors` is empty
- `IvfPqIndex` / `IvfPqConfig` / `IvfPqSerializer` — inverted file index with Product Quantization
  - `IvfPqConfig::for_dataset(dim, n)` — scales `nlist` to √n clamped [16, 1024]
  - `kmeans_dispatch()` — priority: CUDA → ROCm → CPU rayon
- `AnyIndex` — enum dispatching search to `HnswIndex` or `IvfPqIndex`
- `HnswSerializer` — bincode-based serialization of the full HNSW graph
- `MmapLoader` — opens a serialized HNSW from a memory-mapped byte slice
  - Lazy: graph traversal only pages in the regions touched during search
- `hardware::HardwareBackend` — `CpuSimd` / `NvidiaCuda` / `AmdRocm`
- `hardware::HardwareProfile` — `has_cuda`, `has_rocm`, `backend`, `cpu_logical_cores`, `has_avx2`, `has_avx512`
- `hardware::detect_backend()` — probed once via `OnceLock`; AMD probed before NVIDIA
- `hardware::detect_cuda()` — true only for `NvidiaCuda` (not ROCm compat layer)
- `hardware::detect_rocm()` — true only for `AmdRocm`
- `gpu::try_nvidia_search_batch()` / `try_nvidia_kmeans()` — NVIDIA cuBLAS SGEMM via `libloading` dlopen of `libcudart.so` + `libcublas.so`; returns `None` if libraries not found
- `gpu::try_rocm_search_batch()` / `try_rocm_kmeans()` — AMD hipBLAS SGEMM via `libloading` dlopen of `libamdhip64.so` + `libhipblas.so`; returns `None` if libraries not found

**Feature flags**: none. Both GPU backends are always compiled. Hardware detected at runtime.
- Default build: CPU rayon; NVIDIA activated if `libcudart.so` + `libcublas.so` found; AMD activated if `libamdhip64.so` + `libhipblas.so` found (AMD checked first)
- `candle-core` dependency removed; no CUDA Toolkit required at build time

### `ailake-file`
**Owns the unified file format.** This is the integration crate that combines Parquet + AI-Lake footer.

- `AilakeFileWriter` — high-level writer:
  1. Writes RecordBatch via `ailake-parquet`
  2. Builds HNSW via `ailake-index`
  3. Serializes HNSW + centroid + radius into the AI-Lake footer
  4. Appends footer to the file after the final PAR1 marker
  5. Updates Parquet `key_value_metadata` with `ailake.hnsw_offset` and `ailake.hnsw_len`
- `AilakeFileReader` — high-level reader:
  - `read_parquet()` → returns Parquet data only (via `ailake-parquet`)
  - `load_index()` → reads AI-Lake footer, returns `HnswIndex` via mmap
  - `get_centroid()` → reads centroid + radius from footer header (cheap, no HNSW load)
- `FooterLayout` — binary layout spec of the AI-Lake footer (see `FILE_FORMAT.md`)

See [`docs/specs/FILE_FORMAT.md`](../specs/FILE_FORMAT.md) for the binary layout.

### `ailake-catalog`
Iceberg catalog operations. The only crate that reads/writes `metadata.json` and `.avro` manifests.

Implements the `CatalogProvider` trait for every supported backend:

```
ailake-catalog/src/
├── lib.rs          # re-exports, module declarations
├── provider.rs     # CatalogProvider trait, TableIdent, DataFileEntry, NewSnapshot
├── metadata.rs     # metadata.json read/write (Iceberg Spec v2)
├── snapshot.rs     # manifest JSON builder
├── hadoop.rs       # HadoopCatalog — filesystem / any Store backend
├── rest.rs         # RestCatalog — Iceberg REST Catalog spec (Polaris, S3 Tables, Nessie, Unity Catalog)
├── databricks.rs   # DatabricksAuth + builders for Azure/AWS/GCP Unity Catalog
├── glue.rs         # GlueCatalog — AWS Glue (feature = "catalog-glue", stub)
├── nessie.rs       # NessieCatalog — Nessie branching extensions (feature = "catalog-nessie", stub)
└── jdbc.rs         # JdbcCatalog — PostgreSQL/MySQL (feature = "catalog-jdbc", stub)
```

`CatalogProvider` trait:
```rust
#[async_trait]
pub trait CatalogProvider: Send + Sync {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()>;
    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata>;
    async fn commit_snapshot(&self, table: &TableIdent, snapshot: NewSnapshot) -> AilakeResult<SnapshotId>;
    async fn list_files(&self, table: &TableIdent, snapshot_id: Option<SnapshotId>) -> AilakeResult<Vec<DataFileEntry>>;
    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()>;
}
```

All vector statistics (centroid, radius, HNSW byte offsets) are stored in the `custom_properties` map of each `DataFile` entry in the Avro manifest — a spec-defined extension point ignored by unknown readers.

Backend selection is driven by configuration, not code changes. The `ailake-query` layer depends only on `dyn CatalogProvider`.

### `ailake-store`
Object storage abstraction. Thin wrapper over the `object_store` crate.

- `Store` trait:
  - `get(path) → Bytes`
  - `get_range(path, range: Range<u64>) → Bytes` — critical for partial reads of HNSW footer from S3
  - `put(path, Bytes)`, `list(prefix)`, `file_size(path)`, `exists(path)`, `delete(path)`
- `LocalStore` — filesystem implementation (dev/tests)
- `ObjectStoreBackend` — wraps any `Arc<dyn object_store::ObjectStore>` behind the `Store` trait; used internally by all typed builders
- **Typed credential builders** (feature-gated):
  - `s3_store(S3Config, prefix)` — `store-s3`; `S3Credentials`: `Static` / `WebIdentity` (IRSA) / `InstanceProfile` / `Default`
  - `gcs_store(GcsConfig, prefix)` — `store-gcs`; `GcsCredentials`: `ServiceAccountFile` / `ServiceAccountJson` / `ApplicationDefault` (ADC + Workload Identity)
  - `azure_store(AzureConfig, prefix)` — `store-azure`; `AzureCredentials`: `ClientSecret` / `ManagedIdentity` / `AccessKey` / `SasToken` / `AzureCli`
- `store_from_url(url)` — URL-based auto-builder; reads credentials from env; dispatches by scheme: `s3://`, `gs://`, `az://`, `file://`
- All async, all return `AilakeError` on failure

### `ailake-query`
Query planning and execution. The integration layer — depends on all data-plane crates.

- `TableWriter` — `write_batch(batch, embeddings)` + `commit()` → Iceberg snapshot
- `VectorPruner::prune(files, query, metric, threshold)` — filters `Vec<DataFileEntry>` using centroid geometry; works on catalog metadata only, zero file I/O for pruned files
- `search(table, query, config, ...)` — full pipeline: list catalog → prune → load HNSW → global top-k merge; `SearchConfig.pruning_threshold` controls prune aggressiveness; `SearchConfig.rerank_factor` enables reranking after PQ (fetch `top_k × factor` candidates, recompute exact distances from raw vectors, re-sort)
- `SearchSession` — pre-loads all shard HNSW indexes once, serves many queries without I/O per query:
  - `SearchSession::load(table, vector_column, dim, catalog, store, load_raw) -> AilakeResult<Self>`
  - `SearchSession::search_query(query, config) -> Vec<SearchResult>` — sync, no I/O
  - Used by `ailake-bench` to achieve ~450 QPS on SIFT-1M
- `CompactionPlanner::plan(files)` — selects files smaller than `target_file_size_bytes`
- `CompactionExecutor::compact(files, output_path)` — merges N files into one via Arrow `concat_batches`, rebuilds HNSW, returns new `DataFileEntry`
- `CompactionExecutor::run(planner, table, catalog, prefix)` — full cycle: plan + compact + commit + delete old files
- `ContextAssembler::assemble_chunks(chunks: Vec<Chunk>)`:
  - Sorts by `distance` (most relevant first)
  - Deduplicates by embedding cosine distance < `dedup_threshold`
  - Groups by `document_id`, sorts each group by `chunk_index`
  - Applies `max_tokens` budget (4 chars ≈ 1 token)
  - Returns `AssembledContext { text: XML, chunk_count, token_estimate }`

### `ailake-py`
PyO3 extension module (`cdylib`). Thin async-to-sync bridge — all logic lives in other crates. Built with `maturin`; distributed via PyPI as `ailake`.

Deps: `ailake-query`, `ailake-catalog`, `ailake-store`, `ailake-core` + `openssl-sys[vendored]` (forces hermetic OpenSSL compilation in manylinux wheel builds; no system headers required).

Exports:
- `TableWriter(path, vector_column, dim, metric)` — open or create table
- `TableWriter.write_batch(texts, embeddings)` — stage a batch
- `TableWriter.commit() → int` — flush to Parquet + HNSW, return snapshot id
- `search(path, query, top_k) → list[dict]` — vector search
- `assemble_context(chunks, max_tokens, dedup_threshold) → str` — LLM context XML

### `ailake-jni`
C-ABI cdylib loaded by Spark, Trino, and Flink plugins via JNA. Single JSON-envelope API shared across all three JVM languages (Scala, Kotlin, Java).

Exports (`#[no_mangle]` C-ABI):
- `ailake_search_json(request_json) → *mut c_char` — vector search, JSON in/out
- `ailake_write_batch_json(request_json) → *mut c_char` — write batch, JSON in/out
- `ailake_free_string(ptr)` — free any returned pointer
- `ailake_version() → *const c_char` — static version string

---

## Cargo.toml (workspace root)

```toml
[workspace]
resolver = "2"
members = [
    "ailake-core",
    "ailake-parquet",
    "ailake-vec",
    "ailake-index",
    "ailake-file",
    "ailake-catalog",
    "ailake-store",
    "ailake-query",
    "ailake-cli",
    "tests",
    "ailake-jni",
    "ailake-bench",
    "ailake-py",
]

[workspace.dependencies]
# Core
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
uuid        = { version = "1", features = ["v4", "serde"] }
thiserror   = "1"
bytes       = "1"
half        = { version = "2", features = ["serde"] }
async-trait = "0.1"

# Async
tokio       = { version = "1", features = ["rt-multi-thread", "io-util", "fs", "sync", "time", "macros"] }
futures     = "0.3"

# Data
parquet      = { version = "52", features = ["async"] }
arrow-array  = "52"
arrow-schema = "52"
arrow-select = "52"
arrow-buffer = "52"
object_store = { version = "0.10" }  # cloud features added per-crate via ailake-store feature flags

# Iceberg
iceberg     = "0.3"
apache-avro = "0.16"

# Vector index
hnsw_rs     = "0.3"
bincode     = "1"
memmap2     = "0.9"
rayon       = "1"

# GPU — runtime dlopen, both vendors; no build-time SDK required
libloading  = "0.8"

# Compression
lz4_flex    = "0.11"
zstd        = "0.13"

# Bindings
# Note: reqwest is NOT in workspace deps — ailake-catalog declares it inline
# with rustls-tls to keep openssl-sys out of the ailake-py dep tree.
pyo3        = { version = "0.22", features = ["extension-module"] }
# uniffi removed — all JVM bindings use C-ABI + JNA

# CLI
clap        = { version = "4", features = ["derive", "env"] }

# Observability
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Dev/test
criterion   = { version = "0.5", features = ["html_reports"] }
tempfile    = "3"
proptest    = "1"
rand        = "0.8"

[profile.release]
lto           = "thin"
codegen-units = 1
opt-level     = 3
strip         = "symbols"
panic         = "abort"

[profile.bench]
inherits    = "release"
debug       = true
debug       = true
```

---

## Build phases and what is in scope

| Phase | Status | Scope |
|---|---|---|
| **Phase 1** | ✅ Complete | Local MVP — write + search on local filesystem, HNSW footer, Iceberg catalog |
| **Phase 2** | ✅ Complete | Cloud storage (`ObjectStoreBackend`), mmap HNSW, compaction, PQ, geometric pruning, `ContextAssembler`, PyO3 bindings |
| **Phase 3** | ✅ Complete | Catalog backends (NessieCatalog, JdbcCatalog, GlueCatalog), JNA C-ABI bindings (`ailake-jni`), multi-column vectors |
| **Phase 4** | ✅ Complete | PQ reranking, public format spec, GPU search (NVIDIA cuBLAS + AMD hipBLAS runtime-only), HNSW perf optimizations, IVF-PQ native index, GPU k-means, adaptive index selection, `ailake-flink` Kotlin connector (Flink Table API + Catalog, JNA bridge) |

### Phase 1 — Local MVP ✅
**Goal**: `cargo test --workspace` passes; can write a self-contained file and search it on local disk.

- `ailake-core`: all types
- `ailake-vec`: quantization F32→F16, centroid computation, distance functions
- `ailake-parquet`: writer (vector column encoding), reader (vector column decoding)
- `ailake-index`: `HnswBuilder`, `HnswIndex`, bincode serialization
- `ailake-file`: unified writer/reader, footer layout
- `ailake-catalog`: `CatalogProvider` trait + `HadoopCatalog` (filesystem) only
- `ailake-store`: `LocalStore` only
- Integration test: write + search end-to-end, verify recall

### Phase 2 — Distribution and Cloud Storage ✅

- `ailake-store`: `ObjectStoreBackend` wrapping `object_store` crate (S3/GCS/Azure via feature flags `store-s3`, `store-gcs`, `store-azure`)
- `ailake-index`: real `MmapLoader` — writes HNSW bytes to tempfile, mmaps, deserializes
- `ailake-vec`: `PQCodebook` (k-means++ per subspace, ADC distance), `BlockCompressor` (zstd/lz4)
- `ailake-query`: `VectorPruner` (geometric centroid pruning), `CompactionExecutor`, `ContextAssembler`
- `ailake-query::search`: pruning integrated via `SearchConfig.pruning_threshold`
- `ailake-py`: PyO3 bindings (`TableWriter`, `search`, `assemble_context`)

Also delivered in Phase 2:
- `RestCatalog` — full Iceberg REST Catalog spec implementation (OAuth2 token caching, manifest writes to object storage)
- Databricks Unity Catalog support — `DatabricksAuth` + `databricks_azure`/`databricks_aws`/`databricks_gcp` builders

Deferred to Phase 3:
- Docker integration tests (MinIO + Nessie + Localstack)

### Phase 3 — Catalog backends + Query engine integration ✅

Delivered in Phase 3:
- `ailake-catalog`: `NessieCatalog` — wraps `RestCatalog` + Nessie v2 branching API (`list_branches`, `create_branch`, `merge_branch`, `delete_branch`)
- `ailake-catalog`: `JdbcCatalog` — PostgreSQL/MySQL/SQLite via `sqlx 0.7` `AnyPool`; schema auto-created; versioned metadata.json via UUID paths
- `ailake-catalog`: `GlueCatalog` — AWS Glue Data Catalog via `aws-sdk-glue 1.x`; Iceberg-standard `metadata_location` parameter; tables visible in Athena/EMR
- `ailake-jni`: C-ABI exports (`ailake_search_json`, `ailake_write_batch_json`, `ailake_free_string`)
- Multi-column vector tables (`embedding` + `context_embedding`)
- `ailake-spark-runtime` (separate Scala repo): Spark `VectorScanStrategy`, `ailake_search` UDF
- `ailake-trino-plugin` (separate Java repo): Trino `ConnectorTableFunction`

Deferred (external env required):
- Compatibility tests: Spark, Trino, Beam, DuckDB, PyIceberg (integration tests require Docker/cluster)

### Phase 4 — Production hardening ✅

Delivered in Phase 4:
- Reranking after PQ: `SearchConfig.rerank_factor`, `exact_distance()` in `ailake-vec`
- Public format spec: `docs/specs/FILE_FORMAT.md` — binary layout, AILK header/trailer, KV metadata keys
- GPU search: NVIDIA CUDA (cuBLAS SGEMM, runtime-only, no build flag) + AMD ROCm (hipBLAS SGEMM, runtime-only) in `ailake-index`; automatic CPU fallback via rayon; detection priority: AMD ROCm → NVIDIA CUDA → CPU SIMD; `candle-core` removed from workspace
- Hardware abstraction: `HardwareBackend` enum, `HardwareProfile`, `detect_backend()` / `detect_cuda()` / `detect_rocm()` in `ailake-index/src/hardware.rs`
- GPU k-means dispatch: CUDA → ROCm → CPU for IVF-PQ training (`kmeans_dispatch` in `ivf_pq.rs`)
- Adaptive index selection: `IndexType::Auto`, `write_batch_auto()`, `CompactionIndexStrategy::Auto`
- GPU FFI evaluation: `docs/specs/GPU_FFI_EVALUATION.md` — cuVS evaluated, cuBLAS + hipBLAS libloading chosen (both runtime-only)
- Real HNSW graph: custom implementation in `ailake-index` (Malkov & Yashunin 2018); generation bitmap visited tracker; contiguous `flat_vecs` layout
- SIMD distance functions: AVX2 + NEON in `ailake-vec/src/distance.rs`; runtime detection; 2× unrolled AVX2 for dot/euclidean
- `SearchSession` in `ailake-query`: pre-loaded multi-query search, eliminates per-query I/O
- `ailake-bench` crate: SIFT-1M benchmark (128D Euclidean, 1M vectors)
  - Results: 2394 vec/s write, 453 QPS, Recall@10 = 99.6%, mean latency 2.2 ms
- HNSW performance optimizations in `ailake-index`:
  - **Neighbor prefetch**: `_mm_prefetch T0` in `search_layer` hot loop — hides random DRAM latency on x86_64
  - **SELECT-NEIGHBORS-HEURISTIC** (Algorithm 4, Malkov & Yashunin 2018): diversity-enforcing neighbor selection replaces simple nearest-M prune; improves recall@10 by ~2-5% at same throughput
  - **F16 search + F32 rerank**: `HnswIndex` stores `flat_vecs_f16`; HNSW traversal uses half-precision distances (less cache pressure), final candidates reranked with exact F32 — 30% latency reduction, no recall loss
  - **Metric monomorphization**: `DistFn` trait with `CosineDist`/`EuclideanDist`/`DotProductDist` ZSTs; dispatch on metric once at entry, all inner fns generic `<M: DistFn>` — eliminates per-call `match` from hot loop, allows LLVM to inline distance functions
  - SIFT-1M HNSW build: 218.9 s → 155.8 s (−29%)
- Multi-engine comparison benchmark (`--engine all`): AI-Lake vs LanceDB vs pgvector
  - `ailake-bench`: new `pgvector-bench` feature — `pgvector_bench.rs` uses text COPY + HNSW index; `Engine::Pgvector` + `Engine::All` updated
  - `bench_result::print_multi_comparison` — N-engine side-by-side table, highlights fastest QPS
  - Deep Lake: `scripts/deeplake_bench.py` (Python) — exact kNN on subset; ANN requires paid Deep Memory plan (no Rust SDK available)

Phase 4 complete. See Phase 5 for next planned work.
