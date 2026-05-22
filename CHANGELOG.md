# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] — 2026-05-22

### Added
- **IVF-PQ native index**: `IvfPqIndex` for S3 workloads — coarse IVF quantizer + PQ ADC; `TableWriter::write_batch_ivf_pq` (`ailake-index`)
- **GPU k-means for IVF-PQ training**: k-means++ centroid training offloaded to CUDA via `candle-core` when GPU is available
- **Adaptive index selection**: `HardwareCapability` detection at startup; `TableWriter` and compaction automatically choose HNSW vs IVF-PQ based on dataset size and hardware
- **Runtime CUDA detection**: `libloading`-based dynamic loader with `OnceLock` cache; zero-cost when no GPU present (`ailake-index`)
- **NVIDIA GPU backend** (`nvidia_impl`): replaces `candle-core` direct dep — loaded at runtime via `libloading` from system CUDA libs
- **AMD ROCm backend** (`rocm_impl`): hipBLAS SGEMM compute path; auto-detected alongside CUDA
- **GPU batch search**: `try_gpu_search_batch` in `SearchSession`; falls back to rayon CPU if GPU unavailable
- **MemTable write buffer**: `MemTable` accumulates rows in memory before HNSW/IVF-PQ build, enabling streaming ingestion (`ailake-query`)
- **Multi-vector column support**: `List<FixedSizeBinary>` Parquet encoding for parallel vector columns (`ailake-parquet`)
- **AVX-512 + FMA + F16C SIMD**: extended distance kernels on top of existing AVX2/NEON paths (`ailake-vec`)
- **`ailake-cli`**: binary crate with `create`, `insert`, `search`, `compact`, `info` subcommands; `--store` global flag accepts `s3://`, `gs://`, `az://`, local paths
- **Cloud credential builders**: typed `S3Config`/`S3Credentials`, `GcsConfig`/`GcsCredentials`, `AzureConfig`/`AzureCredentials` with feature-gated modules (`store-s3`, `store-gcs`, `store-azure`)
  - S3: static keys, WebIdentity (IRSA), EC2 IMDSv2, or full default chain
  - GCS: service account file/JSON, Application Default Credentials (Workload Identity)
  - Azure: client secret, Managed Identity (system/user-assigned), access key, SAS token, Azure CLI
- **`store_from_url()`**: zero-config URL dispatch — infers provider and credentials from scheme + env vars
- **Dual license**: MIT + Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`)
- **`ailake-auto` bench engine**: benchmark harness selects index automatically, matching production behavior
- **`CHANGELOG.md`**: release notes for all versions (this file)

### Changed
- CUDA backend decoupled from compile-time `candle-core` dep — now fully runtime-loaded via `libloading`
- Compaction job uses adaptive index selection instead of always rebuilding HNSW

### Fixed
- Redundant closure replaced with fn pointer in `PQCodebook::train` (clippy)
- `cargo fmt` violations in `ivf_pq`, `pq`, `writer`, `lib`, `gpu`, `scanner`, `ailake-cli`, `ailake-store`

---

## [0.4.0] — 2026-05-21

### Added
- **HNSW graph search**: `SearchSession` with configurable `ef` parameter and layered graph traversal (`ailake-index`)
- **Parallel HNSW build**: multi-threaded index construction via `rayon`; `ef_construction` default raised to 150
- **Deferred HNSW indexing**: build index after all row groups are written, avoiding partial-write inconsistencies
- **Generation bitmap + contiguous vector storage**: tighter memory layout in HNSW nodes reduces cache misses (~15% speedup)
- **AVX2 + NEON SIMD**: hand-written distance kernels for dot product, Euclidean, and cosine — x86-64 and AArch64 (`ailake-vec`)
- **GPU search with CPU fallback**: `candle-core` + CUDA backend; auto-detects GPU, falls back to `rayon` parallel CPU scan
- **Automatic PQ reranking**: after approximate HNSW/PQ search, re-scores top candidates with exact F32 distances
- **Flink connector**: `VectorScanSource` + `VectorScanTableFactory` for Apache Flink streaming ingestion (`ailake-jni`)
- **Extended JNI C-ABI**: additional entry points — `ailake_search_filtered`, `ailake_get_stats`, `ailake_compact`
- **Multi-engine benchmarks**: LanceDB, pgvector, Deep Lake comparison suite with `criterion` (`ailake-bench`)
- **Public format specification**: `docs/architecture/FILE_FORMAT.md` v1 — normative description of the binary layout

### Changed
- HNSW prefetch hints (`std::hint::prefetch_read`) inserted in graph traversal hot path
- Small Neighbor Heuristic (SNH) replaces simple distance sort during layer construction

### Fixed
- Unused `RowId` import in `ailake-index` (CI clippy)
- `&mut Vec` → `&mut [u64]` clippy::ptr_arg in bench
- Spurious `mut` on Parquet reader (unused-mut CI error)
- `too_many_arguments` clippy lint in JNI bindings

---

## [0.3.0] — 2026-05-19

### Added
- **Trino `VectorScanConnector`**: full Trino plugin with `VectorScanMetadata`, `VectorScanSplitManager`, and `VectorScanRecordSetProvider` (`ailake-jni`)
- **Spark `VectorScanStrategy`**: custom `SparkStrategy` that injects a `VectorScanExec` physical node into the query plan
- **Multi-column vector support**: tables can declare multiple vector columns (e.g. `embedding` + `context_embedding`); each generates its own HNSW in the file footer
- **`ailake-jni` uniffi bindings**: full C-ABI layer exposing `write_batch`, `search`, `compact`, `assemble_context` to JVM/Kotlin/Swift callers
- **`RestCatalog`**: Iceberg REST Catalog client for multi-cloud catalog federation (`ailake-catalog`)
- **`DatabricksAuth`** + config builders for `databricks_azure`, `databricks_aws`, `databricks_gcp` — Unity Catalog integration
- **`NessieCatalog`**, **`JdbcCatalog`**, **`GlueCatalog`**: three additional catalog backends (`ailake-catalog`)
- JVM plugin setup guides: step-by-step Trino and Spark integration docs (`docs/integrations/`)

### Changed
- Manifest Avro entries extended to carry `ailake.vector_columns` (JSON array) when multiple vector columns are present

---

## [0.2.0] — 2026-05-19

### Added
- **`ailake-store`**: unified object storage abstraction over S3, GCS, Azure Blob, and local filesystem via `object_store` 0.10
  - `S3Config` / `S3Credentials` — static keys, WebIdentity (IRSA), IMDSv2 instance profile, or full default chain
  - `GcsConfig` / `GcsCredentials` — service account file, inline JSON, or Application Default Credentials (Workload Identity)
  - `AzureConfig` / `AzureCredentials` — client secret, Managed Identity (system/user-assigned), access key, SAS token, Azure CLI
  - `store_from_url()` — zero-config URL-based dispatch (`s3://`, `gs://`, `az://`, `file://`)
  - Cargo feature flags: `store-s3`, `store-gcs`, `store-azure` (individually opt-in)
- **Async compaction**: `CompactionPlanner` identifies small files; `CompactionExecutor` merges and rewrites with fresh HNSW
- **Product Quantization (PQ)**: `PQCodebook` with k-means++ training and Asymmetric Distance Computation (ADC) for 32–128× vector compression (`ailake-vec`)
- **`BlockCompressor`**: zstd/lz4 block compression layer for raw vector blobs
- **Geometric pruning**: `VectorPruner` reads per-file centroid + radius from Iceberg manifest properties; prunes without opening Parquet
- **`ContextAssembler`**: deduplication, document grouping, token-budget allocation, XML rendering for LLM context windows (`ailake-query`)
- **PyO3 bindings** (`ailake-py`): `TableWriter`, `search()`, `assemble_context()` — returns zero-copy PyArrow `RecordBatch`

### Changed
- Parquet writer now records `ailake.centroid` and `ailake.radius` as base64-encoded custom properties in Iceberg manifest entries

---

## [0.1.0] — 2026-05-18

### Added
- **AI-Lake file format**: self-contained Parquet file carrying row group data, HNSW graph, and centroid in a single physical file
  - Binary layout: `PAR1` header → columnar row groups → AILK section (64-byte header + centroid + HNSW bytes) → Parquet footer → `PAR1`
  - HNSW section is invisible to standard Parquet readers; `ailake.footer_offset` key in Parquet file metadata bootstraps the AI-Lake reader
- **`ailake-core`**: base types — `VectorColumn`, `VectorMetric` (cosine / dot / euclidean), `LlmContextSchema`, `RowId`
- **`ailake-parquet`**: Parquet reader/writer with `FIXED_LEN_BYTE_ARRAY` vector column and custom field metadata
- **`ailake-vec`**: scalar quantization pipeline — F32 → F16 → I8 symmetric; `VectorPrecision` enum
- **`ailake-index`**: HNSW construction via `hnsw_rs`; `bincode` serialization; integrity check (`parquet_record_count == hnsw_graph.node_count`)
- **`ailake-file`**: unified writer/reader — atomic single-pass write; mmap-based HNSW loading via `memmap2` + tempfile
- **`ailake-catalog`**: Iceberg Spec v2 `metadata.json` writer + Avro manifest; custom `ailake.*` properties on snapshot entries
- **`ailake-query`**: `VectorScanner` — parallel file scan with `tokio`, partial S3 GET for footer/HNSW, global top-k merge
- Criterion benchmark for write throughput (`ailake-file/benches/write.rs`)
- `SETUP.md` with local filesystem quickstart guide

### Fixed
- AILK section placement corrected — lives between row groups and Parquet footer, not after `PAR1` trailer
- Clippy and `rustfmt` clean baseline for CI

---

[0.5.0]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/releases/tag/v0.1.0
