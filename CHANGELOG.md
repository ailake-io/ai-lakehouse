# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **`compat-heavy.yml` BigQuery job**: validates that AI-Lake Parquet files are readable by BigQuery without errors from the AILK footer section. Uses `fsouza/fake-gcs-server:1.47.2` to simulate GCS and `ghcr.io/goccy/bigquery-emulator:0.6.6` (Go-based, compliant Parquet reader); `STORAGE_EMULATOR_HOST` wires the emulator's internal GCS client to fake-gcs so no real GCP credentials are required. Test creates a BigQuery external Parquet table pointing to the fixture files and asserts row count, schema (`id`, `text`, `embedding`), and id range.

### Fixed
- **`ailake-jni` global static Tokio runtime**: `rt()` previously created a new multi-threaded Tokio runtime on every JNA call and dropped it on return; repeated creation/destruction of the runtime's OS thread pool conflicted with the JVM's signal handlers on Linux, causing SIGABRT (exit code 134) in `compat-jvm-plugins`. Runtime is now created once via `OnceLock` and reused for the process lifetime; falls back to a single-threaded runtime if multi-thread init fails.
- **`VectorScanRecordSetTest` uses `getCompletedBytes()` not `getTotalBytes()`**: `RecordCursor.getTotalBytes()` was removed in Trino SPI 430; test call site at line 81 missed in previous fix passes.
- **`trino-plugin` compiles with Trino SPI 430**: `isRemotelyAccessible()` is now abstract in `ConnectorSplit` — added `override fun isRemotelyAccessible(): Boolean = true` to `VectorScanSplit`; `getSplitInfo()` removed — replaced with no-op; `getTotalBytes()` renamed to `getCompletedBytes()` in `RecordCursor`. Follow-up: `ConnectorSplit` added two more abstract methods in 430 — `getAddresses()` (returns `List<HostAddress>`, returns `emptyList()` since native lib handles file-level parallelism) and `getInfo()` (returns `Any?`, returns `null`); `RecordCursor` added abstract `getReadTimeNanos()` — returns `0L`; `ConnectorSplitManager.getSplits` signature in 430 requires `Constraint` as 5th parameter (previous fix incorrectly removed it) — re-added with `Constraint.alwaysTrue()` in tests.
- **`spark-plugin` compiles with Scala 2.12 + Spark 3.5**: removed unused `scala.jdk.CollectionConverters` and `scala.util.Using` imports (Scala 2.13-only); replaced `Dataset[Row](spark, plan)(RowEncoder(schema))` with `createDataFrame` (both `Dataset.apply(LogicalPlan)` and `RowEncoder.apply(StructType)` are private/removed in Spark 3.5); test now uses `spark.sessionState.executePlan(plan).executedPlan` instead of private Spark APIs.
- **`ailake-py` uses `openssl-sys[vendored]` directly**: replaced `openssl = { features = ["vendored"] }` with `openssl-sys = { version = "0.9", features = ["vendored"] }` to target the exact package that was failing in manylinux containers. The indirect path through the `openssl` wrapper crate was not reliably propagating the `vendored` feature to `openssl-sys` during Cargo resolution inside the manylinux Docker container.
- **`publish-pypi.yml` Linux job removes conflicting `OPENSSL_DIR`/`OPENSSL_LIB_DIR`/`OPENSSL_INCLUDE_DIR`/`OPENSSL_STATIC` env vars**: these vars never reached the manylinux Docker container (confirmed by `OPENSSL_DIR unset` in build log) and would have overridden vendored mode if they had, causing pkg-config lookup against a path with no OpenSSL dev headers.
- **`publish-pypi.yml` `before-script-linux` installs `perl-core make gcc`**: these are the build tools required for vendored OpenSSL compilation (`openssl-src` runs OpenSSL's configure + make internally). Replaced the previous `openssl-devel openssl-static` install which is only needed for system (non-vendored) OpenSSL.

---

## [0.0.7] - 2026-05-25

### Fixed
- **`ailake_search_json` / `ailake_vector_search_json` now surfaces errors**: `do_search` previously used `unwrap_or_default()`, silently converting any internal error (Avro parse failure, path resolution issue, HNSW load error) into empty results and `{"ok":true,"results":[]}`. Both C-ABI functions now return `{"ok":false,"error":"..."}` on failure so callers see the actual root cause.
- **`avx512::hsum512` no longer uses `_mm512_reduce_add_ps`**: that intrinsic was stabilized in Rust 1.89 and caused `exit status: 101` in older manylinux Docker containers used by `maturin-action`. Replaced with a store-and-reload reduction using only `avx512f` + `avx` intrinsics (stable since Rust 1.27/1.72).
- **`publish-pypi.yml` Linux job now pins `rust-toolchain: stable`**: maturin-action's bundled Rust in the manylinux Docker can lag behind; pinning to stable ensures the same toolchain used in `ci.yml` / `release.yml`.
- **`publish-pypi.yml` all build jobs set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`**: macOS runners ship Homebrew Python 3.14 which `--find-interpreter` picks up; PyO3 0.22 caps at 3.13 and fails with "interpreter version newer than maximum supported". The env var builds against the stable ABI (abi3), which is forward-compatible with any 3.x version.
- **`reqwest` workspace dependency uses `native-tls-vendored`**: the previous `features = ["json"]` without `default-features = false` enabled `default-tls` → system `openssl-sys`, failing in manylinux (no `openssl.pc`). `rustls-tls` was tried next but iceberg-rust 0.3 (transitive dep) re-introduces `native-tls` via feature unification, so system OpenSSL was still required. Final fix: `default-features = false, features = ["json", "native-tls-vendored"]` — compiles OpenSSL from source via `openssl-src` (needs only gcc/make/perl, all present in manylinux containers).
- **`ailake-vec` AVX-512 kernels gated behind `avx512` Cargo feature**: all `_mm512_*` intrinsics were stabilised in Rust 1.89 and caused `exit status: 101` in the manylinux Docker container whose bundled Rust predates that release. The `avx512` feature is opt-in and disabled by default; manylinux / PyPI builds compile and fall through to the AVX2 kernels. Enable with `--features ailake-vec/avx512` on Rust ≥ 1.89.
- **`reqwest` removed from workspace dependencies; `ailake-catalog` uses inline `rustls-tls`**: the workspace-level `reqwest = { features = ["native-tls-vendored"] }` definition caused `openssl-sys` to appear in the workspace resolution graph even when `reqwest` was optional and unused in `ailake-py`. Removed `reqwest` from `[workspace.dependencies]` entirely and inlined `reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"], optional = true }` in `ailake-catalog`. `rustls-tls` is pure Rust — zero C/OpenSSL dependencies — eliminating `openssl-sys` from the manylinux build unconditionally.
- **`ailake-py` adds `openssl = { features = ["vendored"] }` for hermetic wheel builds**: manylinux and CI environments lack system OpenSSL headers; adding `openssl` with `vendored` feature forces `openssl-sys` to compile from source via `openssl-src` (requires only `cc`/`make`/`perl`, all present in manylinux containers). Cargo feature unification ensures any other transitive pull of `openssl-sys` also activates vendored mode, making wheel builds fully hermetic.
- **`publish-pypi.yml` Linux job sets `OPENSSL_DIR`/`OPENSSL_LIB_DIR`/`OPENSSL_INCLUDE_DIR`/`OPENSSL_STATIC` env vars**: maturin-action passes `env:` variables into the manylinux Docker container; setting these vars makes openssl-sys skip pkg-config lookup and link directly against system OpenSSL. `before-script-linux` installs `openssl-devel openssl-static` to ensure headers and static libs are present.

---

## [0.0.6] - 2026-05-25

### Added
- **Automatic Iceberg schema propagation on `commit()`**: `TableWriter.commit()` now calls `arrow_schema_to_iceberg_update` internally — no manual metadata patching required. The generated `IcebergSchemaUpdate` carries all Arrow fields (including vector columns) correctly typed as Iceberg types (`"long"`, `"string"`, `"bytes"`, `"timestamptz"`, `List`, `Struct`, `Map`), plus a complete `schema.name-mapping.default` so PyIceberg resolves Parquet columns by name when field-ids are absent.
- **`write_fixture` example simplified**: removed the 37-line manual metadata patch block; schema propagation is now entirely automatic via `commit()`.
- **`ailake-py` Python SDK compat test expanded**: `check_ailake_py.py` covers cosine search (self-distance ≈ 0), `top_k` enforcement, euclidean metric, multi-batch before commit, `assemble_context` with chunk presence, token budget enforcement, and `dedup_threshold` parameter acceptance; added error-path coverage (missing table → exception). CI job pins `python-version: '3.12'` and builds the wheel with `--interpreter python3.12`.
- **`HadoopCatalog` versioned metadata layout**: catalog now writes `vN.metadata.json` + `version-hint.text` instead of `current.json`, matching Iceberg Hadoop catalog spec and enabling `PyIceberg.StaticTable.from_metadata` to locate the current metadata file via the version hint
- **Absolute table location in metadata**: `create_table` now records the full absolute path as `location` (and `manifest-list` paths) when `write_fixture` passes the absolute warehouse path; PyIceberg and other readers can now resolve data file paths without additional config
- **Default schema entry in `IcebergMetadata`**: `schemas` array now includes `[{"schema-id": 0, "type": "struct", "fields": []}]` so PyIceberg's `StaticTable` does not fail with `current-schema-id 0 can't be found in the schemas` before reaching the manifest stage
- **Phase 2 Avro manifests — full PyIceberg `StaticTable.scan()` PASS**: replaced apache-avro 0.16 writer (strips `field-id` from schema JSON) with raw Avro OCF writer (`avro_raw.rs`) that embeds schema verbatim; manifest files now carry `logicalType: "map"` on map-typed fields and correct field-ids per Iceberg spec; `check_pyiceberg.py` reports `PASS (StaticTable)` with full scan of 1000 rows
- **PyPI publish workflow** (`.github/workflows/publish-pypi.yml`): builds `ailake` wheels on push of `v*` tags — Linux x86_64 + aarch64 (manylinux), macOS x86_64 + arm64, Windows x86_64, sdist; Python 3.9–3.13; publishes via `PYPI_API_TOKEN` secret
- **Version bump**: all crates `0.0.5` → `0.0.6`
- **README**: PyPI badge, `pip install ailake` snippet, `SETUP.md` link, workspace map updated with `databricks.rs`
- **`tests/tests/iceberg_compat.rs`**: three integration tests — `metadata_json_is_iceberg_spec_v2`, `parquet_files_have_valid_magic_and_ailake_section`, `data_files_referenced_in_metadata`
- **`ailake-cli` subcommands implemented**: `create`, `insert`, `search`, `compact`, `info` — wired to real engine calls
- **`ailake-py` re-enabled**: PyO3 bindings compile and pass `check_ailake_py.py` end-to-end
- **Compatibility test suite** (`tests/compat/`): `check_pyarrow.py`, `check_duckdb.py`, `check_pyiceberg.py`, `check_ailake_py.py`, `check_jni_cabi.py`; `write_fixture` example generates deterministic 1000-row fixture; Flink/Spark/Trino JNA integration tests in Gradle subprojects

### Fixed
- `HadoopCatalog::create_table`: `location` field was computed as `/{namespace}.db/{table}` (leading `/` with empty warehouse) instead of using `table_root()` — now consistent
- `iceberg_compat` integration tests: `find_json_named(..., "current.json")` replaced with `find_current_metadata()` that follows `version-hint.text` to locate the current `vN.metadata.json`
- `write_fixture` example: uses `fs::canonicalize` to pass absolute path as warehouse, fixing relative `location` field in generated fixture metadata
- `avro_manifest.rs`: `upper_bounds` field-id corrected 124 → 128; `key_metadata` field-id corrected 132 → 131 per Iceberg Spec v2 §4.1.7
- `avro_manifest.rs`: all six map-typed manifest fields (`column_sizes`, `value_counts`, `null_value_counts`, `nan_value_counts`, `lower_bounds`, `upper_bounds`) now carry `"logicalType": "map"` in the Avro schema so PyIceberg resolves them as `MapType` instead of `list`
- `avro_raw.rs`: removed trailing zero-count block terminator from `write_avro_container`; apache-avro Reader handles EOF at block-count read (clean) but errors on block-byte-count read after count=0 (UnexpectedEof)
- `HadoopCatalog::commit_snapshot`: data file paths are prefixed with warehouse root only when warehouse is an absolute path (starts with `/` or contains `://`); relative warehouse strings (used in unit tests) keep paths unchanged
- `ailake-jni`: `ailake_write_batch_json` used `write_batch_deferred` — background HNSW task raced with immediate search, producing empty results; switched to synchronous `write_batch`
- `ailake-query`: `scanner::search` now falls back to exact flat scan for `IndexStatus::Indexing` files and Parquet-only files missing the AILK footer, consistent with `SearchSession` behavior
- `ailake-py`: missing deps (`ailake-catalog`, `ailake-store`, `arrow-array`, `arrow-schema`) added to `Cargo.toml`; `HadoopCatalog::new` signature corrected; upgraded PyO3 0.21 → 0.22 (`Bound` API, Python 3.13 support); `maturin develop` replaced with `maturin build` + `pip install` in CI
- `ailake-catalog`: `HadoopCatalog::table_root()` with empty warehouse no longer produces absolute path

### Changed
- **`compat-heavy.yml` now triggers on `push: [main]` and weekly schedule** in addition to `workflow_dispatch`. Spark job upgraded to real Spark+Iceberg integration test (`iceberg-spark-runtime-3.5_2.12:1.5.2`). Trino job rewritten to use `tabulario/iceberg-rest:0.10.0` + `trinodb/trino:436` via Docker.
- `CLAUDE.md` roadmap: Phase 1 all items marked complete; Phase 4 extended with IVF-PQ, GPU, Flink, SIMD, MemTable items
- CI: added `compat-pyarrow`, `compat-duckdb`, `compat-pyiceberg`, `compat-ailake-py` jobs to `ci.yml`; Python pinned to 3.12 for wheel builds

---

## [0.0.5] — 2026-05-22

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

## [0.0.4] — 2026-05-21

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

## [0.0.3] — 2026-05-19

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

## [0.0.2] — 2026-05-19

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

## [0.0.1] — 2026-05-18

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

[Unreleased]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/releases/tag/v0.0.1
