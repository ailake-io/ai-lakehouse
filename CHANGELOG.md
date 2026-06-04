# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- **`ailake-catalog/src/hadoop.rs`**: `HadoopCatalog::commit_snapshot` for `Replace`/`Overwrite` operations no longer inherits manifests from previous snapshots — new manifest IS the complete state. Previously, all operations unconditionally appended to the manifest list, causing `list_files` to return duplicate `DataFileEntry` records. With 10 concurrent deferred HNSW background tasks all racing to commit `Replace` snapshots, the accumulated duplicates prevented `IndexStatus::Ready` entries from reaching the `ready >= num_shards` threshold, causing the bench to block indefinitely.
- **`ailake-vec/src/pq.rs`**: `kmeans_pp_init` complexity reduced from O(n × k²) to O(n × k) by maintaining an incremental `min_dist` array instead of recomputing all distances from scratch at each step. With n=100k, k=256: 3.2B → 25M distance computations for the init phase alone — **17× end-to-end write speedup** on SIFT-1M IVF-PQ benchmark (96s → 5.7s for 10k vectors).
- **`ailake-bench/src/main.rs`**: `--engine ailake-ivf-pq` now derives `nlist`/`nprobe` from `IvfPqConfig::for_dataset(dim, shard_size)` when CLI args are left at default (0). Previous hardcoded defaults `nlist=256 nprobe=8` were calibrated for ~65k-vector datasets; with 100k vectors/shard `nprobe=8/256=3.1%` scan coverage produced `Recall@10=0.32`.
- **`ailake-bench/src/main.rs`**: IVF-PQ multi-shard search now loads raw vectors (`load_with_raw=true`) and sets `rerank_factor=Some(3)`. Per-shard PQ codebooks produce ADC distances on different scales — cross-shard merge sorted by incomparable approximations, causing `Recall@10=0.32` even with correct nlist/nprobe. Exact reranking with true L2² distances corrects the merge step.

### Added
- **`VectorMetric::NormalizedCosine` (value `3`) + `VectorStoragePolicy::pre_normalize`**: New fast-path distance metric for cosine workloads. When `pre_normalize = true`, vectors are normalized to unit L2 at write time and HNSW uses `1 - dot(a, b)` instead of full cosine — eliminates the `sqrt` of norms from every edge traversal (~12–20% faster search on dim=1536). Query vectors are automatically normalized at search time in all bindings — callers need no changes. Exposed via `ailake create --pre-normalize` (CLI), `TableWriter(pre_normalize=True)` (Python), `MetricNormalizedCosine` (Go), and `Metric::NormalizedCosine` (C++). All metric match arms updated across `gpu`, `ivf_pq`, `serialize`, `pruner`, `scanner`, `parquet schema`, `footer`, and `reader`.
- **`ailake-index/src/ivf_pq.rs`**: `IvfPqCodebook` struct — sharable coarse quantizer + PQ codebook trainable once and reused across all shards. New methods: `IvfPqIndex::train_codebook(vectors, metric, config) -> IvfPqCodebook` (k-means only, no inverted lists) and `IvfPqIndex::build_with_codebook(row_ids, vectors, codebook) -> IvfPqIndex` (assign + encode, no k-means). When all shards share the same codebook, ADC distances are numerically comparable across shards — cross-shard merge is correct without exact reranking.
- **`ailake-file/src/writer.rs`**: `AilakeFileWriter::with_shared_ivf_codebook(Arc<IvfPqCodebook>)` builder — bypasses k-means training and calls `IvfPqIndex::build_with_codebook` instead of `IvfPqIndex::train`.
- **`ailake-query/src/writer.rs`**: `TableWriter::write_batch_ivf_pq_deferred` — async variant of `write_batch_ivf_pq`. Persists Parquet immediately (~200k vec/s, same as HNSW deferred), spawns background tokio task to train IVF-PQ index, rewrite file with AILK section, and transition `IndexStatus::Indexing → Ready`. Shared codebook is coordinated via `Arc<tokio::sync::OnceCell<IvfPqCodebook>>` — first task trains, all others await and skip k-means.
- **`ailake-query/src/writer.rs`**: `TableWriter` now caches `cached_ivf_codebook: Option<Arc<IvfPqCodebook>>` (synchronous path) and `deferred_ivf_codebook: Arc<tokio::sync::OnceCell<IvfPqCodebook>>` (deferred path).
- **`ailake-bench/src/main.rs`**: new `--engine ailake-ivf-pq-deferred` — exercises `write_batch_ivf_pq_deferred`, waits for `IndexStatus::Ready`, searches with `rerank_factor=3`.

### Changed
- **`ailake-vec/src/pq.rs`**: k-means assignment loop now uses `rayon::par_iter()` — parallel assignment across all CPU cores. `kmeans_pp_init` initial and incremental distance computations also parallelized via `par_iter`/`par_iter_mut`.
- **`ailake-vec/Cargo.toml`**: added `rayon` workspace dependency.
- **`ailake-index/src/ivf_pq.rs`**: `IvfPqConfig::for_dataset` now sets `nprobe = nlist/4` (25% coverage) instead of `nlist/8` (12.3%) — better candidate quality per shard, needed alongside reranking for `Recall@10 ≥ 0.90`.

### Fixed
- **`ailake-py/src/lib.rs`**: `local_catalog_store` now passes `file://{canonical_path}` as warehouse to `HadoopCatalog` so Iceberg `metadata.json` writes absolute `file://` URIs for `location` and manifest paths — required by Trino's Iceberg connector
- **`ailake-store/src/local.rs`**: `LocalStore::full_path` strips `file://` prefix before `PathBuf::join` so absolute `file://` URIs resolve correctly on the local filesystem
- **`tests/docker/compose-demo.yml`**: 9 DX issues fixed in demo stack — Trino 446 Nessie catalog (hadoop type removed in 400+), correct property names (`default-warehouse-dir`, `ref`), removed `:ro` on Trino volume (blocked `/data/trino/var`), BQ emulator healthcheck uses `bash /dev/tcp` (no curl in image), BQ host port 19050 (avoids Tor default 9050), Nessie registration uses real snapshot/schema IDs, direct Nessie API v1 via `urllib` (pyiceberg dropped nessie catalog in 0.8+), SQL `"table"` quoted in notebook 04 (reserved keyword in Trino)
- **`tests/parquet_trailing_bytes.rs`**: `pyarrow_ignores_ailake_footer` de-ignored — PyArrow 24.0.0 available

### Changed
- **`tests/docker/compose-demo.yml`**: Trino and BigQuery emulator moved to `profiles: ["engines"]`; `compose-demo-engines.yml` overlay deleted — single-file command: `docker compose -f compose-demo.yml --profile engines up -d`

### Added
- **`ailake-index/src/gpu.rs`**: 3 GPU unit tests gated on `AILAKE_GPU_BACKEND` env var — `gpu_search_batch_cosine_top1_exact` (cosine SGEMM, top-1 == query), `gpu_search_batch_euclidean_top1_exact` (euclidean SGEMM, dist-to-self ≈ 0), `gpu_kmeans_returns_k_centroids` (k-means produces k centroids of correct dim); skip silently when `AILAKE_GPU_BACKEND=none`
- **`ailake-index/tests/gpu_data.rs`**: 3 GPU data integration tests fired against realistic synthetic datasets — `gpu_search_recall_vs_cpu_baseline` (2 000 vecs × dim 128, 20 queries, recall@10 ≥ 99% vs CPU brute-force), `gpu_search_exact_hit_in_large_db` (5 000 vecs × dim 64, query == db[1337], top-1 exact match), `gpu_kmeans_converges_on_clustered_data` (8 clusters × 50 vecs × dim 32, each centroid maps unique cluster within ε = 1.0); all skip when `AILAKE_GPU_BACKEND=none`
- **`ci-gpu-data.yml`**: new `workflow_dispatch` workflow — runs `cargo test -p ailake-index --test gpu_data` on `[self-hosted, Windows, X64]` runner with CUDA or ROCm; same DLL-detection logic as `ci-gpu.yml`
- **`docs/specs/FILE_FORMAT.md`**: added §15 "Bincode v1 Wire Format (Language-Agnostic)" — encoding rules table + field-by-field byte layout for HnswSnapshot and IvfPqSnapshot so any language can decode the index blob without the Rust crate; added §16 "Cross-Language Implementations" — Rust/C++/Go comparison table and language-agnostic 10-step bootstrap sequence
- **`ailake-cpp/CMakeLists.txt`**: added `SPDX-License-Identifier: MIT OR Apache-2.0` header and inline licensing note — NVIDIA CUDA Toolkit (`-DAILAKE_CUDA=ON`) and AMD ROCm are third-party proprietary SDKs not bundled by default; binary distributors must comply with vendor EULAs
- **`SETUP.md`**: added "Licensing note — third-party GPU SDKs" table in section 8F documenting NVIDIA/AMD SDK ownership, licenses, and per-language binding strategy (runtime dlopen vs. opt-in static link for C++)
- **`README.md`**: added "Interactive demo" section with `docker compose up -d` quick start, notebook table, and engines profile (`--profile engines`) command; updated repository layout to include all `tests/docker/demo/` files
- **`SETUP.md`**: added "Fastest path — Docker demo" section at the top pointing to `compose-demo.yml` and engines profile (`--profile engines`)
- **`docs/contributing/TESTING.md`**: added `index-cpu-fallback` job to `ci.yml` matrix; added `ci-gpu.yml` workflow section (Windows self-hosted GPU runner); updated `secret-scan.yml` note to document that automatic triggers are disabled while repo is private
- **`tests/docker/compose-demo.yml` `engines` profile**: Trino 446 + BigQuery emulator added as optional services under `--profile engines`; activated with `docker compose -f compose-demo.yml --profile engines up -d`
- **`tests/docker/demo/trino-catalog/ailake.properties`**: Trino Iceberg HadoopCatalog config pointing at the demo-data volume (`file:///data/ailake_demo`)
- **`tests/docker/demo/notebooks/02_duckdb.ipynb`**: DuckDB demo — direct Parquet glob scan, filtered queries, aggregations, embedding as BLOB, optional Iceberg extension
- **`tests/docker/demo/notebooks/03_spark.ipynb`**: Spark demo — PySpark local[*] mode (no cluster), direct Parquet read, Iceberg HadoopCatalog SQL, snapshot history
- **`tests/docker/demo/notebooks/04_trino.ipynb`**: Trino demo — connection via `trino` Python driver, schema/catalog discovery, SQL queries, `$snapshots` and `$files` Iceberg system tables
- **`tests/docker/demo/notebooks/05_bigquery.ipynb`**: BigQuery demo — PyArrow reads AI-Lake Parquet, streaming inserts to BQ emulator, SQL queries and schema inspection
- **`tests/docker/demo/Dockerfile`**: added `pyspark`, `trino`, `google-cloud-bigquery`, and `google-auth` packages
- **`tests/docker/compose-demo.yml`**: single-command onboarding demo (`docker compose up -d`) — starts MinIO, Nessie, and a JupyterLab container pre-loaded with 500 synthetic documents; `ailake-py` wheel is built from source via maturin on first run and cached by Docker layer cache
- **`tests/docker/demo/Dockerfile`**: two-stage build — stage 1 compiles the ailake-py wheel with Rust + maturin; stage 2 installs JupyterLab, pyiceberg, DuckDB, and the wheel
- **`tests/docker/demo/init_demo.py`**: fixture generator run at container startup; writes 500 documents (dim=16, cosine, F16) using `ailake.TableWriter` and persists a demo query vector; idempotent (skips if table already present)
- **`tests/docker/demo/notebooks/01_ailake_demo.ipynb`**: interactive demo notebook covering vector search, PyIceberg compatibility, DuckDB SQL scan, RAG context assembly, and optional MinIO S3 upload/read
- **`CONTRIBUTING.md`**: expanded from minimal stub to full contributor guide — prerequisites table (Rust/JDK/Gradle/Python/maturin/cargo-deny), per-component setup steps (Rust workspace, ailake-py, JVM plugins, Go, C++), test commands per language, code style gates, branch/commit/CHANGELOG strategy, PR workflow, and issue reporting
- **`.github/ISSUE_TEMPLATE/bug_report.yml`**: added `engine_versions` field for exact Spark/Trino/Flink/Python/Java versions; made `logs` field required; added per-engine instructions for capturing backtraces and stack traces (`RUST_BACKTRACE=1`, `RUST_LOG=debug`, JVM full stack trace, Python traceback)
- **`ci.yml`**: added `index-cpu-fallback` job — runs `ailake-index` tests on a CPU-only Linux runner, verifying that `hardware::detect_backend()` returns `CpuSimd` and all index tests pass without CUDA/ROCm libraries present
- **`ci-gpu.yml`**: new workflow (`workflow_dispatch`) — runs `ailake-index` GPU tests on `[self-hosted, Windows, X64]` runner; detects CUDA (`cudart64_*.dll` + `cublas64_*.dll`) or ROCm (`amdhip64.dll` + `hipblas.dll`) at runtime and reports backend selected
- **`publish-pypi.yml`**: added `workflow_run` trigger so PyPI publish runs automatically after the `Release` workflow succeeds on `main`; added `guard` job that aborts the pipeline when triggered by a failed release; all build jobs (`linux`, `windows`, `sdist`) now depend on `guard`
- **CI**: `publish-jvm.yml` — manual workflow that builds Spark/Trino/Flink fat-JARs + `libailake_jni.so` and uploads them to an existing GitHub Release
- **`airflow-providers-ailake/README.md`**: PyPI package page with install instructions, hook/operator/sensor usage, and requirements
- **`docs/contributing/TESTING.md`**: manual Actions trigger order table (pre-release checklist, steps 1–8)
- **CI**: all workflows opt into Node.js 24 via `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` — removes Node.js 20 deprecation warning ahead of forced switch on 2026-06-02

### Changed
- **`actions/checkout`**: bumped from `@v4` to `@v6` across all 9 workflows — eliminates Node.js 20 deprecation warning introduced by GitHub's September 2025 runner update

### Changed
- **`publish-pypi.yml`**: replaced deprecated `maturin upload` with `twine` (`maturin upload/publish` removed per PyO3/maturin#2334)
- **`publish-pypi.yml`**: release tag now read from `ailake-core/Cargo.toml` (single source of truth); previously read from `ailake-py/pyproject.toml` which caused version drift
- **`ailake-py/pyproject.toml`**: replaced static `version` field with `dynamic = ["version"]` — maturin reads version from `Cargo.toml` at build time, eliminating manual sync

### Fixed
- **`ci-gpu.yml`**: PowerShell DLL detection replaced `Get-Command` (rejects non-executables) with `Find-Dll` helper using `Test-Path` across `$env:PATH` entries — fixes `ArgumentList parameter can be specified only when retrieving a single cmdlet` error on Windows runner
- **`spark-plugin/src/main/scala/io/ailake/spark/AilakeNative.scala`**: resolved SLF4J overload ambiguity (`error(String, Any, Any)` vs `error(String, Object*)`) in Scala 2.12 by replacing format-string calls with string interpolation (`s"..."`) for all three affected logger statements
- **`trino-plugin/build.gradle.kts`**: added `compileOnly("org.slf4j:slf4j-api:2.0.9")` — `trino-spi` is `compileOnly` so its transitive SLF4J dependency was absent from the compile classpath, causing `Unresolved reference: LoggerFactory`
- **`trino-plugin/build.gradle.kts`**: added `testRuntimeOnly("org.slf4j:slf4j-simple:2.0.9")` — `compileOnly` does not populate the test runtime classpath; `AilakeNative` object initialization was failing at test time with `NoClassDefFoundError: org/slf4j/LoggerFactory`, cascading to 4 test failures (`AilakeNativeTest` × 3 + `AilakeNativeIntegrationTest`)
- **`ailake-bench/Cargo.toml`**: added missing `repository` field — was the only crate of 13 without it
- **Cargo formatting**: applied `cargo fmt --all` across `ailake-index/src/mmap_loader.rs`, `ailake-jni/src/lib.rs`, `ailake-py/src/lib.rs`, `ailake-query/src/writer.rs` to fix CI `fmt` check failures
- **`ailake-py/pyproject.toml`**: version was stuck at `0.0.8`; bumped to `0.0.10` to match workspace before switching to dynamic version
- **`publish-pypi.yml`**: `actions/checkout@v4` was placed after `dist/` was populated, causing it to wipe the downloaded wheels; moved checkout before download steps
- **`publish-pypi.yml`**: Docker pre-checkout cleanup must run before `actions/checkout@v4` to avoid EACCES on root-owned files left by maturin build jobs; added `if: always()` cleanup at end of publish job to prevent workspace pollution for subsequent workflows
- **`secret-scan.yml`**: added pre-checkout Docker cleanup to handle root-owned `dist/` files left by previous publish-pypi runs

---

## [0.0.10] - 2026-05-29

### Added
- **CI**: TruffleHog secret scanning on every push and PR (`secret-scan.yml`)
- **CI**: Go build + vet for `ailake-go` (`ci-go.yml`)
- **CI**: C++17 cmake build for `ailake-cpp` (`ci-cpp.yml`)
- **CI**: `bench-build` job validates `ailake-bench` compiles
- **`ailake-cpp/src`**: `catalog.cpp` and `search.cpp` compilation units

### Changed
- **pyo3**: upgraded `0.22 → 0.24`; fixes RUSTSEC-2025-0020 (PyString buffer overflow)
- **sqlx**: upgraded `0.7 → 0.8`; fixes RUSTSEC-2024-0363 (protocol truncation SQL injection); feature `runtime-tokio-rustls` split into `runtime-tokio` + `tls-rustls`
- **`deny.toml`**: added `0BSD`, `BSL-1.0`, `MPL-2.0`, `CDLA-Permissive-2.0` to license allow-list; skipped unfixable transitive advisories (bincode, encoding, paste, rustls-pemfile, rsa, rustls-webpki)
- **Airflow provider tests**: removed `hook.log = MagicMock()` — `BaseHook.log` is read-only in Airflow 2.x and 3.x

### Fixed
- `cargo fmt` violations in `ailake-catalog/src/avro_manifest.rs`, `ailake-cli/src/main.rs`, `ailake-cli/src/serve.rs`

---

## [0.0.9] - 2026-05-28

### Changed
- **`ailake-jni` dead uniffi code removed**: `uniffi::setup_scaffolding!()`, `#[uniffi::export]` on `vector_search`/`assemble_context`, `#[derive(uniffi::Record)]` on `RowResult`, and `uniffi = "0.27"` workspace dep all removed. All JVM plugins use `ailake_search_json` C-ABI via JNA — uniffi was declared but generated no bindings and no plugin consumed it.
- **Workspace `Cargo.toml`**: `uniffi = "0.27"` removed from `[workspace.dependencies]` (no crate depends on it).
- **Trino plugin**: `VectorScanSplit` field `queryVector` (CSV String) → `queryBytes` (Base64 LE f32); CSV→Base64 conversion moved to planning phase (`VectorScanSplitManager.csvFloatsToBase64`) to eliminate 1536-element string split on every worker execution.
- **Spark plugin**: `AilakeNative` now uses direct `import com.fasterxml.jackson.databind.ObjectMapper` instead of `Class.forName(...)` reflection; single shared `ObjectMapper` instance (thread-safe); `jackson-databind` added as `compileOnly` dep in `build.gradle.kts`.

### Added
- **`ailake-jni/README.md`**: crates.io page with all 4 C-ABI exports, request/response JSON schemas, Kotlin + Scala JNA usage examples, library path setup.
- **`ailake-py/README.md`**: PyPI page with install, `TableWriter`, `search`, `assemble_context`, Iceberg compatibility matrix.
- **GitHub repository topics**: `lakehouse`, `iceberg`, `vector-search`, `rust`, `embeddings`, `rag`, `hnsw`, `parquet`.

---

## [0.0.8] - 2026-05-27

### Added
- **`compat-heavy.yml` BigQuery job**: validates that AI-Lake Parquet files are readable by BigQuery without errors from the AILK footer section. Uses `fsouza/fake-gcs-server:1.47.2` to simulate GCS and `ghcr.io/goccy/bigquery-emulator:0.6.6` (Go-based, compliant Parquet reader); `STORAGE_EMULATOR_HOST` wires the emulator's internal GCS client to fake-gcs so no real GCP credentials are required. Test creates a BigQuery external Parquet table pointing to the fixture files and asserts row count, schema (`id`, `text`, `embedding`), and id range.

### Fixed
- **`AilakeNative.search` double-free fixed**: `ptr` was freed in the success path before `mapper.readValue` ran; if `readValue` threw (e.g., error-response JSON is an object, not an array), the `catch` block freed `ptr` a second time — `free(): double free detected in tcache 2` killed the JVM after the integration test passed. Fixed by moving `ailake_free_string` to a `finally` block so `ptr` is freed exactly once regardless of parse outcome.
- **`compat-bigquery` drops file upload, uses pyarrow + streaming inserts**: both `load_table_from_file()` (resumable upload — emulator resets connection on chunk PUT with `ConnectionResetError 104`) and `uploadType=multipart` (emulator returns 500) are broken in `goccy/bigquery-emulator` 0.6.6. The verification step now has two explicit stages: (1) **pyarrow reads all AILK Parquet files** — validates that the AILK footer appended after PAR1 does not break a standard Parquet reader (the same guarantee required for BQ compatibility); (2) **BQ emulator streaming inserts** (`insertAll` API) load the rows (id, text, embedding as base64 BYTES), followed by `SELECT COUNT(*)`, schema inspection, and `MIN/MAX(id)` queries — validates BQ SQL and schema compat. The `insertAll` endpoint is the reliably-supported write path in the emulator.
- **`compat-bigquery` Python verification step fixed**: (1) `python3 -u -` forces unbuffered stdout so logs appear immediately; (2) `BIGQUERY_EMULATOR_HOST` set to `host:port` format (no `http://` scheme — the client adds it) and set before importing the BigQuery library; (3) explicit `ClientOptions(api_endpoint=...)` passed to `bigquery.Client` as belt-and-suspenders; (4) all API calls have explicit `timeout=` parameters; (5) wait loop fails loudly with `exit 1` if a service never becomes ready; (6) BQ emulator health check uses TCP-connect-only curl (accepts any HTTP status, not `-f`) since `/` returns non-200 on a fresh emulator.
- **`compat-bigquery` uses random host port for BigQuery emulator**: fixed "address already in use" on port 9050 by switching to `127.0.0.1::PORT` (Docker-assigned random host port); actual port captured via `docker port` and exported as `BQ_EMULATOR_PORT`. Eliminates conflicts between concurrent workflow runs and system services on self-hosted runners.
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
- **`ailake-jni` C-ABI layer**: `ailake_search_json`, `ailake_write_batch_json`, `ailake_free_string` — JSON-envelope API consumed by all JVM plugins via JNA
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

[0.0.9]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/ThiagoLange/ai-lakehouse/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ThiagoLange/ai-lakehouse/releases/tag/v0.0.1
