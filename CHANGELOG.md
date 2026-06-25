# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed (demo)

- **`07_multimodal.ipynb` cell `ff4798c26ba84498`** — replaced invalid conditional `from ... import (...) if ... else ...` syntax (SyntaxError in CPython) with `try/except ImportError` guard.
- **`09_hybrid_search.ipynb` cells `cell-7`, `cell-9`** — `ailake.search()` returns `SearchQuery`, not an iterable; added `.to_list()` before iteration in both cells.
- **`10_gpu_demo.ipynb` cell `cell-10`** — `BENCH_PATH` pointed to `/data/gpu_bench_ivfpq` (never written); corrected to `/data/gpu_bench_deferred` (written in cell-8 via `write_batch_auto_deferred`).
- **`12_airflow.ipynb` cell `read-back-code`** — `search_text()` returns dicts with `distance` field (negated BM25 score), not `score`; fixed `h.get('score', 0)` → `-h.get('distance', 0)`, added `text_column='text'`, added guard for missing table.
- **`04_trino.ipynb` cell `cell-26`** — SyntaxError: `split_part(file_path, '/', -1)` embedded in single-quoted Python string; changed outer quote to double-quote.
- **`compose-demo.yml`** — notebooks were baked into the Docker image only; new notebooks added after initial build were invisible until `--build`. Added bind-mount `./demo/notebooks:/notebooks:ro` to `jupyter` and `jupyter-gpu` services so notebooks appear immediately without rebuilding.

---

## [0.0.26] — 2026-06-24

### Security

- **`ailake-jni` JNI `query_len` upper-bound guard** (`ailake-jni/src/lib.rs`) — `ailake_search_legacy` accepted any `u32` value for `query_len`; `u32::MAX` cast to `usize` would instruct `from_raw_parts` to read ~16 GB from the caller's address space, causing OOM or out-of-bounds memory access. Now returns `{"ok":false,"error":"query_len exceeds maximum supported dimension (65536)"}` for values > 65,536.
- **`ailake-jni` `ef_search` DoS cap in JNI search path** (`ailake-jni/src/lib.rs`) — the HTTP server already clamped `ef_search` via `top_k.min(10_000).saturating_mul(5)`, but JNI callers (`ailake_search_json`, `ailake_search_text_json`) accepted arbitrary `ef_search` values, allowing a buggy or malicious Spark/Trino executor to trigger an arbitrarily expensive HNSW traversal. Both JNI search paths now apply `.min(100_000)` before passing to `do_search`.
- **Airflow hook: truncate CLI output in error messages** (`airflow-providers-ailake/airflow_providers_ailake/hooks/ailake.py`) — `run_cli` raised a `RuntimeError` with full `stdout`+`stderr` from the subprocess; cloud SDKs in verbose/debug mode can print credential-adjacent context (error messages referencing access key IDs, SDK configuration) to stderr. Both outputs are now truncated to 4096 characters in the error message. Operational debugging is not affected — the full output is still accessible via Airflow's task log if the subprocess itself writes to stderr incrementally.
- **HTTP server: documented trusted-network requirement** (`ailake-cli/src/serve.rs`) — `ailake serve` has no authentication by design (sidecar / VPC-internal use case). Added a file-level comment and a startup `eprintln!` warning to make this explicit: the server must not be exposed on a public interface without an authenticating reverse proxy (nginx + mTLS, API gateway, etc.).

### Added (demo)

- **`12_airflow.ipynb`** — new notebook demonstrating Airflow 2.9 integration: trigger DAGs via REST API, poll run status, pull XCom results, read Airflow-written tables from Jupyter. Requires `--profile airflow`.
- **`--profile airflow`** in `compose-demo.yml` — new `ailake-demo-airflow` service (port 8090) built from `Dockerfile.airflow`; two-stage image (Rust/maturin wheel build + `apache/airflow:2.9.2`); `SequentialExecutor` + SQLite for demo simplicity; shares `demo-data` volume with Jupyter.
- **`Dockerfile.airflow`** — two-stage Docker build: same `rust:1.78-slim` builder stage as main Dockerfile (includes `ailake-fts` COPY fix) → `apache/airflow:2.9.2` runtime with ailake wheel + providers + numpy + pyarrow.
- **`airflow-entrypoint.sh`** — `airflow db migrate` + `users create` (admin/admin) + scheduler + webserver startup in single entrypoint; fixed credentials for demo usability.
- **`dags/dag_ailake_ingest_search.py`** — `@daily` DAG: `write_docs → vector_search → fts_search → assemble_context` using TaskFlow API + `import ailake` directly (no CLI binary required in Airflow container).
- **`dags/dag_ailake_compaction.py`** — `@weekly` DAG: `compact_table → table_info`; reads Iceberg `metadata.json` and logs `ailake.*` properties.
- **`docs/guides/DEMO_NOTEBOOKS.md`** — new complete step-by-step demo walkthrough (10 sections): prerequisites, service/port map, optional profiles, fixture table reference (11 tables), per-notebook guide (01–12 with required profiles, fixtures, and section maps), recommended execution order, stop/cleanup commands, troubleshooting, env var reference.
- **FTS intro section in `01_ailake_demo.ipynb`** — §32 added linking to `11_fts.ipynb` as next-steps entry; `12_airflow.ipynb` row added to Next Steps table.

### Fixed

- **`ailake-py` local path resolution for new tables** (`ailake-py/src/lib.rs`) — `std::fs::canonicalize` fails when the table directory does not yet exist, producing a relative `file://` URI in Iceberg metadata that Trino/Spark cannot resolve. Replaced with `std::path::absolute` (Rust 1.79+), which resolves against CWD without requiring the path to exist.
- **`ailake-jni` per-table commit serialization** (`ailake-jni/src/lib.rs`) — each JNI call created a fresh `HadoopCatalog` instance with its own in-process mutex, so concurrent Spark/Trino executors writing to the same table raced on `metadata.json` (last write wins, earlier snapshot silently lost). Added `jni_table_lock` — a process-level static `HashMap<path, Arc<Mutex<()>>>` — that serializes all four mutating JNI functions (`write_batch`, `delete_where`, `evolve_schema`, `compact`) per warehouse/namespace/table key.
- **`ailake-jni` empty query guard** (`ailake-jni/src/lib.rs`) — `ailake_search_json` with `query_len == 0` previously passed a zero-length slice to `from_raw_parts`, producing undefined behaviour. Now returns `{"ok":false,"error":"query_len must be > 0"}` before any unsafe code.
- **`ailake-vec` distance function dimension assertions** (`ailake-vec/src/distance.rs`) — added `debug_assert_eq!` to all six public distance functions (`dot_product`, `euclidean_distance`, `cosine_distance` and their `_f16` variants). Dimension mismatches now panic in debug/test builds with a clear message instead of silently reading out-of-bounds.
- **`ailake-file` centroid decode panic message** (`ailake-file/src/reader.rs`) — `.unwrap()` in `get_centroid` replaced with `.expect("invariant")` to surface the invariant violated on panic.
- **`ailake-query` geometric pruner panics on multimodal secondary column** (`ailake-query/src/pruner.rs`) — `VectorPruner::prune` compared the secondary column's query (e.g. `dim=2`) against the file's centroid for the primary column (`dim=4`), causing a `debug_assert_eq` panic in the distance function. Fix: when `centroid.values.len() != query.len()`, pruning is skipped and the file is kept (conservative fallback). Exposed and fixed by the distance-assertion fix above.
- **`ailake-cli` HTTP server DoS and overflow** (`ailake-cli/src/serve.rs`) — three issues: (1) no request body size limit allowed OOM via oversized POST `/write`; added `DefaultBodyLimit::max(32 MB)`. (2) `ef_search = req.top_k * 5` overflowed for large `top_k`; replaced with `top_k.min(10_000).saturating_mul(5)`. (3) empty `query` vector reached the search engine with `dim=0`; now returns HTTP 400 before dispatch.
- **`ailake-query` multi-column embedding dim validation** (`ailake-query/src/writer.rs`) — `write_batch_multi` only validated the first column's dimension, silently accepting mismatched dims in secondary columns and corrupting Parquet row groups. Refactored into `validate_embedding_dim_for_policy` (static); all `MultiVectorBatch` columns now validated against their own `policy.dim` before any I/O.
- **`ailake-catalog` centroid decode panic messages** (`ailake-catalog/src/hadoop.rs`, `ailake-catalog/src/provider.rs`) — two remaining `.unwrap()` calls on `chunks_exact(4).try_into()` replaced with `.expect("invariant")` for consistency with the same fix in `ailake-file`.
- **`ailake-cli` embed-cmd stdin pipe** (`ailake-cli/src/main.rs`) — `child.stdin.take().unwrap()` panicked if the OS did not allocate a pipe handle. Replaced with `ok_or_else(|| io::Error::new(BrokenPipe, ...))`. Added explicit `drop(stdin)` before `wait_with_output()` to ensure child receives EOF on all code paths.

### Changed (docs)

- **`SETUP.md` §1** — "Fastest path — Docker demo" updated: notebook count 10 → 12; full notebook table with profiles; `--profile airflow` command block added; link to `docs/guides/DEMO_NOTEBOOKS.md`.
- **`README.md` + `README.pt-BR.md`** — `12_airflow.ipynb` row added to notebook table; `--profile airflow` added to profile commands block; `docs/guides/DEMO_NOTEBOOKS.md` row added to Quick Orientation table.
- **`docs/architecture/WORKSPACE.md`** — Phase 7 table row: `🚧 In progress` → `✅ Complete`; demo entry updated (01–12 notebooks, 11 fixture tables); Phase T deliverables section adds `11_fts.ipynb`, `12_airflow.ipynb`, `Dockerfile.airflow`, `--profile airflow` compose service.
- **`docs/specs/JVM_PLUGINS.md`** — `VERSION=0.0.17` example updated to `0.0.25`.
- **`docs/specs/GPU_FFI_EVALUATION.md`** — §1 intro: callout box added noting `candle-core`/`--features gpu` replaced in Phase 4 (document preserved as decision record); §7 recommendation: stale "Option D (candle + cublas)" replaced with Phase 4 outcome (libloading cuBLAS/hipBLAS implemented); cuVS remains recommended for large-file ANN.
- **`docs/contributing/TESTING.md`** — `compat-ailake-py` description: adds `fts_text_columns` write + `search_text()` Tantivy fast path + `search_multimodal` RRF; `compat-jvm-plugins` description: adds FTS write (`fts_columns[]`) + `ailake_search_text_json` Spark/Trino round-trip tests.

---

## [0.0.25] — 2026-06-23

### Fixed

- **Flink `AilakeJniIntegrationTest` always SKIPPED in CI** (`ailake-flink`, `.github/workflows/compat-heavy.yml`) — `-Dailake.native.lib=...` passed to Gradle sets a property in the Gradle daemon JVM but is not propagated to the test worker JVM, so `System.getProperty("ailake.native.lib")` always returned null and `assumeTrue` skipped all three tests (`writeAndSearch`, `deleteWhere`, `evolveSchema`). Fixed: CI now passes `AILAKE_NATIVE_LIB` as an environment variable (inherited by the test JVM automatically); `build.gradle.kts` `tasks.test` block also forwards the system property via `systemProperty()` for local dev (`gradle test -Dailake.native.lib=...`).

### Added

- **FTS `writeBatch` + `searchText` integration tests for Spark and Trino** (`spark-plugin`, `trino-plugin`) — `AilakeNativeTest` had `writeBatch(ftsColumns=[...])` and `searchText()` tests that correctly SKIP in CI when the native library is present (they test graceful degradation when absent), but there was no positive coverage of the FTS path when the library IS present. Added `writeBatchWithFtsColumnsAndSearchTextRoundtrip` to `AilakeWriteBatchIntegrationTest` in both plugins: writes 3 rows with `ftsColumns=["chunk_text"]` and `columns={"chunk_text": [...]}`, then calls `searchText("rust")` and asserts `rowId=0` is the top result.

### Changed (CI)

- **GPU CI consolidated** (`.github/workflows/`) — `ci-gpu-data.yml` deleted; its test (`cargo test -p ailake-index --test gpu_data`) was a strict subset of what `ci-gpu.yml` already runs (`cargo test -p ailake-index`). Single entry point: `ci-gpu.yml`. Linux GPU jobs (`index-gpu-linux-cuda`, `index-gpu-linux-rocm`) set to `if: false` — no Linux GPU runner registered; only Windows self-hosted GPU runner available.

---

## [0.0.24] — 2026-06-23

### Fixed

- **`HadoopCatalog` lost-update under concurrent in-process writes** (`ailake-catalog`) — `save_metadata()` performed two non-atomic `store.put()` calls (versioned JSON + version-hint.text). Two concurrent tokio tasks could both read version N and both write version N+1, causing one commit to be silently discarded. Fixed: added `Arc<tokio::sync::Mutex<()>>` to `HadoopCatalog` that serializes all `commit_snapshot` calls within the process. Cross-process concurrent writers (multi-JVM Spark) require a REST or Nessie catalog — same documented limitation as upstream Apache Iceberg `HadoopCatalog`.
- **`GlueCatalog` lost-update under concurrent writes** (`ailake-catalog`) — `commit_snapshot` called `update_table()` without the Glue version_id OCC guard. Two concurrent writers could both read the same Glue table version and overwrite each other's `metadata_location`. Fixed: new `get_table_state()` method reads both `metadata_location` and Glue's `version_id`; `build_table_input()` now accepts `version_id: Option<&str>` and passes it to `update_table()`; on `ConcurrentModificationException` the commit retries up to 5 times with exponential back-off (100ms, 200ms, 400ms, …).
- **`JdbcCatalog` lost-update under concurrent writes** (`ailake-catalog`) — `commit_snapshot` executed `UPDATE iceberg_tables SET metadata_location = ?` with no CAS condition, allowing concurrent writers to silently overwrite each other's commits. Fixed: `AND metadata_location = old_location` added as CAS predicate; `rows_affected() == 0` triggers up to 5 retries with 50ms exponential back-off; compatible with PostgreSQL, MySQL, and SQLite.

- **`ailake_free_string` non-nullable in Trino JNA interface** (`trino-plugin`) — `Lib.ailake_free_string(ptr: Pointer)` was non-nullable; if Rust ever returns a null pointer under OOM, JNA would NPE before reaching the native call. Fixed: `Pointer?` matches Flink's `AilakeNativeLib.ailake_free_string(ptr: Pointer?)`.
- **`searchMultimodal` `dropLast(1)` JSON hack in Trino** (`trino-plugin`) — `AilakeNative.searchMultimodal` built the JSON payload as `mapper.writeValueAsString(mapOf(...)).dropLast(1) + """,...}"""`, the same fragile string-concatenation pattern that was fixed in `evolveSchema` last sprint but not applied here. JSON would be malformed if `warehouse`/`namespace`/`table` contained `}` or `"`. Fixed: proper `mutableMapOf` + `mapper.writeValueAsString(payload)`.
- **Trino/Spark missing `AILAKE_NATIVE_LIB` / `ailake.native.lib` override** (`trino-plugin`, `spark-plugin`) — Flink's `AilakeNativeLoader` supported explicit library path via `System.getProperty("ailake.native.lib")` and `System.getenv("AILAKE_NATIVE_LIB")` since sprint 4; Trino and Spark only used the JNA default search path, making it impossible to point to a specific lib path without changing `LD_LIBRARY_PATH` globally. Both now support the same discovery order: system property → env var → JNA default search path. Log message updated to match Flink style.

### Added

- **GPU Docker images for reproducible local and CI testing** (`docker/gpu-cuda/Dockerfile`, `docker/gpu-rocm/Dockerfile`, `docker-compose.gpu.yml`) — two purpose-built images replacing the need to manually install CUDA Toolkit or ROCm on a Linux development machine. `gpu-cuda` is based on `nvidia/cuda:12.6.0-runtime-ubuntu22.04` (runtime-only, not devel — ailake-index uses libloading so no compile-time SDK is needed); `gpu-rocm` on `rocm/dev-ubuntu-22.04:6.2`. Both images vendor Rust stable, pre-fetch workspace deps, and pre-build the test harness. `docker-compose.gpu.yml` provides `gpu-cuda` and `gpu-rocm` services with correct device passthrough flags. Local usage: `docker compose -f docker-compose.gpu.yml run --rm gpu-cuda`.
- **Linux GPU CI jobs** (`.github/workflows/ci-gpu.yml`, `.github/workflows/ci-gpu-data.yml`) — new `index-gpu-linux-cuda` and `index-gpu-linux-rocm` jobs run the Docker images on `[self-hosted, Linux, gpu-nvidia]` / `[self-hosted, Linux, gpu-amd]` runners. GPU CI previously covered Windows bare-metal only; `hardware.rs` Linux paths (`libcuda.so.1`, `libamdhip64.so`) were never exercised in CI.
- **Composite action `locate-rust-windows`** (`.github/actions/locate-rust-windows/action.yml`) — extracts the ~60-line PowerShell Rust-discovery block duplicated across `ci-gpu.yml` and `ci-gpu-data.yml` into a reusable composite action. Both workflows now call `uses: ./.github/actions/locate-rust-windows`.

- **Concurrent-write stress tests** (`tests/tests/concurrent_writes.rs`) — three new integration tests: `hadoop_8_concurrent_appends_no_lost_update` (8 parallel tokio tasks, verifies all 8 files survive via `list_files`), `hadoop_overwrite_and_append_no_corruption` (4 Append + 2 Overwrite tasks race, verifies metadata integrity), `jdbc_4_concurrent_commits_no_lost_update` (4 SQLite writers, CAS retry, each snap_id individually findable). Requires `features = ["catalog-jdbc"]` in `tests/Cargo.toml`.

- **ADR-017 — Arrow Flight rejected; Fase 10 Arrow IPC write_batch planned** (`docs/contributing/DECISIONS.md`, `CLAUDE.md`) — architectural decision record documenting evaluation of Apache Arrow Flight as unified interop layer; decision: not adopted. `ailake serve` (HTTP/JSON) already covers language-agnostic access; PyO3 and Go SDK are already zero-FFI; JVM FFI surface (JNA, 10 functions) is manageable post sprint-4 fixes; distributed Spark favors per-executor local search over centralised Flight server; Arrow Flight Java client would add 50MB+ to Spark classpath with version-conflict risk. Identified next incremental improvement: replace JSON embedding payload in `ailake_write_batch_json` with Arrow IPC bytes (12MB JSON → 3MB IPC binary per 1k×1536-dim batch) — Fase 10 in roadmap. Conditions that would reopen Flight: GPU co-location mandatory, >8 JVM plugins, multi-tenant shared inference cluster, or streaming >10k results.

### Changed (docs)

- **Version strings** — all `0.0.20` references updated to `0.0.23` in `README.md`, `README.pt-BR.md`, and `ailake-py/README.md`.
- **Repository layout** — `ailake-fts/` and `airbyte-destination-ailake/` added to layout sections in `README.md` and `README.pt-BR.md`.
- **`docs/specs/FILE_FORMAT.md`** — §7 renamed from "Phase T" to "Phase 7 — Full-Text Search"; new §8.2 subsection documents `index_status` / `index_error` in `key_metadata` JSON with full status table and failure JSON example.
- **`docs/specs/COMPACTION.md`** — new "Failed index recovery" subsection: explains `IndexStatus::Failed` lifecycle, flat-scan fallback, and automatic self-healing at next compaction run.
- **`docs/architecture/DATA_FLOW.md`** — `IndexStatus` lifecycle updated to include `Failed` state, `patch_index_failed()` reference, and flat-scan fallback for both `Indexing` and `Failed` files.
- **`ailake-py/README.md`** — added `search_text()` API doc (BM25 + Tantivy fast path); added `info()` API doc showing `index_status`/`index_error` per-file fields; added **Version: 0.0.23** to header.
- **`ailake-go/README.md`** — `DataFileEntry.IndexStatus` comment updated to include `"failed"`; `IndexError string` field added.

### Changed (demo)

- **Demo notebooks + init_demo.py updated to v0.0.23** — version strings updated from v0.0.20 in `01_ailake_demo.ipynb`, `09_hybrid_search.ipynb`, and `init_demo.py`.
- **`01_ailake_demo.ipynb` — new feature demos**:
  - §15 (HNSW tuning): added `ef_search` and `pruning_threshold` to markdown table and code cell — shows `search(..., ef_search=400)` and `search(..., pruning_threshold=0.7)`.
  - §31 (new): `ailake.compact()` — merges small files and rebuilds index; post-compaction search verification.
  - §30 (schema evolution): added `ailake.evolve_schema()` combined wrapper demo alongside `add_column` + `rename_column`.

---

## [0.0.23] — 2026-06-22

### Fixed (release infrastructure)

- **`ailake-fts` crates.io publish order** (`release`) — added `ailake-fts` to workspace crates.io publish sequence; unblocked v0.0.23 crates.io release after v0.0.22 publish failure.

---

## [0.0.22] — 2026-06-22

### Fixed (release infrastructure)

- **`ailake-fts` workspace dependency version** (`workspace`) — added explicit `version` pin to the workspace-level `ailake-fts` dependency entry; fixed missing version that blocked workspace publish.

---

## [0.0.21] — 2026-06-22

### Fixed

- **Hybrid BM25 fusion score misalignment** (`ailake-query`) — `bm25_scores` was computed from `raw_candidates` before `sort_by()` but indexed by position after the sort; the candidate order changed but the score array did not track the shuffle, so every hybrid search returned wrong RRF fusion scores. Fixed: BM25 scores zipped into `candidates_with_bm25` tuples before sorting so each score travels with its candidate.
- **Equality delete AND semantics** (`ailake-query`) — `EqualityDeleteFilter::should_delete_row` and `apply` used OR logic (deleted a row when *any* predicate column matched); Iceberg spec requires AND — row deleted only when *all* predicate columns match. Fixed: `apply` now delegates to `should_delete_row`; eliminated duplicated logic; fixes over-deletion in multi-column equality delete files.
- **Memory decay drops secondary vector indexes** (`ailake-query`) — `MemoryDecayJob::make_data_file_entry` (single-column path) discarded `extra_vector_indexes`, making secondary HNSW columns permanently inaccessible after any decay run. Fixed: rewrites via `write_multi` to preserve all HNSW sections; rebuilds `ExtraVectorIndex` from new file headers; calls `make_multi_column_data_file_entry`.
- **`ailake_vector_search_json` null guard returns `[]`** (`ailake-jni`) — a prior sprint entry changed null-pointer input to return `{"ok":false,"error":"..."}`, but this broke Spark/Trino init-time callers that treat any non-`[]` response as a parse error. Reverted: `table_uri = null` or `query_ptr = null` returns `[]` (empty JSON array), matching `cabi_null_guard` test expectation. Error envelope is only returned for invalid non-null inputs.
- **Python type stubs completeness** (`ailake-py`) — `_ailake.pyi` was missing `delete_rows` and `now_ns` stubs (mypy `attr-defined` errors); `search` missing `pruning_threshold` and `ef_search` params (mypy `too many arguments`); `add_column` missing `write_default` and `doc` params and had a superfluous keyword-only `*` causing `too many positional arguments` errors. All stubs updated to match PyO3 binding signatures.
- **Spark SLF4J `warn` overload ambiguity** (`spark-plugin`) — Scala 2.12 cannot resolve `warn(String, Any, Any)` vs `warn(String, Object*)` when both args are `String`, causing a compile error in the version-check log statement in `AilakeNative.scala`. Fixed: replaced positional `{}` log format with Scala string interpolation (`s"..."`).
- **Spark write+search integration test assertion** (`spark-plugin`) — `AilakeWriteBatchIntegrationTest` expected `best.rowId == 5` but rows 5 and 13 share identical embeddings (`5 % 8 == 13 % 8 == 5`), making both equally valid top-1 results. Fixed: assertion changed to `best.rowId % dim == queryIdx` so any row with the same spike position passes.
- **DuckDB `ailake_write_batch` table-name convention** (`duckdb-ailake`) — a prior sprint entry claimed the fix derived table name from the path tail; in practice the 3-arg and full `AilakeWriteExecFull` handlers now both hardcode the Iceberg table name to `"table"` (namespace `"default"`). The `table_path` argument is the warehouse root directory. This matches the `ailake_search` DuckDB function and the `ailake_vector_search_json` C-ABI convention — `default.table` is the fixed Iceberg table within the warehouse. Updated description replaces the earlier incorrect "derived from path" claim.

### Added

- **Sprint 4 DX improvements**:
  - **`AilakeIndexStatusSensor`** (`airflow-providers-ailake`) — new sensor that polls `ailake info <table> --json` until `index_status == "ready"`; useful for gating downstream tasks on async IVF-PQ/HNSW deferred builds.
  - **`AilakeHook.compact()` and `AilakeHook.decay_memories()`** (`airflow-providers-ailake`) — typed hook methods for running compaction and memory-decay jobs from Airflow DAGs; return file counts parsed from CLI output.
  - **`evolve_schema()` Python wrapper** (`ailake-py`) — top-level convenience combining `add_column()` + `rename_column()` in one call; `add_columns=` / `rename_columns=` accept list-of-dicts.
  - **`ef_search` in Python search** (`ailake-py`) — `search(…, ef_search=N)`, `Table.search(…, ef_search=N)`, and `SearchQuery` now accept `ef_search`; previously hardcoded to 50. Propagated through PyO3 Rust binding.
  - **`delete_rows`, `now_ns`, `search_with_data`, `evolve_schema` added to `__all__`** (`ailake-py`) — previously exported from `_ailake` but not importable via `from ailake import *`.
  - **JVM startup version check** — Spark, Trino, and Flink plugins call `ailake_version()` on native lib load and log a warning when major version does not match expected. `ailake_version()` added to `Lib`/`Lib`/`AilakeNativeLib` interfaces.
  - **`ailake_version()` added to Trino `Lib` interface** (`trino-plugin`) — was missing; now used in startup version check.
  - **`columns` field documented in `AilakeNativeLib.kt` KDoc** (`ailake-flink`) — `ailake_write_batch_json` docs now describe `columns` map for FTS content; `ailake_search_json` docs add `pruning_threshold` and `ef_search`; `ailake_vector_search_json` response format corrected to `{"ok":true,"results":[...]}`.
  - **Go `EvolveSchema` `-1` return documented** (`ailake-go`) — doc comment now states that `-1` is returned when the CLI emits no `new_schema_id` (no-op evolution).

### Fixed

- **`jsonStr()` in Spark** (`spark-plugin`) — hand-rolled `jsonStr(s: String)` only escaped `\` and `"`, missing control characters. Fixed: delegates to `mapper.writeValueAsString(s)`.
- **`evolveSchema` `dropLast(1)` JSON hack in Trino and Flink** — fragile string-concatenation JSON assembly replaced with proper Jackson `ObjectNode` + `mapper.readTree(ac.initialDefault)` to embed raw JSON literals without re-quoting.
- **JNI warn/error logs missing context** (`ailake-jni`) — `ailake_search_json` and `ailake_write_batch_json` error logs now include `warehouse`, `namespace`, and `table` to identify failing table in multi-tenant deployments.
- **`ailake_vector_search_json` bare array return** (`ailake-jni`) — on success returned `[...]` instead of `{"ok":true,"results":[...]}`, inconsistent with all other `*_json` functions. Fixed: wraps result in `{"ok":true,"results":[...]}`. (Null/invalid input behavior was subsequently updated — see top-level Fixed entry for `125a60b`.)
- **Go `resolveBin()` no executable check** (`ailake-go`) — `os.Stat` verified file exists but not that it is executable. Fixed: on non-Windows, checks `info.Mode()&0111 != 0`.
- **Go `isLocalPath()` missing Windows UNC guard** (`ailake-go`) — warehouse path guards checked `filepath.IsAbs` and `strings.Contains("://")` but not UNC paths (`\\server\share`), causing them to be passed to `filepath.Abs()` on Linux cross-builds. New `isLocalPath()` helper also checks `strings.HasPrefix("\\")`.
- **C++ `pclose` exit code on POSIX** (`ailake-cpp`) — `pclose()` returns wait-status (not exit code) on POSIX; comparing to `!= 0` would flag signals/stops as errors with wrong code. Fixed: `WIFEXITED(rc) ? WEXITSTATUS(rc) : rc` under `#ifndef _WIN32`; `#include <sys/wait.h>` added.

### Added

- **`hnsw_m`, `hnsw_ef_construction`, `pre_normalize`, `deferred` in all write paths** — parameters previously tunable only via Rust CLI are now available in every plugin and integration:
  - **JNI C-ABI** (`ailake-jni`): added to `ailake_write_batch_json` Req struct; `pre_normalize` and `hnsw_m`/`hnsw_ef_construction` propagate to `VectorStoragePolicy`; `deferred=true` calls `write_batch_auto_deferred()` instead of `write_batch_auto()`.
  - **Spark** (`AilakeNative.scala`): `writeBatch()` gains `hnswM`, `hnswEfConstruction`, `preNormalize`, `deferred` params.
  - **Trino** (`AilakeNative.kt`): same params added to `writeBatch()`.
  - **Flink** (`AilakeNativeLoader.kt`): same params added to `writeBatch()`.
  - **DuckDB** (`ailake_write_batch`): arities 13–16 added (`hnsw_m INTEGER`, `hnsw_ef_construction INTEGER`, `pre_normalize BOOLEAN`, `deferred BOOLEAN`); `AilakeLib::write_batch()` signature updated.
  - **Airflow** (`AilakeWriteOperator`): `hnsw_m`, `hnsw_ef_construction`, `pre_normalize`, `deferred` constructor params pass `--hnsw-m`, `--hnsw-ef`, `--pre-normalize`, `--deferred` CLI flags.
  - **Go** (`ailake-go`): new `WriteBatch(catalog, namespace, table, parquetFile, WriteBatchOptions)` function with all tuning params; `WriteBatchOptions` struct documents every flag.
  - **C++** (`ailake-cpp`): new `write_batch(warehouse, table_id, parquet_file, WriteBatchOptions)` inline function in `include/ailake/write.hpp`; `WriteBatchOptions` struct mirrors Go API.

- **Hybrid BM25+vector search in Spark, Trino, and Airflow** — `hybridText`/`hybrid_text`, `textColumn`/`text_column`, `bm25Weight`/`bm25_weight` params (already present in Flink, DuckDB, Go, C++) now exposed uniformly:
  - **Spark** (`AilakeNative.scala`): `search()` gains `hybridText: Option[String]`, `textColumn: String`, `bm25Weight: Float`.
  - **Trino** (`AilakeNative.kt`): `search()` gains `hybridText: String?`, `textColumn: String`, `bm25Weight: Float`.
  - **Airflow** (`AilakeSearchOperator`): `hybrid_text`, `text_column`, `bm25_weight` constructor params; `AilakeHook.search()` updated to pass `--hybrid-text`, `--text-column`, `--bm25-weight` CLI flags.

- **`SearchHybrid()` in Go** (`ailake-go`): new function for hybrid BM25+vector RRF search, delegating to `ailake search --hybrid-text` CLI. Returns `[]SearchHybridResult`.
- **Airbyte destination** (`airbyte-destination-ailake`): `hnsw_m`, `hnsw_ef_construction`, `deferred` added to `AilakeDestinationConfig`, `spec.json`, and `writer.py`. Propagated to `ailake.open_table()` kwargs.

### Fixed

- **`cstr_err_json` fallback returns empty string** (`ailake-jni`) — `CString::new(s).unwrap_or_default()` returned an empty C string (just `\0`) when the error message contained a null byte, causing callers to receive unparseable JSON. Fixed: `unwrap_or_else(|_| CString::new(r#"{"ok":false,"error":"internal: error message contained null byte"}"#).unwrap())`.
- **JNI UTF-8 byte-index slice panic** (`ailake-jni`) — `&req.query_text[..len.min(60)]` could panic when byte index 60 fell inside a multi-byte UTF-8 character (CJK, emoji). Fixed: `req.query_text.chars().take(60).collect::<String>()`.
- **Flink `AilakeNativeLib` `Pointer` non-nullable** (`ailake-flink`) — JNA returns `null` when native functions return null pointers; all `*_json` return types declared as non-nullable `Pointer` caused NPE / segfault at `ptr.getString(0)`. Fixed: all `*_json` functions now return `Pointer?`; `ailake_free_string` also accepts `Pointer?`. All 6 call sites in `AilakeNativeLoader.kt` now check for null with `?: throw RuntimeException(...)` before `try/finally`.
- **DuckDB `ailake_write_batch` incorrect table name** (`duckdb-ailake`) — both the 3-arg simple form and the full `AilakeWriteExecFull` were deriving the Iceberg table name from a random temp-directory suffix in `warehouse`, causing write and search to use different table names. Fixed: both forms now unconditionally use Iceberg table name `"table"` (namespace `"default"`), consistent with `ailake_search` and `ailake_vector_search_json` defaults. The `table_path` argument is the warehouse root; the Iceberg address is always `default.table` within it. (See also the top-level Fixed entry with the final correction.)
- **DuckDB `search_text()` reads `r["distance"]` for FTS results** (`duckdb-ailake`) — FTS search returns `score` field (higher = more relevant), not `distance`. `r["distance"]` throws a JSON key exception (caught silently), returning zero results. Fixed: `r.contains("score") ? r["score"] : r.value("distance", 0.0f)`.
- **Airflow `AilakeWriteOperator` crash on first run** (`airflow-providers-ailake`) — `hook.get_table_info()` called `run_cli("info", ..., check=True)`, which raises `RuntimeError` when the table doesn't yet exist (ailake exits non-zero). Fixed: `check=False` + graceful fallback to `{}` on non-zero exit or unparseable output.
- **Airflow `evolve_schema` falsy `initial_default` check** (`airflow-providers-ailake`) — `if ac.get("initial_default"):` silently skipped valid defaults of `0`, `0.0`, `""`, or JSON `null`. Fixed: `if ac.get("initial_default") is not None:`.
- **Airbyte `CmdEmbedder` uses `shell=True`** (`airbyte-destination-ailake`) — `subprocess.run(cmd, shell=True)` is a command injection risk when `embed_cmd` is user-controlled. Fixed: `shlex.split(self._cmd)` + `shell=False`.
- **Go `SearchHybrid`/`SearchText` lose stderr on CLI failure** (`ailake-go`) — `exec.Command(...).Output()` wraps non-zero exit as `*exec.ExitError` but the error message loses the CLI's stderr. Fixed: `errors.As(err, &exitErr)` check; when stderr is non-empty it is appended to the error: `fmt.Errorf("...: %w\nstderr: %s", err, exitErr.Stderr)`.

- **`Agent.recall()` column name mismatch** (`ailake-py`) — `search_with_data` returns columns named `_distance` and `text`; `Agent.recall()` was checking for `distance` and `chunk_text`, causing silent wrong scoring (recency/importance always divided by 1.0) and empty memory text on every recall. Fixed to match actual output schema.
- **`parse_metric` missing `NormalizedCosine`** (`ailake-query`) — `scanner.rs::parse_metric` fell through to `Cosine` for `normalized_cosine`/`normalizedcosine` metric strings, causing wrong dot-product distances for pre-normalized tables. Added explicit arm: `"normalized_cosine" | "normalizedcosine" => VectorMetric::NormalizedCosine`.
- **Airbyte `_flush()` buffer cleared before embed** (`airbyte-destination-ailake`) — `self._buffer = []` was called before `self._embedder.embed()`, so any exception in embed caused permanent record loss (at-least-once delivery violated). Fixed: snapshot buffer contents first, clear only after successful `table.insert()`.
- **Airbyte `deferred` flag passed to `open_table()`** (`airbyte-destination-ailake`) — `open_table()` does not accept a `deferred` kwarg; passing it caused `TypeError` on first write for any connector configured with `deferred=true`. Removed the erroneous kwarg passthrough.
- **Flink `flush()` no try/finally** (`ailake-flink`, `AilakeVectorTableSink`) — an exception in `AilakeNativeLoader.writeBatch()` killed the subtask with buffers intact, causing data duplication on checkpoint restart (Flink redelivers the in-flight records). Fixed: `writeBatch()` now wrapped in `try/finally`; `idsBuffer` and `embeddingsBuffer` cleared unconditionally.
- **Spark `search()`/`searchMultimodal()` hardcoded `namespace:"default","table":"table"`** (`spark-plugin`) — all vector searches routed to wrong table. Fixed: `namespace: String = "default"` and `tableName: String = ""` params added (empty `tableName` derives from last path segment of `tableUri`, matching `AilakeCatalog`/`AilakeDataSource` behavior). `VectorScanExec` callers use derived defaults automatically; explicit overrides available for advanced use.
- **Spark `search()` double-free of native pointer** (`spark-plugin`) — `ailake_free_string(ptr)` called in `try` block, then again in `catch` when `parseResponse` threw, risking use-after-free. Restructured: `getString` and `ailake_free_string` now isolated in their own try-catch; `parseResponse` runs after free with only the String value.
- **Trino `search()`/`searchMultimodal()` hardcoded `namespace:"default","table":"table"`** (`trino-plugin`) — same bug as Spark. Fixed: `namespace: String = "default"` and `tableName: String = ""` params added (empty `tableName` derives from `tableUri`). `VectorScanRecordSetProvider` uses defaults; callers may override.

- **`fts-stemmer-langs` Cargo feature** (`ailake-fts`) — opt-in registration of 17 Snowball language stemmers + stop-word-filtered pipelines in every Tantivy index build. Bare stemmers: `ar_stem`, `da_stem`, `nl_stem`, `fi_stem`, `fr_stem`, `de_stem`, `el_stem`, `hu_stem`, `it_stem`, `no_stem`, `pt_stem`, `ro_stem`, `ru_stem`, `es_stem`, `sv_stem`, `ta_stem`, `tr_stem`. Stop-word-filtered: `pt_br` (Portuguese Snowball + PT stop words — recommended for Brazilian Portuguese workloads; ~10-15% smaller blobs); `en_stop` (English Snowball + EN stop words — use `en_stem` for standard English; `en_stop` when index size matters). Feature also enables `tantivy/stopwords` for stop word lists. English: built-in `en_stem` (always available, no feature needed) is the standard; no action required for EN workloads. Enable with `ailake-fts = { features = ["fts-stemmer-langs"] }`. Use via `FtsConfig { tokenizer: "pt_br", .. }`.
- **`cjk_ngram` tokenizer** (`ailake-fts`) — always registered, zero extra deps. `NgramTokenizer(min=1, max=2, prefix_only=false)` + `LowerCaser`. Tokenizes CJK text into unigrams and bigrams so BM25 matches sub-word characters (unigram "知" and bigram "知能"). ~85% recall vs. dictionary-based segmenters (Lindera/jieba). For production CJK, register a custom tokenizer and pass its name as `FtsConfig::tokenizer`. Documented in `ailake-fts/src/tokenizers.rs` with limitations (Thai/Khmer, false-positive unigrams, compound recall gap).
- **FTS text field upgraded to `WithFreqsAndPositions`** (`ailake-fts`) — previously `WithFreqs`; positions required for NgramTokenizer phrase queries and user phrase search (e.g. `"quick brown fox"`). ~25-40% larger uncompressed term postings; negligible after zstd. **Breaks blobs written by prior releases when phrase queries are used** — point queries unaffected; rewrite blobs to regain phrase-query support.

### Fixed (security/correctness follow-up — `fix/security-and-correctness`)

- **`write_batch_multi_deferred` never called `patch_index_failed`** (`ailake-query`) — when multi-column HNSW background build failed, the catalog entry stayed `IndexStatus::Indexing` forever; compaction never retried. Fixed: spawn block clones `catalog`, `table`, `fp` and calls `patch_index_failed(catalog, &table, &fp, &e.to_string()).await` on error, consistent with single-column and IVF-PQ deferred variants.
- **`DataFileEntry` missing `index_error` in 18 struct literals** (`ailake-catalog`, `ailake-query`) — workspace failed to compile after `index_error: Option<String>` was added to the struct but not propagated to test initializers in `avro_manifest.rs`, `hadoop.rs`, `snapshot.rs`, `compaction.rs`, `delete.rs`, and `pruner.rs`. Fixed: `index_error: None` added to all 18 literal sites; `index_failed_roundtrip` test added to verify full Avro round-trip.
- **`ailake-fts` blob deserializer missing adversarial guards** (`ailake-fts`) — `blob_to_ram_dir` did not validate file count, magic bytes, or per-entry bounds; a crafted AILK_FTS blob could allocate unlimited memory. Added: `MAX_FTS_FILES = 65_536` check, magic `"AFTS"` guard, checked arithmetic for entry lengths, and 4 adversarial unit tests.
- **`AilakeIndexStatusSensor` polled forever on `"failed"` status** (`airflow-providers-ailake`) — sensor only branched on `"ready"` (return `True`) and everything else (return `False`); a permanently-failed index caused infinite polling. Fixed: `"failed"` branch raises `RuntimeError` with `index_error` detail. 5 tests added covering `ready`, `indexing`, absent, `failed`-with-detail, and `failed`-without-detail.
- **`redundant_closure` clippy errors in JNI** (`ailake-jni`) — 5 occurrences of `.unwrap_or_else(|e| cstr_err_json(e))` flagged by `cargo clippy -D warnings`; simplified to `.unwrap_or_else(cstr_err_json)`.
- **`cargo fmt` failures** — long method chains in `ailake-cli/src/main.rs`, `ailake-file/src/reader.rs`, and `ailake-query/src/writer.rs` were not formatted per rustfmt style. `cargo fmt --all` applied.

### Fixed (Sprint 3 — P2)

- **`pruning_threshold` hardcoded as `INFINITY`** (`ailake-jni`, `ailake-py`) — geometric file pruning was always disabled because `pruning_threshold: f32::INFINITY` was hardcoded in both the JNI `do_search` helper and the PyO3 `search` function. Fixed: `ailake_search_json` Req struct gains `pruning_threshold: Option<f32>` (default absent = no pruning, preserves backward compat); passed to `SearchConfig`. PyO3 `search()` gains `pruning_threshold: Option<f32>` keyword param (`None` = no pruning). Python `SearchQuery.__init__`, `Table.search()`, and module-level `search()` all updated.
- **`Table.search()` missing hybrid/FTS/partition/score_fn params** (`ailake-py`) — `Table.search()` only accepted `query`, `top_k`, and `fetch_data`; callers needing hybrid search or partition isolation had to bypass `Table` entirely. Fixed: `Table.search()` now accepts the full param set: `partition_filter`, `score_fn`, `hybrid_text`, `text_column`, `bm25_weight`, `pruning_threshold` — all forwarded to `SearchQuery`.
- **FTS text data missing in Spark/Trino/Flink writes** (`spark-plugin`, `trino-plugin`, `ailake-flink`) — `writeBatch()` accepted `fts_columns` (which columns to index) but never sent actual text content; the JNI `columns` field was empty, producing empty Tantivy indexes. Fixed: `writeBatch()` gains `columns: Map[String, Seq[String]]` (Scala) / `columns: Map<String, List<String>>` (Kotlin) — serialized as `"columns": {...}` in JSON payload. Flink sink additionally captures text column values per-row via `ftsColumnIndices` and buffers them in `textBuffers`, then passes the snapshot to `writeBatch` in `flush()`.
- **`ailake_compact_json` missing from JNI** (`ailake-jni`) — compaction was only reachable via CLI, blocking JVM and Python callers from triggering it programmatically. Fixed: new `#[no_mangle] pub unsafe extern "C" fn ailake_compact_json(...)` reads `dim`/`vec_col` from table metadata, builds `CompactionConfig`/`CompactionPlanner`/`CompactionExecutor`, and calls `run()` or `run_deferred()`. Exposed in Flink (`AilakeNativeLib`, `AilakeNativeLoader.compact()`), Trino (`Lib`, `AilakeNative.compact()`), Spark (`Lib`, `AilakeNative.compact()`), and Python (`ailake.compact()`).
- **`compact()` missing from Python SDK** (`ailake-py`) — no `ailake.compact()` function existed; callers had to exec the CLI manually. Fixed: new `compact(path, *, min_files=4, target_size_bytes=..., max_files_per_pass=20, deferred=False)` function delegates to `ailake compact` CLI binary via subprocess (same pattern as `delete_where`/`evolve_schema` in Airflow/Go). Added to `__all__`.

### Fixed

- **FTS capability preservation on compaction** (`ailake-fts`, `ailake-query`) — `CompactionExecutor::run()` now auto-detects `FtsConfig` from Iceberg table properties (`ailake.fts.enabled`, `ailake.fts.text-columns`, `ailake.fts.tokenizer`) when the caller did not explicitly call `with_fts_config()`. Prevents silent FTS index loss when compacting tables that were created with FTS enabled. `compact_incremental()` now also rebuilds FTS on the merged batch (previously the incremental path never embedded FTS regardless of config). `compact_deferred()` remains FTS-free by design (Parquet-only immediate write). `FtsConfig::from_table_props()` added as public API for catalog-driven config reconstruction. 4 unit tests in `ailake-fts`.

---

## [0.0.20] — 2026-06-20

### Added

### Added

- **Phase T — Tantivy per-file FTS index (`ailake-fts`)** — opt-in inverted index embedded in each AI-Lake file as a separate `AILK_FTS` section, enabling `search_text()` O(log N) fast path vs. O(N) BM25 brute-force. New crate `ailake-fts` with `builder.rs` (`build_fts_blob_from_batch`, `merge_fts_blobs`, `FtsConfig`), `blob.rs` (zstd-compressed Tantivy `ManagedDirectory` round-trip serialization), and `searcher.rs` (`FtsSearcher::from_blob`, `search()`). Section layout: `AFTS`(4 bytes magic) | version(2 LE) | reserved(2) | blob_len(8 LE) | blob; located after vector AILK sections, referenced via `ailake.fts_offset` Parquet KV. **Write**: `AilakeFileWriter::with_fts(FtsConfig)` / `with_prebuilt_fts_blob(Vec<u8>)` — builds and embeds AILK_FTS on `write_multi()`; `TableWriter::with_fts_config(FtsConfig)` propagates to every file write. **Read**: `AilakeFileReader::load_fts_blob()` — returns `Bytes` of FTS blob keyed by `ailake.fts_offset`, or `Ok(None)` when not present. **Search**: `scanner::search_text()` now checks `load_fts_blob()` first; on hit, runs `FtsSearcher::search()` and skips O(N) BM25 fallback; files without AILK_FTS fall through to existing BM25. **Compaction**: `CompactionExecutor::with_fts_config(FtsConfig)` — rebuilds FTS via `merge_fts_blobs()` after merge, attaches via `with_prebuilt_fts_blob()`; graceful degradation on error. **CLI**: `ailake create --fts-columns <col1,col2> [--fts-tokenizer default]` stores FTS properties in Iceberg metadata; `ailake insert --fts-columns` embeds AILK_FTS section in every written file; `ailake search --text <query> [--text-columns <cols>]` routes to `search_text()` (Tantivy fast path when available). **Python**: `TableWriter(fts_text_columns=["chunk_text"], fts_tokenizer="default")` wires `with_fts_config()`. Zero overhead by default — AILK_FTS section only written when FTS is configured.

### Changed

- **Documentation audit — full sync with codebase state** — all project docs updated to match current feature set. Changes across 18 files:
  - **SETUP.md**: removed §8I (RaBitQ flat index), §8J (Binary Hamming flat index); updated crate table to drop removed index types; added Python examples for `partition_fields`, `format_version=3`, `delete_where`/`add_column`/`rename_column`, `hardware_info()`.
  - **CONTRIBUTING.md**: fixed C++ test table — replaced `test_binary.cpp` row with `test_write.cpp` (covers `delete_where`, `evolve_schema`, `shell_quote`, `resolve_bin`); fixed `foreach` line to match actual CMakeLists.
  - **docs/specs/FILE_FORMAT.md**: removed `precision=3` Binary row from precision table; added deprecation note (removed in v0.0.14).
  - **docs/specs/JVM_PLUGINS.md**: added uniffi removal note; rewrote C-ABI API section with all 6 exported functions including `ailake_delete_where_json` and `ailake_evolve_schema_json`; updated JSON field table with `partition_fields` and `format_version`; added full Flink connector section (build, SQL DDL, Kotlin API, options table); added `## Delete and schema evolution` section.
  - **docs/specs/GPU_FFI_EVALUATION.md**: added hardware thresholds table (`MIN_VECTORS_FOR_IVF_PQ=5_000`, `MIN_CORES_FOR_IVF_PQ=8`, GPU priority order) and dlopen library names table.
  - **docs/specs/INTEGRATIONS.md**: fixed cloud platform rows that still said "Phase 3" → "✅"; updated `ailake_write_batch` signature with `partition_fields` and `format_version`; expanded optional params description.
  - **docs/specs/CLOUD_DEPLOY.md**: added §5 Flink deployment guide (KDA, EMR, SQL DDL, Kubernetes Flink operator); §5 Troubleshooting bumped to §6.
  - **docs/specs/LLM_CONTEXT.md**: added GPU flat-scan limitation note for `score_fn` (not applied during deferred build window).
  - **docs/WHY_AILAKE.md**: added "AI-Lake is also the right choice when" block covering BM25 hybrid search, `delete_where` atomicity, `partition_by` agent isolation; updated "not the right choice" block to accurately describe BM25 scope.
  - **ailake-py/README.md**: added `partition_fields`, `format_version` params to `open_table()` table; added `TableWriter` params note; added `delete_where`, `add_column`, `rename_column`, `hardware_info()` sections.
  - **ailake-jni/README.md**: updated `ailake_write_batch_json` request JSON with `partition_fields` and `format_version`; documented all optional fields; added full `ailake_delete_where_json` and `ailake_evolve_schema_json` sections.
  - **ailake-go/README.md**: expanded `TableInfo` struct to show all fields including `FormatVersion`, `PartitionFields`, `SchemaFields`; added `PartitionDef` and `SchemaField` struct docs.
  - **ailake-cpp/README.md**: added `## Write operations` section with `delete_where` and `evolve_schema` examples.
  - **duckdb-ailake/README.md**: added `partition_fields` and `format_version` arities to `ailake_write_batch` with examples.
  - **airflow-providers-ailake/README.md**: added `partition_fields` and `format_version` to `AilakeWriteOperator`; added `AilakeDeleteWhereOperator` and `AilakeEvolveSchemaOperator` sections.
  - **airbyte-destination-ailake/README.md**: added `partition_fields` and `format_version` fields to config table.
  - **README.md** + **README.pt-BR.md**: updated notebook list (10 notebooks, accurate descriptions); updated Phase 7 scope; updated fixture table count (5 → 8); added GPU profile note.

### Added

- **Demo update — Phase L-R features, GPU demo notebook** — `tests/docker/demo/` fully updated with new features. `init_demo.py` adds three fixture tables: `ailake_partitioned_v3` (format_version=3, `partition_fields=[topic_id:identity:int]`), `ailake_delete_demo` (100 rows, 10 pre-deleted via `delete_where`), `ailake_schema_evo` (100 rows, `add_column source_url` applied); `_save_query_payload` records paths in `demo_query.json`. `compose-demo.yml` gains `--profile gpu` service `jupyter-gpu` with NVIDIA Container Toolkit device reservation (1 GPU). Notebook `01_ailake_demo.ipynb` gains §29 (`delete_where` before/after demo), §30 (schema evolution `add_column`/`rename_column`), §31 (Iceberg v3 partitioned table with `format_version=3`, bucket[4] demo inline). New notebook `10_gpu_demo.ipynb`: 7 sections covering `hardware_info()`, `write_batch_auto_deferred` auto-index-selection, write timing (immediate HNSW vs deferred auto), search throughput, recall comparison (auto-deferred vs HNSW), graceful CPU fallback path, large-scale deferred write (20k vectors).

- **`ailake-py`: expose `hardware_info()`, `delete_where`, `add_column`, `rename_column` in Python module** — all four functions existed in `ailake-py/src/lib.rs` but were not imported in `ailake-py/python/ailake/__init__.py`. Added to import block from `ailake._ailake` and to `__all__`. Added `ailake-index` as explicit dependency in `ailake-py/Cargo.toml` (already transitive via `ailake-query`; made explicit so `hardware_info()` can import `ailake_index::HardwareProfile` directly). `hardware_info()` PyO3 function added to `lib.rs` — returns `HashMap<String,String>` with keys `backend`, `has_cuda`, `has_rocm`, `cpu_logical_cores`, `has_avx2`, `has_avx512`, `recommend_ivf_pq` (at N=5000 threshold).



- **Phase R — JVM connector public surfaces: `partition_fields`, `format_version` wired end-to-end** — closes the gap where Phase P added JNA wrappers but public-facing connector APIs did not expose the new capabilities. **(Spark)** `AilakeWriteHandle` gains `partitionFields: Seq[PartitionFieldDef] = Seq.empty` and `formatVersion: Int = 2`; `AilakeDataWriter.commit()` forwards both to `AilakeNative.writeBatch`; `AilakeDataSource.getTable()` parses `partition-fields` (JSON string via Jackson `ObjectMapper.readTree`) and `format-version` from DataSourceV2 options; `AilakeSparkExtensions.ailakeWrite()` gains `partitionFields` and `formatVersion` params — serializes partition fields to JSON inline and passes as `.option("partition-fields", ...)` and `.option("format-version", ...)`; 5 new tests. **(Trino)** `AilakeIngestTableHandle` gains `partitionFields: List<PartitionFieldDef> = emptyList()` and `formatVersion: Int = 2` (both `@JsonProperty`); `AilakePageSink.finish()` forwards both to `AilakeNative.writeBatch`; `VectorScanMetadata` carries and injects both; `VectorScanConnector` carries both; `VectorScanConnectorFactory.create()` parses `ailake.partition-fields` (JSON via Jackson) and `ailake.format-version`; 4 new tests. **(Flink)** `AilakeVectorConnectorFactory` adds `PARTITION_FIELDS` (`partition.fields`, default `"[]"`) and `FORMAT_VERSION` (`format.version`, default `2`) to `optionalOptions()`; parses JSON in `createDynamicTableSink`; `AilakeVectorTableSink` and `AilakeSinkFunction` gain `partitionFields` and `formatVersion`; `flush()` passes both to `AilakeNativeLoader.writeBatch`; `copy()` propagates both; 4 new tests.

- **Phase Q — Airflow + Airbyte connectors: `partition_fields`, `format_version`, `delete_where`, `evolve_schema`** — closes the connector gap for Phase K/L capabilities. **(Q1) Airflow `AilakeHook`**: `delete_where(table, column, values)` wraps `ailake delete-where <table> --col <col> --vals <csv>` (no-op on empty list); `evolve_schema(table, add_columns, rename_columns)` wraps `ailake evolve <table> --add name:type [--initial-default JSON] [--rename old:new]`, returns `new_schema_id` parsed from stdout or `-1` when not present, `0` for no-op. **(Q2) Airflow operators**: `AilakeWriteOperator` gains `partition_fields: list[dict] | None` (forwarded as `--partition-fields <json>`) and `format_version: int = 2` (forwarded as `--format-version N`, omitted when `2`); new `AilakeDeleteWhereOperator(table, column, values, values_xcom_task_id?, values_xcom_key?)` — calls `hook.delete_where`, no-op on empty list, raises on missing values source; new `AilakeEvolveSchemaOperator(table, add_columns?, rename_columns?)` — calls `hook.evolve_schema`, pushes `schema_id` to XCom, no-op when both lists empty. **(Q3) Airflow tests**: 6 new `TestAilakeHookDeleteEvolve` tests; 4 new `TestAilakeWriteOperatorPhaseQ` tests; 5 new `TestAilakeDeleteWhereOperator` tests; 3 new `TestAilakeEvolveSchemaOperator` tests. **(Q4) Airbyte `config.py`**: `partition_fields: list[dict]` (default `[]`) and `format_version: int` (default `2`) added to `AilakeDestinationConfig`; `from_dict` populates both; `validate()` checks `format_version ∈ {2,3}` and that each `partition_fields` entry has `column`, `transform`, `column_type`. **(Q5) Airbyte `writer.py`**: `StreamWriter._get_table()` forwards `partition_fields` (when non-empty) and `format_version` (when ≠ 2) to `ailake.open_table()`. **(Q6) Airbyte `spec.json`**: `partition_fields` (array of `{column, transform, column_type}` objects, order 21) and `format_version` (integer enum [2,3], order 22) added to `connectionSpecification`. **(Q7) Airbyte tests**: 8 new `TestAilakeDestinationConfig` assertions for partition_fields/format_version config + validate paths; 4 new `TestStreamWriter` assertions verifying partition_fields and format_version are forwarded (or omitted) correctly.

- **Phase P — JVM plugins: `partition_fields`, `format_version`, `deleteWhere`, `evolveSchema`** — Spark, Trino, and Flink JNA bridges updated to expose all Phase L C-ABI capabilities. **(P1) Spark (`AilakeNative.scala`)**: new case classes `PartitionFieldDef(column, transform, columnType)`, `AddColReq(name, colType, initialDefault?)`, `RenameColReq(from, to)`; `writeBatch` gains `partitionFields: Seq[PartitionFieldDef]` (default `Seq.empty`) and `formatVersion: Int` (default `2`) — serialized as `"partition_fields"` array + `"format_version"` in the JSON envelope; new `deleteWhere(tableUri, namespace, tableName, column, values): Boolean` calls `ailake_delete_where_json` (returns `false` for empty values or absent lib); new `evolveSchema(tableUri, namespace, tableName, addCols, renameCols): Int` calls `ailake_evolve_schema_json` (returns `0` for empty no-op, `-1` on error); `initial_default` embedded as raw JSON literal in `add_columns` array. **(P2) Trino (`AilakeNative.kt`)**: identical API in Kotlin data classes and methods; `evolveSchema` builds `add_columns` JSON manually to embed `initial_default` as a raw JSON literal without Jackson re-quoting; `deleteWhere` uses Jackson's `writeValueAsString` for the full payload. **(P3) Flink (`AilakeNativeLib.kt` + `AilakeNativeLoader.kt`)**: `AilakeNativeLib` gains `ailake_delete_where_json` and `ailake_evolve_schema_json` method declarations with full doc comments; `writeBatch` doc updated with `partition_fields` and `format_version` fields; `AilakeNativeLoader` gains `PartitionFieldDef`, `AddColReq`, `RenameColReq` data classes, `DeleteWhereResponse`/`EvolveSchemaResponse` POJOs, extended `writeBatch` (throws `RuntimeException` on error, consistent with existing API), and `deleteWhere`/`evolveSchema` wrappers (throw on native error). All three plugins: `ailake_delete_where_json` and `ailake_evolve_schema_json` declared in the JNA `Lib` interface / `AilakeNativeLib`; graceful degradation for absent library preserved (Spark/Trino return sentinel values; Flink throws — consistent with existing `writeBatch` contract).

- **Phase O — C++ SDK: `PartitionDef`, `SchemaField`, `TableInfo` fields, `write.hpp`** — `ailake-cpp` updated to expose Phase K capabilities in the C++ catalog reader and adds write-operation delegation. (1) **O1** new structs `PartitionDef {column, transform, column_type}` and `SchemaField {id, name, type, required}` in `catalog.hpp`. `TableInfo` gains three new fields: `format_version int` (default `2`), `partition_fields std::vector<PartitionDef>` (empty for unpartitioned), and `schema_fields std::vector<SchemaField>`. `load_table()` extended with depth-aware JSON brace-walking to parse `"schemas"` (current schema fields) and `"partition-specs"` (current default spec) arrays from `metadata.json`. `PartitionDef::column_type` resolved via `source-id → field_type_by_id` map (fallback: `"string"`). `<map>` added to includes. (2) **O2** new header `write.hpp`: `delete_where(warehouse, table_id, column, values)` and `evolve_schema(warehouse, table_id, add_cols, rename_cols)` delegate to `ailake` CLI via `popen()`/`_popen()` (cross-platform). Binary resolution: `AILAKE_BIN` env → `"ailake"` (PATH). `shell_quote` escapes single-quote-embedded args. Both are no-ops on empty inputs. `evolve_schema` parses `new_schema_id:` from stdout. Header included in `ailake.hpp` umbrella. (3) **O3** new `tests/test_write.cpp`: 18 tests — struct field validation, `shell_quote` edge cases (embedded single-quote, spaces), `resolve_bin` env override/fallback, no-op paths, `TableInfo` default field values, plus 3 integration tests guarded by `AILAKE_BIN`/`AILAKE_FIXTURE` env. Registered in `CMakeLists.txt` as `ailake_test_write`. All 4 C++ test suites pass.

- **Phase N — Go client: `FormatVersion`, `PartitionFields`, `SchemaFields`, `DeleteWhere`, `EvolveSchema`** — `ailake-go` updated to expose all Phase K capabilities via the Go catalog reader and adds write-operation delegation. (1) **N1** `TableInfo` gains three new fields: `FormatVersion int` (from `"format-version"` in `metadata.json`; default `2`), `PartitionFields []PartitionDef` (parsed from `"partition-specs"` + `"schemas"`; empty for unpartitioned tables), and `SchemaFields []SchemaField` (parsed from current schema entry). `LoadTable()` populates all three. `PartitionDef {Column, Transform, ColumnType}` resolves `ColumnType` via `source-id` → schema field lookup; falls back to `"string"` when not found. `SchemaField {ID, Name, Type, Required}` mirrors the Iceberg schema field. (2) **N2** new `write.go`: `DeleteWhere(catalog, namespace, table, col, values)` and `EvolveSchema(catalog, namespace, table, addCols, renameCols)` delegate to the `ailake` CLI binary. Binary resolution order: `AILAKE_BIN` env var → `ailake` in PATH → `ErrNoBinary`. `DeleteWhere` calls `ailake --store <warehouse> delete-where <table> --col <col> --vals <v1,v2>`. `EvolveSchema` calls `ailake --store <warehouse> evolve <table> --add name:type [--initial-default JSON] [--rename old:new]`; parses `new_schema_id:` from stdout. Both are no-ops with nil/empty inputs (no binary call). (3) **N3** 26 new unit + integration tests: `str`/`boolVal` helpers, `resolveBin` no-binary/bad-env paths, struct field coverage, `DeleteWhere`/`EvolveSchema` no-op paths, `parseTableInfoFromMeta` helper for FS-free unit testing of `LoadTable` logic, plus 7 new catalog_test assertions covering `FormatVersion`, `SchemaFields`, and multi-column `PartitionFields` parsing. Integration tests guarded by `AILAKE_FIXTURE` + `AILAKE_BIN`.

- **Phase M — DuckDB plugin: `partition_fields`, `format_version`, `ailake_delete_where`, `ailake_evolve_schema`** — DuckDB extension (`duckdb-ailake`) updated to expose all Phase K/L capabilities via DuckDB SQL. (1) **M1** `ailake_write_batch` gains two new optional arities: arity 9 `(…, partition_fields_json VARCHAR)` accepts a JSON array `[{"column":"x","transform":"identity","column_type":"string"}]` for multi-column partition specs; arity 10 `(…, format_version INTEGER)` adds V3 opt-in (`2` default, `3` for V3 tables). Both are forwarded as-is in the JSON envelope to `ailake_write_batch_json`. (2) **M2** new `ailake_delete_where(table_path VARCHAR, column VARCHAR, values VARCHAR[]) → BOOLEAN` scalar function: registers in `ailake_delete.cpp`, calls `ailake_delete_where_json` via dlopen, returns `TRUE` on success. (3) **M3** new `ailake_evolve_schema(table_path VARCHAR, add_columns_json VARCHAR, rename_columns_json VARCHAR) → INTEGER` scalar function: registers in `ailake_evolve_schema.cpp`, calls `ailake_evolve_schema_json`, returns `new_schema_id` or `-1` on error. `AilakeLib` singleton gains `delete_where_fn_t` and `evolve_schema_fn_t` pointer types + two new member pointers; `load()` resolves them with `dlsym` (graceful nullptr for older builds — `is_delete_ready()` / `is_evolve_ready()` guard call sites). `ailake_init` registers two new functions. `CMakeLists.txt` adds `ailake_delete.cpp` and `ailake_evolve_schema.cpp` to the MODULE target.

- **Phase L — JNI C-ABI: `partition_fields`, `format_version`, `ailake_delete_where_json`, `ailake_evolve_schema_json`** — three gaps in the C-ABI surface (used by DuckDB, Flink, Spark, Trino via JNA/dlopen) closed: (1) **L1** `ailake_write_batch_json` now accepts `partition_fields: [{column, transform, column_type}]` (multi-column Phase K spec) and `format_version: 2|3` (V3 opt-in); `partition_fields` maps to `ailake_core::PartitionDef[]`; `format_version` is forwarded to `TableWriter::create_or_open`; both fields default to their V2/empty equivalents — fully backward-compatible. (2) **L2** new `ailake_delete_where_json(json) -> json` endpoint: `{warehouse, namespace, table, column, values[]}` → calls `ailake_query::delete_where` → returns `{ok:true}` or `{error}`. Writes Iceberg equality delete + delete manifest; no data files rewritten. (3) **L3** new `ailake_evolve_schema_json(json) -> json` endpoint: `{warehouse, namespace, table, add_columns[{name,type,initial_default?}], rename_columns[{from,to}]}` → `HadoopCatalog::evolve_schema`; returns `{ok:true, new_schema_id}` or `{error}`. All three endpoints exported from `ailake-jni` cdylib — DuckDB, Flink, Spark, Trino gain these features after rebuild without plugin changes. 7 new tests: null-guard + bad-warehouse tests for each endpoint, plus JSON-parse validation for `partition_fields`.

- **`HnswIndex::insert_node`** — online single-node insertion into a live HNSW graph (`ailake-index/src/hnsw.rs`). Mirrors the `build_serial_typed` algorithm (Algorithm 1, Malkov & Yashunin 2018): random level assignment, greedy descent above the insertion layer, bidirectional connections with `select_neighbors_heuristic`, and connection pruning. O(log N) per call. Invalidates the F16 cache (call `quantize_to_f16()` after bulk inserts). Used by incremental compaction.

- **`AilakeFileWriter::write_with_prebuilt_hnsw`** — write path that accepts a pre-built `HnswIndex` instead of rebuilding from scratch (`ailake-file/src/writer.rs`). Same two-pass Parquet + KV injection as `write()` but serializes the provided HNSW bytes directly into the AILK section. `build_ailk_section_from_index_bytes` is the private helper that assembles the AILK header + centroid + pre-serialized index + trailer.

- **`CompactionExecutor::compact_incremental`** — incremental HNSW compaction (`ailake-query/src/compaction.rs`). Identifies the *dominant file* (≥ 40 % of total rows), loads its existing HNSW from the AILK section via `AilakeFileReader::load_index`, appends smaller files' vectors via `HnswIndex::insert_node`, then writes the merged file via `write_with_prebuilt_hnsw`. Falls back to `compact` (full rebuild) when: no dominant file exists, or the dominant file's HNSW cannot be loaded (IVF-PQ, `IndexStatus::Indexing`, corrupt). `run()` now calls `compact_incremental` by default.

- **Speedup**: for a 90 % / 10 % dominant split at N = 1 M vectors (dim = 1536), incremental compaction reduces HNSW build cost from O(N log N) to O(N_dom) deserialization + O(N_small × log N_dom) — approximately **7× faster** than full rebuild.

- **Iceberg V3 format-version support (Phase A)** — `TableProperties::format_version: u8` (default `2`) propagated through all catalog backends and `TableWriter::create_or_open`. `IcebergMetadata::new()` and `write_manifest_file()` emit `"format-version": 3` when `format_version=3`. CLI: `ailake create --format-version 3`. Python: `TableWriter(format_version=3)`. V3 tables are append/update compatible out of the box; equality deletes not implemented (Phase B+). V2 default preserves full backward compatibility.

- **Multi-column partition specs + truncate transform (Phase K)** — `VectorStoragePolicy::partition_fields: Vec<PartitionDef>` replaces the single-column `partition_by` path for tables that need compound partition keys or non-identity transforms. When non-empty, `partition_fields` takes precedence over `partition_by` at table creation time and generates a multi-field Iceberg schema + partition spec in `metadata.json`. `PartitionDef { column, transform, column_type }` in `ailake-core/src/schema.rs` supports `"identity"` (pass-through) and `"truncate[W]"` (string: first W chars; int/long: round down to multiple of W). Write path: `apply_partition_transforms` in `ailake-query/src/writer.rs` splits `partition_value` by `\x1f`, applies each field's transform in order, and rejoins — so a two-column spec with `partition_value = "agt-007\x1f20250618"` stored as `"agt-007\x1f2025"` when the second field uses `truncate[4]`. Avro encode: `write_manifest_file` now iterates all spec fields and emits one partition union slot per field. Avro decode: `read_manifest_file` collects all partition field values and joins them with `\x1f` into the single `partition_value: Option<String>` field — backward-compatible with single-column manifests (no join needed). Scan path unchanged: `partition_filter` is a compound `\x1f`-separated string matched via full equality against `DataFileEntry::partition_value`. Python: `TableWriter(partition_fields=[("agent_id", "identity", "string"), ("ts", "truncate[4]", "string")], partition_values={"agent_id": "agt-007", "ts": "20250618"})` — `partition_values` dict is converted to compound string in field order; `partition_value` string overrides if both provided. `PartitionDef` exported from `ailake-core`. 1 new test: `multi_column_partition_roundtrip` verifies two-column string+truncate encode/decode roundtrip in `avro_manifest::tests`.

- **`first_row_id` continuity post-compaction** — `commit_snapshot` previously overwrote `first_row_id` on every file unconditionally, causing `next_row_id` to balloon with each compaction cycle (35 rows compacted 10× → `next_row_id` = 350 instead of 35). Fix: (1) `commit_snapshot` (hadoop.rs) now skips files with a pre-set `first_row_id` and only advances the counter for files without one; (2) `compact`, `compact_incremental`, and `compact_deferred` in `ailake-query/src/compaction.rs` compute `min(source_files.first_row_id)` and set it on the merged output entry before committing — `commit_snapshot` respects it and does not advance `next_row_id`. For `compact_incremental` specifically, the dominant file's `first_row_id` is used (dominant rows go first in the merged output). V2 tables and source files without `first_row_id: None` fall back to the existing fresh-allocation path (unchanged). 1 new test: `compaction_preserves_first_row_id_and_next_row_id_does_not_balloon` verifies that after two writes (10+25 rows) and one compaction, the merged file keeps `first_row_id=0` and a subsequent fresh write starts at `first_row_id=35`, not 70.

- **Partition Statistics Files (Phase J)** — `HadoopCatalog::commit_snapshot` now writes an Iceberg-compliant partition statistics Parquet file (`metadata/partition-stats-<snap_id>.parquet`) for every partitioned table commit. The file contains one row per distinct partition value with aggregate stats (`record_count`, `file_count`, `total_size_bytes`) covering ALL data files in the snapshot (reads every data manifest to compute cumulative totals, not just the current batch). Schema: `partition` group with one column per partition field (string type for identity partitions), plus `record_count`, `file_count`, `total_size_bytes` as `int64`. The stats path is referenced under `"partition-statistics"` in `metadata.json` (Iceberg spec §3.6), enabling Spark/Trino to skip partition scans for aggregation queries (`SELECT COUNT(*) WHERE partition_col = X`, `GROUP BY partition_col`) without reading data files. `IcebergPartitionStatsRef { snapshot_id, statistics_path, file_size_in_bytes }` added to `ailake-catalog/src/metadata.rs`; `IcebergMetadata::partition_statistics: Vec<IcebergPartitionStatsRef>` serialized as `"partition-statistics"` (omitted for unpartitioned tables). `write_partition_stats_parquet(spec, files)` in `ailake-catalog/src/avro_manifest.rs` uses Arrow + parquet-rs to produce a Snappy-compressed Parquet with the nested `partition` struct. Unpartitioned tables: no stats file written (unchanged). 3 new tests in `avro_manifest::tests` (basic roundtrip, empty files, aggregation). `parquet`, `arrow-array`, `arrow-schema` added to `ailake-catalog` dev/runtime deps.

- **Real Iceberg PartitionSpec (Phase I)** — `TableWriter(partition_by="agent_id")` (Python) and `ailake create --partition-by agent_id` (CLI) now emit a complete, Iceberg-compliant identity partition spec that Spark, Trino, and PyIceberg can use for partition pruning. Previously, `partition_specs` in `metadata.json` had an incorrect `source-id: 1000` that didn't reference any schema field. Phase I fixes three gaps: (1) the Iceberg schema in `metadata.json` now includes the partition column as a real field (`{"id": 1, "name": col, "required": false, "type": type}`) so `source-id: 1` resolves correctly; (2) `write_manifest_file` uses a dynamically-generated Avro schema for the `data_file.partition` record (`r102`) that matches the spec — the field is no longer always empty; (3) partition values are encoded natively in each manifest entry's `data_file.partition` record so Iceberg-aware engines can read them without the AI-Lake SDK. Backward compat: `read_manifest_file` still reads partition values from `key_metadata` JSON for old manifests; native Avro field takes priority when present. `PartitionField { source_id, field_id, name, transform, source_type }` and `PartitionSpec { spec_id, fields }` added to `ailake-catalog/src/provider.rs` and exported from `ailake-catalog`. `TableMetadata::partition_spec: Option<PartitionSpec>` populated by `to_table_metadata()` by joining `partition_specs` JSON with `schema_fields` for type resolution. `VectorStoragePolicy::partition_column_type: Option<String>` (default `"string"`) sets the Iceberg type of the partition column; passed through `TableProperties` to `IcebergMetadata::new`. Python: `TableWriter(partition_by="agent_id", partition_column_type="string")`. Supported types: "string", "uuid" (both encoded as Avro string), "int", "long". `build_manifest_entry_schema(spec: Option<&PartitionSpec>)` exported from `ailake-catalog` for downstream integrations.

- **Equality Delete Files (Phase H)** — `delete_where(catalog, store, table, col, values)` in `ailake-query/src/delete.rs` logically deletes all rows where `col` equals any value in `values` without rewriting data files. Writes an Iceberg equality delete Avro file (`metadata/eq-del-<snap_id>.avro`) containing one row per delete predicate, then commits a `Delete` snapshot that inherits existing data manifests and appends a new delete manifest (`content=1` in the manifest list) pointing to the delete file (`content=2` in the manifest entry). `EqualityDeleteFile { path, equality_ids, record_count, file_size_bytes }` struct added to `ailake-catalog`. `CatalogProvider::list_equality_deletes(table, snapshot_id)` default method returns `Vec<EqualityDeleteFile>`; `HadoopCatalog` implements it by reading delete manifests from the snapshot's manifest list. `write_manifest_list_multi_typed` replaces `write_manifest_list_multi` with per-manifest `content` type (0=data, 1=delete); `read_manifest_list_typed` returns `(path, content)` pairs; `list_files` filters to content=0 manifests only. `write_equality_delete_avro(col_name, field_id, iceberg_type, values)` writes the Avro delete file with a dynamic schema embedding `field-id` for Spark/Trino compatibility. `write_equality_delete_manifest(files, snap_id, seq)` writes a delete manifest with `content=2` entries and `equality_ids` populated. `EqualityDeleteFilter` in `ailake-query/src/equality_delete.rs`: loads all delete files from the store, builds `HashMap<col, HashSet<values>>`, exposes `should_delete_row(batch, row_idx)` for per-row HNSW checks and `apply(batch)` for batch-level filtering. Scanner integration: `EqualityDeleteFilter::from_files` called once per `search()` / `search_text()` call; in HNSW path, `need_parquet` extended to include `!eq_del_filter.is_empty()` so parquet data is available for per-row checks; filter applied in flat-scan loop, HNSW result loop, and `search_text` row loop. `NewSnapshot::equality_delete_files: Vec<EqualityDeleteFile>` added (all existing struct literals updated with `equality_delete_files: vec![]`). Python: `ailake.delete_where(path, column, values)`. CLI: `ailake delete-where --col document_id --vals "doc-a,doc-b"`. Field-id lookup from `schema_fields` for Iceberg-compatible Avro schema; falls back to `id=0` / `"string"` for old tables without schema fields.

- **Field defaults / Schema evolution without compaction (Phase G)** — `CatalogProvider::evolve_schema(&table, evolution)` applies `AddColumnRequest` / `RenameColumnRequest` in a single metadata-only `metadata.json` rewrite — no data files touched. `HadoopCatalog` implements it: clones current schema, assigns fresh field-ids from `last-column-id`, stores `initial-default` + `write-default` in the Iceberg field JSON, pushes a new schema entry with incremented `schema-id`, and atomically updates `current-schema-id` + `last-column-id`. Other backends return a "not supported" error (overridable). `SchemaField` struct added to `TableMetadata`; `to_table_metadata()` parses the current schema fields including `initial-default` and `write-default` values. `SchemaFiller::fill(batch, &schema_fields)` in `ailake-query/src/schema_filler.rs` detects columns absent from a file's Parquet schema and injects Arrow arrays filled with `initial_default` (or null); supports all Iceberg primitive types via `iceberg_type_to_arrow`. Wired into three scanner paths: flat-scan fallback, HNSW `need_parquet` read, and `search_text`. `SchemaEvolution` builder (`schema_evolution.rs`) provides a fluent API: `SchemaEvolution::new().add_column(req).rename_column(old, new)`. Python: `ailake.add_column(path, name, type, initial_default=…)`, `ailake.rename_column(path, old, new)`. CLI: `ailake evolve --add name:type [--initial-default JSON] [--rename old:new]`. `serde_json` added to `ailake-py` deps.

- **Puffin stats file with vector stats + BM25 Bloom filters (Phase F)** — V3 tables now emit an Apache Puffin stats file (`metadata/stats-<snap_id>.puffin`) on every `commit_snapshot`. The file contains two blob types: `ailake-vector-stats-v1` (centroid + radius per data file, enabling future cross-tool geometric pruning tooling) and `ailake-bm25-bloom-v1` (per-file Bloom filters for BM25 term presence, enabling file-level pruning for hybrid queries). `AilakePuffinWriter::write_stats` assembles the Puffin binary (`PFAc` magic, blobs, footer JSON, `footer_len` LE, `PFAc` footer); `AilakePuffinReader` decodes both blob types. `IcebergStatisticsRef` + `BlobRef` added to `IcebergMetadata` (V3 `statistics` field in `metadata.json`). V2 tables: no Puffin file written (backward-compatible). `BloomFilter` in `ailake-query/src/bloom.rs`: FNV-64a double-hashing (k=4), `with_capacity(n, fpr)`, `insert`, `may_contain`, `to_bytes`/`from_bytes` (no new dep). `BloomPruner::prune` in `pruner.rs`: skips files where no query term can appear (zero false negatives). `load_bloom_map` in `scanner.rs` fetches the Puffin file and applies bloom pruning after geometric pruning for hybrid queries. `TableWriter` builds one `BloomFilter` per `write_batch*` call when `bm25_text_column` is set; flushed as `bloom_filters` in `NewSnapshot` at `commit()`. `NewSnapshot::bloom_filters: Vec<(String, Vec<u8>)>` added (existing struct literals updated with `bloom_filters: vec![]`). `HadoopCatalog::commit_snapshot` writes the Puffin file and appends `IcebergStatisticsRef` to `IcebergMetadata`; `collect_vector_stats` decodes `centroid_b64` + `radius` from `DataFileEntry`. `tracing` added to `ailake-catalog` deps.

- **Timestamp nanosecond for `created_at` / `last_accessed_at` (Phase E)** — `created_at` (in `llm_columns`) and `created_at` / `last_accessed_at` (in `episodic_columns`) are now specified as `Timestamp(Nanosecond, Some("UTC"))` in Arrow — Iceberg maps this to `timestamptz`. `ailake_core::now_ns() -> i64` returns Unix epoch nanoseconds for populating these columns. Python binding: `ailake.now_ns()`. `MemoryDecayJob::apply_decay` now accepts `TimestampNanosecondArray` and `TimestampMicrosecondArray` in addition to the legacy `Utf8` ISO-string format (backward compatible — old files still processed correctly). Internal helper `days_old_vec()` centralises the multi-format dispatch.

- **Iceberg V3 Row Lineage (Phase D)** — `DataFileEntry::first_row_id: Option<i64>` carries the globally unique first row ID for each data file. `IcebergMetadata::next_row_id: i64` tracks the next available row ID at the table level (serialised as `"next-row-id"` in `metadata.json`, V3 only). `HadoopCatalog::commit_snapshot` assigns `first_row_id` to each new file for V3 tables by consuming from `next_row_id`; V2 tables leave the field `None`. `MANIFEST_ENTRY_SCHEMA_STR` gains field `first_row_id` (field-id 141, `union(null, long)`); `write_manifest_file` emits it as a native Avro field; `read_manifest_file` recovers it via `parse_v3_first_row_id` with `AilakeEntryExt` JSON as fallback. Row IDs are monotonically increasing and non-overlapping across files in the same table — enables batch audit, CDC pipelines, and cross-file row deduplication without opening Parquet files.

- **Iceberg V3 Deletion Vector read support (Phase B)** — `DataFileEntry::deletion_vector: Option<DeletionVector>` carries DV pointer (Puffin path + offset + length + cardinality). `ailake-catalog`: `parse_v3_deletion_vector()` extracts native V3 Avro `deletion_vector` field from manifests written by Spark/Trino/PyIceberg; AI-Lake-written DVs stored in `AilakeEntryExt` JSON (Phase C write support planned). `ailake-query/src/dv.rs`: `load_deletion_vector(store, dv)` fetches the Roaring Bitmap blob via range GET (`offset..offset+length`), no full Puffin footer parse needed. Scanner (`scanner.rs`) loads DV bitmap once per file and masks deleted `row_id`s in both flat-scan and HNSW result paths. DV fetch failure: warn + continue without mask (safe degradation). Zero impact on V2 tables — `deletion_vector` field defaults to `None`.

### Tests

- `equality_delete::tests::empty_filter_is_no_op` — `EqualityDeleteFilter` with no predicates leaves batch unchanged.
- `equality_delete::tests::single_value_deleted` — one delete predicate removes the matching row; remaining rows intact with correct indices.
- `equality_delete::tests::multiple_values_deleted` — two delete predicates in the same set remove both matching rows.
- `equality_delete::tests::column_absent_from_batch_is_skipped` — filter column not in batch → no rows removed (no false deletes after schema evolution).
- `equality_delete::tests::numeric_column_deletion` — Int32 column values matched by string-normalised predicate set.
- `avro_manifest::tests::equality_delete_manifest_roundtrip` — `write_equality_delete_manifest` + `read_equality_delete_manifest` round-trips path, `equality_ids`, `record_count`, `file_size_bytes` for a single-column delete entry.
- `avro_manifest::tests::read_manifest_list_typed_returns_content` — `write_manifest_list_multi_typed` with content=0 + content=1 entries; `read_manifest_list_typed` returns correct `(path, content)` pairs.
- `metadata::tests::partition_spec_written_to_metadata_json` — `IcebergMetadata::new` with `partition_by="agent_id"` emits correct `schemas`, `partition-specs`, `default-spec-id`, `last-column-id`, `last-partition-id`; `to_table_metadata()` reconstructs `PartitionSpec` with correct `source_type` resolved from schema fields.
- `metadata::tests::unpartitioned_table_has_no_partition_spec` — unpartitioned table returns `None` for `partition_spec`.
- `metadata::tests::partition_spec_int_type` — `partition_column_type="int"` written to schema and resolved via `to_table_metadata()`.
- `avro_manifest::tests::partition_spec_native_roundtrip` — string identity partition value encoded in native `data_file.partition` Avro record and decoded back by `read_manifest_file`.
- `avro_manifest::tests::partition_spec_int_native_roundtrip` — int identity partition value ("7") encoded as Avro `int` union and decoded to string via `to_string()`.
- `avro_manifest::tests::build_manifest_entry_schema_no_spec` — `None` spec produces empty `r102` partition record.
- `avro_manifest::tests::build_manifest_entry_schema_with_string_spec` — string spec injects `tenant_id` field with `field-id=1000` and `["null","string"]` union type.
- `schema_filler::tests::no_op_when_no_schema_fields` — `SchemaFiller::fill` is a no-op when `schema_fields` is empty.
- `schema_filler::tests::no_op_when_all_columns_present` — no extra columns added when all schema fields match.
- `schema_filler::tests::injects_missing_column_with_null_default` — missing `Float32` column injected with all-null values when `initial_default` is `None`.
- `schema_filler::tests::injects_missing_column_with_value_default` — missing `Float32` column filled with `0.5` from `initial_default`.
- `schema_filler::tests::injects_string_column_with_default` — missing `Utf8` column filled with `"uncategorized"` from `initial_default`.
- `puffin::tests::vector_stats_roundtrip` — `AilakePuffinWriter::write_stats` + `AilakePuffinReader::read_vector_stats` round-trip preserves centroid, radius, and path for 2 files.
- `puffin::tests::bm25_bloom_roundtrip` — Puffin round-trip with BM25 bloom entries; recovered bloom bytes header (`num_bits=1024`) verified intact.
- `puffin::tests::empty_bloom_produces_no_bloom_blob` — empty bloom list → no `ailake-bm25-bloom-v1` blob in footer.
- `puffin::tests::footer_size_matches_actual` — `footer_size` declared in `PuffinStatsResult` matches actual bytes in file.
- `bloom::tests::*` — 5 tests covering `with_capacity`, `insert`, `may_contain`, FPR ≤ 5%, `to_bytes`/`from_bytes` roundtrip.
- `memory_decay::tests::apply_decay_handles_timestamp_nanosecond` — `apply_decay` on a batch with `Timestamp(Nanosecond, UTC)` column; verifies day-count matches the legacy Utf8 path for the same date.
- `memory_decay::tests::now_ns_is_recent` — `now_ns()` returns a value greater than 2025-01-01 UTC in nanoseconds.
- `hadoop::tests::v3_assigns_first_row_id_monotonically` — two-commit sequence on a V3 table; second file's `first_row_id` equals the first file's `record_count` (= 10).
- `hadoop::tests::v2_does_not_assign_first_row_id` — V2 table commit leaves `first_row_id: None` on all files.
- `avro_manifest::tests::first_row_id_roundtrip_v3` — `write_manifest_file` → `read_manifest_file` preserves `first_row_id = Some(5000)`.
- `avro_manifest::tests::first_row_id_none_for_v2` — V2 manifest round-trips with `first_row_id = None`.
- `metadata::tests::format_version_v3_emitted` — `IcebergMetadata::new(..., 3)` serialises `"format-version": 3` and round-trips correctly.
- `metadata::tests::format_version_defaults_to_v2` — V2 is the default when `format_version=2`.
- `hnsw::tests::insert_node_extends_existing_graph` — inserts a 4th node and verifies nearest-neighbour correctness.
- `hnsw::tests::insert_node_normalized_cosine` — insert with unnormalised input; node is pre-normalised internally.
- `hnsw::tests::insert_node_into_single_node_graph` — insert into a 1-node graph (edge case: entry point with no neighbours yet).
- `compaction::tests::compact_incremental_merges_dominant_plus_small` — 6-row dominant + 2-row small file; verifies merged row count, dominant rows first, HNSW searchable with correct RowIds after incremental insertion.
- `compaction::tests::compact_incremental_falls_back_when_no_dominant` — 50/50 split triggers full-rebuild fallback; merged file still valid.
- `dv::tests::load_dv_roundtrip` — writes bitmap bytes at a simulated Puffin offset; `load_deletion_vector` fetches via range GET and verifies all deleted row IDs.
- `dv::tests::has_deletions_detects_overlap` — `has_deletions` returns true iff any row_id in the candidate set appears in the bitmap.

- **Iceberg V3 Deletion Vector write support (Phase C)** — `ailake-query/src/delete.rs`: `delete_rows(catalog, store, table, file_path, &[u32])` logically deletes rows from a V3 table without modifying the Parquet file. Flow: (1) verifies `format-version=3`; (2) reads current file list; (3) merges new row IDs into existing DV bitmap (or creates new one); (4) serializes via `PuffinWriter::write_single_dv` into a minimal Puffin `.dvd` file at `{table_location}/metadata/dv-{snap_id}.dvd`; (5) commits `SnapshotOperation::Replace` snapshot carrying all files with the updated `deletion_vector` pointer. `PuffinWriter` produces a spec-compliant Puffin file (magic `PFAc` + blob + footer JSON + footer_len LE + magic). Multiple `delete_rows` calls accumulate — each call reads the existing bitmap and merges. CLI: `ailake delete-rows --table t --file data/part-00001.parquet --rows 5,10,42`. Python: `ailake.delete_rows(table_path, file_path, [5, 10, 42])`. V2 tables: returns `InvalidArgument` error.

### Tests (Phase C additions)

- `delete::tests::writes_dv_and_manifest_reflects_cardinality` — end-to-end: commit file, delete 3 rows, verify DV in manifest, verify bitmap content via `load_deletion_vector`.
- `delete::tests::merges_with_existing_dv_across_calls` — two sequential `delete_rows` calls accumulate into a single bitmap (4 total deleted rows).
- `delete::tests::rejects_v2_table` — `InvalidArgument` error when table is format-version 2.
- `delete::tests::noop_when_row_ids_empty` — empty `row_ids` slice returns immediately without writing any DV.
- `delete::tests::puffin_magic_and_structure_valid` — verifies Puffin file starts/ends with magic `PFAc` and bitmap bytes at declared offset decode correctly.

### Docs

- **`docs/guides/DBT_INTEGRATION.md`** — complete dbt integration guide covering: project layout; global vars (`ailake_vec_col`, `ailake_dim`, `ailake_metric`, `ailake_precision`); `ailake_write_batch` adapter macro (Spark / Trino / DuckDB); `ailake_compact` operation macro; full model chain `stg_documents → int_chunks → ailake_embeddings` (incremental append); three embedding generation patterns (Spark UDF, pre-computed table, Python dbt model); dbt recall assertion test via `ailake_search()`; Spark cluster configuration; Trino plugin deployment; known limitations table.
- **`docs/architecture/WORKSPACE.md`** — dbt guide marked delivered; DuckLake deferred with rationale (C++ dep + `HadoopCatalog` coverage).


- **Flink `ailake_search_text_json` binding** — `AilakeNativeLib.kt` now declares `ailake_search_text_json` JNA function. `AilakeNativeLoader.kt` adds `searchText(warehouse, namespace, table, queryText, topK, textColumn, partitionFilter)` Kotlin wrapper. Mirrors the C-ABI function added to `ailake-jni`. AilakeVectorTableSource unaffected (vector-only Flink source remains unchanged).

- **Flink `search()` hybrid params** — `AilakeNativeLoader.search()` gains `hybridText: String?`, `textColumn: String`, `bm25Weight: Float` optional params. When `hybridText != null`, includes `hybrid_text`/`text_column`/`bm25_weight` in the `ailake_search_json` request payload. Backward-compatible — all existing callers pass defaults.

- **DuckDB `ailake_search_text()` table function** — new SQL function in `duckdb-ailake/src/ailake_search_text.cpp`. Pure BM25 full-text search from DuckDB: `SELECT * FROM ailake_search_text('path', 'rust programming', 10, text_column:='chunk_text') ORDER BY distance`. Backed by new `ailake_search_text_json` C-ABI function in `ailake-jni`. Returns `(row_id BIGINT, distance FLOAT, file_path VARCHAR)` where `distance` = negated BM25 score. Graceful degradation (0 rows) when native lib not loaded.

- **DuckDB `ailake_search()` hybrid BM25+vector** — `ailake_search()` now accepts three new named params: `hybrid_text VARCHAR`, `text_column VARCHAR` (default `'chunk_text'`), `bm25_weight FLOAT` (default `0.5`). Backed by new fields in `ailake_search_json` C-ABI protocol (backward-compatible; missing = `null` = pure vector). Example: `SELECT * FROM ailake_search('path', query, 10, hybrid_text:='rust programming', bm25_weight:=0.4)`.

- **`ailake_search_json` protocol extended** — `ailake-jni/src/lib.rs`: `do_search()` accepts `hybrid_text`, `text_column`, `bm25_weight`; `ailake_search_json` Req struct adds these three optional fields with serde defaults (backward-compatible). All existing callers (Spark/Trino/Flink/Go) pass `null` and get pure vector search unchanged.

- **`ailake_search_text_json` C-ABI function** — new `#[no_mangle]` export in `ailake-jni/src/lib.rs`. JSON protocol: `{"warehouse","namespace","table","query_text","top_k","text_column","partition_filter"}`. Returns `{"ok":true,"results":[{"row_id","distance","file_path"}]}`.

- **Python type stubs updated** (`ailake-py/python/ailake/_ailake.pyi`) — all new Phase 5/8/9 APIs now reflected in stubs: `bm25_text_column` param in `TableWriter.__init__`; `extra_columns` param in `write_batch` / `write_batch_auto_deferred` / `write_batch_idempotent`; `hybrid_text` / `text_column` / `bm25_weight` params in `search()`; new `search_text()` function; new `WorkingMemoryBuffer` class; new `decay_memories()` function. `SearchQuery` and module-level `search()` in `__init__.py` now forward `hybrid_text` / `text_column` / `bm25_weight` to the Rust `_search_raw()` binding.

- **`WorkingMemoryBuffer`** — bounded in-memory FIFO queue for agent short-term memory (`ailake-query/src/mem_table.rs`). Stores at most `max_rows` `(text, embedding, importance)` tuples; evicts oldest on overflow. `search(query, top_k)` brute-force cosine flat scan; `drain_to_table(&mut TableWriter)` persists all entries and clears the buffer. Python: `ailake.WorkingMemoryBuffer(max_rows=1000)`. Replaces MemTable for single-session agents; cascade pattern: buffer first, drain to AI-Lake when full.

- **`MemoryDecayJob`** — async recomputation of `recency_weight = exp(-λ × days_since_access)` for episodic memory tables (`ailake-query/src/memory_decay.rs`). Reads `last_accessed_at` column (ISO 8601 string), rewrites each data file with updated `recency_weight`, commits a new `Overwrite` snapshot. Python: `ailake.decay_memories(path, decay_lambda=0.1)` returns number of updated files. No new crate deps (JDN date arithmetic inline, no chrono).

- **`extra_columns` support in `write_batch` / `write_batch_auto_deferred` / `write_batch_idempotent`** — all Python write methods now accept `extra_columns: dict[str, list]` keyword argument. Column types are inferred from the first element: `bool` → `Boolean`, `float` → `Float32`, `int` → `Int64`, `str` / other → `Utf8`. Enables writing `EpisodicMemorySchema`, `ToolCallSchema`, and custom agent columns without constructing PyArrow schemas manually.

- **`search_text` and `WorkingMemoryBuffer` exported from `ailake` Python module** — `ailake.search_text(path, query_text, top_k, text_column, partition_filter)` and `ailake.WorkingMemoryBuffer` now in `__all__`. `ailake.decay_memories` added to module.

- **BM25 integration tests** — `tests/tests/hybrid_search.rs` (6 tests): `search_text_returns_most_relevant_doc`, `search_text_returns_top_k_limit`, `hybrid_search_rrf_returns_top_k`, `idf_stats_serialization_roundtrip`, `bm25_scorer_ranks_rust_docs_above_python`, `write_batch_auto_deferred_creates_file`.

- **`WorkingMemoryBuffer` unit tests** — 4 tests in `ailake-query/src/mem_table.rs`: eviction correctness, cosine ranking, empty buffer, drain-to-table roundtrip.

- **`MemoryDecayJob` unit tests** — 4 tests in `ailake-query/src/memory_decay.rs`: ISO date parse (epoch, known date, short string error), `apply_decay` correct weight.

- **Demo fixture `ailake_bm25`** — `init_demo.py` now writes a BM25-indexed table (`_write_bm25`, 200 rows, `bm25_text_column="text"`). Path exposed in `demo_query.json` as `table_paths.bm25`. `main()` calls `_write_bm25` alongside other fixtures.

- **Notebook `09_hybrid_search.ipynb`** — 7 sections: BM25 write, `search_text` pure lexical, hybrid RRF, weight ablation, pre-built fixture, `WorkingMemoryBuffer` demo, `decay_memories` demo.

- **Hybrid BM25+vector search** — `SearchConfig::hybrid: Option<HybridConfig>` adds first-class BM25 lexical scoring to the vector search pipeline, eliminating the need for external FTS infrastructure for RAG/hybrid workloads:
  - `BM25Scorer` in pure Rust (no Tantivy dep) — BM25+ formula (k1=1.2, b=0.75), always-positive IDF, 50k-term vocabulary cap with automatic pruning.
  - `IdfStats` accumulated at write time from `TableWriter::with_bm25("chunk_text")` — serialized as zstd-compressed bincode, persisted to `metadata/ailake_bm25_stats.bin` alongside the Iceberg catalog. Updated on every `write_batch` / `write_batch_deferred` call. Compaction rebuilds stats accurately.
  - Hybrid pipeline: HNSW retrieves `candidate_pool` (default `10 × top_k`) candidates → BM25 scores each candidate using global IDF → fuses via **RRF** (default: `w_vec/(60+rank_vec) + w_bm25/(60+rank_bm25)`) or **linear combination** (min-max normalized). `HybridFusion::Rrf` and `HybridFusion::Linear` available.
  - `search_text()` — pure BM25 brute-force scan (no HNSW required): scans all Parquet files, scores rows by BM25, returns top-k. O(N) per call; documented trade-off vs. inverted index at scale.
  - Python: `ailake.TableWriter(bm25_text_column="chunk_text")` + `ailake.search(path, query, top_k, hybrid_text="my query", text_column="chunk_text", bm25_weight=0.5)` + `ailake.search_text(path, "my query", top_k)`.
  - Rust: `SearchConfig::default().with_hybrid(HybridConfig::new("my query").with_text_column("chunk_text"))`.
  - All other bindings (JNI, CLI, serve) pass `hybrid: None` (backward-compatible).

- **`NormalizedCosine` + F16 in-memory HNSW** — `quantize_to_f16()` no longer skips `NormalizedCosine`. F16 is now used during HNSW graph traversal (half memory bandwidth, ~2× cache efficiency) for all metrics. After traversal, `HnswIndex::search()` re-scores the final `top_k` candidates with exact F32 from `flat_vecs` to correct F16 rounding errors (~0.001 error vs ~0.0002 true `1-dot` distance for very similar unit vectors). Re-score cost is O(top_k × dim) — negligible vs. O(ef × dim) traversal. Users can now pair `NormalizedCosine` + `pre_normalize=true` with `precision=F16` Parquet storage and F16 in-memory quantization simultaneously without precision trade-offs.

### Fixed

- **`quantize_to_f16()` was no-op for `NormalizedCosine`** — previous implementation returned early for this metric, forcing in-memory HNSW search to use F32 (double memory bandwidth). Fixed by removing the guard and adding an exact F32 re-score pass over the final top-k candidates inside `HnswIndex::search()` when `flat_vecs_f16` is populated.

- **`AilakeFileWriter::write_single_pass()` / `write_multi_single_pass()`** — single-pass streaming write that emits a valid AI-Lake file without any footer seek or second Parquet write. Safe for append-only / write-once destinations (HDFS strict mode, piped stdout, object stores that do not support partial PUT). Readers bootstrap the AILK section offset from the `AilakeTrailer` (24 bytes immediately before the Parquet footer); no `ailake.footer_offset` KV injection needed.
- **`AilakeFileReader` trailer bootstrap fallback** — `ailk_offset()` / `ailk_offset_for_column()` now fall back to reading the `AilakeTrailer` when `ailake.footer_offset` is absent from the Parquet KV. Trailer bytes are always present in the initial footer range-GET (same bytes already fetched to read the Parquet footer), so the fallback costs no extra I/O vs the KV path. Backward-compatible: existing files with KV still use the faster KV path.
- **`parquet_footer_start()` promoted to `pub`** in `ailake-file::footer` — shared by writer and reader; re-exported from `ailake-file` root.
- **`ToolCallSchema`** — extends `LlmContextSchema` with agent fields: `agent_id: Uuid`, `session_id: Uuid`, `step_index: u32`, `tool_name: String`, `tool_input_json: String`, `tool_output_json: String`, `outcome: Enum(Success, Failure, Timeout)`, `latency_ms: u32`. Enables vector search over tool call history ("when did tool X fail in similar contexts?"). Exposed in `ailake-core/src/schema.rs` as `ToolCallSchema` + `ToolCallOutcome` enum.
- **`EpisodicMemorySchema`** — extends `LlmContextSchema` with episodic memory fields: `recency_weight: f32` (decays over time via `exp(-λ * days_since_access)`), `access_count: u32`, `last_accessed_at: Timestamp`, `importance_score: f32` (agent-defined). Scoring at search time: `final_score = distance * recency_weight * importance_score`. Schema struct in `ailake-core`.
- **Injectable `ScoreFn` for hybrid search scoring** — `SearchConfig` gains `score_fn: Option<ScoreFn>` where `ScoreFn = Arc<dyn Fn(f32, &RecordBatch) -> f32 + Send + Sync>`. Agents inject recency, importance, or any contextual signal into the final ranking without rewriting the index. Applied in `ailake-query` merge loop after HNSW candidates collected.
- **Python `ailake.Agent` helper** — `ailake.Agent(table_path, embed_fn, agent_id=None)` with methods: `remember(text, importance=1.0)`, `recall(query, top_k=5)`, `log_tool_call(name, input, output, outcome="success", latency_ms=0)`, `assemble_context(query, max_tokens=4096)`. High-level abstraction over `TableWriter` + `search` + `ContextAssembler` for agent frameworks (LangChain, CrewAI, AutoGen). Hybrid scoring (distance × recency × importance) applied automatically in `recall()`.
- **Partition by `agent_id` — manifest-level file pruning** — `VectorStoragePolicy::partition_by: Option<String>` stores an Iceberg identity partition spec in `metadata.json`. At write time, `partition_value: Option<String>` (runtime-only, `#[serde(skip)]`) tags each `DataFileEntry` via `key_metadata` JSON in Avro manifests. At search time, `SearchConfig::partition_filter: Option<String>` prunes the manifest file list BEFORE geometric centroid pruning and HNSW load — per-agent isolated search with zero post-scan filtering. Python: `TableWriter(partition_by="agent_id", partition_value=uuid)` and `search_with_data(path, query, top_k, partition_value=uuid)`. `Agent.recall()` passes `self._agent_id` automatically.
- **Phase 9 propagated to all native plugins** — `partition_by`, `partition_value`, and `partition_filter` added to every SDK and connector (Spark, Trino, Flink, Go, C++, DuckDB, Airbyte, Airflow).
- **Demo fixture + notebooks Phase 9** — `_write_agent_memory` fixture, §24–§28 in `01_ailake_demo.ipynb`, new `08_agents.ipynb` (26 cells).
- **`CompactionConfig::max_files_per_pass`** — new field (default 20) caps the number of files merged in a single compaction run. `CompactionPlanner::plan()` now sorts candidates by size ascending and truncates to this limit, bounding peak RAM and HNSW rebuild CPU cost to O(max_files_per_pass × avg_file) rather than O(whole_table). Set to `usize::MAX` to restore prior unbounded behaviour.
- **`CompactionExecutor::compact_deferred()` + `run_deferred()`** — deferred compaction variant: merges input files and persists the merged Parquet immediately via `write_parquet_only()`, registers the output as `IndexStatus::Indexing`, commits the catalog snapshot, then spawns a background Tokio task to build the HNSW / IVF-PQ index and patch the entry to `IndexStatus::Ready`. Same pattern as `write_batch_deferred`. Use for large tables where inline HNSW rebuild (O(N log N)) would block the compaction job for minutes.
- **`CompactionIndexStrategy::ForceIvfPq` doc update** — documents the recommendation to use `ForceIvfPq` for large compactions (N > 100 000) on CPU-only machines where HNSW rebuild is prohibitive.

### Fixed

- **`write_multi()` double row-group write allocation** — previously, `write_multi()` allocated and wrote Parquet row groups twice (pass 1 for `footer_start` measurement, pass 2 for KV injection). The final assembly now takes row groups from pass 1 (`parquet_v1[..footer_start]`) and only the footer thrift from pass 2 (`parquet_v2[footer_start_v2..]`), freeing the pass-1 buffer before the large AILK index allocation. Peak resident memory reduced from `2 × parquet_size + ailk_size` to `parquet_size + ailk_size + footer_size` for the two-pass path.
- **`TableWriter::create_or_open` part_counter collision** — fix: initialize `part_counter` from `catalog.list_files().len()`.
- **DuckDB `ailake_search_multimodal` missing `partition_filter`** — added `partition_filter=` named parameter.
- **Python SDK `partition_filter` missing from `search()` and `search_multimodal()`** — all three Rust functions now accept `partition_filter: Option<String>` and pass it to `SearchConfig`.
- **Python SDK `score_fn` not wired in `SearchQuery`** — `_apply_score_fn()` helper added; `score_fn: Callable | None` exposed on `search()` and `SearchQuery`.
- **GPU flat-scan (`SearchSession.search_batch`) does not apply `score_fn`** — documented in `GPU_FFI_EVALUATION.md §9` and `SETUP.md §8F-7`.
- **Compaction parallel file reads** — `CompactionExecutor::compact()` (and `compact_deferred()`) now reads input files concurrently via `futures::future::try_join_all` instead of sequentially. Eliminates per-file latency stacking on S3 or high-latency object stores.
- **Stale `partition_value` / `partition_by` fields removed from compaction tests** — `DataFileEntry` and `VectorStoragePolicy` struct literals in `compaction::tests` referenced fields that no longer exist in the structs; test code was never compiled (cargo check skips test modules). All stale fields removed; tests now compile and pass under `cargo test`.

---

## [0.0.19] — 2026-06-17

### Added

- **`write_batch_multi_deferred`** — deferred variant of `write_batch_multi` for N-column multimodal ingest with GPU acceleration. Persists Parquet immediately and builds all N column HNSW indexes in a single background tokio task (`build_and_patch_multi_index`). During the build window, `SearchSession` serves the shard via GPU flat scan (CUDA/ROCm, exact) with automatic fallback to CPU flat scan. CAS retry loop patches both primary HNSW offsets and `extra_vector_indexes[].hnsw_offset/len` atomically on completion. Python: `writer.write_batch_multi_deferred(texts, [(spec, embs), ...])`. All N column embeddings are cloned into the task; recommended batch size: N×rows×dim×4 bytes fits in RAM. The gap between `write_batch_multi` (synchronous HNSW on CPU) and `write_batch_auto_deferred` (single-column) is now closed for multi-column GPU workloads.


- **`VectorModality` + `ailake.modality-<col>` property** — new `VectorModality` enum (`Text`, `Image`, `Audio`, `Video`) added to `ailake-core`. `VectorStoragePolicy` gains optional `modality: Option<VectorModality>` field (`#[serde(default)]`, backward-compatible). Written to Iceberg table properties as `ailake.modality-<col>` at `create_table` time. CLI `ailake create --modality text|image|audio|video` sets the tag on the primary column. Allows readers to select the correct HNSW by modality tag without inspecting vector data.
- **N generalized vector columns** — `AilakeFileWriter::write_multi` (existing) plus new CLI/Python surface: `ailake create --vector-cols embedding,image_embedding` is now possible via `VectorColSpec` in Python. Each column gets its own independent AILK section (header + centroid + HNSW + trailer) appended sequentially before the Parquet footer. Backward-compatible: existing single-column files unchanged.
- **Cross-modal fusion search (RRF)** — `search_multimodal()` in `ailake-query` accepts `&[ModalQuery]` (column + query vec + weight + dim) and fuses per-column ranked lists via Reciprocal Rank Fusion (`score = Σ weight_i / (60 + rank_i)`). Result `SearchResult.distance = -rrf_score` preserves sort-ascending semantics. Python: `ailake.search_multimodal(path, [(col, query, weight)], top_k)` returns `[{"row_id", "rrf_score", "file"}]`. `FusionMethod::Rrf` is the only method; enum is extensible. Per-column dims stored as `ailake.dim-<col>` / `ailake.metric-<col>` in Iceberg properties on first multi-column write commit; Python auto-detects per-query dim from these properties. `search()` dim validation now uses the right property key per column (fixes false `ModelMismatch` for secondary columns with different dims).
- **Python `VectorColSpec` class** — `ailake.VectorColSpec(column, dim, metric="cosine", modality=None)` specifies a vector column for multi-column writes and searches. Exposed in `_ailake` module alongside `TableWriter`.
- **`MultimodalContextSchema` + `multimodal_columns`** — `ailake-core` gains `MultimodalContextSchema` marker struct and `multimodal_columns` module with canonical column name constants: `MEDIA_URI`, `MEDIA_MIME`, `MEDIA_CAPTION`, `IMAGE_EMBEDDING`, `AUDIO_TRANSCRIPT`, `THUMBNAIL_B64`. Extends `LlmContextSchema` convention — schema enforced by column names, not code-gen. AI-Lake stores only URIs and embeddings; raw media lives in object storage.
- **Python `TableWriter.write_batch_multi()`** — `writer.write_batch_multi(texts, [(VectorColSpec, embeddings), ...])` writes a batch with N independent vector columns in a single call. Each column gets its own AILK section (HNSW + centroid) in the file footer. Modality tag from `VectorColSpec.modality` propagated to both Iceberg properties and Parquet field metadata.
- **CLI `ailake insert --vector-cols`** — multi-column insert: `ailake insert my_table file.parquet --vector-cols "embedding:1536:cosine,image_embedding:512:cosine:image"`. Each spec is `col:dim:metric[:modality]`. Reads each column independently from the source Parquet and calls `write_batch_multi`. Falls back to single-column `--embeddings` mode when `--vector-cols` is absent (no breaking change).

- **`airbyte-destination-ailake` package** — Airbyte CDK v3 destination connector that writes Airbyte records to AI-Lake vector tables. Each stream maps to one AI-Lake table at `{table_base_path}/{stream_name}/`. Embedding backends: `cmd` (external process via stdin/stdout JSON protocol), `openai` (OpenAI Embeddings API), `cohere` (Cohere Embed API), `http` (any OpenAI-compatible endpoint: Ollama, vLLM, LM Studio, Together.ai, Azure OpenAI). Config fields: `table_base_path`, `embed_mode`, `text_field` (dot-notation for nested), `embedding_dim`, `embedding_metric`, `embedding_model` / `embedding_model_version` (propagated to Iceberg properties), `batch_size`, `pre_normalize`, `pq_only`. Airbyte state messages trigger intermediate commits (durability guarantee). Ships with connector spec JSON, Dockerfile for Airbyte registry, and unit tests.
- **`http` embed mode** (`airbyte-destination-ailake`) — `HttpEmbedder` posts to any OpenAI-compatible endpoint using stdlib `urllib` (no extra dependencies). Request: `{"model": "...", "input": [...]}`. Response: `{"data": [{"embedding": [...]}]}`. Config: `http_url` (required), `http_model` (optional), `http_auth_header` (optional, `Authorization` header, `airbyte_secret`), `http_timeout` (default 60 s).
- **Airbyte destination demo notebook** — `tests/docker/demo/notebooks/06_airbyte_destination.ipynb`: config → CmdEmbedder → StreamWriter batch flush → simulate `Destination.write()` with state messages → vector search → DuckDB/Iceberg compat → embedding model tracking → migration → Airbyte platform usage snippets.
- **Standalone demo scripts** (`airbyte-destination-ailake/demo/`) — `demo_local.py` (no Docker/API keys, CmdEmbedder with `embed_cmd.py`), `demo_openai.py` (real OpenAI embeddings, `--table-path` / `--model` args), `embed_cmd.py` (shell-protocol helper: deterministic unit vectors from text hash).
- **Roadmap Phase 8 — Multimodal** — `ailake.modality` property per vector column, N generalized `VECTOR` columns with independent HNSW per file, cross-modal fusion search (RRF), `MultimodalContextSchema` with `image_embedding` / `media_uri` / `audio_transcript`. Note: raw `MEDIA` binary column excluded — AI-Lake is not a blob store; images live in S3, only vectors belong in AI-Lake. **Phase 8 complete.**
- **Roadmap Phase 9 — Agents / Episodic Memory** — `ToolCallSchema` (agent_id, tool_name, tool_input/output, step_index), `EpisodicMemorySchema` (recency_weight with exponential decay, importance_score), injectable hybrid scoring fn (`distance * recency * importance`), `agent_id` Iceberg hidden partitioning, `WorkingMemoryBuffer`, `MemoryDecayJob`, `ailake.Agent` Python helper for LangChain/CrewAI/AutoGen.
- **Phase 8 multimodal propagated to all native plugins** — `ailake_search_multimodal_json` C-ABI entry point added to `ailake-jni` (shared foundation for all JVM and DuckDB callers). Propagated to: Spark plugin (`searchMultimodal()` via JNA), Trino plugin (same pattern), Flink (`AilakeNativeLoader.searchMultimodal()`), DuckDB extension (`ailake_search_multimodal()` table function returning `row_id, rrf_score, file_path`). Go SDK gains `SearchMultimodal()` with `ModalQuery`/`RRFResult` types, `searchFileAtOffset()` helper for per-column HNSW lookup, and `ExtraVectorIndex` parsing in the catalog reader. C++ SDK gains `ExtraVectorIndex` struct in `catalog.hpp` (parsed from `key_metadata` JSON), and `search_multimodal()` in `ailake.hpp` with geometric pruning + per-column HNSW dispatch + RRF fusion.
- **Demo expanded — multimodal fixtures + notebooks** — `init_demo.py` gains `_write_multimodal()` fixture (200 rows, `embedding` dim=32 text + `image_embedding` dim=16 image via `VectorColSpec` + `write_batch_multi`; path saved to `demo_query.json`). `compose-demo.yml` adds `DEMO_MULTIMODAL_PATH=/data/ailake_multimodal` env var. `01_ailake_demo.ipynb` gains sections 21–23 (N vector columns + modality tagging, cross-modal `search_multimodal` + weight ablation, `MultimodalContextSchema` column constants); ToC updated to v0.0.19. New `07_multimodal.ipynb` (26 cells): full Phase 8 demo — `VectorColSpec`, `write_batch_multi`, Iceberg metadata inspection, single-column vs RRF search, weight ablation (100/0 → 0/100), pre-gen fixture, `MultimodalContextSchema` constants, multimodal LLM context assembly.

---

## [0.0.18] — 2026-06-15

### Added

- **`EmbeddingModelInfo.dim` + `.metric` fields** — `EmbeddingModelInfo` gains optional `dim: Option<u32>` and `metric: Option<VectorMetric>` fields with builder methods `with_dim()` / `with_metric()`. When set, stored as `ailake.embedding-model-dim` and `ailake.embedding-model-metric` in Iceberg properties (separate from the `ailake.embedding-model` name string). Backward-compatible: `dim`/`metric` default to `None`.
- **Per-file `embedding_model` in Avro manifests** — `DataFileEntry.embedding_model` and `AilakeEntryExt.embedding_model` added; per-file model identifier serialized to Avro `key_metadata` JSON. Enables detection of mixed-model tables during migration without reading `metadata.json`. Writer sets it from `VectorStoragePolicy.embedding_model` at every write site.
- **Writer model-name mismatch warning** — `TableWriter::create_or_open` warns via `tracing::warn!` when the incoming `embedding_model.name` differs from the model stored in existing Iceberg properties. Fires at open time (not per-write) since it's a table-level concern.
- **Query dim validation** — `search()` in `ailake-query/src/scanner.rs` now validates `query.len()` against `ailake.vector-dim` from table metadata and returns `AilakeError::ModelMismatch` before any I/O if they differ. Surfaces the stored model name (or dim) for clear error messages.
- **Pattern B: `TableWriter(embed_fn=callable)`** — Python `TableWriter.__init__` and `Table.__init__` / `open_table()` accept `embed_fn: Optional[Callable[[list[str]], list[list[float]]]]`. When set, `write_batch(texts)` and `Table.insert(texts)` may omit `embeddings`; the callable is invoked automatically. Rust side stores `embed_fn: Option<Py<PyAny>>` in `TableWriter` and calls it via `Python::attach` when `embeddings=None`.
- **`migrate_embeddings(on_progress=...)`** — Python `migrate_embeddings()` gains optional `on_progress: Optional[Callable]` kwarg. Called after each file with keyword args `files_done`, `files_total`, `rows_migrated`. Wraps the Python callable as `Arc<dyn Fn(MigrationProgress) + Send + Sync>` and passes to `MigrationJob.on_progress`.
- **`EmbeddingModelInfo` + `ModelMismatch` — model tracking across all bindings** — `EmbeddingModelInfo { name, version }` is stored in Iceberg properties as `ailake.embedding-model` (format: `"<name>"` or `"<name>@<version>"`). `TableWriter.validate_embedding_dim` now returns `ModelMismatch` error when vector dimensions differ, surfacing the conflicting model names. All bindings updated: Python `TableWriter(embedding_model=..., embedding_model_version=...)` and `open_table(embedding_model=...)` propagate model info; JNI `ailake_write_batch_json` accepts `"embedding_model"` field in the JSON envelope; CLI `ailake create` already stored `embedding_model: None` (unchanged); Go `TableInfo` gains `EmbeddingModel string` populated from `ailake.embedding-model` property.
- **`MigrationJob` — embedding model migration** (`ailake-query`) — `MigrationJob` re-embeds all chunks in a table via a user-supplied `embed_fn` and commits the result as a new Iceberg snapshot. Two strategies: `AtomicReplace` (lower peak storage, brief mixed-model window) and `DualWriteThenCutover` (2× peak storage, zero downtime). Optional `new_model: EmbeddingModelInfo` updates `ailake.embedding-model` in properties after migration. Progress callback included.
- **Python `migrate_embeddings()`** — `ailake.migrate_embeddings(path, old_column, new_column, embed_fn, ...)` wraps `MigrationJob` for Python callers. `embed_fn` is a Python callable `list[str] → list[list[float]]`. Exposed in `ailake.__all__`, `_ailake.pyi` stub updated with full docstring and signature.
- **CLI `ailake migrate`** — `ailake migrate <table> --embed-cmd <shell-cmd> [--old-column ...] [--new-column ...] [--strategy ...] [--model-name ...] [--model-version ...]`. The `--embed-cmd` shell command receives a JSON array of strings on stdin and must write a JSON array of float arrays to stdout. Prints per-file progress to stderr.
- **JNI `NormalizedCosine` metric** — `parse_metric` in `ailake-jni` now maps `"normalized_cosine"` / `"normalizedcosine"` to `VectorMetric::NormalizedCosine` (was falling through to `Cosine`). Spark/Trino/Flink callers can now use the pre-normalized fast path via `"metric": "normalized_cosine"` in the JSON envelope.
- **Go: `EmbeddingModel` tracking + dim validation** — `DataFileEntry` gains `EmbeddingModel string` populated from `embedding_model` in Avro `key_metadata` JSON. `ailakeEntryExt` gains `EmbeddingModel *string` field. `Search()` in `ailake-go/ailake.go` validates `len(query)` against `TableInfo.VectorDim` before any I/O and returns an error naming the stored model when mismatched.
- **C++: `embedding_model` tracking + dim validation** — `DataFileEntry` gains `std::string embedding_model` parsed from `key_metadata`. `TableInfo` gains `std::string embedding_model` loaded via `load_table()`. `search()` in `ailake.hpp` validates query dim against `info.vector_dim` and throws with model name when mismatched.
- **Trino: `embedding_model` propagation** — `ailake.embedding-model` catalog property flows through `VectorScanConnectorFactory` → `VectorScanConnector` → `VectorScanMetadata` → `AilakeIngestTableHandle.embeddingModel` → `AilakePageSink` → `AilakeNative.writeBatch()`. `AilakeNative.writeBatch` accepts `embeddingModel: String? = null` and includes `"embedding_model"` in the JSON envelope when set.
- **Flink: `embedding_model` propagation** — `embedding.model` config option added to `AilakeVectorConnectorFactory` (optional, no default). Flows through `AilakeVectorTableSink.embeddingModel` → `AilakeSinkFunction.embeddingModel` → `AilakeNativeLoader.writeBatch(embeddingModel = ...)`. `AilakeNativeLoader.writeBatch` accepts `embeddingModel: String? = null` and includes `"embedding_model"` in the JSON envelope when set.

- **Python SDK: `pq_only` + `ivf_residual` + `write_batch_auto_deferred` exposed** — `open_table()` and `Table` now accept `pq_only=True` (discard raw vectors post-index) and `ivf_residual=True` (residual encoding); `Table.write_batch_auto_deferred()` + async variant `write_batch_auto_deferred_async()` added; all three propagated to the underlying `_TableWriter` Rust extension. `_ailake.pyi` stubs updated with full parameter signatures and docstrings.
- **`EmbedFn` + `ProgressFn` type aliases** (`ailake-query`) — `pub type EmbedFn = Arc<dyn Fn(&[String]) -> AilakeResult<Vec<Vec<f32>>> + Send + Sync>` and `pub type ProgressFn = Arc<dyn Fn(MigrationProgress) + Send + Sync>` extracted into `ailake-query/src/migration.rs` and re-exported from `ailake_query`. Replaces all inline complex `Arc<dyn Fn(...)>` annotations in `MigrationJob.on_progress`, `ailake-cli`, and `ailake-py` — satisfies `clippy::type_complexity` (`-D warnings` in CI).
- **CI: Flink unit tests in `test-jvm` job** — `ci.yml` `test-jvm` job now runs `gradle -p ailake-flink test` alongside the existing Trino and Spark test steps. New test: `AilakeVectorConnectorFactoryTest.optionalOptionsIncludesEmbeddingModel` verifies `embedding.model` key is present in `optionalOptions()`.
- **CI: `check_jni_cabi.py` embedding_model coverage** — write-with-`"embedding_model"` test block added; verifies that `ailake_write_batch_json` with `"embedding_model": "test-model@v2"` returns `{"ok": true}` and produces a valid `snapshot_id`.
- **Tests: Trino embedding model propagation** (`trino-plugin`) — `AilakeIngestMetadataTest` gains `embeddingModelPropagatedToIngestHandle` and `embeddingModelNullByDefault`; `AilakePageSinkTest` gains `sinkWithEmbeddingModelFinishesGracefully` with `embeddingModel = "text-embedding-3-small@v1"`.
- **Demo notebooks — embedding model tracking sections** — `01_ailake_demo.ipynb` gains §18 (write with `embedding_model=`, inspect `ailake.embedding-model` in Iceberg metadata, trigger `ModelMismatch`), §19 (Pattern B via `embed_fn=`), §20 (`migrate_embeddings()` with `on_progress` and post-migration property verification); `02_duckdb.ipynb` gains §9 listing `ailake.embedding*` from `metadata.json`; `03_spark.ipynb` gains §8 filtering `ailake.embedding*` from `SHOW TBLPROPERTIES`; `04_trino.ipynb` gains §9 filtering `ailake.embedding*` from Trino `$properties`.
- **`init_demo.py` — `ailake_model_tracked` fixture** — fifth fixture table written at container startup: 100 rows, dim=32, `embedding_model="synthetic-embed-v1"`, `embedding_model_version="1.0"`. Path added to `demo_query.json` under `table_paths.model_tracked`.
- **Docker demo expanded** — `init_demo.py` now generates 4 fixture tables at startup: HNSW (standard), PQ-only (`pq_only=True`), Residual-PQ (`ivf_residual=True`), Deferred (`write_batch_auto_deferred`). `demo_query.json` extended with paths to all tables. Notebooks updated: `01_ailake_demo` gains sections 12–17 (IVF-PQ/PQ-only, Residual PQ, `write_batch_auto_deferred`, HNSW tuning, async API, storage estimator); `02_duckdb` gains per-file storage stats, F16 BLOB decode, Iceberg metadata JSON; `03_spark` gains time-travel `VERSION AS OF` and manifest file stats; `04_trino` gains AI-Lake table properties via `$properties`, `$files`, `$manifests`; `05_bigquery` gains F16 BYTES decode and production GCS + BigQuery Omni pattern.
- **Residual PQ** (`ivf_residual=true`) — encodes `vec - coarse_centroid` per cluster instead of raw vector; zero storage overhead vs. standard PQ; ~2–4 pp recall@10 improvement on typical embeddings (dim=1536, cosine). Enabled via `VectorStoragePolicy::ivf_residual`, `ailake create --ivf-residual` (CLI), `TableWriter(ivf_residual=True)` (Python), `IvfPqConfig::with_residual()` (Rust). Search uses per-cluster ADC table in all bindings.
- **`write_batch_auto_deferred`** — deferred variant of auto index selection; persists Parquet immediately (~200k vec/s) and builds HNSW or IVF-PQ index in a background Tokio task. Hardware detection: CUDA GPU / AMD ROCm / ≥8 CPU cores + ≥5k vectors → IVF-PQ (deferred); else HNSW (deferred). Shard served via flat scan until index is ready. Exposed via `TableWriter.write_batch_auto_deferred()` (Python and Rust).

### Fixed

- **CI `cargo fmt` diffs** — multiple Rust files had lines exceeding rustfmt line length after the embedding model tracking additions (`avro_manifest.rs`, `hadoop.rs`, `metadata.rs`, `snapshot.rs`, `main.rs`, `lib.rs`, `types.rs`, `ailake-py/src/lib.rs`, `compaction.rs`, `migration.rs`, `pruner.rs`, `scanner.rs`, `writer.rs`). Fixed by running `cargo fmt --all`.
- **CI `E0063` missing `embedding_model` field** — ten test-only `DataFileEntry` struct initializers across `ailake-catalog/src/avro_manifest.rs`, `hadoop.rs`, `snapshot.rs`, `ailake-query/src/compaction.rs`, and `pruner.rs` were missing the newly-added `embedding_model: None` field, causing compile errors. All initializers updated; spurious duplicate assignments accidentally introduced in production code of `avro_manifest.rs` were removed.
- **CI `clippy::type_complexity`** — three locations across `ailake-query/src/migration.rs`, `ailake-cli/src/main.rs`, and `ailake-py/src/lib.rs` used the same complex `Arc<dyn Fn(...) + Send + Sync>` type inline. Extracted into `EmbedFn` and `ProgressFn` aliases; `unused import: AilakeResult` in `ailake-py` removed after the alias replaced the inline type annotation.
- **Residual PQ backward compat** (`ailake-index`) — `#[serde(default)]` does not work with bincode v1 (positional serialization); old files missing trailing byte caused "unexpected EOF". Fixed by removing `residual` from `IvfPqSnapshot` struct (renamed `IvfPqSnapshotCore`) and appending a single trailing byte (`0x01` = residual) after bincode payload. `from_bytes` uses `bincode::deserialize_from` with a `Cursor` to detect the trailing byte; absence defaults to `false`.
- **Go IVF-PQ search correctness** (`ailake-go`) — `Search()` was using a single global ADC LUT for all probed clusters regardless of `Config.Residual`. For residual indexes this computed distance against raw query instead of per-cluster residual (`q - coarse_centroid`), producing wrong rankings. Fixed: per-cluster ADC via new `buildADCTable` helper; global LUT pre-computed once for non-residual path.
- **C++ IVF-PQ search correctness** (`ailake-cpp`) — same bug as Go: `ivfpq_search` built one global ADC LUT; residual indexes returned wrong results. Fixed: `IvfPqConfig` gains `residual` field; `deserialize_ivfpq` reads trailing byte via `r.remaining() > 0`; `ivfpq_search` uses per-cluster LUT via new `build_adc_lut` helper when `config.residual=true`.
- **Go `sqEuclidean` returns squared distance** (`ailake-go`) — `sqEuclidean()` was calling `math.Sqrt(sum)`, returning euclidean distance instead of squared euclidean. ADC lookup tables require squared distances; the sqrt caused wrong ranking for all IVF-PQ searches regardless of residual mode.
- **CLI `--ivf-residual` flag wired** (`ailake-cli`) — `ailake create --ivf-residual` was documented but the flag and its handler were absent; added to `Create` subcommand and propagated to `VectorStoragePolicy`.
- **JNI `ivf_residual` from JSON** (`ailake-jni`) — `ailake_write_batch_json` Req struct gains `#[serde(default)] ivf_residual: bool`; Spark/Trino/Flink callers can now enable residual PQ via `"ivf_residual": true` in the JSON envelope.
- **NaN-safe sort in IVF-PQ search** (`ailake-index`) — `partial_cmp(...).unwrap()` in sort closures panics if any distance is NaN (degenerate zero-vector input). Replaced with `total_cmp()` (Rust 1.62+) which imposes a total order over all f32 values including NaN.
- **`assert_eq!` → `Err` in `write_batch_multi_vec`** (`ailake-parquet`) — internal invariant check now returns `AilakeError::Parquet` instead of aborting the process on dim/precision mismatch.
- **footer.rs: remove `try_into().unwrap()`** (`ailake-file`) — `AilakeHeader::from_bytes` and `AilakeTrailer::from_bytes` used `b[x..y].try_into().unwrap()` to convert fixed-size slices; replaced with explicit `[b[x], b[x+1], ...]` array literals — no runtime failure path, no unwrap.

---

## [0.0.17] — 2026-06-12

### Added

- **PQ-only mode** (`keep_raw_for_reranking = false`) — when enabled, the raw F16 vector column is omitted from Parquet files entirely; only the AILK index blob is written. Storage reduction: ~98% for vector column (1M × dim=1536 F16: 3 GB → ~47 MB). Trade-off: reranking disabled, recall@10 ~93-95%. Exposed via `ailake create --pq-only` (CLI) and `TableWriter(pq_only=True)` (Python). Reader detects `ailake.pq_only=true` KV metadata and returns empty embeddings vec instead of erroring.
- **`ailake estimate` CLI** — pure-math storage estimator, zero I/O. Shows vectors + index bytes for all modes (F32, F16, I8, F16+IVF-PQ, I8+IVF-PQ, PQ-only) with reduction factor and recall@10. Supports K/M/B row-count suffixes and `--format json`. Example: `ailake estimate --rows 10M --dim 1536`.

### Fixed

- **pyo3 upgraded 0.24 → 0.29** — resolves RUSTSEC-2026-0176 (OOB read in `PyList`/`PyTuple` `nth`/`nth_back` iterators). `PyObject` (removed from prelude in 0.29) replaced with `Py<PyAny>`.
- **deny.toml** — removed stale `MPL-2.0` license allowance and `RUSTSEC-2021-0153` ignore entry; both were transitive via `lancedb` which moved to the separate `ailake-benchmark` repository.
- **`keep_raw_for_reranking` default corrected** — all production paths (CLI insert/compact/serve, JNI write, demo, integration tests, compat fixture) now correctly default to `true`; `false` is only set when `--pq-only` / `pq_only=True` is explicitly requested. Fixes compat CI failure where the `embedding` column was missing from the fixture Parquet file.
- **clippy `too_many_arguments`** — suppressed via `#[allow(clippy::too_many_arguments)]` on `TableWriter::new` in `ailake-py` (8 params required by PyO3 `#[new]` signature).
- **clippy `print_literal`** — `"Recall@10"` moved from `println!` argument into format string literal in `ailake estimate` table header; satisfies `-D clippy::print_literal` (CI was failing).

---

## [0.0.16] — 2026-06-11

### Added

- **Python full-read after search** — `ailake.search(..., fetch_data=True)` and `Table.search(..., fetch_data=True)` return a `SearchQuery` whose `.to_arrow()` / `.to_pandas()` / `.to_polars()` / async variants materialise a full `pyarrow.Table` with all columns including the embedding decoded as `FixedSizeList<Float32>` + `_distance: float32`. Backward-compatible: default `fetch_data=False` behaviour unchanged.
- **DuckDB extension** (`duckdb-ailake`) — C++ community extension exposing `ailake_search(table_path, query FLOAT[], top_k) → TABLE(row_id, distance, file_path)` and `ailake_write_batch(table_path, ids BIGINT[], embeddings FLOAT[][]) → BIGINT`. Bridges DuckDB to `libailake_jni.so` via `dlopen`/C-ABI — same JSON-envelope protocol as Spark and Trino plugins. Graceful degradation: search returns 0 rows when native lib not found. CI workflow `ci-duckdb.yml`.
- **DuckDB `ailake_scan()` — full-row table function** — `ailake_scan(path, query FLOAT[], top_k) → TABLE(col1, col2, ..., _distance)` returns all Parquet columns alongside distance. Schema inferred at bind time; streams STANDARD_VECTOR_SIZE chunks; graceful degradation when native lib not loaded. Backed by new `ailake_scan_json` C-ABI in `ailake-jni`.
- **Go `Scan()` — full-row fetch** (`ailake-go/scan.go`) — `Scan(catalog, namespace, table, query, opts)` = `Search()` + `FetchRows()`; reads Parquet rows for HNSW hits via `parquet-go` (pure Go, zero CGO); skips row groups with no target row IDs; auto-decodes F16 vector column to `[]float32`; returns `[]ScanRow{RowID, Distance, FilePath, Fields map[string]any}`.
- **Go unit tests for all packages** — `footer_test.go` (9 tests), `ailake_test.go` (10 unit + 3 integration tests), `distance_test.go` (6 tests), `catalog_test.go` (4 tests), `scan_test.go` (6 unit + 2 integration tests). 33 unit tests pass without fixture; 5 integration tests require `AILAKE_FIXTURE`.

### Fixed

- **DuckDB extension metadata format** — `append_extension_metadata.py` now writes the correct 8×32-byte field layout; fixes `InvalidInputException: metadata at the end of the file is invalid` when loading the extension.
- **DuckDB extension RTLD_GLOBAL / RTLD_DEFAULT** — `AilakeLib::load()` falls back to `dlsym(RTLD_DEFAULT, …)`; test files set `sys.setdlopenflags(RTLD_GLOBAL)` before `import duckdb`; fixes `undefined symbol` errors at dlopen time.
- **DuckDB extension C++ ABI** — `CMakeLists.txt` adds `_GLIBCXX_USE_CXX11_ABI=0` to match DuckDB manylinux wheels; fixes ABI mismatch undefined symbols.
- **DuckDB `allow_unsigned_extensions`** — must be passed via `duckdb.connect(config={...})`, not `SET` after connection starts.
- **DuckDB fixture path** — `FIXTURE.resolve()` now called in `test_scan.py` to convert relative env var to absolute path.
- **`LocalStore::new` file:// URI root** — strips the `file://` scheme before constructing the root `PathBuf`; fixes files landing in CWD instead of the intended directory.
- **`HadoopCatalog::list_files` on fresh table** — returns empty `Vec` when `current_snapshot_id` is `None`; previously errored on brand-new tables before any commit.
- **HNSW F16 quantization disabled for NormalizedCosine** — `HnswIndex::quantize_to_f16` skips F16 downcast for `NormalizedCosine`; F16 rounding error exceeded true inter-vector distance for pre-normalized unit vectors.
- **Python `SearchQuery` repr** — pending state renders as `SearchQuery(top_k=N, pending)`, executed state as `SearchQuery(N results, top_k=K)`.
- **Python `to_arrow()` pointer-only** — returns `pyarrow.Table` (was `RecordBatch`); distance column is `distance` (was `_distance`); columns are `row_id, distance, file`.
- **Go `HadoopCatalog.tableDir`** — removed `.db` suffix; standard Iceberg HadoopCatalog uses `{warehouse}/{namespace}/{table}` not `{namespace}.db`.
- **Go `searchFile` path resolution** — same `.db` bug fixed in relative path fallback.
- **Go `key_metadata` Avro union** — `goavro` v2 returns `["null","bytes"]` union as `map[string]interface{}{"bytes": []byte{...}}`; raw `[]byte` assertion always failed → `HnswOffset` nil → all files silently skipped.
- **Go `decodeCentroid`** — Rust encodes `centroid_b64` as dim×4 bytes (vector only); radius is a separate JSON field. Old code stripped last float as radius → centroid had dim-1 elements → index-out-of-range panic.
- **Go `searchFile` AILK header offset** — `key_metadata.hnsw_offset` is absolute position of HNSW blob (after header + centroid); Go was reading header at blob position → "bad magic". Fixed: `ailk_header = hnsw_offset - HeaderSize - (dim+1)*4`.

### Tests

- **`check_ailake_py.py` §8–13** — full-read mode (`fetch_data=True`), `write_batch_idempotent`, `to_polars()`, multiple commits, `pre_normalize=True`, HNSW tuning, edge cases, pointer-only column schema.
- **`tests/fixtures/write_fixture.py`** — fixture writer for `ci-duckdb.yml`: 1 000 rows dim=128 cosine F16.
- **Docker demo (`tests/docker/`)** — all 5 notebooks (`01_ailake_demo` through `05_bigquery`) execute cleanly via `nbconvert`; verified with Spark 3.5 local mode, Trino 446 + Nessie, and goccy BigQuery emulator.

### CI

- `ci-duckdb.yml`: cmake build + Python integration tests for DuckDB extension + `ailake_scan` integration tests.
- `ci-go.yml`: unit step runs `go test ./...` (integration tests auto-skip without `AILAKE_FIXTURE`); integration step runs all tests with fixture.

---

## [0.0.15] — 2026-06-09

### Added

- **Python fluent API** — `open_table(path, **kwargs) → Table`, `Table.insert(texts, embeddings)`, `Table.search(query, top_k) → SearchQuery`, `SearchQuery.limit(n)`, `.to_list()`, `.to_pandas()`, `.to_polars()`. Chainable, DataFrame-native; accepts numpy arrays anywhere a vector is expected.
- **Python async API** — `Table.insert_async`, `Table.commit_async`, `SearchQuery.to_list_async`, `to_pandas_async`, `to_polars_async`; backed by `run_in_executor` so asyncio event loop is never blocked; supports `asyncio.gather` for parallel searches.
- **Python Jupyter repr** — `Table._repr_html_()` renders a styled card with path and vector config; `SearchQuery._repr_html_()` renders pending state or results table inline in notebooks.
- **Python type stubs** (`ailake/_ailake.pyi`) — full stubs for `TableWriter`, `search`, `assemble_context` with `Sequence`-based input types; `_Embeddings`/`_Vector` aliases in `__init__.py`; `py.typed` PEP 561 marker; `mypy` passes with zero errors.
- **Python mixed module layout** — Rust extension compiled as `ailake._ailake`; public Python surface at `ailake-py/python/ailake/__init__.py`; maturin `python-source = "python"` picks up the layout automatically; wheels include both Rust extension and Python wrapper.
- **`ailake.TableWriter` backward-compat re-export** — existing code using `ailake.TableWriter(path, ...)` continues to work unchanged.
- **Spark INSERT INTO** (`ailake-spark`) — `AilakeWriteBuilder`, `AilakeBatchWrite`, `AilakeDataWriter`, `AilakeDataWriterFactory` via Spark DataSourceV2 `WriteBuilder`; `AilakeCatalog` implements `StagingTableCatalog`; `INSERT INTO ailake_table SELECT ...` triggers native write path.
- **Trino INSERT INTO** (`ailake-trino`) — `AilakePageSink`, `AilakePageSinkProvider`, `AilakeIngestTableHandle` via Trino SPI `ConnectorPageSink`; `INSERT INTO` DML routes through `ailake_write_batch_json` JNA bridge.

### Fixed

- **Trino SPI 430**: `ConnectorPageSinkContext` → `ConnectorPageSinkId` in `AilakePageSinkProvider` (removed in Trino 430+).
- **Spark/Scala 2.12**: `def buildForBatch()` → `override def buildForBatch()` — Scala 2.12 requires explicit `override` for concrete Java default methods.
- **Scala 2.12 compat**: `scala.jdk.CollectionConverters` → `scala.collection.JavaConverters` in test files (`jdk.CollectionConverters` requires Scala 2.13+).
- **`release.yml`: sync version bump back to develop** — after bumping `Cargo.toml` on `main`, the action now merges `origin/main → develop` automatically.
- **`release.yml`: idempotent `publish-crates`** — exit code 10 (crate already exists on crates.io) treated as success; re-runs skip already-published crates.
- **`release.yml`: idempotent tag + GitHub Release creation** — both steps check for existing tag/release and skip if already present.
- **`release.yml`: non-fast-forward push rejection** — `git pull --rebase origin main` before push in version bump step.

### Tests

- Unit and integration tests for Trino INSERT INTO (`AilakePageSinkTest`, `AilakeIngestMetadataTest`, `AilakeWriteBatchIntegrationTest`).
- Unit tests for Spark INSERT INTO (`AilakeWriteSupportTest`, `AilakeCatalogTest`, `AilakeWriteBatchIntegrationTest`).
- `check_ailake_py.py` updated: covers legacy `TableWriter` API, fluent chain, `SearchQuery` repr + `_repr_html_`, context manager, async API, `asyncio.gather`.

### CI

- `test-jvm` job in `ci.yml` runs Trino and Spark plugin unit tests on every push.
- `compat-ailake-py` job installs `mypy + pandas` and runs `mypy` type check before the compat script.
- `compat-heavy.yml`: `AILAKE_WRITE_DIR` injected into Spark and Trino integration test steps.

---

## [0.0.14] — 2026-06-09

### Removed

- **RaBitQ index** (`RaBitQIndex`, `RaBitQSerializer`, `RaBitQCodebook`, `RaBitQVec`, `ailake-vec/src/rabitq.rs`, `ailake-index/src/rabitq.rs`) removed from all layers. Recall ≈ 0 on general float embeddings (orthonormal rotation does not help without training data alignment); adds significant complexity for no practical benefit over HNSW or IVF-PQ.
  - Removed `RaBitQConfig` from `ailake-core/src/schema.rs` and `rabitq` field from `VectorStoragePolicy`.
  - Removed `FLAG_INDEX_RABITQ = 0x0002` from `ailake-file/src/footer.rs`.
  - Removed `AnyIndex::RaBitQ` variant; `AnyIndex` now dispatches only `Hnsw` and `IvfPq`.
  - Removed `IndexType::RaBitQ` from `ailake-file/src/writer.rs`.
  - Removed `--rabitq`, `--rabitq-seed`, `--rabitq-keep-raw` CLI flags.
  - Removed `rabitq=`, `rabitq_seed=`, `rabitq_keep_raw=` parameters from `ailake-py` `TableWriter`.
  - Removed `"rabitq"`, `"rabitq_seed"`, `"rabitq_keep_raw"` from `ailake-jni` JSON API.
  - Removed `rabitq.go`, `chacha12.go` from `ailake-go`; removed `FlagIndexRaBitQ`, `IsRaBitQ()`.
  - Removed `rabitq.hpp`, `chacha12.hpp` from `ailake-cpp`; removed `kFlagIndexRaBitQ`, `is_rabitq()`, `rabitq_rerank_factor`.
  - Removed `rabitq_write_search_returns_correct_top_result` integration test.

- **Binary Hamming index** (`BinaryIndex`, `BinarySerializer`, `ailake-vec/src/binary_quant.rs`, `ailake-index/src/binary.rs`) removed from all layers. Recall 0.50–0.70 without reranking on general float embeddings is too low for production use; no advantage over IVF-PQ which achieves 0.90–0.95 recall at comparable or smaller storage.
  - Removed `BinaryConfig` from `ailake-core/src/schema.rs` and `binary` field from `VectorStoragePolicy`.
  - Removed `FLAG_INDEX_BINARY = 0x0004` from `ailake-file/src/footer.rs`.
  - Removed `AnyIndex::Binary` variant.
  - Removed `IndexType::Binary` from `ailake-file/src/writer.rs`.
  - Removed `--binary`, `--binary-keep-raw` CLI flags.
  - Removed `binary=`, `binary_keep_raw=` parameters from `ailake-py` `TableWriter`.
  - Removed `"binary"`, `"binary_keep_raw"` from `ailake-jni` JSON API.
  - Removed `binary.go` from `ailake-go`; removed `FlagIndexBinary`, `IsBinary()`.
  - Removed `binary.hpp` from `ailake-cpp`; removed `kFlagIndexBinary`, `is_binary()`.
  - Removed `ailake-cpp/tests/test_binary.cpp` (14 tests).

- **`ailake-bench`** removed from workspace `Cargo.toml` members. Benchmarks live in the separate [`ailake-benchmarks`](https://github.com/ThiagoLange/ailake-benchmarks) repository.

### Changed

- `AnyIndex` enum now contains only `Hnsw(HnswIndex)` and `IvfPq(IvfPqIndex)`.
- `VectorStoragePolicy` index auto-selection: checks `policy.pq.is_some()` → `IvfPq`; default → `Hnsw`. Binary and RaBitQ checks removed.
- `ailake-file` reader flag dispatch: `FLAG_INDEX_IVF_PQ = 0x0001` only; unknown flags default to HNSW.
- File format spec §3 `flags` field: only bit 0 (`IVF-PQ`) defined; bits 1–15 reserved.
- File format spec: removed §6.3 (RaBitQ Index Blob), §6.4 (Binary Hamming Index Blob), §15.4 (BinarySnapshot wire layout).
- Cross-language table in file format spec reduced to Rust, C++17, Go columns for HNSW and IVF-PQ only.

---

## [0.0.13] — 2026-06-08

### Added
- **Binary Hamming flat index** — `IndexType::Binary` / `FLAG_INDEX_BINARY = 0x0004`. Binarizes each vector dimension via sign (positive = 1), packs to `ceil(dim/8)` bytes. Distance = Hamming (`popcount(a XOR b)`). 32× smaller than F32 (1 bit/dim vs 32 bits/dim). Designed for models trained to produce binary-compatible vectors (Cohere embed-v3 binary, Jina ColBERT). For general float embeddings use RaBitQ — it applies a random rotation before binarization and achieves much better recall at the same storage cost.
  - **`ailake-vec/src/binary_quant.rs`**: `f32_to_bits` (sign packing, MSB-first), `hamming_distance` with AVX2/SSSE3 Mula nibble-LUT + PSADBW (32 bytes/iter), NEON `vcntq_u8` (16 bytes/iter), scalar u64-chunk fallback (maps to `popcnt`).
  - **`ailake-index/src/binary.rs`**: `BinaryIndex` flat scan, `BinarySerializer` (bincode). Optional `keep_raw: bool` for exact F16 reranking; partial-select O(N) top-k with optional rerank. `rerank_factor ≥ 3` recommended.
  - **`ailake-file`**: `FLAG_INDEX_BINARY = 0x0004` in footer; writer builds `BinaryIndex` when `policy.binary.is_some()`; reader dispatches on `FLAG_INDEX_BINARY` before RaBitQ/IVF-PQ/HNSW checks.
  - **`ailake-core/src/schema.rs`**: `BinaryConfig { keep_raw: bool }` added to `VectorStoragePolicy`.
  - **CLI**: `ailake create --binary [--binary-keep-raw]`.
  - **Python**: `TableWriter(binary=True, binary_keep_raw=True)`.
  - **JVM plugins** (Trino / Spark / Flink): search dispatches automatically via `AnyIndex::search()` — no plugin code changes needed. `ailake_write_batch_json` in `ailake-jni` now accepts `"binary":true,"binary_keep_raw":true` so JVM plugins can write Binary tables.
  - **Go SDK** (`ailake-go/binary.go`): `BinaryIndex`, `DeserializeBinary` (bincode wire format), `hammingBinary` (u64-chunk XOR + `bits.OnesCount64` → POPCNT on x86_64 / VCNT+UADDLV on aarch64), `f32ToBits` (MSB-first), `BinaryIndex.Search` (Hamming scan + optional F16 rerank). `FlagIndexBinary = 0x0004` and `IsBinary()` in `footer.go`; dispatch in `searchFile()` before RaBitQ check.
  - **C++ SDK** (`ailake-cpp/include/ailake/binary.hpp`): `BinaryIndex`, `deserialize_binary`, `f32_to_bits`, `hamming_distance` (AVX2+SSSE3 nibble-LUT / NEON `vcntq_u8` / scalar `__builtin_popcountll`), `binary_search` (O(N) scan + `std::nth_element` + optional F16 reranking). `kFlagIndexBinary = 0x0004` and `is_binary()` in `footer.hpp`; dispatch in `search_file()` before RaBitQ check.
  - **C++ SDK tests** (`ailake-cpp/tests/`): `test_binary.cpp` — 14 tests covering `f32_to_bits` MSB-first packing, `hamming_distance` (single byte / multibyte / 32-byte AVX2 chunk), `binary_search` top-k, F16 reranking, and edge cases. Also created `test_footer.cpp`, `test_hnsw.cpp`, `test_ivfpq.cpp` (first C++ unit test suite — CMakeLists previously referenced non-existent files). `CMakeLists.txt` updated to per-module `foreach` loop.

---

## [0.0.12] — 2026-06-07

### Fixed
- `.github/workflows/publish-pypi.yml`: remove duplicate `runs-on` key in `linux` job
- `.github/workflows/release.yml`: all downstream jobs (`publish-crates`, `publish-jvm`, `publish-airflow`, `pypi-linux/macos/windows/sdist`) now checkout `ref: ${{ needs.release.outputs.tag }}` — prevents publishing stale pre-bump version to crates.io/PyPI
- `.github/workflows/release.yml`: fix cascade-skip — `pypi-windows` and `pypi-sdist` depended on `pypi-macos` (`if: false`); skipped job propagated to Windows, sdist, and `pypi-publish`, blocking PyPI release entirely; both now depend on `pypi-linux` instead; removed `pypi-macos` from `pypi-publish` needs
- `.github/workflows/release.yml`, `publish-pypi.yml`: Windows Rust install — `dtolnay/rust-toolchain` uses bash internally (fails on Windows self-hosted); replaced with inline PowerShell that downloads `rustup-init.exe` if rustup absent, otherwise runs `rustup toolchain install`
- `.github/workflows/release.yml` (`pypi-sdist`), `publish-pypi.yml` (`sdist`): add `dtolnay/rust-toolchain@stable` before `maturin sdist` — `maturin sdist` runs natively on Linux runner (no manylinux Docker), so cargo must be in PATH explicitly
- `tests/docker/demo/Dockerfile`: remove `COPY ailake-bench` (crate lives in separate repo; line caused Docker build failure)
- `notebooks/04_trino.ipynb`, `notebooks/05_bigquery.ipynb`: fix pre-flight error message — wrong `-f compose-demo-engines.yml` replaced with `--profile engines`

### Changed
- `.github/workflows/ci.yml`: disable automatic push/PR triggers — manual `workflow_dispatch` only while repo is private
- `.github/workflows/release.yml`: manual-only trigger (`workflow_dispatch`); fix JAR glob pattern

### Docs
- `README.md`: remove duplicate `ailake-cli/` lines in repo layout; add `ailake-go/`, `ailake-cpp/`, `airflow-providers-ailake/` to directory tree
- `docs/architecture/WORKSPACE.md`: document `axum = "0.7"` workspace dependency (`ailake serve` REST server)
- `docs/specs/INTEGRATIONS.md`: add Python, Go, and C++ SDK rows to compatibility matrix

---

## [0.0.11] — 2026-06-05

### Changed
- **`release.yml`**: Restructured into a single sequential publish chain — `release` → `publish-crates` → `publish-jvm` → `publish-airflow` → `pypi-linux` (max-parallel:1) → `pypi-macos` (disabled) → `pypi-windows` → `pypi-sdist` → `pypi-publish`. All publish jobs run automatically after the release job using `needs:` — no separate manual triggers needed. `publish-pypi.yml`, `publish-jvm.yml`, and `publish-airflow-provider.yml` demoted to manual fallback workflows for re-publishing without rerunning the full pipeline. Triggers: `push: branches: [main]` (automatic on merge) and `workflow_dispatch` (manual). The `release` job auto-bumps the patch version by reading the latest git tag (`v*.*.*`) and incrementing the patch component, updating all `Cargo.toml` files and committing with `[skip ci]` before tagging — no manual version edits required.
- **`.github/workflows/compat-heavy.yml` (`compat-spark`)**: `pip install pyspark` now uses `--index-url https://pypi.org/simple/` to bypass the runner's pip mirror configuration.

### Fixed
- **`ailake-go/chacha12.go` + `ailake-cpp/include/ailake/chacha12.hpp`**: Cross-language RaBitQ search was producing recall ≈ 0% because Go (`math/rand` LCG) and C++ (`std::mt19937_64`) generated completely different projection matrices than Rust's `StdRng` (ChaCha12) for the same seed. Fixed by implementing the full Rust PRNG: splitmix64 seed expansion (`u64 → 32-byte key`, 4 rounds) + ChaCha12 block function (6 double rounds, Bernstein state layout) + Standard float distribution (`f32::from_bits((u32>>9)|0x3f800000) - 1.0`). Go and C++ now regenerate bit-identical matrices to the Rust SDK for any seed.
- **`ailake-cpp/include/ailake/`**: Added `kFlagIndexRaBitQ = 0x0002`, `AilakeHeader::is_rabitq()`, `RaBitQIndex`, `deserialize_rabitq`, `rabitq_search` (O(N) scan + `std::nth_element` partial select + optional F16 reranking), `SearchOptions::rabitq_rerank_factor`. C++ SDK previously silently misrouted RaBitQ files as HNSW. New `BincodeReader` methods: `read_u16()`, `read_u8_vec_flat()`, `read_u16_vec()`.
- **`ailake-index/src/rabitq.rs` + `ailake-index/src/lib.rs`**: `RaBitQIndex::search` now takes `&self` instead of `&mut self` — the unsafe raw-pointer cast workaround in `AnyIndex::RaBitQ` is removed. Shard-level parallelism via rayon in `SearchSession` is now fully safe with no `unsafe` code.
- **`ailake-index/src/rabitq.rs`**: Inner binary scan is now sequential (`iter().enumerate()`; no `into_par_iter()`). Outer shard parallelism in `SearchSession` already handles concurrency — nesting `par_iter` inside each shard spawned O(shards × N) micro-tasks (1M+ with 10 shards × 100k entries), making rayon scheduler overhead dominate actual work. **QPS on SIFT-1M: 48 → 101 (+2.1×)**.
- **`ailake-index/src/rabitq.rs`**: Top-k candidate selection replaced full O(N log N) sort with O(N) `select_nth_unstable_by(candidates − 1)` + sort of `candidates` elements only. For `candidates = rerank_factor × top_k ≪ N` this eliminates most comparison work.
- **`ailake-catalog/src/hadoop.rs`**: `HadoopCatalog::commit_snapshot` for `Replace`/`Overwrite` operations no longer inherits manifests from previous snapshots — new manifest IS the complete state. Previously, all operations unconditionally appended to the manifest list, causing `list_files` to return duplicate `DataFileEntry` records. With 10 concurrent deferred HNSW background tasks all racing to commit `Replace` snapshots, the accumulated duplicates prevented `IndexStatus::Ready` entries from reaching the `ready >= num_shards` threshold, causing the bench to block indefinitely.
- **`ailake-vec/src/pq.rs`**: `kmeans_pp_init` complexity reduced from O(n × k²) to O(n × k) by maintaining an incremental `min_dist` array instead of recomputing all distances from scratch at each step. With n=100k, k=256: 3.2B → 25M distance computations for the init phase alone — **17× end-to-end write speedup** on SIFT-1M IVF-PQ benchmark (96s → 5.7s for 10k vectors).
- **`ailake-bench/src/main.rs`**: `--engine ailake-ivf-pq` now derives `nlist`/`nprobe` from `IvfPqConfig::for_dataset(dim, shard_size)` when CLI args are left at default (0). Previous hardcoded defaults `nlist=256 nprobe=8` were calibrated for ~65k-vector datasets; with 100k vectors/shard `nprobe=8/256=3.1%` scan coverage produced `Recall@10=0.32`.
- **`ailake-bench/src/main.rs`**: IVF-PQ multi-shard search now loads raw vectors (`load_with_raw=true`) and sets `rerank_factor=Some(3)`. Per-shard PQ codebooks produce ADC distances on different scales — cross-shard merge sorted by incomparable approximations, causing `Recall@10=0.32` even with correct nlist/nprobe. Exact reranking with true L2² distances corrects the merge step.

### Added
- **`ailake-vec/src/rabitq.rs`**: `RaBitQCodebook::estimate_ip_binary(b_q: &[u8], q_scale: f32, entry: &RaBitQVec) -> f32` — new public method that accepts pre-binarized query codes instead of raw `q_proj`. Eliminates repeated `bits_from_signs` calls in the search hot path (query is binarized once per search call, not once per entry). `estimate_ip` now wraps `estimate_ip_binary` for backwards compatibility.
- **RaBitQ (Random Binary Quantization)**: new flat index type for extreme storage compression — 1 bit/dim = 16× smaller than F16, with better recall than naive binary quantization via random rotation + unbiased XOR/popcount IP estimator. Key types: `ailake_vec::rabitq::RaBitQCodebook` (random rotation matrix, seed-regenerated), `ailake_index::RaBitQIndex` (flat search + optional F16 reranking), `ailake_core::schema::RaBitQConfig`. File format flag `FLAG_INDEX_RABITQ = 0x0002`. `RaBitQConfig` re-exported from `ailake_core` crate root. `AilakeFileWriter::new` auto-selects `IndexType::RaBitQ` when `policy.rabitq` is set — callers using `write_batch`/`write_batch_idempotent` get RaBitQ automatically without calling `with_index_type`. Exposed via CLI `ailake create --rabitq [--rabitq-seed N] [--rabitq-keep-raw]` and Python `TableWriter(rabitq=True, rabitq_seed=0, rabitq_keep_raw=True)`. Use with `rerank_factor ≥ 3` at search time for best recall.
- **`VectorStoragePolicy::hnsw_m` + `VectorStoragePolicy::hnsw_ef_construction`**: Per-table HNSW tuning parameters. `hnsw_m` controls connections per node (default 16; higher → better recall, more memory); `hnsw_ef_construction` controls candidate pool during build (default 150; higher → better graph quality, slower build). Both stored as `ailake.hnsw-m` / `ailake.hnsw-ef-construction` in Iceberg metadata properties. Exposed via `ailake create --hnsw-m 32 --hnsw-ef 400` (CLI) and `TableWriter(hnsw_m=32, hnsw_ef_construction=400)` (Python). `None` = use defaults (fully backwards-compatible).
- **`VectorMetric::NormalizedCosine` (value `3`) + `VectorStoragePolicy::pre_normalize`**: New fast-path distance metric for cosine workloads. When `pre_normalize = true`, vectors are normalized to unit L2 at write time and HNSW uses `1 - dot(a, b)` instead of full cosine — eliminates the `sqrt` of norms from every edge traversal (~12–20% faster search on dim=1536). Query vectors are automatically normalized at search time in all bindings — callers need no changes. Exposed via `ailake create --pre-normalize` (CLI), `TableWriter(pre_normalize=True)` (Python), `MetricNormalizedCosine` (Go), and `Metric::NormalizedCosine` (C++). All metric match arms updated across `gpu`, `ivf_pq`, `serialize`, `pruner`, `scanner`, `parquet schema`, `footer`, and `reader`.
- **`ailake-index/src/ivf_pq.rs`**: `IvfPqCodebook` struct — sharable coarse quantizer + PQ codebook trainable once and reused across all shards. New methods: `IvfPqIndex::train_codebook(vectors, metric, config) -> IvfPqCodebook` (k-means only, no inverted lists) and `IvfPqIndex::build_with_codebook(row_ids, vectors, codebook) -> IvfPqIndex` (assign + encode, no k-means). When all shards share the same codebook, ADC distances are numerically comparable across shards — cross-shard merge is correct without exact reranking.
- **`ailake-file/src/writer.rs`**: `AilakeFileWriter::with_shared_ivf_codebook(Arc<IvfPqCodebook>)` builder — bypasses k-means training and calls `IvfPqIndex::build_with_codebook` instead of `IvfPqIndex::train`.
- **`ailake-query/src/writer.rs`**: `TableWriter::write_batch_ivf_pq_deferred` — async variant of `write_batch_ivf_pq`. Persists Parquet immediately (~200k vec/s, same as HNSW deferred), spawns background tokio task to train IVF-PQ index, rewrite file with AILK section, and transition `IndexStatus::Indexing → Ready`. Shared codebook is coordinated via `Arc<tokio::sync::OnceCell<IvfPqCodebook>>` — first task trains, all others await and skip k-means.
- **`ailake-query/src/writer.rs`**: `TableWriter` now caches `cached_ivf_codebook: Option<Arc<IvfPqCodebook>>` (synchronous path) and `deferred_ivf_codebook: Arc<tokio::sync::OnceCell<IvfPqCodebook>>` (deferred path).
- **`ailake-bench/src/main.rs`**: new `--engine ailake-ivf-pq-deferred` — exercises `write_batch_ivf_pq_deferred`, waits for `IndexStatus::Ready`, searches with `rerank_factor=3`.

### Changed
- **`ailake-vec/src/rabitq.rs`**: `RaBitQCodebook::rebuild_proj` now generates a **modified Gram-Schmidt orthonormal matrix** (P^T · P = I) instead of a column-normalized Gaussian. Orthonormal projection preserves inner products exactly — unit-sphere vectors map to unit-sphere after rotation — improving recall fidelity on cosine workloads. The `seed` in `RaBitQCodebook` is still the only persisted field; readers regenerate P via `rebuild_proj(seed, dim)` as before.
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
