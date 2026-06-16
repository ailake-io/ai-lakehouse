# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

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
