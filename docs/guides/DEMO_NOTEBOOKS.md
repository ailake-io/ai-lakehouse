# Demo Notebooks — Step-by-Step Guide

Complete walkthrough for running the AI-Lake interactive demo environment: from first `docker compose up` to executing every notebook.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker Engine | 24+ | `docker --version` |
| Docker Compose | 2.20+ (plugin, not standalone) | `docker compose version` |
| Free disk space | ≥ 6 GB | Image + wheel build cache |
| Free RAM | ≥ 4 GB (basic) / ≥ 8 GB (+ engines) / ≥ 12 GB (+ airflow) | |
| NVIDIA GPU + Container Toolkit | optional | Only for `--profile gpu` |

> **No Rust toolchain, Python, or cloud account required.** Everything builds and runs inside Docker.

---

## 2. Clone and start

```bash
git clone https://github.com/ThiagoLange/ai-lakehouse.git
cd ai-lakehouse

# Build ailake-py wheel + ailake-cli (with catalog-ducklake) + start all core
# services (~8-12 min on first run — the DuckLake catalog build pulls duckdb's
# bundled C++ build; instant after)
docker compose -f tests/docker/compose-demo.yml up -d
```

First-run output (abridged):
```
[+] Building 240.3s
 => [jupyter builder] compiling ailake-py…      ✓
 => [jupyter] pip install /wheels/*.whl          ✓
[+] Running 5/5
 ✓ Container ailake-demo-minio       Started
 ✓ Container ailake-demo-minio-init  Exited (0)
 ✓ Container ailake-demo-nessie      Started
 ✓ Container ailake-demo-jupyter     Started
```

> Subsequent `up -d` calls use Docker layer cache — start in seconds.

---

## 3. Building the Docker images

The demo uses two Docker images built locally from source:

| Image | Built from | Contains |
|---|---|---|
| `docker-jupyter` | `tests/docker/demo/Dockerfile` | Rust toolchain → `maturin build` → ailake-py wheel + `cargo build --features catalog-ducklake` → `ailake` CLI binary → JupyterLab |
| `docker-airflow` | `tests/docker/demo/Dockerfile.airflow` | Same Rust builder stage → ailake-py wheel → `apache/airflow:2.9.2` |

Build happens automatically on first `docker compose up -d`. The Airflow image is only built when `--profile airflow` is used.

### What triggers a rebuild

| Change | Rebuild needed? | Command |
|---|---|---|
| Notebook files (`tests/docker/demo/notebooks/`) | **No** — bind-mounted live from the repo | — |
| `airflow-entrypoint.sh`, DAG files | **No** — bind-mounted live | — |
| `ailake-py/python/ailake/__init__.py` (pure Python) | **Yes** — baked into the wheel | See below |
| Any Rust source (`ailake-*/src/`) | **Yes** — requires full recompile | See below |
| `tests/docker/demo/init_demo.py` | **Yes** — COPY'd into image | See below |
| `tests/docker/demo/Dockerfile` | **Yes** | See below |

### Rebuild commands

```bash
# Rebuild Jupyter image only (most common — Rust or Python SDK change)
docker compose -f tests/docker/compose-demo.yml build jupyter

# Rebuild without layer cache (force full recompile)
docker compose -f tests/docker/compose-demo.yml build --no-cache jupyter

# Rebuild Airflow image (after SDK or DAG-infrastructure change)
docker compose -f tests/docker/compose-demo.yml build airflow

# Rebuild both images then restart
docker compose -f tests/docker/compose-demo.yml build jupyter airflow
docker compose -f tests/docker/compose-demo.yml up -d
```

> **Tip:** `build` without `--no-cache` reuses cached layers up to the first changed file — so a pure Python change to `ailake-py/python/` skips the Rust recompile (the heaviest layer) and finishes in ~30 s instead of ~3-5 min.

### Build time reference

| Scenario | Approx time |
|---|---|
| First build (no cache) | 8–12 min (Rust + wheel + `ailake` CLI w/ `catalog-ducklake` + JupyterLab) |
| Python-only change (`__init__.py`) | ~30 s |
| Rust source change (any `ailake-*/src/`) | 8–12 min (full recompile — wheel + CLI binary) |
| Airflow image, first build | 5–8 min |
| Subsequent `up -d` (no rebuild) | < 5 s |

---

## 4. Services and ports

| Service | URL | Profile | Description |
|---|---|---|---|
| **JupyterLab** | http://localhost:8888 | always-on | Notebooks, demo data, ailake-py pre-installed |
| **MinIO console** | http://localhost:9001 | always-on | Local S3 (user: `minioadmin` / pass: `minioadmin`) |
| **Nessie catalog** | http://localhost:19120 | always-on | Iceberg REST catalog |
| **Trino** | http://localhost:8080 | `--profile engines` | SQL engine with Iceberg connector |
| **BigQuery emulator** | http://localhost:19050 | `--profile engines` | BigQuery-compatible SQL endpoint |
| **Airflow UI** | http://localhost:8090 | `--profile airflow` | DAG scheduler (user: `admin` / pass: `admin`) |
| **JupyterLab (GPU)** | http://localhost:8889 | `--profile gpu` | Same as :8888, NVIDIA GPU exposed |
| **Flink Web UI** | http://localhost:8082 | `--profile flink` | Standalone session cluster (JobManager + TaskManager, single container) |

---

## 5. Optional profiles

Profiles add heavyweight services on demand. Core notebooks (01-03, 07-12) work without any profile.

### `--profile engines` — Trino + BigQuery emulator

Required for notebooks **04** and **05**.

```bash
docker compose -f tests/docker/compose-demo.yml --profile engines up -d
```

Added services: a custom-built Trino image (`Dockerfile.trino`, pinned to **Trino
430** — Trino 460 breaks the plugin outright with a real SPI signature change)
that bundles the real `trino-plugin` JAR alongside the stock Iceberg connector.
Two catalogs are registered: `ailake` (stock Iceberg connector pointing at
Nessie) and `ailake_native` (the real ailake plugin, `io.ailake.trino.VectorScanConnectorFactory`,
backed directly by `libailake_jni.so`) — plus `goccy/bigquery-emulator:0.6.6`.

### `--profile flink` — Flink standalone cluster

Required for notebook **14**.

```bash
docker compose -f tests/docker/compose-demo.yml --profile flink up -d
# Web UI: http://localhost:8082
```

Added service: a custom-built Flink image (`Dockerfile.flink`, `flink:1.18.1-scala_2.12-java17`)
bundling the real `ailake-flink` connector JAR, running as a single-container
standalone session cluster (JobManager + TaskManager in one process via
`start-cluster.sh`). The main `jupyter` image also gets a plain Flink **client**
install (`FLINK_CLIENT_HOME=/opt/flink-client`) so the notebook can submit SQL
to the remote cluster via `sql-client.sh` — PyFlink was evaluated and rejected
(`apache-flink==1.18.1` pins a `numpy`/Python version incompatible with this
image's Python 3.12).

### `--profile gpu` — NVIDIA GPU JupyterLab

Required for notebook **10**. Requires NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```bash
docker compose -f tests/docker/compose-demo.yml --profile gpu up -d
# Opens on http://localhost:8889 (separate port — can run alongside :8888)
```

### `--profile airflow` — Apache Airflow 2.9

Required for notebook **12**. Builds a second Docker image (~5-8 min on first run).

```bash
docker compose -f tests/docker/compose-demo.yml --profile airflow up -d
# Airflow UI: http://localhost:8090  (admin / admin)
```

### Multiple profiles at once

```bash
docker compose -f tests/docker/compose-demo.yml --profile engines --profile airflow up -d
```

---

## 6. Demo fixture tables

When the Jupyter container starts for the first time, `init_demo.py` writes fixture tables to `/data/` (Docker volume `demo-data`). This runs once and is skipped on restart.

| Table path | Rows | Description |
|---|---|---|
| `/data/ailake_demo` | 500 | Main HNSW table — used by most notebooks |
| `/data/ailake_pq` | 500 | PQ-only (vectors discarded, codes only) |
| `/data/ailake_residual_pq` | 500 | IVF-PQ with residual encoding |
| `/data/ailake_deferred` | 200 | `write_batch_auto_deferred` — Parquet immediate, index async |
| `/data/ailake_multimodal` | 200 | Text (dim=32) + image (dim=16) dual embeddings |
| `/data/ailake_agent` | 100 | Two agents (agent-A / agent-B), partition isolation |
| `/data/ailake_bm25` | 200 | BM25 stats written at ingest time (legacy hybrid) |
| `/data/ailake_fts` | 200 | Tantivy per-file FTS index (`AILK_FTS` section) |
| `/data/ailake_partitioned_v3` | 200 | `partition_fields=[topic_id:identity]`, `format_version=3` |
| `/data/ailake_delete_demo` | 100 | 10 rows pre-deleted via equality delete file |
| `/data/ailake_schema_evo` | 100 | `add_column(source_url)` pre-applied |

All tables live in the shared `demo-data` Docker volume — they persist across container restarts and are accessible from both Jupyter and Airflow containers simultaneously.

`13_ducklake.ipynb` is one exception — it doesn't use an `init_demo.py`
fixture. It creates its own table live (via the `ailake` CLI, `--catalog ducklake`)
under `DEMO_DUCKLAKE_STORE` (`/data/ailake_ducklake`).

`dag_ailake_hook_ops.py` (triggered from `12_airflow.ipynb` §8B–8C) is the other
exception — it writes its own two tables (`/data/ailake_hooks_delete_demo`,
`/data/ailake_hooks_evolve_demo`) at task-run time rather than reading an
`init_demo.py` fixture.

---

## 7. Notebook walkthrough

Open **http://localhost:8888** and execute notebooks top-to-bottom. Each notebook is self-contained — cells load the demo fixture paths from environment variables set by Docker Compose.

### `01_ailake_demo.ipynb` — Python SDK comprehensive demo

**Profile required:** none  
**Fixture dependency:** `/data/ailake_demo` + sub-tables  

The main SDK reference notebook. 32 sections covering:

| Sections | Topics |
|---|---|
| 1–5 | `open_table()`, `insert()`, `commit()`, `SearchQuery`, `fetch_data`, fluent API |
| 6–7 | `pre_normalize`, `normalized_cosine`, `hnsw_m`, `hnsw_ef_construction`, idempotent writes |
| 8–10 | Iceberg compat (PyArrow + PyIceberg), DuckDB SQL, `assemble_context()` |
| 11–14 | MinIO upload, IVF-PQ `pq_only` + `rerank_factor`, Residual-PQ, `write_batch_auto_deferred` |
| 15–17 | HNSW tuning (`ef_search`, `pruning_threshold`), async API, storage estimator |
| 18–20 | Embedding model tracking, `embed_fn` pattern B, `migrate_embeddings()` |
| 21–23 | `VectorColSpec`, `write_batch_multi`, `search_multimodal` RRF, `MultimodalContextSchema` |
| 24–28 | `ailake.Agent`, partition isolation, hybrid `ScoreFn`, `ToolCallSchema`, `EpisodicMemorySchema` |
| 29–31 | `delete_where`, schema evolution (`add_column`/`rename_column`/`evolve_schema`), `compact()` |
| 32 | Tantivy FTS intro (`fts_text_columns`, `search_text`) — full demo in `11_fts.ipynb` |

---

### `02_duckdb.ipynb` — DuckDB Parquet compatibility

**Profile required:** none  
**Fixture dependency:** `/data/ailake_demo`

Shows that AI-Lake Parquet files are standard DuckDB-readable without any plugin. The HNSW footer is invisible — DuckDB stops at `PAR1`.

| Section | Topic |
|---|---|
| 1–3 | `read_parquet(glob)`, schema, aggregations |
| 4 | Topic distribution via SQL `LIKE` filter |
| 5 | DuckDB Iceberg extension (optional) |
| 6–7 | Per-file storage breakdown, F16 BLOB → numpy decode |
| 8–9 | Iceberg `metadata.json` properties, embedding model tracking |
| 10 | `duckdb-ailake` C++ extension — loads `ailake.duckdb_extension` + `libailake_jni.so` |
| 11 | `ailake_search` + `ailake_scan` — native vector search / search+full-row over SQL |
| 12 | `ailake_search_text` — Tantivy FTS over SQL |
| 13 | `ailake_search_multimodal` — cross-modal RRF over SQL |
| 14 | Write lifecycle from SQL — `ailake_write_batch`, `ailake_delete_where`, `ailake_evolve_schema`, `ailake_compact` |

---

### `03_spark.ipynb` — PySpark + Iceberg

**Profile required:** none (Spark runs in local[*] mode inside Jupyter)  
**Fixture dependency:** `/data/ailake_demo`, `/data/ailake_partitioned_v3`, `/data/ailake_delete_demo`, `/data/ailake_schema_evo`

> Takes ~30-60 s for Spark JVM startup on first cell.

| Section | Topic |
|---|---|
| 1 | `SparkSession` with Iceberg JAR, `HadoopCatalog` |
| 2 | Direct Parquet read (no Iceberg) |
| 3 | Iceberg `HadoopCatalog` SQL interface — `COUNT(*)`, schema |
| 4 | Aggregations — `MIN/MAX` |
| 5 | Iceberg snapshot history |
| 6 | Time-travel — `VERSION AS OF <snapshot_id>` |
| 7 | Snapshot metadata + file manifests |
| 8 | Embedding model tracking via `SHOW TBLPROPERTIES` |
| 9 | Iceberg v3 partitioned table (`ailake_partitioned_v3`, `topic_id` identity) |
| 10 | `AilakeNative` py4j bridge — helpers (`Seq`/`Option`/`float[]` conversions raw py4j needs) |
| 11 | `AilakeNative.deleteWhere` — Iceberg equality delete (real call, against `ailake_delete_demo`) |
| 12 | `AilakeNative.evolveSchema` — metadata-only schema change (real call, against `ailake_schema_evo`) |
| 13 | `AilakeNative.search` / `.scan` / `.searchText` / `.compact` (real calls) |
| 14 | `AilakeNative.writeBatch` — write from Spark (real call) |

> `searchMultimodal`/`writeBatchMulti` are **not** demonstrated in this notebook — both take a Scala `Float`-boxed field that raw py4j has no way to marshal (a Python `float` always crosses as `java.lang.Double`); this is a genuine raw-py4j limitation, not a plugin bug. Production Scala/Java Spark code calls them directly with no issue.

---

### `04_trino.ipynb` — Trino SQL

**Profile required:** `--profile engines` (Trino)  
**Fixture dependency:** `/data/ailake_demo`, `/data/ailake_partitioned_v3`, `/data/ailake_delete_demo`, `/data/ailake_schema_evo`

```bash
# Start with engines profile first
docker compose -f tests/docker/compose-demo.yml --profile engines up -d
```

Wait ~30 s for Trino health check. Then open the notebook.

| Section | Topic |
|---|---|
| 1 | Discover catalogs and tables |
| 2 | Schema inspection |
| 3 | Basic scan |
| 4 | Filtered query + aggregation |
| 5 | Iceberg metadata — snapshots and manifests |
| 6 | Table properties — AI-Lake custom metadata via `$properties` |
| 7 | File-level manifest statistics via `$files` |
| 8 | Manifest files via `$manifests` |
| 9 | Embedding model tracking via Trino `$properties` |
| 10 | Iceberg v3 partitioned table via Trino — `partitioned_v3` (format_version=3, `topic_id` identity), `delete_demo` (equality delete files visible in `$manifests`/`$files`), `schema_evo` (evolved schema visible in `DESCRIBE`) |
| 11 | ailake Trino plugin — `ailake_native` catalog (real connector, `io.ailake.trino.VectorScanConnectorFactory`, backed by `libailake_jni.so`), exposing `search`/`search_full`/`search_multimodal`/`ingest` |
| 12 | Session properties + query plan — `SET SESSION ailake_native.query_vector/top_k` and `EXPLAIN` both execute; the query vector is passed as a session property, not a SQL function argument |
| 13 | `SELECT` execution against `search`/`search_full` — **fully works end-to-end** (two real Jackson serialization bugs in Trino's internal task codec found and fixed: a bare `@JsonProperty` on a Kotlin data-class `val` never reaching a getter, and a Kotlin `object` transaction handle with a private synthetic constructor — see `CHANGELOG.md`) |

Sections 1–10 use Trino's stock Iceberg connector (`ailake` catalog, no AI-Lake code runs inside Trino). Sections 11–13 use the real plugin (`ailake_native` catalog) — planning **and** query execution both work against the live Trino 430 server this image builds.

---

### `05_bigquery.ipynb` — BigQuery emulator

**Profile required:** `--profile engines` (BigQuery emulator)

| Section | Topic |
|---|---|
| 1 | BigQuery client → emulator on port 19050 |
| 2 | Stream inserts from Parquet; `COUNT`, `MIN/MAX` validation |
| 3 | F16 BYTES column → float32 decode |
| 4 | Production pattern: real GCS bucket + BigQuery Omni |

---

### `06_airbyte_destination.ipynb` — Airbyte destination connector

**Profile required:** none  
**Fixture dependency:** none (writes its own data)

Shows the `airbyte-destination-ailake` connector: accepts Airbyte record stream, calls `ailake.TableWriter` and commits Iceberg snapshots.

---

### `07_multimodal.ipynb` — Multi-vector and cross-modal search

**Profile required:** none  
**Fixture dependency:** `/data/ailake_multimodal`

| Section | Topic |
|---|---|
| 1–2 | `VectorColSpec` declaration, `write_batch_multi` (text dim=32 + image dim=16) |
| 3 | Modality tags in Iceberg properties |
| 4 | `search_multimodal` — weight ablation (100/0 → 70/30 → 50/50 → 0/100) |
| 5 | RRF fusion formula — `Σ weight_i / (60 + rank_i)` |
| 6 | `MultimodalContextSchema` column name constants |

---

### `08_agents.ipynb` — Agent memory (Phase 9)

**Profile required:** none  
**Fixture dependency:** `/data/ailake_agent`

| Section | Topic |
|---|---|
| 1 | `ailake.Agent(path, embed_fn, agent_id)` — `remember()`, `recall()` |
| 2 | Partition isolation — `partition_by` / `partition_filter` |
| 3 | `ToolCallSchema` — semantic search over tool call history |
| 4 | `EpisodicMemorySchema` — `recency_weight`, `importance_score` |
| 4b | `ailake.TimestampNs` + native `decay_memories()` — recomputes `recency_weight` from `last_accessed_at` |
| 5 | `ScoreFn` — hybrid ranking (distance × recency × importance) |
| 6 | `assemble_context()` for agent memory |

---

### `09_hybrid_search.ipynb` — BM25 + vector hybrid

**Profile required:** none  
**Fixture dependency:** `/data/ailake_bm25`

| Section | Topic |
|---|---|
| 1 | Write with `bm25_text_column` — IDF stats at ingest |
| 2 | `search_text()` pure lexical (BM25 brute-force O(N)) |
| 3 | Hybrid search — vector HNSW + BM25 RRF fusion |
| 4 | Weight ablation: `bm25_weight` 0.0 → 0.5 → 1.0 |
| 5 | Comparison with Phase T Tantivy (see `11_fts.ipynb`) |

---

### `10_gpu_demo.ipynb` — GPU acceleration

**Profile required:** `--profile gpu`  
**Hardware required:** NVIDIA GPU + Container Toolkit

```bash
docker compose -f tests/docker/compose-demo.yml --profile gpu up -d
# Open http://localhost:8889
```

| Section | Topic |
|---|---|
| 1 | `ailake.hardware_info()` — auto-detected backend (CUDA / ROCm / CPU SIMD) |
| 2 | `write_batch_auto_deferred` throughput on GPU vs CPU |
| 3 | `write_batch_ivf_pq_deferred` / `write_batch_ivf_pq` — force IVF-PQ regardless of the hardware heuristic, vs immediate HNSW |
| 4 | Search QPS comparison |
| 5 | Recall@10 — IVF-PQ (forced, §3) vs HNSW |
| 6 | GPU k-means for IVF-PQ training speedup |
| 7 | CPU fallback — same binary, no recompile |

---

### `11_fts.ipynb` — Tantivy per-file FTS (Phase T)

**Profile required:** none  
**Fixture dependency:** `/data/ailake_fts`

| Section | Topic |
|---|---|
| 1 | Write with `fts_text_columns=["text"]` — `AILK_FTS` section in footer |
| 2 | `search_text()` O(log N) Tantivy fast path |
| 3 | Multi-column FTS (`chunk_text` + `document_title`) |
| 4 | Tantivy query syntax — phrase, wildcard, field-scoped |
| 5 | Legacy BM25 fallback (files without `AILK_FTS` section) |
| 6 | FTS + HNSW hybrid re-ranking — RRF fusion |
| 7 | Storage layout comparison: HNSW (~15 MB) vs FTS (~3-4 MB) per 50k docs |

---

### `12_airflow.ipynb` — Apache Airflow pipelines

**Profile required:** `--profile airflow`

```bash
docker compose -f tests/docker/compose-demo.yml --profile airflow up -d
# Wait ~45-60 s for Airflow scheduler to start
# Airflow UI: http://localhost:8090  (admin / admin)
```

Three pre-loaded DAGs (from `tests/docker/demo/dags/`):

| DAG | Schedule | Pipeline |
|---|---|---|
| `ailake_ingest_search` | `@daily` | `write_docs → vector_search → fts_search → assemble_context` |
| `ailake_compaction` | `@weekly` | `compact_table → table_info` |
| `ailake_hook_ops` | manual only | `run_estimate` (no table); `setup_delete_decay_table → delete_some_rows → decay`; `setup_evolve_table → add_vector_column → backfill_vector_column → migrate_primary` |

| Section | Topic |
|---|---|
| 1 | Airflow REST API health check |
| 2 | List DAGs via API |
| 3 | Trigger `ailake_ingest_search` manual run |
| 4 | Poll run status — completes in ~10-20 s (SequentialExecutor) |
| 5 | Pull task logs (write_docs, vector_search, fts_search) |
| 6 | XCom pull — vector + FTS results from completed tasks |
| 7 | Read Airflow-written data in Jupyter via `ailake.search()` |
| 8 | Trigger `ailake_compaction` |
| 8B | Register the two Airflow Connections (`conn_type="ailake"`) `ailake_hook_ops` needs — `AilakeHook`-based tasks resolve their `--store` warehouse root from a Connection `host`, not an env var, unlike the SDK-based DAGs above |
| 8C | Trigger `ailake_hook_ops` — real run exercising the six `AilakeHook` methods that shell out to the `ailake` CLI binary: `estimate`, `delete_rows`, `decay_memories`, `add_vector_column`, `backfill_vector_column`, `migrate` |
| 8D | Inspect `ailake_hook_ops` task logs (`run_estimate`, `add_vector_column`, `backfill_vector_column`, `decay`) |
| 9 | Direct PythonOperator demo — no Airflow needed |
| 10 | `AilakeWriteOperator` production pattern + connection setup — `dag_ailake_hook_ops.py` (§8B–8D above) is a real, running example of the same CLI-based operator pattern |

> `ailake_ingest_search`/`ailake_compaction` use `import ailake` (Python SDK) directly via TaskFlow API and need no CLI binary. `ailake_hook_ops` is the opposite case: it exercises `AilakeHook` methods that shell out to the `ailake` CLI binary, which `Dockerfile.airflow` now builds and installs alongside the `ailake-py` wheel (previously only the wheel was installed, so these hook methods had no binary to call).

---

### `13_ducklake.ipynb` — DuckLake catalog backend

**Profile required:** none  
**Fixture dependency:** none (creates its own table via the CLI)  
**Requires:** the `ailake` CLI binary baked into the `jupyter` image (always built — see [§3](#3-building-the-docker-images))

`DuckLakeCatalog` is CLI-only (no `ailake-py` binding, local filesystem `--store`
only) — this notebook drives it via `subprocess` instead of `import ailake`.

| Section | Topic |
|---|---|
| 0 | Locate the `ailake` binary, define a `run_cli()` helper |
| 1 | `create` — new table on `--catalog ducklake` |
| 2 | Seed a byte-correct AI-Lake Parquet file via `ailake.TableWriter` (source file for `insert`) |
| 3 | `insert` — load the seed file into the DuckLake table |
| 4 | `search` — vector similarity, `--format json` |
| 5 | `evolve` — `ALTER TABLE ADD COLUMN` without rewriting data files |
| 6 | Insert a file older than the new column — `allow_missing`/`ignore_extra_columns` |
| 7 | `compact` — merge files, rebuild HNSW |
| 8 | `info` — table statistics |
| 9 | Open the sidecar (`catalog/ailake_root.db`) directly with `duckdb` — `ailake_vector_index` |
| 10 | Open the DuckLake attachment (`catalog/ducklake_meta.db`) directly — real row data, no AI-Lake code |
| 11 | Known v1 limitations (local-fs only, single-writer, no reclamation) |

See `docs/guides/DUCKLAKE_CATALOG.md` for the full design writeup.

---

### `14_flink.ipynb` — Apache Flink SQL

**Profile required:** `--profile flink` (Flink standalone cluster)  
**Fixture dependency:** `/data/ailake_demo`, `/data/ailake_fts`

```bash
docker compose -f tests/docker/compose-demo.yml --profile flink up -d
# Web UI: http://localhost:8082
```

Demos the ailake Flink connector (`io.ailake.flink`, `ailake-flink/`) — AI-Lake
tables exposed as Flink SQL `CREATE TABLE ... WITH ('connector'='ailake', ...)`
sources and sinks, backed directly by `libailake_jni.so`. The notebook drives
the bundled `sql-client.sh` via `subprocess` rather than a Python DB-API
client — the query vector/text is a Flink **job parameter**
(`-Dpipeline.global-job-parameters.ailake.query.vector=...`, a process-launch
flag with no `SET SESSION` equivalent), and PyFlink was evaluated and
rejected (`apache-flink==1.18.1` pulls a `numpy`/Python version incompatible
with this image's Python 3.12).

| Section | Topic |
|---|---|
| 0 | `run_flink_sql()` helper — submits SQL to the remote cluster via `sql-client.sh` |
| 1 | `search` table — pointer-only vector search (`ailake_search_json`), `(row_id, distance, file_path)` |
| 2 | `search.mode='full'` — search + full row, no `JOIN` (`ailake_scan_json`, Fase 11); last declared column must be `_distance` |
| 3 | FTS / hybrid search via the `ailake.query.text` job parameter — pure BM25/Tantivy alone, hybrid RRF fusion combined with `ailake.query.vector` |
| 4 | Write — `INSERT INTO` an `ailake` sink table (batch-mode, polls the REST API for job completion) |

Of the three JVM plugins demoed across this stack, Flink and Trino both work
fully end-to-end; Spark works for 7 of 9 native methods (see `03_spark.ipynb`
§14). Two real bugs were found and fixed getting this notebook working
against a live (non-local) Flink 1.18 cluster — neither previously caught by
any test in this repo, since none exercised a real multi-process Flink
cluster before this: a `NotSerializableException` from `AilakeScanInputFormat`
holding a non-serializable `ResolvedSchema` field, and `search.mode='full'`
silently dropping the first (lowest-distance) result row due to a manual
`position: Int` indexing scheme. See `CHANGELOG.md` for the full write-up.

---

## 8. Recommended execution order

For first-time exploration:

```
01 → 02 → 03                    (core: write, DuckDB, Spark)
       ↓ need --profile engines
       04 → 05                  (Trino, BigQuery)

01 §21-23 → 07                  (multimodal prerequisite)
01 §24-28 → 08                  (agent memory prerequisite)
09 → 11                         (BM25 legacy → Tantivy FTS)
13                               (DuckLake catalog — self-contained)

       ↓ need --profile airflow
       12                       (Airflow pipelines)

       ↓ need --profile gpu + NVIDIA GPU
       10                       (GPU acceleration)

       ↓ need --profile flink
       14                       (Flink SQL)
```

Notebooks 01, 02, 03, 06, 07, 08, 09, 11, 13 can run in any order without profiles.

---

## 9. Stopping and cleanup

```bash
# Stop all services (data volumes preserved)
docker compose -f tests/docker/compose-demo.yml down

# Stop with a specific profile
docker compose -f tests/docker/compose-demo.yml --profile airflow down

# Full cleanup — removes containers, networks, AND volumes (destroys fixture data)
docker compose -f tests/docker/compose-demo.yml --profile engines --profile airflow --profile gpu down -v
```

After `down -v`, the next `up -d` re-runs `init_demo.py` and rebuilds all fixture tables (~1-2 min).

---

## 10. Troubleshooting

### JupyterLab blank or connection refused

```bash
docker logs ailake-demo-jupyter --tail 30
```

Common causes: wheel build still running (wait for `maturin build` to finish), port 8888 already in use.

### Fixture tables not found (`/data/ailake_demo` missing)

```bash
docker exec ailake-demo-jupyter python3 /opt/init_demo.py
```

### Airflow DAGs not appearing

DAGs are scanned every 10 s (`AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=10`). Check:

```bash
docker logs ailake-demo-airflow --tail 50 | grep -i "dag\|error"
```

### Trino: `CONNECTION_REFUSED` in notebook 04

Trino takes ~30 s to become ready. Check health:

```bash
curl -sf http://localhost:8080/v1/info | python3 -m json.tool | grep starting
# Expected: "starting": false
```

### Notebook 13: `ailake: command not found` or DuckLake extension install hangs

`ailake --version` (notebook §0) confirms the CLI binary is present — if missing,
rebuild the `jupyter` image (see [§3](#3-building-the-docker-images); a stale image
built before this binary was added won't have it). If `create`/`insert` hang on
first run, the container is fetching the `ducklake` DuckDB extension over the
network (`INSTALL ducklake; LOAD ducklake;`, one-time, cached after) — confirm the
container has outbound internet access.

### Rebuild after code changes

See [§3 Building the Docker images](#3-building-the-docker-images) for the full rebuild reference.

### Port conflicts

| Port | Default service | Override |
|---|---|---|
| 8888 | JupyterLab | Edit `ports` in `compose-demo.yml` |
| 8090 | Airflow | Edit Airflow service `ports` |
| 9000/9001 | MinIO | Edit MinIO service `ports` |
| 8080 | Trino | Edit Trino service `ports` |
| 8082 | Flink Web UI | Edit Flink service `ports` |

---

## 11. Environment variables reference

All variables are set by `compose-demo.yml` and consumed by `init_demo.py` and the notebooks.

| Variable | Default | Used by |
|---|---|---|
| `DEMO_TABLE_PATH` | `/data/ailake_demo` | All notebooks |
| `DEMO_MULTIMODAL_PATH` | `/data/ailake_multimodal` | `07_multimodal.ipynb` |
| `DEMO_AGENT_PATH` | `/data/ailake_agent` | `08_agents.ipynb` |
| `DEMO_FTS_PATH` | `/data/ailake_fts` | `11_fts.ipynb`, `14_flink.ipynb` §3 |
| `DEMO_DUCKLAKE_STORE` | `/data/ailake_ducklake` | `13_ducklake.ipynb` (`--store` for `--catalog ducklake`) |
| `AILAKE_HOOKS_DELETE_PATH` | `/data/ailake_hooks_delete_demo` | `dag_ailake_hook_ops.py`, `12_airflow.ipynb` §8B |
| `AILAKE_HOOKS_EVOLVE_PATH` | `/data/ailake_hooks_evolve_demo` | `dag_ailake_hook_ops.py`, `12_airflow.ipynb` §8B |
| `DEMO_DIM` | `32` | All notebooks (vector dimension) |
| `MINIO_ENDPOINT` | `http://minio:9000` | Notebook 01 §11, MinIO upload |
| `NESSIE_URI` | `http://nessie:19120/api/v1` | `init_demo.py` Nessie registration |
| `TRINO_HOST` | `trino` | `04_trino.ipynb` |
| `BQ_EMULATOR_HOST` | `bigquery-emulator` | `05_bigquery.ipynb` |
| `FLINK_HOST` | `flink` | `14_flink.ipynb` |
| `FLINK_PORT` | `8081` | `14_flink.ipynb` |
| `AIRFLOW_URL` | `http://ailake-demo-airflow:8080` | `12_airflow.ipynb` |
| `AIRFLOW_USER` | `admin` | `12_airflow.ipynb` |
| `AIRFLOW_PASSWORD` | `admin` | `12_airflow.ipynb` |
| `AILAKE_GPU_DEMO` | `1` (gpu profile only) | `10_gpu_demo.ipynb` |
