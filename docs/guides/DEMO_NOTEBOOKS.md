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

# Build ailake-py wheel + start all core services (~3-5 min on first run, instant after)
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
| `docker-jupyter` | `tests/docker/demo/Dockerfile` | Rust toolchain → `maturin build` → ailake-py wheel → JupyterLab |
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
| First build (no cache) | 3–5 min (Rust + wheel + JupyterLab) |
| Python-only change (`__init__.py`) | ~30 s |
| Rust source change (any `ailake-*/src/`) | 3–5 min (full recompile) |
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

---

## 5. Optional profiles

Profiles add heavyweight services on demand. Core notebooks (01-03, 07-12) work without any profile.

### `--profile engines` — Trino + BigQuery emulator

Required for notebooks **04** and **05**.

```bash
docker compose -f tests/docker/compose-demo.yml --profile engines up -d
```

Added services: `trinodb/trino:446` (Iceberg connector pointing at Nessie) + `goccy/bigquery-emulator:0.6.6`.

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
| 11–14 | MinIO upload, IVF-PQ `pq_only`, Residual-PQ, `write_batch_auto_deferred` |
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

---

### `03_spark.ipynb` — PySpark + Iceberg

**Profile required:** none (Spark runs in local[*] mode inside Jupyter)  
**Fixture dependency:** `/data/ailake_demo`, `/data/ailake_partitioned_v3`, `/data/ailake_delete_demo`, `/data/ailake_schema_evo`

> Takes ~30-60 s for Spark JVM startup on first cell.

| Section | Topic |
|---|---|
| 1 | SparkSession with Iceberg JAR, `HadoopCatalog` |
| 2 | `COUNT(*)`, `MIN/MAX`, schema inspection via SQL |
| 3 | Snapshot history + time-travel `VERSION AS OF` |
| 4 | Partitioned v3 table — partition spec visible |
| 5 | Equality delete visibility (rows 0-9 masked) |
| 6 | Schema evolution — `source_url` column added without rewrite |

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
| 1 | Trino connection (`trino` Python client, `TRINO_HOST` env var) |
| 2 | `COUNT(*)`, `SHOW TBLPROPERTIES` — `ailake.*` properties visible |
| 3 | `$files` + `$manifests` system tables (HNSW offset, centroid in key_metadata) |
| 4 | `partition_fields` DDL inspection |
| 5 | Equality delete files — 5 eq-del manifests committed (verified via `$manifests`); Trino 446 / Iceberg 1.5.2 does not apply them in MOR scan (COUNT stays 100); requires Iceberg 1.7+ / Trino 450+ for full MOR support |

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
| 1 | `ailake.Agent(path, embed_fn, agent_id)` |
| 2 | `remember()`, `recall()` — partition-scoped search |
| 3 | `WorkingMemoryBuffer` — in-memory with `drain_to_table()` |
| 4 | `EpisodicMemorySchema` — `recency_weight`, `importance_score` |
| 5 | `MemoryDecayJob` — exponential decay λ |
| 6 | `ToolCallSchema` — semantic search over tool call history |

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
| 3 | Search QPS comparison, recall@10 |
| 4 | GPU k-means for IVF-PQ training speedup |
| 5 | CPU fallback — same binary, no recompile |

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

Two pre-loaded DAGs (from `tests/docker/demo/dags/`):

| DAG | Schedule | Pipeline |
|---|---|---|
| `ailake_ingest_search` | `@daily` | `write_docs → vector_search → fts_search → assemble_context` |
| `ailake_compaction` | `@weekly` | `compact_table → table_info` |

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
| 9 | Direct PythonOperator demo — no Airflow needed |
| 10 | `AilakeWriteOperator` production pattern + connection setup |

> DAGs use `import ailake` (Python SDK) directly via TaskFlow API — the `ailake` CLI binary is not required inside the Airflow container.

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

       ↓ need --profile airflow
       12                       (Airflow pipelines)

       ↓ need --profile gpu + NVIDIA GPU
       10                       (GPU acceleration)
```

Notebooks 01, 02, 03, 06, 07, 08, 09, 11 can run in any order without profiles.

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

### Rebuild after code changes

See [§3 Building the Docker images](#3-building-the-docker-images) for the full rebuild reference.

### Port conflicts

| Port | Default service | Override |
|---|---|---|
| 8888 | JupyterLab | Edit `ports` in `compose-demo.yml` |
| 8090 | Airflow | Edit Airflow service `ports` |
| 9000/9001 | MinIO | Edit MinIO service `ports` |
| 8080 | Trino | Edit Trino service `ports` |

---

## 11. Environment variables reference

All variables are set by `compose-demo.yml` and consumed by `init_demo.py` and the notebooks.

| Variable | Default | Used by |
|---|---|---|
| `DEMO_TABLE_PATH` | `/data/ailake_demo` | All notebooks |
| `DEMO_MULTIMODAL_PATH` | `/data/ailake_multimodal` | `07_multimodal.ipynb` |
| `DEMO_AGENT_PATH` | `/data/ailake_agent` | `08_agents.ipynb` |
| `DEMO_FTS_PATH` | `/data/ailake_fts` | `11_fts.ipynb` |
| `DEMO_DIM` | `32` | All notebooks (vector dimension) |
| `MINIO_ENDPOINT` | `http://minio:9000` | Notebook 01 §11, MinIO upload |
| `NESSIE_URI` | `http://nessie:19120/api/v1` | `init_demo.py` Nessie registration |
| `TRINO_HOST` | `trino` | `04_trino.ipynb` |
| `BQ_EMULATOR_HOST` | `bigquery-emulator` | `05_bigquery.ipynb` |
| `AIRFLOW_URL` | `http://ailake-demo-airflow:8080` | `12_airflow.ipynb` |
| `AIRFLOW_USER` | `admin` | `12_airflow.ipynb` |
| `AIRFLOW_PASSWORD` | `admin` | `12_airflow.ipynb` |
| `AILAKE_GPU_DEMO` | `1` (gpu profile only) | `10_gpu_demo.ipynb` |
