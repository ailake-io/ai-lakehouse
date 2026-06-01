# AI-Lake Format

[![CI](https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/actions/workflows/ci.yml/badge.svg)](https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ailake-core.svg)](https://crates.io/crates/ailake-core)
[![PyPI](https://img.shields.io/pypi/v/ailake.svg)](https://pypi.org/p/ailake)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-MIT)

Vector-native Lakehouse format built on Apache Iceberg Spec v2, written in Rust.

**Single self-contained file**: tabular data, embeddings, and HNSW index live together in one Parquet-extended file at the S3 layer. ACID transactions via Iceberg. Any Iceberg-compatible framework reads AI-Lake tables without modification — the vector index in the file footer is invisible to standard Parquet readers.

---

## Why AI-Lake?

**No second system.** Traditional stacks split tabular data (Parquet/Iceberg) from vectors (Pinecone, Milvus, Weaviate). Two systems to operate, two consistency models, two billing lines, and a join across a network boundary at query time. AI-Lake collapses both into a single `.parquet` file — one source of truth, one transaction log, one S3 prefix.

**ACID vectors.** Iceberg snapshot isolation applies to vector search the same way it applies to SQL queries. Time-travel, rollback, and concurrent writers work out of the box. No eventual consistency or index rebuild windows.

**Iceberg-compatible by spec, not by convention.** Standard Parquet readers (Spark, Trino, DuckDB, Athena, Snowflake) read AI-Lake tables without any plugin. The HNSW index lives in the file footer past the final `PAR1` magic — invisible to readers that follow the Parquet spec. The vector scan is an additive capability, not a format fork.

**Geometric pruning cuts S3 costs before any I/O.** Each file records its vector centroid and radius in the Iceberg manifest. A query eliminates files whose centroid is geometrically too far — without opening a single Parquet file. On tables with thousands of files, 95–99% of objects are never fetched.

**One binary, zero GPU build flags.** NVIDIA cuBLAS and AMD hipBLAS are loaded at runtime via `libloading` (dynamic FFI — no compile-time dependency). The same release binary auto-selects GPU on CUDA/ROCm machines and falls back to AVX-512/AVX2/NEON SIMD on CPU-only machines. No recompilation, no feature flags, no driver headers required. NVIDIA CUDA Toolkit and AMD ROCm are proprietary software owned by their respective manufacturers; AI-Lake does not bundle or redistribute them. See [`SETUP.md §8F`](./SETUP.md) for the full licensing note.

**Rust core, first-class Python and JVM.** The write/search path is pure Rust (zero GC pauses, no JVM heap pressure). Python gets zero-copy PyArrow `RecordBatch` results. Spark, Trino, and Flink get a JNA C-ABI bridge — four exported functions shared across all three JVM plugins.

**Storage-efficient at scale.** F16 quantization halves vector storage vs. F32. Product Quantization (IVF-PQ) reduces the index footprint 10–100× for S3-resident workloads where sequential reads are cheap. PQ reranking recovers precision with a second pass over the raw F16 column.

| | Iceberg alone | External vector DB | **AI-Lake** |
|---|---|---|---|
| ACID transactions | ✅ | ❌ | ✅ |
| SQL via Spark / Trino | ✅ | ❌ | ✅ |
| Native vector search | ❌ | ✅ | ✅ |
| Single file / single system | ✅ | ❌ | ✅ |
| Geometric file pruning | ❌ | ❌ | ✅ |
| GPU search (NVIDIA + AMD) | ❌ | Vendor-specific | ✅ |
| Time-travel on vectors | ❌ | ❌ | ✅ |

---

## Interactive demo (single command)

Spin up a local environment with MinIO, Nessie, and JupyterLab pre-loaded with 500 synthetic documents and an HNSW index — no cloud account, no credentials:

```bash
# From the repository root — builds ailake-py wheel on first run (~3-5 min, cached after)
docker compose -f tests/docker/compose-demo.yml up -d
```

Then open **http://localhost:8888** and run the notebooks:

| Notebook | What it shows |
|---|---|
| `01_ailake_demo.ipynb` | Vector search, Iceberg compat, RAG context assembly, MinIO upload |
| `02_duckdb.ipynb` | DuckDB direct Parquet scan, filtered queries, aggregations |
| `03_spark.ipynb` | PySpark local[*], Iceberg HadoopCatalog SQL, snapshot history |
| `04_trino.ipynb` | Trino SQL via `trino` Python driver, `$snapshots` / `$files` system tables |
| `05_bigquery.ipynb` | BigQuery emulator streaming inserts, SQL queries |

Notebooks 04 and 05 require the engines overlay (adds Trino + BigQuery emulator):

```bash
docker compose \
  -f tests/docker/compose-demo.yml \
  -f tests/docker/compose-demo-engines.yml \
  up -d
```

See [`tests/docker/`](./tests/docker/) for compose file details.

---

## Quick orientation

| Document | What it answers |
|---|---|
| [`CLAUDE.md`](./CLAUDE.md) | Architecture decisions, format spec, storage strategy, LLM context design |
| [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) | Crate map, dependency graph, build instructions |
| [`docs/architecture/DATA_FLOW.md`](./docs/architecture/DATA_FLOW.md) | Write path, read path, compaction flow end-to-end |
| [`docs/architecture/CATALOG_BACKENDS.md`](./docs/architecture/CATALOG_BACKENDS.md) | `CatalogProvider` trait + Hadoop / REST / Glue / Nessie / JDBC backends |
| [`docs/specs/FILE_FORMAT.md`](./docs/specs/FILE_FORMAT.md) | Binary spec of the unified `.parquet` file with AI-Lake footer |
| [`docs/specs/ICEBERG_COMPAT.md`](./docs/specs/ICEBERG_COMPAT.md) | Exactly how compatibility with Iceberg readers is maintained |
| [`docs/specs/LLM_CONTEXT.md`](./docs/specs/LLM_CONTEXT.md) | `LlmContextSchema`, dual embeddings, `ContextAssembler` |
| [`docs/specs/INTEGRATIONS.md`](./docs/specs/INTEGRATIONS.md) | Spark, Trino, Beam, AWS, GCP, Azure — config snippets and compatibility matrix |
| [`docs/specs/CLOUD_DEPLOY.md`](./docs/specs/CLOUD_DEPLOY.md) | Step-by-step deployment on EMR, Glue, Lambda, Dataproc, Dataflow, Databricks, HDInsight, AzureML |
| [`docs/specs/COMPACTION.md`](./docs/specs/COMPACTION.md) | Compaction job design, triggers, HNSW rebuild strategy |
| [`docs/contributing/TESTING.md`](./docs/contributing/TESTING.md) | Test strategy, fixtures, CI matrix, compat test harness |
| [`docs/contributing/CODING_STANDARDS.md`](./docs/contributing/CODING_STANDARDS.md) | Rust conventions, error handling, unsafe policy, testing rules |
| [`docs/contributing/DECISIONS.md`](./docs/contributing/DECISIONS.md) | ADR log — why each key choice was made |
| [`SETUP.md`](./SETUP.md) | Local dev setup — run the full stack (MinIO, Nessie, compat tests) on your machine |

## Install

**Rust** (add to `Cargo.toml`):
```toml
[dependencies]
ailake-core  = "0.0.10"
ailake-query = "0.0.10"   # search(), TableWriter, ContextAssembler
ailake-store = "0.0.10"   # S3 / GCS / Azure / local backends
```

**Python**:
```bash
pip install ailake
```

```python
import ailake

writer = ailake.TableWriter("s3://my-lake/docs/")
writer.write_batch(arrow_table, embeddings=np.array(..., dtype=np.float32))
writer.commit()

results = ailake.search("s3://my-lake/docs/", query_embedding, top_k=20)
# returns a PyArrow RecordBatch — zero-copy to pandas / polars
```

**Apache Airflow**:
```bash
pip install apache-airflow-providers-ailake
```

**JVM (Spark / Trino / Flink)** — download pre-built JARs from [GitHub Releases](https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/releases):

```bash
VERSION=0.0.10

# Spark plugin
wget https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/releases/download/v${VERSION}/spark-plugin-${VERSION}-plugin.jar

# Trino plugin
wget https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/releases/download/v${VERSION}/trino-plugin-${VERSION}-plugin.jar

# Flink connector
wget https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/releases/download/v${VERSION}/ailake-flink-${VERSION}-plugin.jar

# Native library (required by all three — place on java.library.path)
wget https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/releases/download/v${VERSION}/libailake_jni.so
```

See [`docs/specs/JVM_PLUGINS.md`](./docs/specs/JVM_PLUGINS.md) for installation and configuration.

## Repository layout

```
ailake/
├── CLAUDE.md
├── README.md
├── Cargo.toml                  # workspace root
├── docs/
│   ├── architecture/
│   ├── specs/
│   └── contributing/
├── ailake-core/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── types.rs            # VectorColumn, VectorMetric, Distance, RowId
│       ├── schema.rs           # LlmContextSchema, VectorStoragePolicy
│       └── error.rs            # AilakeError
├── ailake-parquet/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── reader.rs           # Parquet reader (data section only)
│       └── writer.rs           # Parquet writer (data section only)
├── ailake-vec/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── quantize.rs         # F32→F16→I8 scalar quantization
│       ├── distance.rs         # Cosine, Euclidean, DotProduct, centroid computation
│       ├── compress.rs         # BlockCompressor (zstd / lz4 / none)
│       └── pq.rs               # Product Quantization — PQCodebook, ADC distance
├── ailake-file/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── footer.rs           # AI-Lake footer binary layout
│       ├── writer.rs           # writes Parquet + AI-Lake footer
│       └── reader.rs           # reads either section
├── ailake-catalog/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── metadata.rs         # metadata.json read/write
│       ├── snapshot.rs         # Iceberg snapshot with vector stats
│       ├── databricks.rs       # Databricks Unity Catalog — config builders (Azure/AWS/GCP)
│       ├── glue.rs             # AWS Glue catalog backend
│       ├── rest.rs             # REST catalog backend (Polaris, Nessie, Unity)
│       ├── nessie.rs           # Nessie-specific extensions
│       ├── hadoop.rs           # Filesystem catalog (local dev)
│       └── jdbc.rs             # JDBC catalog (PostgreSQL/MySQL)
├── ailake-store/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── store.rs                  # Store trait
│       ├── local.rs                  # LocalStore — filesystem (dev/tests)
│       └── object_store_backend.rs   # ObjectStoreBackend — S3/GCS/Azure via object_store
├── ailake-index/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # AnyIndex enum — dispatches HNSW or IVF-PQ
│       ├── hnsw.rs             # hnsw_rs wrapper
│       ├── ivf_pq.rs           # IvfPqIndex, IvfPqConfig, IvfPqSerializer
│       ├── gpu.rs              # NVIDIA CUDA (cuBLAS libloading) + AMD ROCm (hipBLAS libloading) GPU backends
│       ├── hardware.rs         # HardwareProfile, HardwareBackend detection (CUDA / ROCm / CPU)
│       ├── serialize.rs        # bincode serialization
│       └── mmap_loader.rs      # memmap2 loading
├── ailake-query/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── writer.rs           # TableWriter — write_batch, write_batch_ivf_pq, write_batch_multi
│       ├── mem_table.rs        # MemTableWriter — streaming ingestion write buffer
│       ├── scanner.rs          # search() with geometric pruning; AnyIndex dispatch
│       ├── pruner.rs           # VectorPruner — centroid-based file pruning
│       ├── compaction.rs       # CompactionPlanner + CompactionExecutor (async)
│       └── context_assembler.rs # ContextAssembler — dedup, XML, token budget
├── ailake-cli/
│   ├── Cargo.toml
│   └── src/
│       └── main.rs             # CLI: ailake create / insert / search / compact / info
├── ailake-bench/
│   ├── Cargo.toml
│   └── src/
│       └── main.rs             # SIFT-1M benchmark vs. LanceDB / pgvector (--engine flag)
├── ailake-py/
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs              # PyO3 bindings (abi3-py39 wheel)
├── ailake-jni/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs              # C-ABI cdylib for Spark/Trino/Flink via JNA
├── spark-plugin/               # Scala — Spark 3.5 Catalyst strategy (Gradle)
│   ├── build.gradle.kts
│   └── src/main/scala/io/ailake/spark/
│       ├── AilakeSparkExtensions.scala
│       ├── AilakeNative.scala
│       ├── VectorSearchPlan.scala
│       ├── VectorScanExec.scala
│       └── VectorScanStrategy.scala
├── trino-plugin/               # Kotlin — Trino SPI connector (Gradle)
│   ├── build.gradle.kts
│   └── src/main/kotlin/io/ailake/trino/
│       ├── VectorScanConnector.kt
│       ├── VectorScanMetadata.kt
│       ├── VectorScanSplitManager.kt
│       ├── VectorScanRecordSet.kt
│       └── AilakeNative.kt
└── ailake-flink/               # Kotlin — Flink Table API connector (Gradle)
    ├── build.gradle.kts
    └── src/main/kotlin/io/ailake/flink/
        ├── AilakeCatalog.kt
        ├── AilakeVectorConnectorFactory.kt
        ├── AilakeVectorTableSink.kt
        └── AilakeVectorTableSource.kt
tests/
├── write_read_roundtrip.rs
├── iceberg_compat.rs
├── parquet_trailing_bytes.rs
├── vector_pruning.rs
├── positional_invariant.rs
├── context_assembler.rs
└── docker/
    ├── compose.yml              # MinIO + Nessie + Localstack (Phase 2 integration)
    ├── compose-engines.yml      # + Spark + Trino containers (Phase 3 compat)
    ├── compose-demo.yml         # Single-command onboarding demo (docker compose up -d)
    ├── compose-demo-engines.yml # Overlay: + Trino + BigQuery emulator
    └── demo/
        ├── Dockerfile           # Two-stage: Rust/maturin → JupyterLab
        ├── entrypoint.sh        # Init fixture then start Jupyter
        ├── init_demo.py         # Writes 500-row AI-Lake table at startup
        ├── trino-catalog/
        │   └── ailake.properties # Trino Iceberg HadoopCatalog config
        └── notebooks/
            ├── 01_ailake_demo.ipynb  # Vector search + Iceberg + RAG + MinIO
            ├── 02_duckdb.ipynb       # DuckDB direct Parquet scan
            ├── 03_spark.ipynb        # PySpark local[*] + Iceberg SQL
            ├── 04_trino.ipynb        # Trino SQL (engines overlay required)
            └── 05_bigquery.ipynb     # BigQuery emulator (engines overlay required)
```

## Build

```bash
cargo build --workspace
cargo build --workspace --release
cargo test --workspace
cd ailake-py && maturin develop
cargo check --workspace
```

## Phase status

| Phase | Status | Scope |
|---|---|---|
| **Phase 1** | ✅ Complete | Local MVP — write + search on local filesystem, HNSW footer, Iceberg catalog |
| **Phase 2** | ✅ Complete | Cloud storage (`ObjectStoreBackend`), mmap HNSW loading, compaction, PQ, geometric pruning, `ContextAssembler`, PyO3 bindings |
| **Phase 3** | ✅ Complete | Catalog backends (Nessie/JDBC/Glue), JNA C-ABI bindings, multi-column vectors, Spark/Trino/Flink plugins |
| **Phase 4** | ✅ Complete | PQ reranking, public format spec, GPU search (NVIDIA cuBLAS + AMD hipBLAS, both runtime-only), HNSW optimizations, IVF-PQ native index, GPU k-means, `MemTableWriter`, multi-vector columns, adaptive index selection, `ailake-flink` Kotlin connector (Flink Table API + Catalog) |
| **Phase 5** | ✅ Complete | Multi-language SDKs (`ailake-go`, `ailake-cpp`), `ailake serve` HTTP REST server, Apache Airflow provider, idempotent writes, Compat Heavy CI (Spark+Iceberg, Trino+REST, BigQuery emulator), TruffleHog secret scanning, cloud deployment guides |
| **Phase 6** | ✅ Complete | Public distribution pipeline — crates.io, PyPI (manylinux abi3 wheels), Airflow provider on PyPI, pre-built JVM JARs + `libailake_jni.so` on GitHub Releases, dynamic Python versioning |
| **Phase 7** | 🚧 Planned | DuckLake catalog backend (`DuckLakeCatalog` over DuckDB), dbt integration guide (dbt-spark + dbt-trino with AI-Lake plugins) |

See [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) for the full phase breakdown.
