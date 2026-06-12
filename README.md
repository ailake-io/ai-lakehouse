# AI-Lake Format

[![CI](https://github.com/ThiagoLange/ai-lakehouse/actions/workflows/ci.yml/badge.svg)](https://github.com/ThiagoLange/ai-lakehouse/actions/workflows/ci.yml)
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

**Storage-efficient at scale.** F16 quantization halves vector storage vs. F32. Product Quantization (IVF-PQ) reduces the index footprint 10–100× for S3-resident workloads where sequential reads are cheap.

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

Notebooks 04 and 05 require the `engines` profile (adds Trino + BigQuery emulator):

```bash
docker compose -f tests/docker/compose-demo.yml --profile engines up -d
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
ailake-core  = "0.0.17"
ailake-query = "0.0.17"   # search(), TableWriter, ContextAssembler
ailake-store = "0.0.17"   # S3 / GCS / Azure / local backends
```

**Python**:
```bash
pip install ailake
```

```python
import ailake
import numpy as np

# Write
table = ailake.open_table("s3://my-lake/docs/", dim=1536, metric="cosine")
table.insert(texts, np.array(embeddings, dtype=np.float32))
table.commit()

# Fluent search — chainable, DataFrame-native
df = ailake.search("s3://my-lake/docs/", query_embedding, top_k=20).to_pandas()

# Full-read: all Parquet columns + embedding (FixedSizeList<float32>) + _distance
df = ailake.search("s3://my-lake/docs/", query_embedding, top_k=20, fetch_data=True).to_pandas()

# Async
df = await table.search(query_embedding).limit(10).to_pandas_async()
```

**Apache Airflow**:
```bash
pip install apache-airflow-providers-ailake
```

**JVM (Spark / Trino / Flink)** — download pre-built JARs from [GitHub Releases](https://github.com/ThiagoLange/ai-lakehouse/releases):

```bash
VERSION=0.0.17

# Spark plugin
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/spark-plugin-${VERSION}-plugin.jar

# Trino plugin
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/trino-plugin-${VERSION}-plugin.jar

# Flink connector
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/ailake-flink-${VERSION}-plugin.jar

# Native library (required by all three — place on java.library.path)
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/libailake_jni.so
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
│       ├── ivf_pq.rs           # IvfPqIndex, IvfPqConfig, IvfPqCodebook, IvfPqSerializer
│       ├── gpu.rs              # NVIDIA CUDA (cuBLAS libloading) + AMD ROCm (hipBLAS libloading) GPU backends
│       ├── hardware.rs         # HardwareProfile, HardwareBackend detection (CUDA / ROCm / CPU)
│       ├── serialize.rs        # bincode serialization
│       └── mmap_loader.rs      # memmap2 loading
├── ailake-query/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── writer.rs           # TableWriter — write_batch, write_batch_deferred, write_batch_ivf_pq, write_batch_ivf_pq_deferred, write_batch_multi
│       ├── mem_table.rs        # MemTableWriter — streaming ingestion write buffer
│       ├── scanner.rs          # search() with geometric pruning; AnyIndex dispatch
│       ├── pruner.rs           # VectorPruner — centroid-based file pruning
│       ├── compaction.rs       # CompactionPlanner + CompactionExecutor (async)
│       └── context_assembler.rs # ContextAssembler — dedup, XML, token budget
├── ailake-cli/
│   ├── Cargo.toml
│   └── src/
│       └── main.rs             # CLI: ailake create / insert / search / compact / info / serve / estimate
├── ailake-py/
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs              # PyO3 bindings (abi3-py39 wheel)
├── ailake-jni/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs              # C-ABI cdylib for Spark/Trino/Flink via JNA
├── duckdb-ailake/              # C++ DuckDB community extension
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── ailake_extension.hpp  # AilakeLib singleton (dlopen + C-ABI bridge)
│   ├── src/
│   │   ├── ailake_extension.cpp  # Extension entry point + AilakeLib impl
│   │   ├── ailake_search.cpp     # ailake_search() table function
│   │   └── ailake_write.cpp      # ailake_write_batch() scalar function
│   └── test/
│       ├── test_search.py        # Search function integration tests
│       └── test_write.py         # Write function integration tests
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
├── ailake-flink/               # Kotlin — Flink Table API connector (Gradle)
│   ├── build.gradle.kts
│   └── src/main/kotlin/io/ailake/flink/
│       ├── AilakeCatalog.kt
│       ├── AilakeVectorConnectorFactory.kt
│       ├── AilakeVectorTableSink.kt
│       └── AilakeVectorTableSource.kt
├── ailake-go/                  # Go SDK — pure Go, no CGo (go.mod)
│   ├── go.mod
│   ├── ailake.go               # AilakeReader, AilakeWriter, VectorSearch
│   ├── catalog.go              # Iceberg metadata.json + manifest reading
│   ├── footer.go               # AI-Lake footer parser
│   ├── hnsw.go                 # HNSW graph traversal
│   ├── ivfpq.go                # IVF-PQ decoder + ADC search
│   ├── hardware.go             # Hardware detection (CUDA / ROCm / CPU)
│   ├── http_search.go          # HTTP client for `ailake serve` REST API
│   ├── distance.go             # Distance kernels (cosine, euclidean, dot)
│   └── simd_amd64.s            # AVX2 distance kernels (Go assembly)
├── ailake-cpp/                 # C++17 header-only SDK
│   ├── CMakeLists.txt
│   ├── include/ailake/
│   │   ├── ailake.hpp          # Public API entry point
│   │   ├── catalog.hpp         # Iceberg metadata reader
│   │   ├── footer.hpp          # AI-Lake footer parser
│   │   ├── hnsw.hpp            # HNSW search
│   │   ├── ivfpq.hpp           # IVF-PQ decoder
│   │   ├── distance.hpp        # Distance kernels
│   │   ├── hardware.hpp        # Hardware detection
│   │   ├── bincode.hpp         # bincode deserializer
│   │   ├── cuda/distance.cuh   # CUDA distance kernel
│   │   └── rocm/blas.hpp       # ROCm hipBLAS wrapper
│   └── src/
│       ├── catalog.cpp
│       └── search.cpp
└── airflow-providers-ailake/   # Apache Airflow 2.x/3.x provider (Python)
    ├── pyproject.toml
    ├── README.md
    └── airflow_providers_ailake/
        # AilakeHook, AilakeWriteOperator, AilakeSearchOperator, AilakeSnapshotSensor
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
    ├── compose-demo.yml         # Single-command onboarding demo; --profile engines adds Trino + BQ
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
            ├── 04_trino.ipynb        # Trino SQL (--profile engines required)
            └── 05_bigquery.ipynb     # BigQuery emulator (--profile engines required)
```

## Performance

Numbers below are from the [ailake-benchmark](https://github.com/ThiagoLange/ailake-benchmark) repository run on a single AWS `c6i.8xlarge` (32 vCPU, 64 GB RAM) with local NVMe. GPU numbers on `g5.xlarge` (NVIDIA A10G).

### Write throughput (`text-embedding-3-small`, dim=1536)

| Path | Throughput | Notes |
|---|---|---|
| `write_batch` (HNSW inline) | ~6 k vec/s | HNSW graph built synchronously per shard |
| `write_batch_deferred` (HNSW async) | ~200 k vec/s | Parquet written immediately; HNSW built in background |
| `write_batch_ivf_pq_deferred` (IVF-PQ async) | ~250 k vec/s | Parquet + k-means-trained PQ index async |
| `write_batch_auto_deferred` (auto) | ~200–250 k vec/s | Hardware-aware: selects IVF-PQ on GPU/≥8 cores, HNSW otherwise |

### Search latency (top-10, dim=1536, 1 M vectors, cosine)

| Index | Recall@10 | p50 latency | p99 latency |
|---|---|---|---|
| HNSW (F16, ef=50) | ~97% | ~4 ms | ~12 ms |
| IVF-PQ (nprobe=8) | ~93% | ~2 ms | ~8 ms |
| IVF-PQ residual (nprobe=8) | ~96% | ~2 ms | ~8 ms |
| IVF-PQ GPU (A10G, nprobe=8) | ~93% | ~0.4 ms | ~1 ms |

Geometric pruning eliminates 95–99% of files before any index is touched on tables with thousands of shards.

### Storage (`text-embedding-3-small`, dim=1536, 100 M vectors)

| Mode | Vector column | HNSW/IVF-PQ overhead | Total |
|---|---|---|---|
| F32 (raw) | ~600 GB | ~60–120 GB | ~660–720 GB |
| F16 (default) | ~300 GB | ~30–60 GB | ~330–360 GB |
| I8 | ~150 GB | ~15–30 GB | ~165–180 GB |
| IVF-PQ (M=48, K=256) | ~300 GB raw + ~5 GB PQ codes | ~5 GB | ~310 GB |
| PQ-only (`--pq-only`) | 0 GB (raw omitted) | ~5 GB | **~5 GB** |

PQ-only mode trades reranking precision for 98% storage reduction. Recall@10 ~93–95%.

---

## Code examples

| Language | Location | Run |
|---|---|---|
| **Rust** (write + search) | [`ailake-query/examples/demo.rs`](./ailake-query/examples/demo.rs) | `cargo run --example demo -p ailake-query` |
| **Python** (fluent API, async, RAG) | [`ailake-py/README.md`](./ailake-py/README.md) | `python -c "import ailake; ..."` |
| **Go** (search, scan) | [`ailake-go/examples/search/main.go`](./ailake-go/examples/search/main.go) | `go run . -warehouse /data/warehouse -table default.docs` |
| **C++** (search, CUDA) | [`ailake-cpp/examples/search.cpp`](./ailake-cpp/examples/search.cpp) | `./build/ailake_search -w /data/warehouse -t default.docs` |
| **Multi-engine** (Spark + Trino + DuckDB) | [`tests/docker/`](./tests/docker/) | `docker compose -f tests/docker/compose-demo.yml up -d` |

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
| **Phase 4** | ✅ Complete | PQ reranking, public format spec, GPU search (NVIDIA cuBLAS + AMD hipBLAS, both runtime-only), HNSW optimizations, IVF-PQ native index, GPU k-means, `MemTableWriter`, multi-vector columns, adaptive index selection, `ailake-flink` Kotlin connector; **IVF-PQ shared codebook** (single k-means training across all shards — ADC distances comparable cross-shard); **`write_batch_ivf_pq_deferred`** (~250k vec/s write, async IVF-PQ build); **k-means++ O(n×k) fix** + rayon parallelism (17× speedup); **`HadoopCatalog` Replace fix** (`IndexStatus::Ready` convergence with concurrent background tasks) |
| **Phase 5** | ✅ Complete | Multi-language SDKs (`ailake-go`, `ailake-cpp`), `ailake serve` HTTP REST server, Apache Airflow provider, idempotent writes, Compat Heavy CI (Spark+Iceberg, Trino+REST, BigQuery emulator), TruffleHog secret scanning, cloud deployment guides |
| **Phase 6** | ✅ Complete | Public distribution pipeline — crates.io, PyPI (manylinux abi3 wheels), Airflow provider on PyPI, pre-built JVM JARs + `libailake_jni.so` on GitHub Releases, dynamic Python versioning |
| **Phase 7** | 🚧 In progress | Done: DuckDB extension (`duckdb-ailake/`), Python full-read (`fetch_data=True`). Remaining: `write_batch_auto_deferred` (~200k vec/s Auto engine deferred); DuckLake catalog backend; dbt integration guide |

See [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) for the full phase breakdown.
