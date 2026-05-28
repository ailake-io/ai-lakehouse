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

**One binary, zero GPU build flags.** NVIDIA cuBLAS and AMD hipBLAS are loaded at runtime via `libloading`. The same release binary auto-selects GPU on CUDA/ROCm machines and falls back to AVX-512/AVX2/NEON SIMD on CPU-only machines. No recompilation, no feature flags, no driver headers required.

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
ailake-core  = "0.0.9"
ailake-query = "0.0.9"   # search(), TableWriter, ContextAssembler
ailake-store = "0.0.9"   # S3 / GCS / Azure / local backends
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
    ├── compose.yml             # MinIO + Nessie + Localstack
    └── compose-engines.yml     # + Spark + Trino containers
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

See [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) for the full phase breakdown.
