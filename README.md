# AI-Lake Format

Vector-native Lakehouse format built on Apache Iceberg Spec v2, written in Rust.

**Single self-contained file**: tabular data, embeddings, and HNSW index live together in one Parquet-extended file at the S3 layer. ACID transactions via Iceberg. Any Iceberg-compatible framework reads AI-Lake tables without modification — the vector index in the file footer is invisible to standard Parquet readers.

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
| [`docs/specs/COMPACTION.md`](./docs/specs/COMPACTION.md) | Compaction job design, triggers, HNSW rebuild strategy |
| [`docs/contributing/TESTING.md`](./docs/contributing/TESTING.md) | Test strategy, fixtures, CI matrix, compat test harness |
| [`docs/contributing/CODING_STANDARDS.md`](./docs/contributing/CODING_STANDARDS.md) | Rust conventions, error handling, unsafe policy, testing rules |
| [`docs/contributing/DECISIONS.md`](./docs/contributing/DECISIONS.md) | ADR log — why each key choice was made |

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
│       ├── quantize.rs         # F32→F16→I8, PQ
│       └── distance.rs         # Cosine, Euclidean, DotProduct
├── ailake-index/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── hnsw.rs             # hnsw_rs wrapper
│       ├── serialize.rs        # bincode serialization
│       └── mmap_loader.rs      # memmap2 loading
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
│       ├── glue.rs             # AWS Glue catalog backend
│       ├── rest.rs             # REST catalog backend (Polaris, Nessie, Unity)
│       ├── nessie.rs           # Nessie-specific extensions
│       ├── hadoop.rs           # Filesystem catalog (local dev)
│       └── jdbc.rs             # JDBC catalog (PostgreSQL/MySQL)
├── ailake-store/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── store.rs            # object_store wrapper
├── ailake-query/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── pruner.rs           # vector pruning via centroids
│       ├── scanner.rs          # VectorScan execution
│       └── context_assembler.rs
├── ailake-py/
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs              # PyO3 bindings
└── ailake-jni/
    ├── Cargo.toml
    └── src/
        └── lib.rs              # uniffi bindings for Spark/Trino
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

## Phase 1 scope (current)

Build locally-runnable MVP: write a Parquet file with embedded HNSW footer, search via vector pruning + local HNSW lookup — all on local filesystem. No object storage, no Python bindings yet.

See [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) for the full phase breakdown.
