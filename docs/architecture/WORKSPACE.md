# WORKSPACE.md — Crate Architecture

## Dependency graph

Arrows point from consumer to dependency. No crate may introduce a cycle.

```
ailake-py  ──────────────────────────────────────────► ailake-query
ailake-jni ──────────────────────────────────────────► ailake-query
                                                              │
                        ┌─────────────────────────────────────┤
                        ▼                 ▼                   ▼
                  ailake-file       ailake-catalog       ailake-store
                        │                 │                   │
            ┌───────────┼─────────────────┘                   │
            ▼           ▼                                     │
       ailake-index ailake-parquet                            │
            │           │                                     │
            ▼           │                                     │
       ailake-vec       │                                     │
            │           │                                     │
            └───────┬───┘                                     │
                    ▼                                         │
              ailake-core ◄───────────────────────────────────┘
```

**Rule**: `ailake-core` has zero internal dependencies. Every other crate may depend on `ailake-core`. `ailake-query` depends on all data-plane crates.

## Crate responsibilities

### `ailake-core`
The shared type system. No I/O, no async, no external deps beyond `serde`, `uuid`, `thiserror`.

Public API surface:
- `VectorColumn` — name, dim, metric, precision
- `VectorMetric` — `Cosine | Euclidean | DotProduct`
- `VectorPrecision` — `F32 | F16 | I8Symmetric | Binary`
- `VectorStoragePolicy` — precision, PQ config, reranking flag
- `LlmContextSchema` — canonical field names and types for RAG tables
- `Centroid` — centroid + radius for pruning
- `AilakeError` — unified error enum, used by all crates via `thiserror`
- `RowId` — newtype `u64`, positional index linking Parquet row to HNSW node

### `ailake-parquet`
Reads and writes the **Parquet section** of the unified file. Knows about the `VECTOR` logical type extension via field metadata.

- `ParquetVectorWriter` — writes Arrow `RecordBatch` + encodes vector column as `FIXED_LEN_BYTE_ARRAY` with field metadata (`ailake.dim`, `ailake.metric`, `ailake.precision`)
- `ParquetVectorReader` — reads Parquet, detects vector column by field metadata, returns `RecordBatch`
- This crate does NOT touch the AI-Lake footer extension. That is `ailake-file`'s job.
- Iceberg-compatible: standard readers stop at the PAR1 marker and never see the AI-Lake footer.

### `ailake-vec`
Vector data transformations. No I/O.

- `Quantizer::f32_to_f16(&[f32]) -> Vec<u8>` — lossless precision cast
- `Quantizer::f32_to_i8(&[f32]) -> (Vec<i8>, ScalingParams)` — symmetric min-max
- `PQEncoder::train(vectors, M, k) -> Codebook` — Product Quantization training
- `PQEncoder::encode(&[f32]) -> Vec<u8>`
- `Distance::cosine(a: &[f32], b: &[f32]) -> f32`
- `Distance::euclidean(a: &[f32], b: &[f32]) -> f32`
- `compute_centroid(&[Vec<f32>]) -> (Vec<f32>, f32)` — returns (centroid, radius)

### `ailake-index`
HNSW index lifecycle. Wraps `hnsw_rs`.

- `HnswBuilder` — builds HNSW from `(RowId, &[f32])` pairs
  - Parameters: `M` (max connections), `ef_construction`, `metric`
- `HnswIndex` — searchable index over typed `RowId` keys
  - `search(query: &[f32], top_k: usize, ef_search: usize) -> Vec<(RowId, f32)>`
- `Serializer` — bincode-based serialization of the full HNSW graph
  - `serialize(index: &HnswIndex) -> Vec<u8>`
  - `deserialize(bytes: &[u8]) -> HnswIndex`
- `MmapLoader` — opens a serialized HNSW from a memory-mapped byte slice
  - Lazy: graph traversal only pages in the regions touched during search

### `ailake-file`
**Owns the unified file format.** This is the integration crate that combines Parquet + AI-Lake footer.

- `AilakeFileWriter` — high-level writer:
  1. Writes RecordBatch via `ailake-parquet`
  2. Builds HNSW via `ailake-index`
  3. Serializes HNSW + centroid + radius into the AI-Lake footer
  4. Appends footer to the file after the final PAR1 marker
  5. Updates Parquet `key_value_metadata` with `ailake.hnsw_offset` and `ailake.hnsw_len`
- `AilakeFileReader` — high-level reader:
  - `read_parquet()` → returns Parquet data only (via `ailake-parquet`)
  - `load_index()` → reads AI-Lake footer, returns `HnswIndex` via mmap
  - `get_centroid()` → reads centroid + radius from footer header (cheap, no HNSW load)
- `FooterLayout` — binary layout spec of the AI-Lake footer (see `FILE_FORMAT.md`)

See [`docs/specs/FILE_FORMAT.md`](../specs/FILE_FORMAT.md) for the binary layout.

### `ailake-catalog`
Iceberg catalog operations. The only crate that reads/writes `metadata.json` and `.avro` manifests.

Implements the `CatalogProvider` trait for every supported backend:

```
ailake-catalog/src/
├── lib.rs            # CatalogProvider trait, TableIdent, DataFileEntry, NewSnapshot
├── metadata.rs       # metadata.json read/write (Iceberg Spec v2)
├── snapshot.rs       # snapshot creation, vector stats in custom_properties
├── glue.rs           # AWS Glue Data Catalog (uses aws-sdk-glue)
├── rest.rs           # Iceberg REST Catalog spec (Polaris, Unity Catalog, S3 Tables)
├── nessie.rs         # Project Nessie (wraps REST, adds branch/tag ops)
├── hadoop.rs         # Filesystem catalog — metadata.json on local FS / S3 / GCS
└── jdbc.rs           # JDBC catalog — metadata in PostgreSQL or MySQL
```

`CatalogProvider` trait:
```rust
#[async_trait]
pub trait CatalogProvider: Send + Sync {
    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata>;
    async fn commit_snapshot(&self, table: &TableIdent, snapshot: NewSnapshot) -> AilakeResult<SnapshotId>;
    async fn list_files(&self, table: &TableIdent, snapshot_id: Option<SnapshotId>) -> AilakeResult<Vec<DataFileEntry>>;
    async fn create_table(&self, name: &TableIdent, schema: &Schema, props: &TableProperties) -> AilakeResult<()>;
    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()>;
}
```

All vector statistics (centroid, radius, HNSW byte offsets) are stored in the `custom_properties` map of each `DataFile` entry in the Avro manifest — a spec-defined extension point ignored by unknown readers.

Backend selection is driven by configuration, not code changes. The `ailake-query` layer depends only on `dyn CatalogProvider`.

### `ailake-store`
Object storage abstraction. Thin wrapper over the `object_store` crate.

- `Store` trait:
  - `get(path) → Bytes`
  - `get_range(path, range) → Bytes` — critical for partial reads of HNSW footer
  - `put(path, Bytes)`
  - `list(prefix) → Vec<Path>`
- `LocalStore` — filesystem (Phase 1, tests)
- `S3Store`, `GcsStore`, `AzureStore` — cloud backends (Phase 2)
- All async, all return `AilakeError` on failure

### `ailake-query`
Query planning and execution. The integration layer — depends on all data-plane crates.

- `VectorPruner` — loads centroids from `VectorStatsCatalog`, computes distances to query vector, returns list of candidate file paths
- `VectorScanner` — for each candidate file:
  1. Partial `GET` of Parquet footer to extract `ailake.hnsw_offset/len`
  2. Partial `GET` of AI-Lake footer bytes
  3. Load HNSW via `MmapLoader`
  4. Search top-k locally
  5. Return `(RowId, f32)` pairs
  6. Read full Parquet rows for top results
- `ContextAssembler` — given retrieved `RetrievedChunk` list:
  - Deduplicates by cosine distance threshold
  - Groups by `document_id`, sorts by `chunk_index`
  - Truncates to `max_tokens` budget
  - Returns `AssembledContext` with prompt-ready XML structure

### `ailake-py`
PyO3 extension module. Thin — all logic lives in other crates.

Exports:
- `TableWriter(table_uri: str, policy: StoragePolicy)`
- `TableWriter.write_batch(record_batch: PyArrow, embeddings: np.ndarray)`
- `TableWriter.commit() → SnapshotId`
- `search(table_uri, query_vector, top_k, filter) → PyArrow RecordBatch`
- `assemble_context(chunks, max_tokens) → str`

### `ailake-jni`
uniffi bindings for JVM. Exposes only the hot path needed by Spark/Trino connectors.

Exports (UDL interface):
- `vector_search(table_uri, query_bytes, top_k, filter_sql) → Vec<RowResult>`
- `assemble_context(chunk_jsons, max_tokens) → String`

---

## Cargo.toml (workspace root)

```toml
[workspace]
resolver = "2"
members = [
    "ailake-core",
    "ailake-parquet",
    "ailake-vec",
    "ailake-index",
    "ailake-file",
    "ailake-catalog",
    "ailake-store",
    "ailake-query",
    "ailake-py",
    "ailake-jni",
]

[workspace.dependencies]
# Core
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
uuid        = { version = "1", features = ["v4", "serde"] }
thiserror   = "1"
bytes       = "1"
half        = { version = "2", features = ["serde"] }

# Async
tokio       = { version = "1", features = ["full"] }
futures     = "0.3"

# Data
parquet     = { version = "52", features = ["async"] }
arrow       = "52"
arrow-array = "52"
object_store = { version = "0.10", features = ["aws", "gcp", "azure"] }

# Iceberg
iceberg     = "0.3"
apache-avro = "0.16"

# Catalog backends
aws-sdk-glue       = { version = "1", optional = true }   # feature = "catalog-glue"
aws-config         = { version = "1", optional = true }
reqwest            = { version = "0.12", features = ["json"] }  # REST catalog
sqlx               = { version = "0.7", features = ["postgres", "mysql", "runtime-tokio-rustls"], optional = true }  # JDBC catalog

# Vector index
hnsw_rs     = "0.3"
bincode     = "1"
memmap2     = "0.9"

# Compression
lz4_flex    = "0.11"
zstd        = "0.13"

# Bindings
pyo3        = { version = "0.21", features = ["extension-module"] }
uniffi      = "0.27"

# Observability
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Dev/test
criterion   = { version = "0.5", features = ["html_reports"] }
tempfile    = "3"
proptest    = "1"

[profile.release]
lto         = "thin"
codegen-units = 1
opt-level   = 3

[profile.bench]
inherits    = "release"
debug       = true
```

---

## Build phases and what is in scope

### Phase 1 — Local MVP
**Goal**: `cargo test --workspace` passes; can write a self-contained file and search it on local disk.

In scope:
- `ailake-core`: all types
- `ailake-vec`: quantization F32→F16, centroid computation, distance functions
- `ailake-parquet`: writer (vector column encoding), reader (vector column decoding)
- `ailake-index`: `HnswBuilder`, `HnswIndex`, bincode serialization
- `ailake-file`: unified writer/reader, footer layout
- `ailake-catalog`: `CatalogProvider` trait + `HadoopCatalog` (filesystem) only
- `ailake-store`: `LocalStore` only
- Integration test: write 10k rows into a single file, search top-10, verify recall

Out of scope (Phase 1):
- Cloud storage backends
- `GlueCatalog`, `RestCatalog`, `NessieCatalog`, `JdbcCatalog`
- `ailake-py`, `ailake-jni`
- `MmapLoader` (deserialize fully into RAM for Phase 1)
- Compaction
- PQ quantization
- `ContextAssembler`

### Phase 2 — Distribution and Cloud Storage
- `ailake-store`: S3Store, GcsStore, AzureStore with `get_range` support
- `ailake-catalog`: `GlueCatalog`, `RestCatalog` (covers Polaris, Nessie, Unity Catalog, BigLake, S3 Tables)
- `ailake-index`: full `MmapLoader` with partial S3 reads
- Compaction job: merge N small files into one, rebuild HNSW
- `ailake-vec`: PQ quantization
- `ailake-query`: `VectorPruner`, `VectorScanner`, `ContextAssembler`
- `ailake-py`: full PyO3 bindings
- Integration tests with Docker (MinIO + Nessie + Localstack)

### Phase 3 — Query engine integration
- `ailake-catalog`: `NessieCatalog` (branching ops), `JdbcCatalog`
- `ailake-jni`: uniffi exports
- `ailake-spark-runtime` (separate Scala repo): Spark `VectorScanStrategy`, `ailake_search` UDF
- `ailake-trino-plugin` (separate Java repo): Trino `ConnectorTableFunction`
- Compatibility tests: Spark, Trino, Beam, DuckDB, PyIceberg
- Multi-column vector tables (`embedding` + `context_embedding`)

### Phase 4 — Production hardening
- GPU index via cuVS FFI
- I8 quantization + benchmarks
- Public format spec v1.0
- Reranking after PQ
- `ailake-flink` (separate Java repo): Flink sink/source connector
