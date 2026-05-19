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

- `Quantizer::f32_to_f16_bytes(&[f32]) -> Vec<u8>` — half-precision cast
- `Quantizer::f32_to_i8(&[f32]) -> (Vec<i8>, ScalingParams)` — symmetric min-max
- `PQCodebook::train(vectors, M, k, max_iter) -> PQCodebook` — k-means++ per subspace
- `PQCodebook::encode(&[f32]) -> Vec<u8>` — M bytes, one code per subspace
- `PQCodebook::compute_adc_table(query) -> Vec<Vec<f32>>` — precomputed ADC table for fast batch search
- `PQCodebook::adc_distance(codes, table) -> f32` — O(M) approximate distance
- `cosine_distance(a, b) -> f32`, `euclidean_distance`, `dot_product`
- `compute_centroid_and_radius(&[Vec<f32>], VectorMetric) -> Centroid`
- `BlockCompressor::zstd(level)`, `BlockCompressor::lz4()` — block-level compression

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
  - `get_range(path, range: Range<u64>) → Bytes` — critical for partial reads of HNSW footer from S3
  - `put(path, Bytes)`, `list(prefix)`, `file_size(path)`, `exists(path)`, `delete(path)`
- `LocalStore` — filesystem implementation (dev/tests)
- `ObjectStoreBackend` — wraps any `Arc<dyn object_store::ObjectStore>`:
  - S3: `object_store::aws::AmazonS3Builder`
  - GCS: `object_store::gcp::GoogleCloudStorageBuilder`
  - Azure: `object_store::azure::MicrosoftAzureBuilder`
  - Feature-gated: `store-s3`, `store-gcs`, `store-azure`
- All async, all return `AilakeError` on failure

### `ailake-query`
Query planning and execution. The integration layer — depends on all data-plane crates.

- `TableWriter` — `write_batch(batch, embeddings)` + `commit()` → Iceberg snapshot
- `VectorPruner::prune(files, query, metric, threshold)` — filters `Vec<DataFileEntry>` using centroid geometry; works on catalog metadata only, zero file I/O for pruned files
- `search(table, query, config, ...)` — full pipeline: list catalog → prune → load HNSW → global top-k merge; `SearchConfig.pruning_threshold` controls prune aggressiveness
- `CompactionPlanner::plan(files)` — selects files smaller than `target_file_size_bytes`
- `CompactionExecutor::compact(files, output_path)` — merges N files into one via Arrow `concat_batches`, rebuilds HNSW, returns new `DataFileEntry`
- `CompactionExecutor::run(planner, table, catalog, prefix)` — full cycle: plan + compact + commit + delete old files
- `ContextAssembler::assemble_chunks(chunks: Vec<Chunk>)`:
  - Sorts by `distance` (most relevant first)
  - Deduplicates by embedding cosine distance < `dedup_threshold`
  - Groups by `document_id`, sorts each group by `chunk_index`
  - Applies `max_tokens` budget (4 chars ≈ 1 token)
  - Returns `AssembledContext { text: XML, chunk_count, token_estimate }`

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
    "tests",
    # Phase 3 bindings — excluded until Python/JVM deps are configured
    # "ailake-py",
    # "ailake-jni",
]

[workspace.dependencies]
# Core
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
uuid        = { version = "1", features = ["v4", "serde"] }
thiserror   = "1"
bytes       = "1"
half        = { version = "2", features = ["serde"] }
async-trait = "0.1"

# Async
tokio       = { version = "1", features = ["full"] }
futures     = "0.3"

# Data
parquet      = { version = "52", features = ["async"] }
arrow-array  = "52"
arrow-schema = "52"
arrow-select = "52"
object_store = { version = "0.10", features = ["aws", "gcp", "azure"] }

# Vector index
hnsw_rs     = "0.3"
bincode     = "1"
memmap2     = "0.9"

# Compression
lz4_flex    = "0.11"
zstd        = "0.13"

# Catalog backends (catalog crate adds iceberg/apache-avro directly)
reqwest     = { version = "0.12", features = ["json"] }  # REST catalog

# Bindings (excluded from workspace build until Phase 3)
pyo3        = { version = "0.21", features = ["extension-module"] }
uniffi      = "0.27"

# Observability
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Dev/test
criterion   = { version = "0.5", features = ["html_reports"] }
tempfile    = "3"
proptest    = "1"
rand        = "0.8"

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

| Phase | Status | Scope |
|---|---|---|
| **Phase 1** | ✅ Complete | Local MVP — write + search on local filesystem, HNSW footer, Iceberg catalog |
| **Phase 2** | ✅ Complete | Cloud storage (`ObjectStoreBackend`), mmap HNSW, compaction, PQ, geometric pruning, `ContextAssembler`, PyO3 bindings |
| **Phase 3** | Planned | JVM/Spark/Trino connectors (`uniffi`), multi-column vector tables |
| **Phase 4** | Planned | GPU index (cuVS FFI), PQ reranking, public format spec v1.0 |

### Phase 1 — Local MVP ✅
**Goal**: `cargo test --workspace` passes; can write a self-contained file and search it on local disk.

- `ailake-core`: all types
- `ailake-vec`: quantization F32→F16, centroid computation, distance functions
- `ailake-parquet`: writer (vector column encoding), reader (vector column decoding)
- `ailake-index`: `HnswBuilder`, `HnswIndex`, bincode serialization
- `ailake-file`: unified writer/reader, footer layout
- `ailake-catalog`: `CatalogProvider` trait + `HadoopCatalog` (filesystem) only
- `ailake-store`: `LocalStore` only
- Integration test: write + search end-to-end, verify recall

### Phase 2 — Distribution and Cloud Storage ✅

- `ailake-store`: `ObjectStoreBackend` wrapping `object_store` crate (S3/GCS/Azure via feature flags `store-s3`, `store-gcs`, `store-azure`)
- `ailake-index`: real `MmapLoader` — writes HNSW bytes to tempfile, mmaps, deserializes
- `ailake-vec`: `PQCodebook` (k-means++ per subspace, ADC distance), `BlockCompressor` (zstd/lz4)
- `ailake-query`: `VectorPruner` (geometric centroid pruning), `CompactionExecutor`, `ContextAssembler`
- `ailake-query::search`: pruning integrated via `SearchConfig.pruning_threshold`
- `ailake-py`: PyO3 bindings (`TableWriter`, `search`, `assemble_context`)

Deferred to Phase 3:
- `GlueCatalog`, `RestCatalog`, `NessieCatalog`, `JdbcCatalog`
- Docker integration tests (MinIO + Nessie + Localstack)

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
