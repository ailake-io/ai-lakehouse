// SPDX-License-Identifier: MIT OR Apache-2.0
use ailake_core::{AilakeResult, Centroid, VectorStoragePolicy};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Whether a shard's HNSW index has been built.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum IndexStatus {
    /// HNSW index embedded in the file — normal HNSW search applies.
    #[default]
    Ready,
    /// Parquet written; HNSW build running in background — flat scan applies.
    Indexing,
}

/// Fully-qualified table identifier: namespace.table_name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableIdent {
    pub namespace: String,
    pub name: String,
}

impl TableIdent {
    pub fn new(namespace: &str, name: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            name: name.to_string(),
        }
    }
}

pub type SnapshotId = i64;

/// Iceberg V3 Deletion Vector reference stored in a manifest entry.
///
/// Points to a Roaring Bitmap blob inside a Puffin `.dvd` file.
/// `offset` + `length` address the blob bytes directly — no full Puffin
/// footer parse required for Phase B read support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionVector {
    /// Absolute path to the Puffin `.dvd` file in the object store.
    pub path: String,
    /// Byte offset of the Roaring Bitmap blob within the Puffin file.
    pub offset: u64,
    /// Byte length of the Roaring Bitmap blob.
    pub length: u64,
    /// Number of deleted rows (bitmap popcount; -1 when unknown).
    pub cardinality: i64,
}

/// HNSW index info for one additional (non-primary) vector column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraVectorIndex {
    pub column: String,
    pub dim: u32,
    pub hnsw_offset: u64,
    pub hnsw_len: u64,
    pub centroid_b64: Option<String>,
    pub radius: Option<f32>,
}

/// Metadata about a single data file in a table snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileEntry {
    /// Relative path within the warehouse (e.g., "data/part-00001.parquet")
    pub path: String,
    pub record_count: u64,
    pub file_size_bytes: u64,
    /// base64-encoded centroid F32 values (primary vector column)
    pub centroid_b64: Option<String>,
    pub radius: Option<f32>,
    pub hnsw_offset: Option<u64>,
    pub hnsw_len: Option<u64>,
    pub vector_column: Option<String>,
    pub vector_dim: Option<u32>,
    /// Additional vector columns beyond the primary (empty for single-column tables).
    #[serde(default)]
    pub extra_vector_indexes: Vec<ExtraVectorIndex>,
    /// Index build status. Defaults to Ready for backward compatibility with old manifests.
    #[serde(default)]
    pub index_status: IndexStatus,
    /// Caller-supplied idempotency key. When set, `write_batch_idempotent` skips the
    /// write if a file with the same batch_id is already committed in the snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    /// Embedding model identifier stored per-file so mixed-model tables (during migration)
    /// can be identified without reading the main metadata.json.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// Partition value for this file (e.g. the agent_id UUID).
    /// Written per-file when `VectorStoragePolicy::partition_by` is set.
    /// Enables manifest-level pruning: search skips files whose partition_value
    /// doesn't match the requested partition filter, avoiding all HNSW I/O.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition_value: Option<String>,
    /// Iceberg V3 Deletion Vector: Roaring Bitmap of deleted row positions.
    /// None for V2 tables or V3 tables with no deletes for this file.
    /// When present, scanner masks these row IDs from HNSW results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deletion_vector: Option<DeletionVector>,
    /// Iceberg V3 Row Lineage: globally unique first row ID assigned to this file.
    /// Computed at commit time from the table's cumulative `next-row-id` counter.
    /// None for V2 tables (row lineage requires format-version=3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,
}

/// One field from the current Iceberg table schema (Phase G).
///
/// Parsed from `schemas[current-schema-id].fields` in `metadata.json`.
/// Used by `SchemaFiller` to inject missing columns when reading old files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    pub id: i32,
    pub name: String,
    pub required: bool,
    /// Iceberg type string, e.g. `"int"`, `"string"`, `"timestamptz"`.
    pub iceberg_type: String,
    /// Value injected when reading old files that predate this field.
    /// `None` → null is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_default: Option<serde_json::Value>,
    /// Default value written to new files. Same as `initial_default` in most cases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_default: Option<serde_json::Value>,
}

/// Iceberg-compatible table metadata read from the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMetadata {
    pub table_uuid: String,
    pub format_version: i32,
    pub location: String,
    /// Ailake-specific properties: ailake.vector-column, ailake.dim, etc.
    pub properties: HashMap<String, String>,
    pub current_snapshot_id: Option<SnapshotId>,
    /// Absolute path to the Puffin stats file for the current snapshot (Phase F).
    /// `None` for V2 tables or V3 tables without any committed statistics yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_statistics_path: Option<String>,
    /// Current schema fields parsed from `metadata.json` (Phase G).
    /// Empty for tables created before Phase G or tables with no schema committed.
    /// Used by `SchemaFiller` in the scanner to inject missing columns with defaults.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schema_fields: Vec<SchemaField>,
}

/// Iceberg schema update carried inside a snapshot commit.
///
/// When present, `HadoopCatalog::commit_snapshot` patches `schemas[0].fields`,
/// `last-column-id`, and `schema.name-mapping.default` in the persisted metadata.
/// REST/Glue/JDBC backends that delegate schema management to the server ignore this.
#[derive(Debug, Clone)]
pub struct IcebergSchemaUpdate {
    /// Iceberg-typed field descriptors, e.g. `[{"id":1,"name":"id","required":false,"type":"int"}]`.
    pub fields: Vec<serde_json::Value>,
    pub last_column_id: i32,
    /// Compact JSON string: `[{"field-id":1,"names":["id"]},...]`.
    pub name_mapping_json: String,
}

/// Snapshot commit request.
#[derive(Debug, Clone)]
pub struct NewSnapshot {
    pub snapshot_id: SnapshotId,
    pub parent_snapshot_id: Option<SnapshotId>,
    pub files: Vec<DataFileEntry>,
    pub operation: SnapshotOperation,
    /// When set, the catalog backend should update the table schema on commit.
    pub iceberg_schema: Option<IcebergSchemaUpdate>,
    /// Additional table-level properties to merge on commit (e.g. secondary column dims).
    /// Keys use `ailake.dim-<col>` / `ailake.metric-<col>` convention.
    pub extra_properties: HashMap<String, String>,
    /// Per-file BM25 Bloom filter bytes for term-level file pruning (Phase F).
    /// Key = data file path (relative, matches `DataFileEntry::path`).
    /// Written to the Puffin stats file on V3 commits; ignored for V2 tables.
    pub bloom_filters: Vec<(String, Vec<u8>)>,
}

#[derive(Debug, Clone)]
pub enum SnapshotOperation {
    Append,
    Overwrite,
    Delete,
    Replace,
}

/// Schema properties passed at table creation time.
#[derive(Debug, Clone)]
pub struct TableProperties {
    pub policy: VectorStoragePolicy,
    pub extra: HashMap<String, String>,
    /// Iceberg format version to write. 2 = default (V2). 3 = opt-in V3.
    /// V3: append/update workloads fully supported. Equality deletes and
    /// equality deletes not implemented (see docs/specs/ICEBERG_V3.md).
    pub format_version: u8,
}

/// Unified catalog interface. All backends implement this trait.
#[async_trait]
pub trait CatalogProvider: Send + Sync {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()>;

    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata>;

    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> AilakeResult<SnapshotId>;

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>>;

    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()>;

    /// Apply schema evolution (add columns / rename columns) without rewriting data files.
    ///
    /// Returns the new `schema-id` assigned in `metadata.json`.
    /// Old files missing new columns will have their values filled at read time using
    /// `AddColumnRequest::initial_default` (Phase G `SchemaFiller`).
    ///
    /// Default implementation returns an error — override in file-based backends.
    async fn evolve_schema(
        &self,
        _table: &TableIdent,
        _evolution: crate::schema_evolution::SchemaEvolution,
    ) -> AilakeResult<i32> {
        Err(ailake_core::AilakeError::Catalog(
            "evolve_schema not supported by this catalog backend".into(),
        ))
    }
}

/// Vector index metadata for a single data file.
pub struct VectorIndexInfo<'a> {
    pub column: &'a str,
    pub dim: u32,
    pub hnsw_offset: u64,
    pub hnsw_len: u64,
}

/// Build DataFileEntry from a centroid, encoding it as base64.
pub fn make_data_file_entry(
    path: &str,
    record_count: u64,
    file_size_bytes: u64,
    centroid: &Centroid,
    index: VectorIndexInfo<'_>,
) -> DataFileEntry {
    make_multi_column_data_file_entry(path, record_count, file_size_bytes, centroid, index, &[])
}

/// Build DataFileEntry for a file with multiple vector columns.
///
/// `primary_centroid` and `primary_index` describe the primary (first) vector column.
/// `extra` contains info for additional columns.
pub fn make_multi_column_data_file_entry(
    path: &str,
    record_count: u64,
    file_size_bytes: u64,
    primary_centroid: &Centroid,
    primary_index: VectorIndexInfo<'_>,
    extra: &[ExtraVectorIndex],
) -> DataFileEntry {
    use base64::Engine;
    let centroid_bytes: Vec<u8> = primary_centroid
        .values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let centroid_b64 = base64::engine::general_purpose::STANDARD.encode(&centroid_bytes);
    DataFileEntry {
        path: path.to_string(),
        record_count,
        file_size_bytes,
        centroid_b64: Some(centroid_b64),
        radius: Some(primary_centroid.radius),
        hnsw_offset: Some(primary_index.hnsw_offset),
        hnsw_len: Some(primary_index.hnsw_len),
        vector_column: Some(primary_index.column.to_string()),
        vector_dim: Some(primary_index.dim),
        extra_vector_indexes: extra.to_vec(),
        index_status: IndexStatus::Ready,
        batch_id: None,
        embedding_model: None,
        partition_value: None,
        deletion_vector: None,
        first_row_id: None,
    }
}

/// Build a DataFileEntry for a file whose HNSW is still being built asynchronously.
///
/// `hnsw_offset` and `hnsw_len` are `None`; `index_status` is `Indexing`.
/// The centroid is included so geometric pruning still works during the build window.
pub fn make_data_file_entry_indexing(
    path: &str,
    record_count: u64,
    file_size_bytes: u64,
    centroid: &Centroid,
    column: &str,
    dim: u32,
) -> DataFileEntry {
    use base64::Engine;
    let centroid_bytes: Vec<u8> = centroid
        .values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let centroid_b64 = base64::engine::general_purpose::STANDARD.encode(&centroid_bytes);
    DataFileEntry {
        path: path.to_string(),
        record_count,
        file_size_bytes,
        centroid_b64: Some(centroid_b64),
        radius: Some(centroid.radius),
        hnsw_offset: None,
        hnsw_len: None,
        vector_column: Some(column.to_string()),
        vector_dim: Some(dim),
        extra_vector_indexes: vec![],
        index_status: IndexStatus::Indexing,
        batch_id: None,
        embedding_model: None,
        partition_value: None,
        deletion_vector: None,
        first_row_id: None,
    }
}

/// Encode a centroid to base64 for use in ExtraVectorIndex.
pub fn encode_centroid_b64(centroid: &Centroid) -> String {
    use base64::Engine;
    let bytes: Vec<u8> = centroid
        .values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

/// Decode centroid bytes from base64 in a DataFileEntry.
pub fn decode_centroid(
    entry: &DataFileEntry,
    metric: ailake_core::VectorMetric,
) -> Option<Centroid> {
    use base64::Engine;
    let b64 = entry.centroid_b64.as_ref()?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Some(Centroid {
        values,
        radius: entry.radius.unwrap_or(0.0),
        metric,
    })
}

pub fn new_snapshot_id() -> SnapshotId {
    // Use timestamp-based ID for simplicity (Iceberg uses i64)
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
