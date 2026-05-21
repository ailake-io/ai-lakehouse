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
}

/// Snapshot commit request.
#[derive(Debug, Clone)]
pub struct NewSnapshot {
    pub snapshot_id: SnapshotId,
    pub parent_snapshot_id: Option<SnapshotId>,
    pub files: Vec<DataFileEntry>,
    pub operation: SnapshotOperation,
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
