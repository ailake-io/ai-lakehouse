use ailake_core::{AilakeResult, Centroid, VectorStoragePolicy};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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

/// Metadata about a single data file in a table snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileEntry {
    /// Relative path within the warehouse (e.g., "data/part-00001.parquet")
    pub path: String,
    pub record_count: u64,
    pub file_size_bytes: u64,
    /// base64-encoded centroid F32 values
    pub centroid_b64: Option<String>,
    pub radius: Option<f32>,
    pub hnsw_offset: Option<u64>,
    pub hnsw_len: Option<u64>,
    pub vector_column: Option<String>,
    pub vector_dim: Option<u32>,
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
    async fn create_table(
        &self,
        name: &TableIdent,
        props: &TableProperties,
    ) -> AilakeResult<()>;

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

/// Build DataFileEntry from a centroid, encoding it as base64.
pub fn make_data_file_entry(
    path: &str,
    record_count: u64,
    file_size_bytes: u64,
    centroid: &Centroid,
    hnsw_offset: u64,
    hnsw_len: u64,
    vector_column: &str,
    vector_dim: u32,
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
        hnsw_offset: Some(hnsw_offset),
        hnsw_len: Some(hnsw_len),
        vector_column: Some(vector_column.to_string()),
        vector_dim: Some(vector_dim),
    }
}

/// Decode centroid bytes from base64 in a DataFileEntry.
pub fn decode_centroid(entry: &DataFileEntry, metric: ailake_core::VectorMetric) -> Option<Centroid> {
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
