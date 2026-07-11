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
    /// Background index build failed permanently; `DataFileEntry::index_error`
    /// holds the cause. Compaction will rebuild the index on next run.
    Failed,
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
    /// Human-readable error from a failed background index build. Set only when
    /// `index_status == IndexStatus::Failed`; None otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_error: Option<String>,
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

impl DataFileEntry {
    /// True iff this file was never written by the AI-Lake SDK — most likely rewritten
    /// by a generic Iceberg engine (Spark/Trino `OPTIMIZE`/`rewrite_data_files`, DuckDB)
    /// with no knowledge of AI-Lake.
    ///
    /// Every AI-Lake write path (`make_data_file_entry`, `make_data_file_entry_indexing`)
    /// computes and stores a centroid unconditionally, even for `IndexStatus::Indexing`
    /// files — before the HNSW build itself completes. A missing centroid is therefore a
    /// reliable "not ours" signal, distinct from `IndexStatus::Failed` (an internal
    /// indexing failure on a file the SDK *did* write — see `writer.rs::patch_index_failed`,
    /// which never touches `centroid_b64`). Shared by `CompactionPlanner::plan()`,
    /// `scanner.rs::search`, and `ailake info` so the three can't drift on the definition.
    pub fn is_foreign(&self) -> bool {
        self.centroid_b64.is_none()
    }
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
    /// Equality delete files active in the current snapshot (Phase H).
    /// Loaded from delete manifests (content=2 in the manifest list).
    /// Populated by `CatalogProvider::list_equality_deletes` — empty until first call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub equality_delete_files: Vec<EqualityDeleteFile>,
    /// Active partition spec for this table (Phase I).
    /// `None` for unpartitioned tables or tables created before Phase I.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition_spec: Option<PartitionSpec>,
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

/// Reference to an Iceberg equality delete file (Phase H).
///
/// The delete file is an Avro file with `content=2` whose rows contain equality
/// predicates — any data row matching one of those rows is logically deleted.
/// `equality_ids` lists the field IDs used for the equality check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualityDeleteFile {
    /// Absolute or warehouse-relative path of the Avro delete file.
    pub path: String,
    /// Field IDs whose values must all match to delete a data row.
    pub equality_ids: Vec<i32>,
    /// Number of predicates (rows) in the delete file.
    pub record_count: u64,
    pub file_size_bytes: u64,
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
    /// Equality delete files to add to this snapshot (Phase H).
    /// Written as a separate delete manifest with content=2 in the manifest list.
    pub equality_delete_files: Vec<EqualityDeleteFile>,
}

#[derive(Debug, Clone)]
pub enum SnapshotOperation {
    Append,
    Overwrite,
    Delete,
    Replace,
}

/// One field in an Iceberg partition spec (Phase I).
///
/// For an `identity` partition on column "agent_id":
/// `source_id=1` (must match the field id in the table schema),
/// `field_id=1000` (Iceberg convention: partition fields start at 1000).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionField {
    /// Field ID of the source column in the table schema.
    pub source_id: i32,
    /// Partition field ID (≥ 1000 by convention).
    pub field_id: i32,
    /// Partition column name (usually same as the source column).
    pub name: String,
    /// Transform function: "identity", "bucket[N]", "truncate[W]", "year", etc.
    pub transform: String,
    /// Iceberg type of the source column ("string", "int", "long", "uuid").
    /// Derived from the table schema at read time; stored here for encoding.
    pub source_type: String,
}

/// Iceberg partition spec (Phase I).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionSpec {
    pub spec_id: i32,
    pub fields: Vec<PartitionField>,
}

impl PartitionSpec {
    /// True when this spec has no partition fields (unpartitioned table).
    pub fn is_unpartitioned(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Schema properties passed at table creation time.
#[derive(Debug, Clone)]
pub struct TableProperties {
    pub policy: VectorStoragePolicy,
    pub extra: HashMap<String, String>,
    /// Iceberg format version to write. 2 = default (V2). 3 = opt-in V3.
    pub format_version: u8,
    /// Iceberg type of the partition column when `policy.partition_by` is set.
    /// Defaults to `"string"` when `None`. Supported: "string", "uuid", "int", "long".
    pub partition_column_type: Option<String>,
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

    /// Whether a retired data file (one dropped by a `Replace`/`Overwrite` commit,
    /// e.g. compaction's merged-away inputs) should be physically deleted from the
    /// object store immediately after that commit succeeds.
    ///
    /// True for manifest-based backends (`HadoopCatalog` and friends): once a file
    /// drops out of the Iceberg manifest, the catalog has no other record of it, so
    /// deleting the bytes right away is safe.
    ///
    /// `DuckLakeCatalog` overrides this to `false` — see "The retirement problem"
    /// in `docs/guides/DUCKLAKE_CATALOG.md`: its `commit_snapshot` only issues a
    /// row-level `DELETE`, which attaches a deletion vector but leaves the file
    /// registered in `ducklake_list_files()` until an operator runs DuckLake's own
    /// `ducklake_expire_snapshots`/`ducklake_cleanup_files`. Deleting the bytes
    /// immediately (the default here) would leave that still-registered path
    /// dangling — breaking any subsequent DuckLake-native read that touches it.
    fn retires_files_physically(&self) -> bool {
        true
    }

    /// Whether data files, once committed at a path, may have their bytes
    /// rewritten in place at that same path.
    ///
    /// True for manifest-based backends: the manifest entry the caller commits
    /// alongside the rewrite is the only statistics record, so a same-path
    /// rewrite is self-consistent.
    ///
    /// `DuckLakeCatalog` overrides this to `false`: DuckLake snapshots record
    /// per-file zone-map stats and the exact footer size at
    /// `ducklake_add_data_files` time and trust both afterwards — verified
    /// against a live extension: a same-length rewrite silently returns wrong
    /// filtered rows (stale zone-map prunes the file), and a changed-length
    /// rewrite breaks every subsequent read of the file outright ("Parquet
    /// footer length stored in file is not equal to footer length provided"),
    /// including the row-`DELETE` needed to retire it — an unrecoverable state
    /// through sanctioned SQL. Writers that rewrite files (memory decay,
    /// deferred index patching) must either write to a fresh path and retire
    /// the old one, or refuse to run against catalogs returning `false` here.
    fn supports_in_place_rewrite(&self) -> bool {
        true
    }

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

    /// Return all equality delete files active in the current (or specified) snapshot.
    ///
    /// Reads delete manifests (manifest list entries with `content=1`) and parses their
    /// `content=2` entries. Default returns empty vec for catalog backends that do not
    /// support Iceberg equality deletes.
    async fn list_equality_deletes(
        &self,
        _table: &TableIdent,
        _snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<EqualityDeleteFile>> {
        Ok(vec![])
    }

    /// Add a new vector column to the table schema without rewriting data files.
    ///
    /// - Adds the column as Iceberg type `binary` (maps to `FIXED_LEN_BYTE_ARRAY` in Parquet).
    /// - Stores `ailake.dim-<col>`, `ailake.metric-<col>`, `ailake.precision-<col>` in properties.
    /// - Old files missing the column return `null` at read time (`initial-default = null`).
    /// - Does NOT commit a snapshot or backfill embeddings — use `BackfillJob` for that.
    ///
    /// Returns the new `schema-id`.
    async fn add_vector_column(
        &self,
        table: &TableIdent,
        spec: &ailake_core::VectorColSpec,
    ) -> AilakeResult<i32> {
        use crate::schema_evolution::{AddColumnRequest, SchemaEvolution};
        use std::collections::HashMap;

        let mut props: HashMap<String, String> = HashMap::new();
        props.insert(
            format!("ailake.dim-{}", spec.column_name),
            spec.dim.to_string(),
        );
        props.insert(
            format!("ailake.metric-{}", spec.column_name),
            format!("{:?}", spec.metric).to_lowercase(),
        );
        props.insert(
            format!("ailake.precision-{}", spec.column_name),
            format!("{:?}", spec.precision).to_lowercase(),
        );
        if spec.pre_normalize {
            props.insert(
                format!("ailake.pre-normalize-{}", spec.column_name),
                "true".to_string(),
            );
        }
        if let Some(m) = spec.hnsw_m {
            props.insert(format!("ailake.hnsw-m-{}", spec.column_name), m.to_string());
        }
        if let Some(ef) = spec.hnsw_ef_construction {
            props.insert(
                format!("ailake.hnsw-ef-construction-{}", spec.column_name),
                ef.to_string(),
            );
        }

        let evolution = SchemaEvolution::new()
            .add_column(AddColumnRequest {
                name: spec.column_name.clone(),
                iceberg_type: "binary".to_string(),
                required: false,
                initial_default: None,
                write_default: None,
                doc: Some(format!(
                    "Vector column {} dim={} metric={:?}",
                    spec.column_name, spec.dim, spec.metric
                )),
            })
            .with_properties(props);

        self.evolve_schema(table, evolution).await
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
        index_error: None,
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
        index_error: None,
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
        .map(|b| {
            f32::from_le_bytes(
                b.try_into()
                    .expect("chunks_exact(4) guarantees 4-byte slices"),
            )
        })
        .collect();
    Some(Centroid {
        values,
        radius: entry.radius.unwrap_or(0.0),
        metric,
    })
}

pub fn new_snapshot_id() -> SnapshotId {
    // Microsecond precision avoids collisions when multiple snapshots are committed
    // within the same millisecond (common in fast local-filesystem tests).
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}
