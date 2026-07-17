// SPDX-License-Identifier: MIT OR Apache-2.0
// HadoopCatalog: stores metadata.json on the local filesystem / any Store backend.
// Table layout: {warehouse}/{namespace}/{table}/metadata/vN.metadata.json

use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::manifest_commit::{
    commit_into_metadata, list_equality_deletes_from_metadata, list_files_from_metadata,
};
use crate::metadata::IcebergMetadata;
use crate::provider::{
    CatalogProvider, DataFileEntry, EqualityDeleteFile, NewSnapshot, SnapshotId, TableIdent,
    TableMetadata, TableProperties,
};
use ailake_store::Store;
use bytes::Bytes;

pub struct HadoopCatalog {
    store: Arc<dyn Store>,
    warehouse: String,
    // Serializes commits within the same process. Cross-process concurrent writes
    // require a REST or Nessie catalog — HadoopCatalog is single-process by design,
    // matching upstream Iceberg's documented limitation.
    commit_lock: Arc<Mutex<()>>,
}

impl HadoopCatalog {
    pub fn new(store: Arc<dyn Store>, warehouse: &str) -> Self {
        Self {
            store,
            warehouse: warehouse.trim_end_matches('/').to_string(),
            commit_lock: Arc::new(Mutex::new(())),
        }
    }

    fn table_root(&self, table: &TableIdent) -> String {
        if self.warehouse.is_empty() {
            format!("{}/{}", table.namespace, table.name)
        } else {
            format!("{}/{}/{}", self.warehouse, table.namespace, table.name)
        }
    }

    fn version_hint_path(&self, table: &TableIdent) -> String {
        format!("{}/metadata/version-hint.text", self.table_root(table))
    }

    fn versioned_metadata_path(&self, table: &TableIdent, version: u32) -> String {
        format!(
            "{}/metadata/v{}.metadata.json",
            self.table_root(table),
            version
        )
    }

    async fn current_version(&self, table: &TableIdent) -> AilakeResult<u32> {
        match self.store.get(&self.version_hint_path(table)).await {
            Ok(bytes) => Ok(std::str::from_utf8(&bytes)
                .unwrap_or("1")
                .trim()
                .parse::<u32>()
                .unwrap_or(1)),
            Err(_) => Ok(0),
        }
    }

    async fn load_raw_metadata(&self, table: &TableIdent) -> AilakeResult<IcebergMetadata> {
        let version = self.current_version(table).await?;
        let path = self.versioned_metadata_path(table, version);
        let bytes = self.store.get(&path).await?;
        let json = std::str::from_utf8(&bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;
        IcebergMetadata::from_json(json)
    }

    async fn save_metadata(&self, table: &TableIdent, meta: &IcebergMetadata) -> AilakeResult<()> {
        let next = self.current_version(table).await? + 1;
        let json = meta.to_json()?;
        self.store
            .put(
                &self.versioned_metadata_path(table, next),
                Bytes::from(json.into_bytes()),
            )
            .await?;
        self.store
            .put(
                &self.version_hint_path(table),
                Bytes::from(next.to_string()),
            )
            .await
    }
}

#[async_trait]
impl CatalogProvider for HadoopCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        // save_metadata() below always writes the *next* version and never checks
        // whether one already exists — without this guard, calling create_table on an
        // existing table silently resets it to a fresh, snapshot-less v(N+1), orphaning
        // every data file the old metadata referenced (still on disk, referenced by
        // nothing). Iceberg's own CREATE TABLE semantics reject a duplicate name (no
        // "IF NOT EXISTS"/"OR REPLACE" was requested here), and RestCatalog already gets
        // this for free from the server (a spec-compliant REST catalog 409s on a
        // duplicate `POST /namespaces/{ns}/tables`) — Hadoop is file-based, so the check
        // has to live here.
        if self.current_version(name).await? > 0 {
            return Err(AilakeError::Catalog(format!(
                "table {}.{} already exists",
                name.namespace, name.name
            )));
        }
        let location = self.table_root(name);
        // Prefer TableProperties.partition_column_type; fall back to policy field (set via Python/CLI).
        let pct = props
            .partition_column_type
            .as_deref()
            .or(props.policy.partition_column_type.as_deref());
        let mut meta = IcebergMetadata::new(
            &location,
            &props.policy,
            props.format_version,
            pct,
            &props.policy.partition_fields,
        );
        for (k, v) in &props.extra {
            meta.properties.insert(k.clone(), v.clone());
        }
        self.save_metadata(name, &meta).await
    }

    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata> {
        let meta = self.load_raw_metadata(name).await?;
        Ok(meta.to_table_metadata())
    }

    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> AilakeResult<SnapshotId> {
        let _guard = self.commit_lock.lock().await;
        let mut meta = self.load_raw_metadata(table).await?;
        let table_root = self.table_root(table);
        let snap_id = commit_into_metadata(
            &*self.store,
            &table_root,
            &self.warehouse,
            &mut meta,
            snapshot,
        )
        .await?;
        self.save_metadata(table, &meta).await?;
        Ok(snap_id)
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let meta = self.load_raw_metadata(table).await?;
        list_files_from_metadata(&*self.store, &meta, snapshot_id).await
    }

    async fn list_equality_deletes(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<EqualityDeleteFile>> {
        let meta = self.load_raw_metadata(table).await?;
        list_equality_deletes_from_metadata(&*self.store, &meta, snapshot_id).await
    }

    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()> {
        let version = self.current_version(name).await?;
        if version > 0 {
            let path = self.versioned_metadata_path(name, version);
            if self.store.exists(&path).await? {
                self.store.delete(&path).await?;
            }
            let hint = self.version_hint_path(name);
            if self.store.exists(&hint).await? {
                self.store.delete(&hint).await?;
            }
        }
        Ok(())
    }

    /// Apply schema evolution without rewriting data files (Phase G).
    ///
    /// Steps:
    /// 1. Load current `metadata.json`.
    /// 2. Clone the current schema; apply renames (field name only, id stable).
    /// 3. Append added fields with fresh field-ids and `initial-default` / `write-default`.
    /// 4. Push new schema entry with `schema-id = current + 1`.
    /// 5. Write new `metadata.json` (no new snapshot — pure metadata change).
    ///
    /// Returns the new `schema-id`.
    async fn evolve_schema(
        &self,
        table: &TableIdent,
        evolution: crate::schema_evolution::SchemaEvolution,
    ) -> AilakeResult<i32> {
        use ailake_core::AilakeError;

        let mut meta = self.load_raw_metadata(table).await?;
        let current_id = meta.current_schema_id;

        // Clone current schema's fields array.
        let current_schema = meta
            .schemas
            .iter()
            .find(|s| s["schema-id"].as_i64() == Some(current_id as i64))
            .ok_or_else(|| AilakeError::Catalog("current schema not found in metadata".into()))?
            .clone();

        let mut fields: Vec<serde_json::Value> = current_schema["fields"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        // Apply renames first (preserves field-ids).
        for rename in &evolution.renames {
            for field in fields.iter_mut() {
                if field["name"].as_str() == Some(rename.old_name.as_str()) {
                    field["name"] = serde_json::Value::String(rename.new_name.clone());
                }
            }
        }

        // Apply column additions.
        let mut last_col_id = meta.last_column_id;
        for add in evolution.adds {
            last_col_id += 1;
            let mut field = serde_json::json!({
                "id": last_col_id,
                "name": add.name,
                "required": add.required,
                "type": add.iceberg_type,
            });
            // Prefer explicit initial_default; fall back to write_default.
            let init_default = add.initial_default.or_else(|| add.write_default.clone());
            if let Some(ref d) = init_default {
                field["initial-default"] = d.clone();
            }
            if let Some(ref wd) = add.write_default {
                field["write-default"] = wd.clone();
            }
            if let Some(doc) = add.doc {
                field["doc"] = serde_json::Value::String(doc);
            }
            fields.push(field);
        }

        let new_schema_id = current_id + 1;
        let new_schema = serde_json::json!({
            "schema-id": new_schema_id,
            "type": "struct",
            "fields": fields,
        });

        meta.schemas.push(new_schema);
        meta.current_schema_id = new_schema_id;
        meta.last_column_id = last_col_id;
        meta.last_updated_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        for (k, v) in evolution.extra_properties {
            meta.properties.insert(k, v);
        }

        self.save_metadata(table, &meta).await?;
        tracing::info!(
            "ailake: schema evolved — table={}/{}, new schema-id={new_schema_id}, \
             last-column-id={last_col_id}",
            table.namespace,
            table.name
        );
        Ok(new_schema_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::new_snapshot_id;
    use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
    use ailake_store::LocalStore;
    use tempfile::TempDir;

    fn make_props() -> TableProperties {
        TableProperties {
            policy: VectorStoragePolicy {
                column_name: "embedding".to_string(),
                dim: 4,
                metric: VectorMetric::Cosine,
                precision: VectorPrecision::F16,
                pq: None,
                keep_raw_for_reranking: true,
                pre_normalize: false,
                hnsw_m: None,
                hnsw_ef_construction: None,
                ivf_residual: false,
                embedding_model: None,
                modality: None,
                partition_by: None,
                partition_value: None,
                partition_column_type: None,
                partition_fields: vec![],
            },
            extra: std::collections::HashMap::new(),
            format_version: 2,
            partition_column_type: None,
        }
    }

    #[tokio::test]
    async fn create_and_load_table() {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let catalog = HadoopCatalog::new(store, "warehouse");
        let table = TableIdent::new("default", "docs");

        catalog.create_table(&table, &make_props()).await.unwrap();
        let meta = catalog.load_table(&table).await.unwrap();
        assert_eq!(meta.format_version, 2); // make_props uses format_version=2
        assert!(meta.properties.contains_key("ailake.vector-column"));
    }

    #[tokio::test]
    async fn commit_snapshot_and_list_files() {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let catalog = HadoopCatalog::new(store, "warehouse");
        let table = TableIdent::new("default", "docs");

        catalog.create_table(&table, &make_props()).await.unwrap();

        let snap = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00001.parquet".to_string(),
                record_count: 10,
                file_size_bytes: 1024,
                centroid_b64: None,
                radius: Some(0.3),
                hnsw_offset: Some(512),
                hnsw_len: Some(256),
                vector_column: Some("embedding".to_string()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        let snap_id = catalog.commit_snapshot(&table, snap).await.unwrap();

        let files = catalog.list_files(&table, Some(snap_id)).await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "data/part-00001.parquet");
    }

    fn make_entry(path: &str) -> DataFileEntry {
        DataFileEntry {
            path: path.to_string(),
            record_count: 10,
            file_size_bytes: 1024,
            centroid_b64: None,
            radius: Some(0.3),
            hnsw_offset: Some(512),
            hnsw_len: Some(256),
            vector_column: Some("embedding".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: crate::provider::IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        }
    }

    /// Documents the `Replace`/`Overwrite` contract: the new manifest becomes the
    /// *complete* file list, with no inheritance from the previous snapshot (see the
    /// comment above `all_manifests` in `commit_snapshot`). Callers that commit a
    /// partial file list under `Replace` — instead of first fetching `list_files()`
    /// and carrying forward untouched entries — silently lose those files. This bit
    /// `CompactionExecutor::run()`/`run_deferred()` in `ailake-query`, which is why
    /// they now rebuild the full file list before committing. `TableWriter`
    /// (deferred index Ready-swap) and `MemoryDecayJob::run()` already followed this
    /// contract correctly.
    #[tokio::test]
    async fn replace_does_not_inherit_previous_manifest() {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let catalog = HadoopCatalog::new(store, "warehouse");
        let table = TableIdent::new("default", "docs");
        catalog.create_table(&table, &make_props()).await.unwrap();

        // Snapshot 1 (Append): two files present — "small.parquet" (compaction candidate)
        // and "big.parquet" (above target size, planner would never select it).
        let snap1 = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![
                make_entry("data/small.parquet"),
                make_entry("data/big.parquet"),
            ],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap1).await.unwrap();

        let files_before = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(
            files_before.len(),
            2,
            "sanity: both files visible after Append"
        );

        // Snapshot 2 (Replace): mirrors CompactionExecutor::run() — commits ONLY the
        // merged output of the compacted subset ("small.parquet" → "merged.parquet"),
        // never re-listing "big.parquet" because it was outside `to_compact`.
        let snap2 = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![make_entry("data/merged.parquet")],
            operation: crate::provider::SnapshotOperation::Replace,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap2).await.unwrap();

        let files_after = catalog.list_files(&table, None).await.unwrap();
        let paths_after: Vec<&str> = files_after.iter().map(|f| f.path.as_str()).collect();

        // By design, `Replace` does NOT inherit "big.parquet" from the previous
        // snapshot — the committing caller is responsible for including it if it
        // should remain visible. This is the sharp edge that bit compaction.rs.
        assert_eq!(
            paths_after,
            vec!["data/merged.parquet"],
            "Replace must reflect exactly the file list it was given, nothing inherited — \
             files_after={paths_after:?}"
        );
    }

    #[tokio::test]
    async fn v3_assigns_first_row_id_monotonically() {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let catalog = HadoopCatalog::new(store, "warehouse");
        let table = TableIdent::new("default", "v3docs");

        let mut props = make_props();
        props.format_version = 3;
        catalog.create_table(&table, &props).await.unwrap();

        // First commit — 10 rows → first_row_id=0, next_row_id advances to 10.
        let snap1 = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00001.parquet".to_string(),
                record_count: 10,
                file_size_bytes: 1024,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: Some("embedding".to_string()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None, // assigned by catalog at commit time
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap1).await.unwrap();

        // Second commit — 25 rows → first_row_id=10.
        let snap2_id = new_snapshot_id();
        let snap2 = NewSnapshot {
            snapshot_id: snap2_id,
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00002.parquet".to_string(),
                record_count: 25,
                file_size_bytes: 2048,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: Some("embedding".to_string()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap2).await.unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(files.len(), 2);
        // File 1: first_row_id=0
        let f1 = files
            .iter()
            .find(|f| f.path.ends_with("part-00001.parquet"))
            .unwrap();
        assert_eq!(f1.first_row_id, Some(0), "first file must start at row 0");
        // File 2: first_row_id=10 (after the 10 rows of file 1)
        let f2 = files
            .iter()
            .find(|f| f.path.ends_with("part-00002.parquet"))
            .unwrap();
        assert_eq!(
            f2.first_row_id,
            Some(10),
            "second file must start after first file's rows"
        );
    }

    #[tokio::test]
    async fn v2_does_not_assign_first_row_id() {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let catalog = HadoopCatalog::new(store, "warehouse");
        let table = TableIdent::new("default", "v2docs");

        catalog.create_table(&table, &make_props()).await.unwrap();

        let snap = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00001.parquet".to_string(),
                record_count: 10,
                file_size_bytes: 1024,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: Some("embedding".to_string()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap).await.unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(
            files[0].first_row_id, None,
            "V2 tables must not have first_row_id"
        );
    }

    /// Compaction pre-sets `first_row_id` on the merged file.
    /// `commit_snapshot` must not overwrite it and must not advance `next_row_id`.
    #[tokio::test]
    async fn compaction_preserves_first_row_id_and_next_row_id_does_not_balloon() {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let catalog = HadoopCatalog::new(Arc::clone(&store) as Arc<dyn Store>, "warehouse");
        let table = TableIdent::new("default", "compact_rowid");

        let mut props = make_props();
        props.format_version = 3;
        catalog.create_table(&table, &props).await.unwrap();

        // Write two files: 10 rows + 25 rows → next_row_id advances to 35.
        let snap1 = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00001.parquet".to_string(),
                record_count: 10,
                file_size_bytes: 1024,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: Some("embedding".into()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap1).await.unwrap();

        let snap2 = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00002.parquet".to_string(),
                record_count: 25,
                file_size_bytes: 2048,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: Some("embedding".into()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap2).await.unwrap();

        // Simulate compaction: merged file pre-sets first_row_id=0 (min of source files).
        // next_row_id is 35; compaction must NOT push it to 70.
        let snap_compact = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-merged.parquet".to_string(),
                record_count: 35,
                file_size_bytes: 3072,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: Some("embedding".into()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: Some(0), // pre-set by compaction — must be respected
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Replace,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap_compact).await.unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(files.len(), 1);
        // first_row_id must remain 0 — commit_snapshot must not overwrite it.
        assert_eq!(
            files[0].first_row_id,
            Some(0),
            "compaction first_row_id=0 must survive commit_snapshot"
        );

        // next fresh write must start at 35 (not 70).
        let snap_new = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00004.parquet".to_string(),
                record_count: 5,
                file_size_bytes: 512,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: Some("embedding".into()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None, // fresh write
                column_stats: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap_new).await.unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        let new_file = files
            .iter()
            .find(|f| f.path.ends_with("part-00004.parquet"))
            .unwrap();
        assert_eq!(
            new_file.first_row_id,
            Some(35),
            "fresh write after compaction must start at 35, not 70"
        );
    }
}
