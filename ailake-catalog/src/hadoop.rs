// SPDX-License-Identifier: MIT OR Apache-2.0
// HadoopCatalog: stores metadata.json on the local filesystem / any Store backend.
// Table layout: {warehouse}/{namespace}/{table}/metadata/vN.metadata.json

use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;

use crate::avro_manifest::{
    read_manifest_file, read_manifest_list, write_manifest_file, write_manifest_list_multi,
};
use crate::metadata::{IcebergMetadata, IcebergSnapshot};
use crate::provider::{
    CatalogProvider, DataFileEntry, NewSnapshot, SnapshotId, TableIdent, TableMetadata,
    TableProperties,
};
use ailake_store::Store;
use bytes::Bytes;

pub struct HadoopCatalog {
    store: Arc<dyn Store>,
    warehouse: String,
}

impl HadoopCatalog {
    pub fn new(store: Arc<dyn Store>, warehouse: &str) -> Self {
        Self {
            store,
            warehouse: warehouse.trim_end_matches('/').to_string(),
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
        let location = self.table_root(name);
        let mut meta = IcebergMetadata::new(&location, &props.policy);
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
        let snap_id = snapshot.snapshot_id;
        let mut meta = self.load_raw_metadata(table).await?;
        let seq = meta.last_sequence_number + 1;
        let table_root = self.table_root(table);

        // Minimal Iceberg schema JSON for this table (empty fields; readers get column info from Parquet)
        let table_schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec_json = r#"[{"spec-id":0,"fields":[]}]"#;

        // Write Avro manifest file for the new data files.
        // Iceberg spec requires absolute file paths in manifests. Prefix relative paths
        // with the warehouse root only when the warehouse is itself an absolute path
        // (starts with '/' or contains a URI scheme). Relative warehouse names (e.g. in
        // unit tests) are left as-is so the store can resolve them normally.
        let warehouse_prefix: Option<&str> =
            if self.warehouse.starts_with('/') || self.warehouse.contains("://") {
                Some(&self.warehouse)
            } else {
                None
            };
        let abs_files: Vec<DataFileEntry> = snapshot
            .files
            .iter()
            .map(|f| {
                let path = if f.path.starts_with('/') || f.path.contains("://") {
                    f.path.clone()
                } else if let Some(prefix) = warehouse_prefix {
                    format!("{}/{}", prefix, f.path)
                } else {
                    f.path.clone()
                };
                DataFileEntry { path, ..f.clone() }
            })
            .collect();
        let added_rows: i64 = abs_files.iter().map(|f| f.record_count as i64).sum();
        let manifest_file_path = format!("{}/metadata/{}-m0.avro", table_root, snap_id);
        let manifest_bytes = write_manifest_file(
            &abs_files,
            snap_id,
            seq,
            table_schema_json,
            partition_spec_json,
        );
        let manifest_len = manifest_bytes.len();
        self.store.put(&manifest_file_path, manifest_bytes).await?;

        // Collect manifest paths from the previous snapshot (if any) for the manifest list.
        // Replace/Overwrite: new manifest IS the complete state — don't inherit old manifests.
        // Append/Delete: inherit previous manifests so old files remain visible.
        let mut all_manifests: Vec<(String, i64)> = Vec::new();
        if matches!(
            snapshot.operation,
            crate::provider::SnapshotOperation::Append | crate::provider::SnapshotOperation::Delete
        ) {
            if let Some(prev_snap) = meta.snapshots.last() {
                if let Ok(ml_bytes) = self.store.get(&prev_snap.manifest_list).await {
                    if let Ok(prev_manifests) = read_manifest_list(&ml_bytes) {
                        for prev_path in prev_manifests {
                            let len = self.store.file_size(&prev_path).await.unwrap_or(0) as i64;
                            all_manifests.push((prev_path, len));
                        }
                    }
                }
            }
        }
        all_manifests.push((manifest_file_path.clone(), manifest_len as i64));

        // Write Avro manifest list for this snapshot
        let manifest_list_path = format!("{}/metadata/snap-{}-1.avro", table_root, snap_id);
        // Build manifest list from all manifests
        let ml_bytes = write_manifest_list_multi(&all_manifests, snap_id, seq, added_rows);
        self.store.put(&manifest_list_path, ml_bytes).await?;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let iceberg_snap = IcebergSnapshot {
            snapshot_id: snap_id,
            parent_snapshot_id: snapshot.parent_snapshot_id,
            sequence_number: seq,
            timestamp_ms: now_ms,
            manifest_list: manifest_list_path,
            summary: std::collections::HashMap::from([
                (
                    "operation".to_string(),
                    format!("{:?}", snapshot.operation).to_lowercase(),
                ),
                (
                    "added-data-files".to_string(),
                    snapshot.files.len().to_string(),
                ),
            ]),
            schema_id: Some(0),
        };
        meta.last_sequence_number = seq;
        meta.last_updated_ms = now_ms;
        meta.current_snapshot_id = Some(snap_id);
        meta.snapshots.push(iceberg_snap);

        if let Some(schema_update) = snapshot.iceberg_schema {
            if let Some(schema) = meta.schemas.first_mut() {
                schema["fields"] = serde_json::Value::Array(schema_update.fields);
            }
            meta.last_column_id = schema_update.last_column_id;
            meta.properties.insert(
                "schema.name-mapping.default".to_string(),
                schema_update.name_mapping_json,
            );
        }

        // Merge secondary-column properties (ailake.dim-<col>, ailake.metric-<col>).
        for (k, v) in snapshot.extra_properties {
            meta.properties.insert(k, v);
        }

        self.save_metadata(table, &meta).await?;
        Ok(snap_id)
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let meta = self.load_raw_metadata(table).await?;
        let snap_id = match snapshot_id.or(meta.current_snapshot_id) {
            Some(id) => id,
            None => return Ok(vec![]), // new table — no snapshots yet, no committed files
        };

        let snap = meta
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snap_id)
            .ok_or_else(|| AilakeError::Catalog(format!("snapshot {snap_id} not found")))?;

        // Read Avro manifest list → manifest file paths
        let ml_bytes = self.store.get(&snap.manifest_list).await?;
        let manifest_paths =
            read_manifest_list(&ml_bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;

        // Read each manifest file → data file entries (with AI-Lake metadata from key_metadata)
        let mut entries: Vec<DataFileEntry> = Vec::new();
        for mpath in manifest_paths {
            let mf_bytes = self.store.get(&mpath).await?;
            let file_entries =
                read_manifest_file(&mf_bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;
            entries.extend(file_entries);
        }
        Ok(entries)
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
            },
            extra: std::collections::HashMap::new(),
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
        assert_eq!(meta.format_version, 2);
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
                batch_id: None,
                embedding_model: None,
            }],
            operation: crate::provider::SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
        };
        let snap_id = catalog.commit_snapshot(&table, snap).await.unwrap();

        let files = catalog.list_files(&table, Some(snap_id)).await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "data/part-00001.parquet");
    }
}
