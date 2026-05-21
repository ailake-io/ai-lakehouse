// HadoopCatalog: stores metadata.json on the local filesystem / any Store backend.
// Table layout: {warehouse}/{namespace}.db/{table}/metadata/vN.metadata.json

use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;

use crate::metadata::{IcebergMetadata, IcebergSnapshot};
use crate::provider::{
    CatalogProvider, DataFileEntry, NewSnapshot, SnapshotId, TableIdent, TableMetadata,
    TableProperties,
};
use crate::snapshot::{build_manifest, manifest_path};
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
        format!("{}/{}.db/{}", self.warehouse, table.namespace, table.name)
    }

    fn current_metadata_path(&self, table: &TableIdent) -> String {
        format!("{}/metadata/current.json", self.table_root(table))
    }

    async fn load_raw_metadata(&self, table: &TableIdent) -> AilakeResult<IcebergMetadata> {
        let path = self.current_metadata_path(table);
        let bytes = self.store.get(&path).await?;
        let json = std::str::from_utf8(&bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;
        IcebergMetadata::from_json(json)
    }

    async fn save_metadata(&self, table: &TableIdent, meta: &IcebergMetadata) -> AilakeResult<()> {
        let json = meta.to_json()?;
        let path = self.current_metadata_path(table);
        self.store.put(&path, Bytes::from(json.into_bytes())).await
    }
}

#[async_trait]
impl CatalogProvider for HadoopCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        let location = format!("{}/{}.db/{}", self.warehouse, name.namespace, name.name);
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

        // Write manifest
        let manifest = build_manifest(&snapshot);
        let manifest_path = format!("{}/{}", self.table_root(table), manifest_path(snap_id));
        let manifest_json = manifest.to_json()?;
        self.store
            .put(&manifest_path, Bytes::from(manifest_json.into_bytes()))
            .await?;

        // Update metadata
        let mut meta = self.load_raw_metadata(table).await?;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let iceberg_snap = IcebergSnapshot {
            snapshot_id: snap_id,
            parent_snapshot_id: snapshot.parent_snapshot_id,
            sequence_number: meta.last_sequence_number + 1,
            timestamp_ms: now_ms,
            manifest_list: manifest_path,
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
        meta.last_sequence_number += 1;
        meta.last_updated_ms = now_ms;
        meta.current_snapshot_id = Some(snap_id);
        meta.snapshots.push(iceberg_snap);

        self.save_metadata(table, &meta).await?;
        Ok(snap_id)
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let meta = self.load_raw_metadata(table).await?;
        let snap_id = snapshot_id
            .or(meta.current_snapshot_id)
            .ok_or_else(|| AilakeError::Catalog("table has no snapshots".to_string()))?;

        let snap = meta
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snap_id)
            .ok_or_else(|| AilakeError::Catalog(format!("snapshot {snap_id} not found")))?;

        let manifest_bytes = self.store.get(&snap.manifest_list).await?;
        let manifest_json = std::str::from_utf8(&manifest_bytes)
            .map_err(|e| AilakeError::Catalog(e.to_string()))?;
        let manifest = crate::snapshot::Manifest::from_json(manifest_json)?;
        Ok(manifest.files)
    }

    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()> {
        // Phase 1: just remove the metadata file
        let path = self.current_metadata_path(name);
        if self.store.exists(&path).await? {
            self.store.delete(&path).await?;
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
                keep_raw_for_reranking: false,
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
            }],
            operation: crate::provider::SnapshotOperation::Append,
        };
        let snap_id = catalog.commit_snapshot(&table, snap).await.unwrap();

        let files = catalog.list_files(&table, Some(snap_id)).await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "data/part-00001.parquet");
    }
}
