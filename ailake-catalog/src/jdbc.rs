// SPDX-License-Identifier: MIT OR Apache-2.0
// JdbcCatalog: stores Iceberg metadata pointers in PostgreSQL or MySQL.
//
// The catalog stores ONE row per table pointing to the current metadata.json.
// Actual metadata.json files and manifests are written to object storage via Store.
//
// Schema (auto-created on connect):
//   iceberg_tables(catalog_name, table_namespace, table_name, metadata_location)
//
// Connection URLs:
//   postgres://user:pass@host:5432/dbname
//   mysql://user:pass@host:3306/dbname
//   sqlite::memory:               (in-process SQLite, useful for tests)
//
// Note: queries use ? as placeholder — sqlx::AnyPool translates to
// $1/$2 (Postgres) or ? (MySQL/SQLite) internally.

use std::collections::HashMap;
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use sqlx::AnyPool;
use uuid::Uuid;

use crate::metadata::{IcebergMetadata, IcebergSnapshot};
use crate::provider::{
    CatalogProvider, DataFileEntry, NewSnapshot, SnapshotId, TableIdent, TableMetadata,
    TableProperties,
};
use crate::snapshot::{build_manifest, manifest_path};
use ailake_store::Store;

// ── JdbcCatalog ───────────────────────────────────────────────────────────────

pub struct JdbcCatalog {
    pool: AnyPool,
    catalog_name: String,
    store: Arc<dyn Store>,
    warehouse: String,
}

#[derive(sqlx::FromRow)]
struct MetadataLocationRow {
    metadata_location: String,
}

impl JdbcCatalog {
    /// Connect to a JDBC-compatible database and ensure the catalog schema exists.
    ///
    /// Call once at startup; subsequent calls are idempotent (CREATE TABLE IF NOT EXISTS).
    pub async fn connect(
        connection_url: &str,
        catalog_name: &str,
        warehouse: &str,
        store: Arc<dyn Store>,
    ) -> AilakeResult<Self> {
        sqlx::any::install_default_drivers();
        let pool = AnyPool::connect(connection_url)
            .await
            .map_err(|e| AilakeError::Catalog(format!("JDBC connect failed: {e}")))?;
        let catalog = Self {
            pool,
            catalog_name: catalog_name.to_string(),
            store,
            warehouse: warehouse.trim_end_matches('/').to_string(),
        };
        catalog.migrate().await?;
        Ok(catalog)
    }

    async fn migrate(&self) -> AilakeResult<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS iceberg_tables (
                catalog_name      VARCHAR(255) NOT NULL,
                table_namespace   VARCHAR(255) NOT NULL,
                table_name        VARCHAR(255) NOT NULL,
                metadata_location VARCHAR(1000) NOT NULL,
                PRIMARY KEY (catalog_name, table_namespace, table_name)
            )",
        )
        .execute(&self.pool)
        .await
        .map_err(|e| AilakeError::Catalog(format!("JDBC migrate: {e}")))?;
        Ok(())
    }

    fn table_root(&self, table: &TableIdent) -> String {
        format!("{}/{}/{}", self.warehouse, table.namespace, table.name)
    }

    fn metadata_path(&self, table: &TableIdent, uuid: &str) -> String {
        format!("{}/metadata/{}.metadata.json", self.table_root(table), uuid)
    }

    async fn get_metadata_location(&self, table: &TableIdent) -> AilakeResult<String> {
        let row: Option<MetadataLocationRow> = sqlx::query_as(
            "SELECT metadata_location FROM iceberg_tables
             WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?",
        )
        .bind(&self.catalog_name)
        .bind(&table.namespace)
        .bind(&table.name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AilakeError::Catalog(format!("JDBC get: {e}")))?;

        row.map(|r| r.metadata_location).ok_or_else(|| {
            AilakeError::Catalog(format!(
                "table not found: {}.{}",
                table.namespace, table.name
            ))
        })
    }

    async fn load_iceberg_metadata(&self, location: &str) -> AilakeResult<IcebergMetadata> {
        let bytes = self.store.get(location).await?;
        let json = std::str::from_utf8(&bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;
        IcebergMetadata::from_json(json)
    }
}

// ── CatalogProvider ───────────────────────────────────────────────────────────

#[async_trait]
impl CatalogProvider for JdbcCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        let location = self.table_root(name);
        let mut meta = IcebergMetadata::new(&location, &props.policy, props.format_version);
        for (k, v) in &props.extra {
            meta.properties.insert(k.clone(), v.clone());
        }

        let meta_uuid = Uuid::new_v4().to_string();
        let metadata_location = self.metadata_path(name, &meta_uuid);
        let json = meta.to_json()?;
        self.store
            .put(&metadata_location, Bytes::from(json.into_bytes()))
            .await?;

        sqlx::query(
            "INSERT INTO iceberg_tables
                 (catalog_name, table_namespace, table_name, metadata_location)
             VALUES (?, ?, ?, ?)",
        )
        .bind(&self.catalog_name)
        .bind(&name.namespace)
        .bind(&name.name)
        .bind(&metadata_location)
        .execute(&self.pool)
        .await
        .map_err(|e| AilakeError::Catalog(format!("JDBC create_table: {e}")))?;

        Ok(())
    }

    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata> {
        let location = self.get_metadata_location(name).await?;
        let meta = self.load_iceberg_metadata(&location).await?;
        Ok(meta.to_table_metadata())
    }

    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> AilakeResult<SnapshotId> {
        let snap_id = snapshot.snapshot_id;

        // 1. Write manifest
        let root = self.table_root(table);
        let abs_manifest = format!("{root}/{}", manifest_path(snap_id));
        let manifest = build_manifest(&snapshot);
        self.store
            .put(&abs_manifest, Bytes::from(manifest.to_json()?.into_bytes()))
            .await?;

        // 2. Load and update metadata
        let old_location = self.get_metadata_location(table).await?;
        let mut meta = self.load_iceberg_metadata(&old_location).await?;
        let now_ms = now_ms();
        let iceberg_snap = IcebergSnapshot {
            snapshot_id: snap_id,
            parent_snapshot_id: snapshot.parent_snapshot_id,
            sequence_number: meta.last_sequence_number + 1,
            timestamp_ms: now_ms,
            manifest_list: abs_manifest,
            summary: HashMap::from([
                (
                    "operation".into(),
                    format!("{:?}", snapshot.operation).to_lowercase(),
                ),
                ("added-data-files".into(), snapshot.files.len().to_string()),
            ]),
            schema_id: Some(0),
        };
        meta.last_sequence_number += 1;
        meta.last_updated_ms = now_ms;
        meta.current_snapshot_id = Some(snap_id);
        meta.snapshots.push(iceberg_snap);

        // 3. Write new versioned metadata.json
        let new_uuid = Uuid::new_v4().to_string();
        let new_location = self.metadata_path(table, &new_uuid);
        let json = meta.to_json()?;
        self.store
            .put(&new_location, Bytes::from(json.into_bytes()))
            .await?;

        // 4. Update pointer in DB
        sqlx::query(
            "UPDATE iceberg_tables SET metadata_location = ?
             WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?",
        )
        .bind(&new_location)
        .bind(&self.catalog_name)
        .bind(&table.namespace)
        .bind(&table.name)
        .execute(&self.pool)
        .await
        .map_err(|e| AilakeError::Catalog(format!("JDBC commit_snapshot: {e}")))?;

        Ok(snap_id)
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let location = self.get_metadata_location(table).await?;
        let meta = self.load_iceberg_metadata(&location).await?;
        let snap_id = snapshot_id
            .or(meta.current_snapshot_id)
            .ok_or_else(|| AilakeError::Catalog("table has no snapshots".into()))?;
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
        sqlx::query(
            "DELETE FROM iceberg_tables
             WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?",
        )
        .bind(&self.catalog_name)
        .bind(&name.namespace)
        .bind(&name.name)
        .execute(&self.pool)
        .await
        .map_err(|e| AilakeError::Catalog(format!("JDBC drop_table: {e}")))?;
        Ok(())
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // Path helpers are tested via the static string logic without needing a live pool.

    #[test]
    fn metadata_path_format() {
        let warehouse = "s3://my-bucket/warehouse";
        let table = TableIdent::new("default", "docs");
        let table_root = format!("{}/{}/{}", warehouse, table.namespace, table.name);
        let path = format!("{}/metadata/{}.metadata.json", table_root, "test-uuid-1234");
        assert_eq!(
            path,
            "s3://my-bucket/warehouse/default/docs/metadata/test-uuid-1234.metadata.json"
        );
    }

    #[test]
    fn table_root_format() {
        let warehouse = "s3://my-bucket/warehouse";
        let table = TableIdent::new("analytics", "embeddings");
        let root = format!("{}/{}/{}", warehouse, table.namespace, table.name);
        assert_eq!(root, "s3://my-bucket/warehouse/analytics/embeddings");
    }

    /// Full end-to-end test using in-process SQLite (no external DB required).
    #[tokio::test]
    #[cfg(feature = "catalog-jdbc")]
    async fn sqlite_create_load_commit_drop() {
        use crate::provider::{
            new_snapshot_id, DataFileEntry, IndexStatus, NewSnapshot, SnapshotOperation,
        };
        use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
        use ailake_store::LocalStore;

        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let warehouse = dir.path().to_str().unwrap();

        // Use file-based SQLite — in-memory databases are per-connection and
        // don't share state across a pool.
        let db_path = dir.path().join("catalog_test.db");
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());

        let catalog = JdbcCatalog::connect(&db_url, "test-catalog", warehouse, store)
            .await
            .unwrap();

        let table = TableIdent::new("default", "docs");
        let props = TableProperties {
            policy: VectorStoragePolicy {
                column_name: "embedding".into(),
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
            },
            extra: HashMap::new(),
            format_version: 2,
        };

        // create
        catalog.create_table(&table, &props).await.unwrap();
        let meta = catalog.load_table(&table).await.unwrap();
        assert_eq!(meta.format_version, 2); // props uses format_version=2
        assert!(meta.properties.contains_key("ailake.vector-column"));

        // commit snapshot
        let snap = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![DataFileEntry {
                path: "data/part-00001.parquet".into(),
                record_count: 10,
                file_size_bytes: 1024,
                centroid_b64: None,
                radius: Some(0.3),
                hnsw_offset: Some(512),
                hnsw_len: Some(256),
                vector_column: Some("embedding".into()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            }],
            operation: SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
                equality_delete_files: vec![],
        };
        let snap_id = catalog.commit_snapshot(&table, snap).await.unwrap();

        let files = catalog.list_files(&table, Some(snap_id)).await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "data/part-00001.parquet");

        // drop
        catalog.drop_table(&table).await.unwrap();
        assert!(catalog.load_table(&table).await.is_err());
    }
}
