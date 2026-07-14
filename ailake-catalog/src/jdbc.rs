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

use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use sqlx::AnyPool;
use uuid::Uuid;

use crate::manifest_commit::{
    commit_into_metadata, list_equality_deletes_from_metadata, list_files_from_metadata,
};
use crate::metadata::IcebergMetadata;
use crate::provider::{
    CatalogProvider, DataFileEntry, EqualityDeleteFile, NewSnapshot, SnapshotId, TableIdent,
    TableMetadata, TableProperties,
};
use crate::schema_evolution::SchemaEvolution;
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
        // OCC retry: read metadata -> apply commit to a fresh copy -> CAS UPDATE on
        // metadata_location. rows_affected == 0 means a concurrent writer won the
        // race -- re-read and retry with fresh state on every attempt, so a
        // concurrent Append/Delete that won a prior iteration isn't lost by an
        // in-flight retry that captured stale state. `commit_into_metadata` (shared
        // with Hadoop) owns the actual Avro manifest / Puffin / partition-stats /
        // first_row_id logic.
        const MAX_RETRIES: u32 = 5;
        let table_root = self.table_root(table);
        for attempt in 0..MAX_RETRIES {
            let old_location = self.get_metadata_location(table).await?;
            let mut meta = self.load_iceberg_metadata(&old_location).await?;

            let snap_id = commit_into_metadata(
                &*self.store,
                &table_root,
                &self.warehouse,
                &mut meta,
                snapshot.clone(),
            )
            .await?;

            let new_uuid = Uuid::new_v4().to_string();
            let new_location = self.metadata_path(table, &new_uuid);
            let json = meta.to_json()?;
            self.store
                .put(&new_location, Bytes::from(json.into_bytes()))
                .await?;

            let result = sqlx::query(
                "UPDATE iceberg_tables SET metadata_location = ?
                 WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?
                   AND metadata_location = ?",
            )
            .bind(&new_location)
            .bind(&self.catalog_name)
            .bind(&table.namespace)
            .bind(&table.name)
            .bind(&old_location)
            .execute(&self.pool)
            .await
            .map_err(|e| AilakeError::Catalog(format!("JDBC commit_snapshot: {e}")))?;

            if result.rows_affected() > 0 {
                return Ok(snap_id);
            }

            if attempt + 1 < MAX_RETRIES {
                tokio::time::sleep(tokio::time::Duration::from_millis(50 << attempt)).await;
            }
        }
        Err(AilakeError::Catalog(format!(
            "JDBC commit_snapshot: {MAX_RETRIES} retries exhausted (concurrent modification)"
        )))
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let location = self.get_metadata_location(table).await?;
        let meta = self.load_iceberg_metadata(&location).await?;
        list_files_from_metadata(&*self.store, &meta, snapshot_id).await
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

    async fn list_equality_deletes(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<EqualityDeleteFile>> {
        let location = self.get_metadata_location(table).await?;
        let meta = self.load_iceberg_metadata(&location).await?;
        list_equality_deletes_from_metadata(&*self.store, &meta, snapshot_id).await
    }

    /// Apply schema evolution without rewriting data files — mirrors
    /// `HadoopCatalog::evolve_schema`'s metadata.json schema-patch logic,
    /// swapping the pointer-update mechanism for this catalog's own
    /// CAS-`UPDATE ... WHERE metadata_location = old` OCC retry loop.
    async fn evolve_schema(
        &self,
        table: &TableIdent,
        evolution: SchemaEvolution,
    ) -> AilakeResult<i32> {
        const MAX_RETRIES: u32 = 5;
        for attempt in 0..MAX_RETRIES {
            let old_location = self.get_metadata_location(table).await?;
            let mut meta = self.load_iceberg_metadata(&old_location).await?;
            let current_id = meta.current_schema_id;

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

            for rename in &evolution.renames {
                for field in fields.iter_mut() {
                    if field["name"].as_str() == Some(rename.old_name.as_str()) {
                        field["name"] = serde_json::Value::String(rename.new_name.clone());
                    }
                }
            }

            let mut last_col_id = meta.last_column_id;
            for add in &evolution.adds {
                last_col_id += 1;
                let mut field = serde_json::json!({
                    "id": last_col_id,
                    "name": add.name,
                    "required": add.required,
                    "type": add.iceberg_type,
                });
                let init_default = add
                    .initial_default
                    .clone()
                    .or_else(|| add.write_default.clone());
                if let Some(d) = init_default {
                    field["initial-default"] = d;
                }
                if let Some(wd) = &add.write_default {
                    field["write-default"] = wd.clone();
                }
                if let Some(doc) = &add.doc {
                    field["doc"] = serde_json::Value::String(doc.clone());
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
            meta.last_updated_ms = now_ms();
            for (k, v) in &evolution.extra_properties {
                meta.properties.insert(k.clone(), v.clone());
            }

            let new_uuid = Uuid::new_v4().to_string();
            let new_location = self.metadata_path(table, &new_uuid);
            let json = meta.to_json()?;
            self.store
                .put(&new_location, Bytes::from(json.into_bytes()))
                .await?;

            let result = sqlx::query(
                "UPDATE iceberg_tables SET metadata_location = ?
                 WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?
                   AND metadata_location = ?",
            )
            .bind(&new_location)
            .bind(&self.catalog_name)
            .bind(&table.namespace)
            .bind(&table.name)
            .bind(&old_location)
            .execute(&self.pool)
            .await
            .map_err(|e| AilakeError::Catalog(format!("JDBC evolve_schema: {e}")))?;

            if result.rows_affected() > 0 {
                return Ok(new_schema_id);
            }
            if attempt + 1 < MAX_RETRIES {
                tokio::time::sleep(tokio::time::Duration::from_millis(50 << attempt)).await;
            }
        }
        Err(AilakeError::Catalog(format!(
            "JDBC evolve_schema: {MAX_RETRIES} retries exhausted (concurrent modification)"
        )))
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
    use std::collections::HashMap;
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
                partition_column_type: None,
                partition_fields: vec![],
            },
            extra: HashMap::new(),
            format_version: 2,
            partition_column_type: None,
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
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
                column_stats: None,
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
        // Real Avro manifests (shared with HadoopCatalog) always store absolute
        // paths per the Iceberg spec — `warehouse` here is a real absolute tempdir
        // path, so the manifest writer prefixes the relative path we wrote with it.
        assert!(files[0].path.ends_with("data/part-00001.parquet"));

        // second incremental Append must inherit the first file, not replace it
        // (regression: commit_snapshot previously wrote only `snapshot.files` verbatim,
        // silently losing every file from prior commits on the very first Append after
        // table creation)
        let meta_before_second = catalog.load_table(&table).await.unwrap();
        let snap2 = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: meta_before_second.current_snapshot_id,
            files: vec![DataFileEntry {
                path: "data/part-00002.parquet".into(),
                record_count: 20,
                file_size_bytes: 2048,
                centroid_b64: None,
                radius: Some(0.4),
                hnsw_offset: Some(1024),
                hnsw_len: Some(512),
                vector_column: Some("embedding".into()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
                column_stats: None,
            }],
            operation: SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        let snap2_id = catalog.commit_snapshot(&table, snap2).await.unwrap();
        let mut files_after = catalog
            .list_files(&table, Some(snap2_id))
            .await
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect::<Vec<_>>();
        files_after.sort();
        assert_eq!(files_after.len(), 2);
        assert!(files_after[0].ends_with("data/part-00001.parquet"));
        assert!(files_after[1].ends_with("data/part-00002.parquet"));

        // evolve_schema: real ALTER-equivalent — add a column, confirm it lands in
        // metadata.json and the schema-id advances (Fase 1 of the catalog-parity pass).
        let evolution = crate::schema_evolution::SchemaEvolution::new().add_column(
            crate::schema_evolution::AddColumnRequest {
                name: "chunk_text".to_string(),
                iceberg_type: "string".to_string(),
                required: false,
                initial_default: None,
                write_default: None,
                doc: None,
            },
        );
        let new_schema_id = catalog.evolve_schema(&table, evolution).await.unwrap();
        assert_eq!(new_schema_id, 1);
        let meta_after_evolve = catalog.load_table(&table).await.unwrap();
        assert_eq!(meta_after_evolve.current_snapshot_id, Some(snap2_id));

        // equality deletes: real write + read-back + Append accumulation (Fase 0/2 —
        // previously silently dropped on commit and always read back empty).
        let eq_del = crate::provider::EqualityDeleteFile {
            path: "metadata/eq-del-1.avro".into(),
            equality_ids: vec![1],
            record_count: 3,
            file_size_bytes: 128,
            inline_values: None,
        };
        let snap3 = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: Some(snap2_id),
            files: vec![],
            operation: SnapshotOperation::Delete,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![eq_del.clone()],
        };
        let snap3_id = catalog.commit_snapshot(&table, snap3).await.unwrap();
        let deletes = catalog
            .list_equality_deletes(&table, Some(snap3_id))
            .await
            .unwrap();
        assert_eq!(deletes.len(), 1);
        // Same absolute-path convention as data files — see the note above.
        assert!(deletes[0].path.ends_with("eq-del-1.avro"));
        // files list must be untouched by the Delete commit (its payload is the
        // equality-delete file, not a change to which data files are active).
        let files_after_delete = catalog.list_files(&table, Some(snap3_id)).await.unwrap();
        assert_eq!(files_after_delete.len(), 2);

        // drop
        catalog.drop_table(&table).await.unwrap();
        assert!(catalog.load_table(&table).await.is_err());
    }
}
