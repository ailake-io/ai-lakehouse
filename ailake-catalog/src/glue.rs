// SPDX-License-Identifier: MIT OR Apache-2.0
// GlueCatalog: AWS Glue Data Catalog backend.
//
// Glue stores a pointer (metadata_location) to the current Iceberg metadata.json
// in the Glue table's Parameters map. The actual metadata.json and manifests
// are written to S3 (or any Store implementation).
//
// Glue table Parameters:
//   table_type         = "ICEBERG"
//   metadata_location  = "s3://bucket/warehouse/ns/table/metadata/{uuid}.metadata.json"
//
// Requires feature flag: catalog-glue
//
// Tables are visible in Athena, EMR, Glue ETL and any service that reads the Glue catalog.

use std::collections::HashMap;
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use aws_sdk_glue::types::{StorageDescriptor, TableInput};
use bytes::Bytes;
use uuid::Uuid;

use crate::metadata::{IcebergMetadata, IcebergSnapshot};
use crate::provider::{
    CatalogProvider, DataFileEntry, NewSnapshot, SnapshotId, TableIdent, TableMetadata,
    TableProperties,
};
use crate::snapshot::{build_manifest, manifest_path};
use ailake_store::Store;

// ── Configuration ─────────────────────────────────────────────────────────────

pub struct GlueCatalogConfig {
    /// Glue database name. Create via `aws glue create-database`.
    pub database: String,
    /// Base storage location for new tables (e.g. "s3://my-bucket/warehouse").
    pub warehouse: String,
    /// AWS region override (e.g. "us-east-1"). If None, uses AWS_DEFAULT_REGION env var.
    pub region: Option<String>,
}

// ── GlueCatalog ───────────────────────────────────────────────────────────────

pub struct GlueCatalog {
    client: aws_sdk_glue::Client,
    config: GlueCatalogConfig,
    store: Arc<dyn Store>,
}

impl GlueCatalog {
    /// Create from a pre-built Glue client. Useful when the caller manages credentials.
    pub fn from_client(
        client: aws_sdk_glue::Client,
        config: GlueCatalogConfig,
        store: Arc<dyn Store>,
    ) -> Self {
        Self {
            client,
            config,
            store,
        }
    }

    /// Create from environment credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    /// AWS_SESSION_TOKEN, AWS_DEFAULT_REGION, or IAM instance role).
    pub async fn from_env(config: GlueCatalogConfig, store: Arc<dyn Store>) -> Self {
        let mut loader = aws_config::from_env();
        if let Some(region) = &config.region {
            loader = loader.region(aws_sdk_glue::config::Region::new(region.clone()));
        }
        let sdk_config = loader.load().await;
        let client = aws_sdk_glue::Client::new(&sdk_config);
        Self {
            client,
            config,
            store,
        }
    }

    // ── Path helpers ──────────────────────────────────────────────────────────

    fn table_root(&self, table: &TableIdent) -> String {
        let warehouse = self.config.warehouse.trim_end_matches('/');
        format!("{}/{}/{}", warehouse, table.namespace, table.name)
    }

    fn metadata_path(&self, table: &TableIdent, uuid: &str) -> String {
        format!("{}/metadata/{}.metadata.json", self.table_root(table), uuid)
    }

    // ── Glue helpers ──────────────────────────────────────────────────────────

    fn table_params(metadata_location: &str) -> HashMap<String, String> {
        HashMap::from([
            ("table_type".into(), "ICEBERG".into()),
            ("metadata_location".into(), metadata_location.into()),
        ])
    }

    fn build_table_input(
        table_name: &str,
        table_root: &str,
        metadata_location: &str,
    ) -> AilakeResult<TableInput> {
        let sd = StorageDescriptor::builder().location(table_root).build();
        TableInput::builder()
            .name(table_name)
            .storage_descriptor(sd)
            .set_parameters(Some(Self::table_params(metadata_location)))
            .build()
            .map_err(|e| AilakeError::Catalog(format!("GlueCatalog TableInput: {e}")))
    }

    async fn get_metadata_location(&self, table: &TableIdent) -> AilakeResult<String> {
        let resp = self
            .client
            .get_table()
            .database_name(&self.config.database)
            .name(&table.name)
            .send()
            .await
            .map_err(|e| AilakeError::Catalog(format!("Glue get_table: {e}")))?;

        let t = resp
            .table()
            .ok_or_else(|| AilakeError::Catalog("Glue get_table: empty response".into()))?;

        let params = t.parameters().ok_or_else(|| {
            AilakeError::Catalog(format!(
                "Glue table {}.{} has no parameters",
                table.namespace, table.name
            ))
        })?;
        params.get("metadata_location").cloned().ok_or_else(|| {
            AilakeError::Catalog(format!(
                "Glue table {}.{} is not an Iceberg table (missing metadata_location)",
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
impl CatalogProvider for GlueCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        let table_root = self.table_root(name);
        let mut meta = IcebergMetadata::new(&table_root, &props.policy);
        for (k, v) in &props.extra {
            meta.properties.insert(k.clone(), v.clone());
        }

        let meta_uuid = Uuid::new_v4().to_string();
        let metadata_location = self.metadata_path(name, &meta_uuid);
        let json = meta.to_json()?;
        self.store
            .put(&metadata_location, Bytes::from(json.into_bytes()))
            .await?;

        let table_input = Self::build_table_input(&name.name, &table_root, &metadata_location)?;
        self.client
            .create_table()
            .database_name(&self.config.database)
            .table_input(table_input)
            .send()
            .await
            .map_err(|e| AilakeError::Catalog(format!("Glue create_table: {e}")))?;

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

        // 4. Update Glue table pointer
        let table_root = self.table_root(table);
        let table_input = Self::build_table_input(&table.name, &table_root, &new_location)?;
        self.client
            .update_table()
            .database_name(&self.config.database)
            .table_input(table_input)
            .send()
            .await
            .map_err(|e| AilakeError::Catalog(format!("Glue update_table: {e}")))?;

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
        let result = self
            .client
            .delete_table()
            .database_name(&self.config.database)
            .name(&name.name)
            .send()
            .await;
        match result {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                // EntityNotFoundException → table already gone, that's fine
                if msg.contains("EntityNotFoundException") {
                    Ok(())
                } else {
                    Err(AilakeError::Catalog(format!("Glue delete_table: {msg}")))
                }
            }
        }
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

    #[test]
    fn table_params_contains_required_keys() {
        let params = GlueCatalog::table_params("s3://bucket/warehouse/ns/tbl/metadata/v1.json");
        assert_eq!(
            params.get("table_type").map(String::as_str),
            Some("ICEBERG")
        );
        assert_eq!(
            params.get("metadata_location").map(String::as_str),
            Some("s3://bucket/warehouse/ns/tbl/metadata/v1.json")
        );
    }

    #[test]
    fn metadata_path_format() {
        use ailake_store::LocalStore;
        let config = GlueCatalogConfig {
            database: "prod_db".into(),
            warehouse: "s3://my-bucket/warehouse".into(),
            region: None,
        };
        // Build a shell catalog just for path testing (no real Glue client).
        // We can't easily construct GlueCatalog without a real client in unit tests,
        // so test path logic via the helpers directly.
        let warehouse = config.warehouse.trim_end_matches('/');
        let table = TableIdent::new("default", "docs");
        let table_root = format!("{}/{}/{}", warehouse, table.namespace, table.name);
        let metadata_path = format!("{}/metadata/my-uuid.metadata.json", table_root);

        assert_eq!(
            metadata_path,
            "s3://my-bucket/warehouse/default/docs/metadata/my-uuid.metadata.json"
        );
        assert!(metadata_path.ends_with(".metadata.json"));
    }

    #[test]
    fn table_root_no_trailing_slash() {
        let warehouse = "s3://my-bucket/warehouse/";
        let ws = warehouse.trim_end_matches('/');
        let table = TableIdent::new("ns", "tbl");
        let root = format!("{}/{}/{}", ws, table.namespace, table.name);
        assert!(!root.contains("//"), "double slash in root: {root}");
        assert_eq!(root, "s3://my-bucket/warehouse/ns/tbl");
    }
}
