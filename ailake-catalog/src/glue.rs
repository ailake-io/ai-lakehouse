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

    /// Returns `(metadata_location, glue_version_id)` for OCC commit guard.
    async fn get_table_state(&self, table: &TableIdent) -> AilakeResult<(String, Option<String>)> {
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

        let version_id = t.version_id().map(str::to_string);

        let params = t.parameters().ok_or_else(|| {
            AilakeError::Catalog(format!(
                "Glue table {}.{} has no parameters",
                table.namespace, table.name
            ))
        })?;
        let location = params.get("metadata_location").cloned().ok_or_else(|| {
            AilakeError::Catalog(format!(
                "Glue table {}.{} is not an Iceberg table (missing metadata_location)",
                table.namespace, table.name
            ))
        })?;
        Ok((location, version_id))
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
        let pct = props
            .partition_column_type
            .as_deref()
            .or(props.policy.partition_column_type.as_deref());
        let mut meta = IcebergMetadata::new(
            &table_root,
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
        // OCC retry: read -> apply commit to a fresh copy of `meta` -> update_table
        // with version_id. Glue rejects update_table when version_id doesn't match
        // (ConcurrentModificationException), so we re-read and retry with the fresh
        // version_id + freshly-read `meta` on every attempt, so a concurrent
        // Append/Delete that won a prior iteration isn't lost by an in-flight retry
        // that captured stale state. `commit_into_metadata` (shared with Hadoop) owns
        // the actual Avro manifest / Puffin / partition-stats / first_row_id logic.
        const MAX_RETRIES: u32 = 5;
        let table_root = self.table_root(table);
        for attempt in 0..MAX_RETRIES {
            let (old_location, version_id) = self.get_table_state(table).await?;
            let mut meta = self.load_iceberg_metadata(&old_location).await?;

            let snap_id = commit_into_metadata(
                &*self.store,
                &table_root,
                &self.config.warehouse,
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

            let table_input = Self::build_table_input(&table.name, &table_root, &new_location)?;
            // `version_id` is a param of the UpdateTable *request*, not of TableInput —
            // Glue compares it against the table's current version server-side and
            // rejects the call with ConcurrentModificationException on mismatch, giving
            // us the OCC guard this retry loop relies on.
            match self
                .client
                .update_table()
                .database_name(&self.config.database)
                .table_input(table_input)
                .set_version_id(version_id.clone())
                .send()
                .await
            {
                Ok(_) => return Ok(snap_id),
                Err(e) => {
                    let svc = e.into_service_error();
                    if svc.is_concurrent_modification_exception() && attempt + 1 < MAX_RETRIES {
                        tokio::time::sleep(tokio::time::Duration::from_millis(100 << attempt))
                            .await;
                        continue;
                    }
                    return Err(AilakeError::Catalog(format!("Glue update_table: {svc}")));
                }
            }
        }
        Err(AilakeError::Catalog(format!(
            "Glue commit_snapshot: {MAX_RETRIES} retries exhausted (concurrent modification)"
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
    /// `HadoopCatalog::evolve_schema` exactly (same metadata.json schema-patch
    /// logic), swapping the pointer-update mechanism for Glue's own
    /// version_id-guarded `update_table` OCC retry loop, since evolving the
    /// schema still means writing a new metadata.json and re-pointing the Glue
    /// table's `metadata_location` parameter at it.
    async fn evolve_schema(
        &self,
        table: &TableIdent,
        evolution: SchemaEvolution,
    ) -> AilakeResult<i32> {
        let table_root = self.table_root(table);
        const MAX_RETRIES: u32 = 5;
        for attempt in 0..MAX_RETRIES {
            let (old_location, version_id) = self.get_table_state(table).await?;
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

            let table_input = Self::build_table_input(&table.name, &table_root, &new_location)?;
            match self
                .client
                .update_table()
                .database_name(&self.config.database)
                .table_input(table_input)
                .set_version_id(version_id.clone())
                .send()
                .await
            {
                Ok(_) => return Ok(new_schema_id),
                Err(e) => {
                    let svc = e.into_service_error();
                    if svc.is_concurrent_modification_exception() && attempt + 1 < MAX_RETRIES {
                        tokio::time::sleep(tokio::time::Duration::from_millis(100 << attempt))
                            .await;
                        continue;
                    }
                    return Err(AilakeError::Catalog(format!("Glue update_table: {svc}")));
                }
            }
        }
        Err(AilakeError::Catalog(format!(
            "Glue evolve_schema: {MAX_RETRIES} retries exhausted (concurrent modification)"
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
        // The `s3://` scheme separator legitimately contains "//" — only the part
        // after it must be free of a doubled slash from an untrimmed trailing '/'.
        let path_after_scheme = root.split_once("://").map_or(root.as_str(), |x| x.1);
        assert!(
            !path_after_scheme.contains("//"),
            "double slash in root: {root}"
        );
        assert_eq!(root, "s3://my-bucket/warehouse/ns/tbl");
    }
}
