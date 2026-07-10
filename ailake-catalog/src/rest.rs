// SPDX-License-Identifier: MIT OR Apache-2.0
// Iceberg REST Catalog (https://iceberg.apache.org/spec/#rest-catalog).
// Covers: Apache Polaris, Azure Databricks Unity Catalog, GCP BigLake Metastore,
// AWS S3 Tables, Project Nessie (REST mode), and any spec-compliant server.
//
// Auth strategies:
//   RestCatalogAuth::None    — open catalogs (local Nessie/Polaris dev setup)
//   RestCatalogAuth::Bearer  — pre-obtained token (CI pipelines, Workload Identity)
//   RestCatalogAuth::OAuth2  — client credentials flow with token caching (production)
//
// Object storage:
//   Manifests are written to the provided Store (LocalStore / ObjectStoreBackend).
//   The REST server manages metadata.json; we write our JSON manifests independently.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::metadata::IcebergMetadata;
use crate::provider::{
    CatalogProvider, DataFileEntry, EqualityDeleteFile, NewSnapshot, SnapshotId, SnapshotOperation,
    TableIdent, TableMetadata, TableProperties,
};
use crate::schema_evolution::SchemaEvolution;
use crate::snapshot::{manifest_path, Manifest};
use ailake_store::Store;

// ── Public configuration types ───────────────────────────────────────────────

/// Authentication strategy for the REST catalog.
#[derive(Debug, Clone)]
pub enum RestCatalogAuth {
    /// No authentication. Works with open Nessie / Polaris dev setups.
    None,

    /// Pre-obtained Bearer token. Use for Workload Identity, CI tokens, etc.
    Bearer(String),

    /// OAuth2 client credentials flow.
    /// Tokens are cached until (expiry − 30 s) to avoid clock-edge failures.
    OAuth2 {
        /// Token endpoint URL (e.g. "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token")
        token_endpoint: String,
        client_id: String,
        client_secret: String,
        /// Optional scope (e.g. "https://management.azure.com/.default")
        scope: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct RestCatalogConfig {
    /// Base URI of the REST catalog (no trailing slash).
    ///
    /// Examples:
    ///   - Polaris (Snowflake):    "https://<account>.snowflakecomputing.com/polaris/api/catalog"
    ///   - Unity Catalog (Azure):  "https://<workspace>.azuredatabricks.net/api/2.1/unity-catalog/iceberg"
    ///   - BigLake (GCP):          "https://biglake.googleapis.com/iceberg/v1beta1"
    ///   - S3 Tables (AWS):        "https://s3tables.<region>.amazonaws.com/iceberg"
    ///   - Nessie:                 "http://localhost:19120/api"
    ///   - Gravitino:              "http://localhost:8090/iceberg"
    pub uri: String,

    /// Optional path prefix inserted between /v1 and /namespaces.
    ///
    /// Polaris: catalog name. Nessie: branch name (e.g. "main").
    /// Unity Catalog / BigLake: leave None.
    pub prefix: Option<String>,

    /// Base storage location for new tables (e.g. "s3://my-bucket/warehouse").
    /// Required for create_table. Unused if the server auto-assigns locations.
    pub warehouse: Option<String>,

    pub auth: RestCatalogAuth,
}

// ── RestCatalog ───────────────────────────────────────────────────────────────

struct CachedToken {
    value: String,
    expires_at: Instant,
}

pub struct RestCatalog {
    config: RestCatalogConfig,
    client: reqwest::Client,
    token_cache: Mutex<Option<CachedToken>>,
    /// Store backend used to write/read manifest JSON files.
    store: Arc<dyn Store>,
}

impl RestCatalog {
    pub fn new(config: RestCatalogConfig, store: Arc<dyn Store>) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            token_cache: Mutex::new(None),
            store,
        }
    }

    // ── URL helpers ──────────────────────────────────────────────────────────

    fn base_url(&self) -> String {
        let uri = self.config.uri.trim_end_matches('/');
        match &self.config.prefix {
            Some(p) if !p.is_empty() => format!("{uri}/v1/{p}"),
            _ => format!("{uri}/v1"),
        }
    }

    fn namespace_tables_url(&self, ns: &str) -> String {
        format!("{}/namespaces/{}/tables", self.base_url(), ns)
    }

    fn table_url(&self, table: &TableIdent) -> String {
        format!(
            "{}/namespaces/{}/tables/{}",
            self.base_url(),
            table.namespace,
            table.name
        )
    }

    fn table_storage_root(&self, table: &TableIdent) -> String {
        let warehouse = self
            .config
            .warehouse
            .as_deref()
            .unwrap_or("")
            .trim_end_matches('/');
        format!("{}/{}/{}", warehouse, table.namespace, table.name)
    }

    // ── Auth ─────────────────────────────────────────────────────────────────

    async fn get_token(&self) -> AilakeResult<Option<String>> {
        match &self.config.auth {
            RestCatalogAuth::None => Ok(None),
            RestCatalogAuth::Bearer(t) => Ok(Some(t.clone())),
            RestCatalogAuth::OAuth2 {
                token_endpoint,
                client_id,
                client_secret,
                scope,
            } => {
                {
                    let cache = self.token_cache.lock().await;
                    if let Some(cached) = &*cache {
                        if cached.expires_at > Instant::now() + Duration::from_secs(30) {
                            return Ok(Some(cached.value.clone()));
                        }
                    }
                }

                let mut params = vec![
                    ("grant_type", "client_credentials"),
                    ("client_id", client_id.as_str()),
                    ("client_secret", client_secret.as_str()),
                ];
                let scope_str;
                if let Some(s) = scope {
                    scope_str = s.clone();
                    params.push(("scope", scope_str.as_str()));
                }

                let resp = self
                    .client
                    .post(token_endpoint)
                    .form(&params)
                    .send()
                    .await
                    .map_err(|e| AilakeError::Store(e.to_string()))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(AilakeError::Catalog(format!(
                        "OAuth2 token request failed: HTTP {status}: {body}"
                    )));
                }

                let token_resp: OAuthTokenResponse = resp
                    .json()
                    .await
                    .map_err(|e| AilakeError::Catalog(format!("OAuth2 token parse: {e}")))?;

                let ttl = token_resp.expires_in.unwrap_or(3600);
                let cached = CachedToken {
                    value: token_resp.access_token.clone(),
                    expires_at: Instant::now() + Duration::from_secs(ttl),
                };
                *self.token_cache.lock().await = Some(cached);
                Ok(Some(token_resp.access_token))
            }
        }
    }

    // ── HTTP helpers ─────────────────────────────────────────────────────────

    async fn get(&self, url: &str) -> AilakeResult<reqwest::Response> {
        let mut req = self.client.get(url);
        if let Some(token) = self.get_token().await? {
            req = req.bearer_auth(token);
        }
        req.send()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn post<T: Serialize>(&self, url: &str, body: &T) -> AilakeResult<reqwest::Response> {
        let mut req = self.client.post(url).json(body);
        if let Some(token) = self.get_token().await? {
            req = req.bearer_auth(token);
        }
        req.send()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn delete(&self, url: &str) -> AilakeResult<reqwest::Response> {
        let mut req = self.client.delete(url);
        if let Some(token) = self.get_token().await? {
            req = req.bearer_auth(token);
        }
        req.send()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn require_ok(resp: reqwest::Response, ctx: &str) -> AilakeResult<reqwest::Response> {
        if resp.status().is_success() {
            return Ok(resp);
        }
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(AilakeError::Catalog(format!(
            "{ctx}: HTTP {status}: {body}"
        )))
    }

    /// Fetch current table state and read back the flat JSON manifest for the
    /// given (or current) snapshot. `Ok(None)` only when no snapshot id can be
    /// resolved at all (fresh table, never committed) — an explicit
    /// `snapshot_id` that doesn't match any known snapshot is a hard error.
    async fn load_manifest(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Option<Manifest>> {
        let resp = self.get(&self.table_url(table)).await?;
        let resp = Self::require_ok(resp, "load_manifest").await?;
        let result: LoadTableResult = resp
            .json()
            .await
            .map_err(|e| AilakeError::Catalog(format!("load_manifest parse: {e}")))?;
        let meta = &result.metadata;
        let snap_id = match snapshot_id.or(meta.current_snapshot_id) {
            Some(id) => id,
            None => return Ok(None),
        };
        let snap = meta
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snap_id)
            .ok_or_else(|| AilakeError::Catalog(format!("snapshot {snap_id} not found")))?;
        let manifest_bytes = self.store.get(&snap.manifest_list).await?;
        let manifest_json = std::str::from_utf8(&manifest_bytes)
            .map_err(|e| AilakeError::Catalog(e.to_string()))?;
        Ok(Some(Manifest::from_json(manifest_json)?))
    }
}

// ── CatalogProvider ───────────────────────────────────────────────────────────

#[async_trait]
impl CatalogProvider for RestCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        let location = self.table_storage_root(name);

        // Reuse the exact same schema/partition-spec/properties construction the
        // file-based backends use (IcebergMetadata::new) instead of duplicating it —
        // guarantees this backend's tables carry the same ailake.* properties and
        // real (not empty) schema/partition-spec as Hadoop/Glue/Jdbc for the same
        // TableProperties. format-version isn't a field the REST CreateTableRequest
        // accepts (checked against the spec's CreateTableRequest schema) — servers
        // create V2 by default, so a V3 request needs a follow-up
        // `upgrade-format-version` commit right after create.
        let pct = props
            .partition_column_type
            .as_deref()
            .or(props.policy.partition_column_type.as_deref());
        let meta = IcebergMetadata::new(
            &location,
            &props.policy,
            props.format_version,
            pct,
            &props.policy.partition_fields,
        );
        let mut properties = meta.properties.clone();
        for (k, v) in &props.extra {
            properties.insert(k.clone(), v.clone());
        }
        let schema = meta
            .schemas
            .first()
            .cloned()
            .unwrap_or_else(RestSchema::empty_value);
        let partition_spec = if meta.default_spec_id > 0 {
            meta.partition_specs
                .get(meta.default_spec_id as usize)
                .cloned()
        } else {
            None
        };

        let req = CreateTableRequest {
            name: name.name.clone(),
            location: Some(location),
            schema,
            partition_spec,
            properties,
        };

        let url = self.namespace_tables_url(&name.namespace);
        let resp = self.post(&url, &req).await?;
        Self::require_ok(resp, "create_table").await?;

        if meta.format_version >= 3 {
            let commit_req = CommitTableRequest {
                identifier: TableIdentifier {
                    namespace: vec![name.namespace.clone()],
                    name: name.name.clone(),
                },
                requirements: vec![],
                updates: vec![TableUpdate::UpgradeFormatVersion {
                    format_version: meta.format_version,
                }],
            };
            let resp = self.post(&self.table_url(name), &commit_req).await?;
            Self::require_ok(resp, "create_table (upgrade-format-version)").await?;
        }
        Ok(())
    }

    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata> {
        let resp = self.get(&self.table_url(name)).await?;
        let resp = Self::require_ok(resp, "load_table").await?;
        let result: LoadTableResult = resp
            .json()
            .await
            .map_err(|e| AilakeError::Catalog(format!("load_table parse: {e}")))?;
        Ok(result.metadata.to_table_metadata())
    }

    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> AilakeResult<SnapshotId> {
        let snap_id = snapshot.snapshot_id;

        // Real OCC: read current state, build the new manifest against it, commit
        // with an `assert-ref-snapshot-id` requirement pinned to the snapshot we
        // read. If another writer moves `main` first, the server rejects the
        // commit (409) and we retry from a fresh read — same retry×5 w/ backoff
        // pattern glue.rs/jdbc.rs already use for their own CAS mechanisms. This
        // replaces the previous "the REST server owns conflict resolution, we
        // never send requirements" assumption, which left concurrent writers free
        // to silently clobber each other's commits.
        const MAX_RETRIES: u32 = 5;
        for attempt in 0..MAX_RETRIES {
            let resp = self.get(&self.table_url(table)).await?;
            let resp = Self::require_ok(resp, "commit_snapshot (read current state)").await?;
            let result: LoadTableResult = resp
                .json()
                .await
                .map_err(|e| AilakeError::Catalog(format!("commit_snapshot parse: {e}")))?;
            let current_snapshot_id = result.metadata.current_snapshot_id;

            // Append/Delete inherit the previous snapshot's full file/eq-delete list
            // (this catalog writes one flat manifest per snapshot, not an Iceberg
            // manifest chain, so the new manifest must already contain the complete
            // resulting state). Replace/Overwrite treat `snapshot.files`/
            // `snapshot.equality_delete_files` as the complete state — callers
            // already rebuild it (see hadoop.rs's identical contract).
            let (effective_files, effective_eq_deletes): (
                Vec<DataFileEntry>,
                Vec<EqualityDeleteFile>,
            ) = if matches!(
                snapshot.operation,
                SnapshotOperation::Append | SnapshotOperation::Delete
            ) {
                let mut prev_files = match current_snapshot_id {
                    Some(id) => self.list_files(table, Some(id)).await?,
                    None => vec![],
                };
                prev_files.extend(snapshot.files.iter().cloned());
                let mut prev_deletes = self
                    .list_equality_deletes(table, current_snapshot_id)
                    .await?;
                prev_deletes.extend(snapshot.equality_delete_files.iter().cloned());
                (prev_files, prev_deletes)
            } else {
                (
                    snapshot.files.clone(),
                    snapshot.equality_delete_files.clone(),
                )
            };

            // 1. Write manifest JSON to object storage
            let root = self.table_storage_root(table);
            let rel_path = manifest_path(snap_id);
            let abs_path = format!("{root}/{rel_path}");
            let manifest = Manifest {
                snapshot_id: snap_id,
                files: effective_files,
                equality_delete_files: effective_eq_deletes,
            };
            self.store
                .put(&abs_path, Bytes::from(manifest.to_json()?.into_bytes()))
                .await?;

            // 2. Register snapshot with the REST catalog, guarded by an OCC requirement.
            let now_ms = now_ms();
            let rest_snap = RestSnapshot {
                snapshot_id: snap_id,
                parent_snapshot_id: snapshot.parent_snapshot_id,
                sequence_number: 1,
                timestamp_ms: now_ms,
                manifest_list: abs_path,
                summary: HashMap::from([
                    (
                        "operation".into(),
                        format!("{:?}", snapshot.operation).to_lowercase(),
                    ),
                    ("added-data-files".into(), snapshot.files.len().to_string()),
                ]),
                schema_id: Some(0),
            };

            let commit_req = CommitTableRequest {
                identifier: TableIdentifier {
                    namespace: vec![table.namespace.clone()],
                    name: table.name.clone(),
                },
                requirements: vec![TableRequirement::AssertRefSnapshotId {
                    r#ref: "main".into(),
                    snapshot_id: current_snapshot_id,
                }],
                updates: vec![
                    TableUpdate::AddSnapshot {
                        snapshot: rest_snap,
                    },
                    TableUpdate::SetSnapshotRef {
                        ref_name: "main".into(),
                        snapshot_type: "branch".into(),
                        snapshot_id: snap_id,
                    },
                ],
            };

            let resp = self.post(&self.table_url(table), &commit_req).await?;
            if resp.status().as_u16() == 409 {
                if attempt + 1 < MAX_RETRIES {
                    tokio::time::sleep(std::time::Duration::from_millis(50 << attempt)).await;
                    continue;
                }
                return Err(AilakeError::Catalog(format!(
                    "commit_snapshot: {MAX_RETRIES} retries exhausted (concurrent modification)"
                )));
            }
            Self::require_ok(resp, "commit_snapshot").await?;
            return Ok(snap_id);
        }
        unreachable!("loop always returns via Ok/Err before exhausting MAX_RETRIES iterations")
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let manifest = self
            .load_manifest(table, snapshot_id)
            .await?
            .ok_or_else(|| AilakeError::Catalog("table has no snapshots".into()))?;
        Ok(manifest.files)
    }

    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()> {
        let resp = self.delete(&self.table_url(name)).await?;
        if resp.status().as_u16() == 404 {
            return Ok(());
        }
        Self::require_ok(resp, "drop_table").await?;
        Ok(())
    }

    async fn evolve_schema(
        &self,
        table: &TableIdent,
        evolution: SchemaEvolution,
    ) -> AilakeResult<i32> {
        let resp = self.get(&self.table_url(table)).await?;
        let resp = Self::require_ok(resp, "evolve_schema (read current state)").await?;
        let result: LoadTableResult = resp
            .json()
            .await
            .map_err(|e| AilakeError::Catalog(format!("evolve_schema parse: {e}")))?;
        let meta = &result.metadata;

        let current_schema = meta
            .schemas
            .iter()
            .find(|s| s["schema-id"].as_i64() == Some(meta.current_schema_id as i64))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"schema-id": 0, "type": "struct", "fields": []}));
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

        let new_schema_id = meta.current_schema_id + 1;
        let new_schema = serde_json::json!({
            "schema-id": new_schema_id,
            "type": "struct",
            "fields": fields,
        });

        let mut updates = vec![
            TableUpdate::AddSchema { schema: new_schema },
            TableUpdate::SetCurrentSchema { schema_id: -1 },
        ];
        if !evolution.extra_properties.is_empty() {
            updates.push(TableUpdate::SetProperties {
                updates: evolution.extra_properties.clone(),
            });
        }

        let commit_req = CommitTableRequest {
            identifier: TableIdentifier {
                namespace: vec![table.namespace.clone()],
                name: table.name.clone(),
            },
            requirements: vec![TableRequirement::AssertCurrentSchemaId {
                current_schema_id: meta.current_schema_id,
            }],
            updates,
        };
        let resp = self.post(&self.table_url(table), &commit_req).await?;
        Self::require_ok(resp, "evolve_schema").await?;
        Ok(new_schema_id)
    }

    async fn list_equality_deletes(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<EqualityDeleteFile>> {
        let manifest = self.load_manifest(table, snapshot_id).await?;
        Ok(manifest
            .map(|m| m.equality_delete_files)
            .unwrap_or_default())
    }
}

// ── REST protocol types ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    expires_in: Option<u64>,
}

#[derive(Serialize)]
struct CreateTableRequest {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<String>,
    schema: serde_json::Value,
    #[serde(rename = "partition-spec", skip_serializing_if = "Option::is_none")]
    partition_spec: Option<serde_json::Value>,
    properties: HashMap<String, String>,
}

impl RestSchema {
    /// Fallback empty struct schema (`{"type":"struct","fields":[]}`) for the
    /// unreachable case `IcebergMetadata::new` produces no schema at all.
    fn empty_value() -> serde_json::Value {
        serde_json::json!({"schema-id": 0, "type": "struct", "fields": []})
    }
}

/// Marker type only used to namespace `empty_value()` — `CreateTableRequest.schema`
/// itself is a raw `serde_json::Value` (reusing `IcebergMetadata::new`'s output
/// directly, see `create_table`), not a fixed Rust shape.
struct RestSchema;

/// Subset of `LoadTableResult` — only fields AI-Lake needs.
#[derive(Deserialize)]
struct LoadTableResult {
    metadata: IcebergMetadata,
}

#[derive(Serialize)]
struct CommitTableRequest {
    identifier: TableIdentifier,
    requirements: Vec<TableRequirement>,
    updates: Vec<TableUpdate>,
}

#[derive(Serialize)]
struct TableIdentifier {
    namespace: Vec<String>,
    name: String,
}

/// `TableRequirement` — Iceberg REST OCC guards sent with a commit. Server
/// rejects (409) if the named ref/schema doesn't match what's asserted here,
/// preventing a commit from silently clobbering a concurrent writer's changes.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum TableRequirement {
    AssertRefSnapshotId {
        r#ref: String,
        #[serde(rename = "snapshot-id")]
        snapshot_id: Option<SnapshotId>,
    },
    AssertCurrentSchemaId {
        #[serde(rename = "current-schema-id")]
        current_schema_id: i32,
    },
}

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "kebab-case")]
enum TableUpdate {
    AddSnapshot {
        snapshot: RestSnapshot,
    },
    SetSnapshotRef {
        #[serde(rename = "ref-name")]
        ref_name: String,
        #[serde(rename = "type")]
        snapshot_type: String,
        #[serde(rename = "snapshot-id")]
        snapshot_id: SnapshotId,
    },
    UpgradeFormatVersion {
        #[serde(rename = "format-version")]
        format_version: i32,
    },
    AddSchema {
        schema: serde_json::Value,
    },
    SetCurrentSchema {
        #[serde(rename = "schema-id")]
        schema_id: i32,
    },
    SetProperties {
        updates: HashMap<String, String>,
    },
}

#[derive(Serialize)]
struct RestSnapshot {
    #[serde(rename = "snapshot-id")]
    snapshot_id: SnapshotId,
    #[serde(rename = "parent-snapshot-id", skip_serializing_if = "Option::is_none")]
    parent_snapshot_id: Option<SnapshotId>,
    #[serde(rename = "sequence-number")]
    sequence_number: i64,
    #[serde(rename = "timestamp-ms")]
    timestamp_ms: i64,
    #[serde(rename = "manifest-list")]
    manifest_list: String,
    summary: HashMap<String, String>,
    #[serde(rename = "schema-id", skip_serializing_if = "Option::is_none")]
    schema_id: Option<i32>,
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
    use ailake_store::LocalStore;

    fn catalog(prefix: Option<&str>) -> RestCatalog {
        let store = Arc::new(LocalStore::new("/tmp"));
        RestCatalog::new(
            RestCatalogConfig {
                uri: "https://catalog.example.com".into(),
                prefix: prefix.map(|s| s.to_string()),
                warehouse: Some("s3://my-bucket/warehouse".into()),
                auth: RestCatalogAuth::None,
            },
            store,
        )
    }

    #[test]
    fn base_url_no_prefix() {
        let c = catalog(None);
        assert_eq!(c.base_url(), "https://catalog.example.com/v1");
    }

    #[test]
    fn base_url_with_prefix() {
        let c = catalog(Some("main"));
        assert_eq!(c.base_url(), "https://catalog.example.com/v1/main");
    }

    #[test]
    fn table_url_format() {
        let c = catalog(Some("main"));
        let tbl = TableIdent::new("default", "docs");
        assert_eq!(
            c.table_url(&tbl),
            "https://catalog.example.com/v1/main/namespaces/default/tables/docs"
        );
    }

    #[test]
    fn namespace_tables_url_format() {
        let c = catalog(None);
        assert_eq!(
            c.namespace_tables_url("prod"),
            "https://catalog.example.com/v1/namespaces/prod/tables"
        );
    }

    #[test]
    fn table_storage_root() {
        let c = catalog(None);
        let tbl = TableIdent::new("default", "docs");
        assert_eq!(
            c.table_storage_root(&tbl),
            "s3://my-bucket/warehouse/default/docs"
        );
    }

    #[test]
    fn commit_request_serializes_correctly() {
        let snap = RestSnapshot {
            snapshot_id: 42,
            parent_snapshot_id: None,
            sequence_number: 1,
            timestamp_ms: 1_000_000,
            manifest_list: "s3://bucket/snap-42.json".into(),
            summary: HashMap::from([("operation".into(), "append".into())]),
            schema_id: Some(0),
        };
        let req = CommitTableRequest {
            identifier: TableIdentifier {
                namespace: vec!["default".into()],
                name: "docs".into(),
            },
            requirements: vec![],
            updates: vec![
                TableUpdate::AddSnapshot { snapshot: snap },
                TableUpdate::SetSnapshotRef {
                    ref_name: "main".into(),
                    snapshot_type: "branch".into(),
                    snapshot_id: 42,
                },
            ],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"action\":\"add-snapshot\""));
        assert!(json.contains("\"action\":\"set-snapshot-ref\""));
        assert!(json.contains("\"snapshot-id\":42"));
        assert!(json.contains("\"ref-name\":\"main\""));
    }

    #[test]
    fn create_table_request_includes_ailake_properties() {
        let properties: HashMap<String, String> = HashMap::from([
            ("ailake.vector-column".into(), "embedding".into()),
            ("ailake.vector-dim".into(), "1536".into()),
            ("ailake.vector-metric".into(), "cosine".into()),
            ("ailake.vector-precision".into(), "f16".into()),
        ]);

        let req = CreateTableRequest {
            name: "docs".into(),
            location: Some("s3://bucket/warehouse/default/docs".into()),
            schema: RestSchema::empty_value(),
            partition_spec: None,
            properties: properties.clone(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("ailake.vector-column"));
        assert!(json.contains("ailake.vector-dim"));
    }

    #[test]
    fn schema_update_serializes_correctly() {
        let req = CommitTableRequest {
            identifier: TableIdentifier {
                namespace: vec!["default".into()],
                name: "docs".into(),
            },
            requirements: vec![TableRequirement::AssertCurrentSchemaId {
                current_schema_id: 0,
            }],
            updates: vec![
                TableUpdate::AddSchema {
                    schema: serde_json::json!({"schema-id": 1, "type": "struct", "fields": []}),
                },
                TableUpdate::SetCurrentSchema { schema_id: -1 },
            ],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"action\":\"add-schema\""));
        assert!(json.contains("\"action\":\"set-current-schema\""));
        assert!(json.contains("\"schema-id\":-1"));
        assert!(json.contains("\"type\":\"assert-current-schema-id\""));
        assert!(json.contains("\"current-schema-id\":0"));
    }

    #[test]
    fn assert_ref_snapshot_id_serializes_correctly() {
        let req = TableRequirement::AssertRefSnapshotId {
            r#ref: "main".into(),
            snapshot_id: Some(42),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"type\":\"assert-ref-snapshot-id\""));
        assert!(json.contains("\"ref\":\"main\""));
        assert!(json.contains("\"snapshot-id\":42"));
    }
}
