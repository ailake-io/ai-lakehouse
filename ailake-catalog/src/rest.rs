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
    CatalogProvider, DataFileEntry, NewSnapshot, SnapshotId, TableIdent, TableMetadata,
    TableProperties,
};
use crate::snapshot::{build_manifest, manifest_path};
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
}

// ── CatalogProvider ───────────────────────────────────────────────────────────

#[async_trait]
impl CatalogProvider for RestCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        let location = self.table_storage_root(name);

        let mut properties: HashMap<String, String> = HashMap::from([
            ("ailake.format-version".into(), "1".into()),
            (
                "ailake.vector-column".into(),
                props.policy.column_name.clone(),
            ),
            ("ailake.vector-dim".into(), props.policy.dim.to_string()),
            (
                "ailake.vector-metric".into(),
                format!("{:?}", props.policy.metric).to_lowercase(),
            ),
            (
                "ailake.vector-precision".into(),
                format!("{:?}", props.policy.precision).to_lowercase(),
            ),
        ]);
        for (k, v) in &props.extra {
            properties.insert(k.clone(), v.clone());
        }

        let req = CreateTableRequest {
            name: name.name.clone(),
            location: Some(location),
            schema: RestSchema::empty(),
            properties,
        };

        let url = self.namespace_tables_url(&name.namespace);
        let resp = self.post(&url, &req).await?;
        Self::require_ok(resp, "create_table").await?;
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

        // 1. Write manifest JSON to object storage
        let root = self.table_storage_root(table);
        let rel_path = manifest_path(snap_id);
        let abs_path = format!("{root}/{rel_path}");
        let manifest = build_manifest(&snapshot);
        self.store
            .put(&abs_path, Bytes::from(manifest.to_json()?.into_bytes()))
            .await?;

        // 2. Register snapshot with the REST catalog
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
            requirements: vec![],
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
        Self::require_ok(resp, "commit_snapshot").await?;
        Ok(snap_id)
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let resp = self.get(&self.table_url(table)).await?;
        let resp = Self::require_ok(resp, "list_files").await?;
        let result: LoadTableResult = resp
            .json()
            .await
            .map_err(|e| AilakeError::Catalog(format!("list_files parse: {e}")))?;

        let meta = &result.metadata;
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
        let resp = self.delete(&self.table_url(name)).await?;
        if resp.status().as_u16() == 404 {
            return Ok(());
        }
        Self::require_ok(resp, "drop_table").await?;
        Ok(())
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
    schema: RestSchema,
    properties: HashMap<String, String>,
}

#[derive(Serialize)]
struct RestSchema {
    #[serde(rename = "type")]
    schema_type: &'static str,
    fields: Vec<serde_json::Value>,
    #[serde(rename = "schema-id")]
    schema_id: i32,
}

impl RestSchema {
    fn empty() -> Self {
        Self {
            schema_type: "struct",
            fields: vec![],
            schema_id: 0,
        }
    }
}

/// Subset of `LoadTableResult` — only fields AI-Lake needs.
#[derive(Deserialize)]
struct LoadTableResult {
    metadata: IcebergMetadata,
}

#[derive(Serialize)]
struct CommitTableRequest {
    identifier: TableIdentifier,
    requirements: Vec<serde_json::Value>,
    updates: Vec<TableUpdate>,
}

#[derive(Serialize)]
struct TableIdentifier {
    namespace: Vec<String>,
    name: String,
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
            schema: RestSchema::empty(),
            properties: properties.clone(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("ailake.vector-column"));
        assert!(json.contains("ailake.vector-dim"));
    }
}
