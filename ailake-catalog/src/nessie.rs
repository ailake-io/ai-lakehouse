// SPDX-License-Identifier: MIT OR Apache-2.0
// NessieCatalog — Project Nessie catalog with branching extensions.
//
// Requires Nessie 0.60+ (REST API v2 + Iceberg REST Catalog spec).
//
// CatalogProvider is fully delegated to an inner RestCatalog.
// Extra public methods expose Nessie-specific branch/tag operations
// via the Nessie v2 API (/api/v2/trees/*).
//
// Example setup (local Docker):
//   docker run -p 19120:19120 ghcr.io/projectnessie/nessie:latest
//
//   NessieCatalogConfig {
//       uri: "http://localhost:19120/api",
//       default_branch: "main",
//       warehouse: Some("/tmp/warehouse".into()),
//       auth: RestCatalogAuth::None,
//   }

use std::sync::Arc;
use std::time::{Duration, Instant};

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::provider::{
    CatalogProvider, DataFileEntry, EqualityDeleteFile, NewSnapshot, SnapshotId, TableIdent,
    TableMetadata, TableProperties,
};
use crate::rest::{RestCatalog, RestCatalogAuth, RestCatalogConfig};
use crate::schema_evolution::SchemaEvolution;
use ailake_store::Store;

// ── Configuration ─────────────────────────────────────────────────────────────

pub struct NessieCatalogConfig {
    /// Nessie server URI, e.g. "http://localhost:19120/api" (no trailing slash).
    pub uri: String,
    /// Branch used for all CatalogProvider operations. Defaults to "main".
    pub default_branch: String,
    /// Base storage location for new tables (e.g. "s3://my-bucket/warehouse").
    pub warehouse: Option<String>,
    pub auth: RestCatalogAuth,
}

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NessieBranch {
    pub name: String,
    pub hash: String,
}

// ── Internal ──────────────────────────────────────────────────────────────────

struct CachedToken {
    value: String,
    expires_at: Instant,
}

// ── NessieCatalog ─────────────────────────────────────────────────────────────

pub struct NessieCatalog {
    inner: RestCatalog,
    nessie_uri: String,
    auth: RestCatalogAuth,
    client: reqwest::Client,
    token_cache: Mutex<Option<CachedToken>>,
}

impl NessieCatalog {
    pub fn new(config: NessieCatalogConfig, store: Arc<dyn Store>) -> Self {
        let rest_config = RestCatalogConfig {
            uri: config.uri.clone(),
            prefix: Some(config.default_branch.clone()),
            warehouse: config.warehouse.clone(),
            auth: config.auth.clone(),
        };
        Self {
            nessie_uri: config.uri.trim_end_matches('/').to_string(),
            auth: config.auth,
            inner: RestCatalog::new(rest_config, store),
            client: reqwest::Client::new(),
            token_cache: Mutex::new(None),
        }
    }

    // ── URL helpers ───────────────────────────────────────────────────────────

    fn trees_url(&self) -> String {
        format!("{}/v2/trees", self.nessie_uri)
    }

    fn ref_url(&self, name: &str) -> String {
        format!("{}/v2/trees/BRANCH,{}", self.nessie_uri, name)
    }

    fn merge_url(&self, into_branch: &str) -> String {
        format!("{}/v2/trees/BRANCH,{}/merge", self.nessie_uri, into_branch)
    }

    // ── Auth ──────────────────────────────────────────────────────────────────

    async fn get_token(&self) -> AilakeResult<Option<String>> {
        match &self.auth {
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
                    if let Some(c) = &*cache {
                        if c.expires_at > Instant::now() + Duration::from_secs(30) {
                            return Ok(Some(c.value.clone()));
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
                        "Nessie OAuth2 failed: HTTP {status}: {body}"
                    )));
                }
                #[derive(Deserialize)]
                struct TokenResp {
                    access_token: String,
                    expires_in: Option<u64>,
                }
                let tr: TokenResp = resp
                    .json()
                    .await
                    .map_err(|e| AilakeError::Catalog(format!("Nessie OAuth2 parse: {e}")))?;
                let ttl = tr.expires_in.unwrap_or(3600);
                *self.token_cache.lock().await = Some(CachedToken {
                    value: tr.access_token.clone(),
                    expires_at: Instant::now() + Duration::from_secs(ttl),
                });
                Ok(Some(tr.access_token))
            }
        }
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────

    async fn get(&self, url: &str) -> AilakeResult<reqwest::Response> {
        let mut req = self.client.get(url);
        if let Some(t) = self.get_token().await? {
            req = req.bearer_auth(t);
        }
        req.send()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn post<T: Serialize>(&self, url: &str, body: &T) -> AilakeResult<reqwest::Response> {
        let mut req = self.client.post(url).json(body);
        if let Some(t) = self.get_token().await? {
            req = req.bearer_auth(t);
        }
        req.send()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn delete(&self, url: &str) -> AilakeResult<reqwest::Response> {
        let mut req = self.client.delete(url);
        if let Some(t) = self.get_token().await? {
            req = req.bearer_auth(t);
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
            "Nessie {ctx}: HTTP {status}: {body}"
        )))
    }

    // ── Nessie branching API ──────────────────────────────────────────────────

    /// Get a single branch by name.
    pub async fn get_branch(&self, name: &str) -> AilakeResult<NessieBranch> {
        let resp = self.get(&self.ref_url(name)).await?;
        let resp = Self::require_ok(resp, "get_branch").await?;
        #[derive(Deserialize)]
        struct ReferenceResp {
            name: String,
            hash: Option<String>,
        }
        let r: ReferenceResp = resp
            .json()
            .await
            .map_err(|e| AilakeError::Catalog(format!("Nessie get_branch parse: {e}")))?;
        Ok(NessieBranch {
            name: r.name,
            hash: r.hash.unwrap_or_default(),
        })
    }

    /// List all branches (not tags).
    pub async fn list_branches(&self) -> AilakeResult<Vec<NessieBranch>> {
        let resp = self.get(&self.trees_url()).await?;
        let resp = Self::require_ok(resp, "list_branches").await?;
        #[derive(Deserialize)]
        struct TreesResp {
            references: Vec<ReferenceItem>,
        }
        #[derive(Deserialize)]
        struct ReferenceItem {
            #[serde(rename = "type")]
            ref_type: String,
            name: String,
            hash: Option<String>,
        }
        let trees: TreesResp = resp
            .json()
            .await
            .map_err(|e| AilakeError::Catalog(format!("Nessie list_branches parse: {e}")))?;
        Ok(trees
            .references
            .into_iter()
            .filter(|r| r.ref_type == "BRANCH")
            .map(|r| NessieBranch {
                name: r.name,
                hash: r.hash.unwrap_or_default(),
            })
            .collect())
    }

    /// Create a new branch pointing to the HEAD of `from_branch`.
    pub async fn create_branch(&self, name: &str, from_branch: &str) -> AilakeResult<()> {
        let source = self.get_branch(from_branch).await?;
        #[derive(Serialize)]
        struct CreateBranchReq<'a> {
            #[serde(rename = "type")]
            ref_type: &'static str,
            name: &'a str,
            hash: &'a str,
            reference: SourceRef<'a>,
        }
        #[derive(Serialize)]
        struct SourceRef<'a> {
            #[serde(rename = "type")]
            ref_type: &'static str,
            name: &'a str,
        }
        let body = CreateBranchReq {
            ref_type: "BRANCH",
            name,
            hash: &source.hash,
            reference: SourceRef {
                ref_type: "BRANCH",
                name: from_branch,
            },
        };
        let resp = self.post(&self.trees_url(), &body).await?;
        Self::require_ok(resp, "create_branch").await?;
        Ok(())
    }

    /// Merge `source_branch` into `into_branch`.
    pub async fn merge_branch(&self, source_branch: &str, into_branch: &str) -> AilakeResult<()> {
        let source = self.get_branch(source_branch).await?;
        #[derive(Serialize)]
        struct MergeReq<'a> {
            #[serde(rename = "fromRefName")]
            from_ref_name: &'a str,
            #[serde(rename = "fromHash")]
            from_hash: &'a str,
        }
        let body = MergeReq {
            from_ref_name: source_branch,
            from_hash: &source.hash,
        };
        let resp = self.post(&self.merge_url(into_branch), &body).await?;
        Self::require_ok(resp, "merge_branch").await?;
        Ok(())
    }

    /// Delete a branch. No-op if branch does not exist.
    pub async fn delete_branch(&self, name: &str) -> AilakeResult<()> {
        let branch = self.get_branch(name).await?;
        let url = format!(
            "{}/v2/trees/BRANCH,{}?expectedHash={}",
            self.nessie_uri, name, branch.hash
        );
        let resp = self.delete(&url).await?;
        if resp.status().as_u16() == 404 {
            return Ok(());
        }
        Self::require_ok(resp, "delete_branch").await?;
        Ok(())
    }
}

// ── CatalogProvider — delegate to inner RestCatalog ───────────────────────────

#[async_trait]
impl CatalogProvider for NessieCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        self.inner.create_table(name, props).await
    }

    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata> {
        self.inner.load_table(name).await
    }

    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> AilakeResult<SnapshotId> {
        self.inner.commit_snapshot(table, snapshot).await
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        self.inner.list_files(table, snapshot_id).await
    }

    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()> {
        self.inner.drop_table(name).await
    }

    // Without these, `evolve_schema`/`list_equality_deletes` would silently fall
    // back to the CatalogProvider trait's own defaults (unsupported / empty)
    // instead of the inner RestCatalog's real implementation — Rust doesn't
    // forward trait methods automatically just because the struct wraps one.
    async fn evolve_schema(
        &self,
        table: &TableIdent,
        evolution: SchemaEvolution,
    ) -> AilakeResult<i32> {
        self.inner.evolve_schema(table, evolution).await
    }

    async fn list_equality_deletes(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<EqualityDeleteFile>> {
        self.inner.list_equality_deletes(table, snapshot_id).await
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_store::LocalStore;

    fn catalog(branch: &str) -> NessieCatalog {
        NessieCatalog::new(
            NessieCatalogConfig {
                uri: "http://localhost:19120/api".into(),
                default_branch: branch.into(),
                warehouse: Some("/tmp/warehouse".into()),
                auth: RestCatalogAuth::None,
            },
            Arc::new(LocalStore::new("/tmp")),
        )
    }

    #[test]
    fn trees_url_format() {
        let c = catalog("main");
        assert_eq!(c.trees_url(), "http://localhost:19120/api/v2/trees");
    }

    #[test]
    fn ref_url_format() {
        let c = catalog("main");
        assert_eq!(
            c.ref_url("main"),
            "http://localhost:19120/api/v2/trees/BRANCH,main"
        );
    }

    #[test]
    fn merge_url_format() {
        let c = catalog("main");
        assert_eq!(
            c.merge_url("main"),
            "http://localhost:19120/api/v2/trees/BRANCH,main/merge"
        );
    }

    #[test]
    fn trailing_slash_stripped_from_uri() {
        let c = NessieCatalog::new(
            NessieCatalogConfig {
                uri: "http://localhost:19120/api/".into(),
                default_branch: "main".into(),
                warehouse: None,
                auth: RestCatalogAuth::None,
            },
            Arc::new(LocalStore::new("/tmp")),
        );
        assert!(
            !c.trees_url().contains("//v2"),
            "double slash in URL: {}",
            c.trees_url()
        );
    }

    #[test]
    fn list_branches_deserialize() {
        let json = r#"{
            "references": [
                {"type": "BRANCH", "name": "main", "hash": "abc123"},
                {"type": "TAG",    "name": "v1.0", "hash": "def456"}
            ]
        }"#;
        #[derive(Deserialize)]
        struct TreesResp {
            references: Vec<ReferenceItem>,
        }
        #[derive(Deserialize)]
        struct ReferenceItem {
            #[serde(rename = "type")]
            ref_type: String,
            name: String,
            hash: Option<String>,
        }
        let resp: TreesResp = serde_json::from_str(json).unwrap();
        let branches: Vec<_> = resp
            .references
            .into_iter()
            .filter(|r| r.ref_type == "BRANCH")
            .collect();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].name, "main");
        assert_eq!(branches[0].hash.as_deref(), Some("abc123"));
    }
}
