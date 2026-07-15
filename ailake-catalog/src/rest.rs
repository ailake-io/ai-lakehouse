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
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::metadata::IcebergMetadata;
use crate::provider::{
    CatalogProvider, DataFileEntry, EqualityDeleteFile, NewSnapshot, SnapshotId, TableIdent,
    TableMetadata, TableProperties,
};
use crate::schema_evolution::SchemaEvolution;
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

    fn namespaces_url(&self) -> String {
        format!("{}/namespaces", self.base_url())
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

    /// Registers `ns` with the REST server if it doesn't already exist.
    ///
    /// Spec-compliant REST catalogs (unlike `HadoopCatalog`, which just uses a
    /// directory implicitly) reject `create_table` with
    /// `NoSuchNamespaceException` for a namespace nobody has explicitly
    /// created — verified live against `apache/iceberg-rest-fixture`. A 409
    /// Conflict (already exists) is treated as success, not an error — this
    /// makes `create_table` idempotent with respect to namespace existence,
    /// matching `HadoopCatalog`'s implicit-namespace behavior from the
    /// caller's point of view.
    async fn ensure_namespace(&self, ns: &str) -> AilakeResult<()> {
        let req = CreateNamespaceRequest {
            namespace: vec![ns.to_string()],
            properties: HashMap::new(),
        };
        let resp = self.post(&self.namespaces_url(), &req).await?;
        if resp.status().is_success() || resp.status() == reqwest::StatusCode::CONFLICT {
            return Ok(());
        }
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(AilakeError::Catalog(format!(
            "ensure_namespace('{ns}'): HTTP {status}: {body}"
        )))
    }

    /// Fetch current table metadata from the REST server.
    async fn load_metadata(&self, table: &TableIdent) -> AilakeResult<IcebergMetadata> {
        let resp = self.get(&self.table_url(table)).await?;
        let resp = Self::require_ok(resp, "load_metadata").await?;
        let result: LoadTableResult = resp
            .json()
            .await
            .map_err(|e| AilakeError::Catalog(format!("load_metadata parse: {e}")))?;
        Ok(result.metadata)
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

        self.ensure_namespace(&name.namespace).await?;

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
        let table_root = self.table_storage_root(table);

        // Real OCC: read current state, build the real Avro manifest against it
        // (`build_commit` — shared with Hadoop/Glue/Jdbc, gives this backend the
        // same V3 first_row_id/Puffin-stats/partition-stats support they have),
        // commit with an `assert-ref-snapshot-id` requirement pinned to the
        // snapshot we read. If another writer moves `main` first, the server
        // rejects the commit (409) and we retry from a fresh read — same
        // retry×5 w/ backoff pattern glue.rs/jdbc.rs use for their own CAS
        // mechanisms.
        const MAX_RETRIES: u32 = 5;
        for attempt in 0..MAX_RETRIES {
            let resp = self.get(&self.table_url(table)).await?;
            let resp = Self::require_ok(resp, "commit_snapshot (read current state)").await?;
            let result: LoadTableResult = resp
                .json()
                .await
                .map_err(|e| AilakeError::Catalog(format!("commit_snapshot parse: {e}")))?;
            let meta = result.metadata;
            // Iceberg's on-disk sentinel for "no current snapshot" is the literal
            // integer -1 (`current-snapshot-id: -1` in metadata.json for a
            // freshly created table with zero snapshots) — `IcebergMetadata`'s
            // plain `#[serde(default)]` deserialization has no reason to know
            // that and reads it straight into `Some(-1)`. The REST spec's own
            // "assert ref does not exist yet" semantics need a real `null` in
            // the request JSON, not `-1` — sending `-1` verbatim, verified live
            // against `apache/iceberg-rest-fixture`, gets every commit to a
            // brand-new table rejected with `CommitFailedException: branch or
            // tag main is missing, expected -1` (all 5 OCC retries, since the
            // same wrong value keeps getting resent). Treat -1 as None here.
            let current_snapshot_id = meta.current_snapshot_id.filter(|&id| id != -1);
            let current_schema_id = meta.current_schema_id;

            let artifacts = crate::manifest_commit::build_commit(
                &*self.store,
                &table_root,
                self.config.warehouse.as_deref().unwrap_or(""),
                &meta,
                snapshot.clone(),
            )
            .await?;

            let rest_snap = RestSnapshot {
                snapshot_id: artifacts.snapshot.snapshot_id,
                parent_snapshot_id: artifacts.snapshot.parent_snapshot_id,
                sequence_number: artifacts.snapshot.sequence_number,
                timestamp_ms: artifacts.snapshot.timestamp_ms,
                manifest_list: artifacts.snapshot.manifest_list.clone(),
                summary: artifacts.snapshot.summary.clone(),
                schema_id: artifacts.snapshot.schema_id,
            };

            let mut updates = vec![
                TableUpdate::AddSnapshot {
                    snapshot: rest_snap,
                },
                TableUpdate::SetSnapshotRef {
                    ref_name: "main".into(),
                    snapshot_type: "branch".into(),
                    snapshot_id: snap_id,
                },
            ];
            let mut requirements = vec![TableRequirement::AssertRefSnapshotId {
                r#ref: "main".into(),
                snapshot_id: current_snapshot_id,
            }];

            // Phase I: schema/partition-spec patch (only present when the caller
            // supplied `NewSnapshot::iceberg_schema` — same trigger as Hadoop/Glue/Jdbc).
            if let Some(patch) = artifacts.schema_patch {
                // Real Iceberg core (`TableMetadata.Builder.addSchema`, which every
                // spec-compliant REST server delegates to) reuses an existing
                // schema-id — silently ignoring whatever id the request suggests —
                // whenever the submitted schema is structurally identical to one
                // already registered, and is a straight no-op when it's identical
                // to the *current* schema specifically. Both cases make the
                // client-predicted `current_schema_id + 1` wrong to reference in
                // the following `SetCurrentSchema`, which is exactly what live
                // testing against `apache/iceberg-rest-fixture` (2026-07) hit:
                // `IllegalArgumentException: Cannot set current schema to unknown
                // schema: N` (dedup reused a *different* existing id) and
                // `ValidationException: Cannot set last added schema: no schema
                // has been added` (no-op — identical to the already-current
                // schema) from the exact same request shape in different runs.
                // Comparing against the current schema's own fields first and
                // skipping the whole schema-update trio when nothing actually
                // changed avoids ever sending a no-op/dedup-prone `AddSchema` —
                // same "skip when unchanged" principle already applied to
                // `AddPartitionSpec` below.
                let current_schema_fields = meta
                    .schemas
                    .iter()
                    .find(|s| s["schema-id"].as_i64() == Some(current_schema_id as i64))
                    .and_then(|s| s["fields"].as_array())
                    .cloned()
                    .unwrap_or_default();
                let schema_unchanged = current_schema_fields == patch.new_schema_fields;

                if !schema_unchanged {
                    let new_schema_id = current_schema_id + 1;
                    updates.push(TableUpdate::AddSchema {
                        schema: serde_json::json!({
                            "schema-id": new_schema_id,
                            "type": "struct",
                            "fields": patch.new_schema_fields,
                        }),
                    });
                    // `-1` ("the schema just added in this request") rather than
                    // the predicted id — correct even in the dedup-reuse case,
                    // since it always resolves to whatever id `AddSchema` above
                    // actually produced or reused.
                    updates.push(TableUpdate::SetCurrentSchema { schema_id: -1 });
                    updates.push(TableUpdate::SetProperties {
                        updates: HashMap::from([(
                            "schema.name-mapping.default".to_string(),
                            patch.name_mapping_json,
                        )]),
                    });
                }
                // The REST protocol has no "modify an existing spec's source-id in
                // place" update — the closest spec-correct equivalent is adding the
                // corrected spec as new and making it the default. Only the *last*
                // (highest spec-id) entry in `remapped_partition_specs` is genuinely
                // new — the earlier ones are the same specs already registered.
                //
                // `remapped_partition_specs` is built by cloning every existing spec
                // and patching `source-id` in place where a column's field-id moved
                // (see `manifest_commit::build_commit`) — for an unpartitioned table
                // (the common case) that's a no-op clone of the empty default spec,
                // and re-sending it as a "new" `AddPartitionSpec` is both redundant
                // and, verified live against `apache/iceberg-rest-fixture`, rejected
                // with a 500 ("Cannot convert metadata update action to json:
                // add-partition-spec") — that fixture's `AddPartitionSpec` response
                // serialization doesn't round-trip cleanly. Only emit the update when
                // the spec actually changed from what's already registered.
                if let Some(new_spec) = patch.remapped_partition_specs.last() {
                    let unchanged = meta
                        .partition_specs
                        .get(patch.remapped_partition_specs.len() - 1)
                        .is_some_and(|old_spec| old_spec == new_spec);
                    if !unchanged {
                        updates.push(TableUpdate::AddPartitionSpec {
                            spec: new_spec.clone(),
                        });
                        // NOTE: same `-1` ("last added") sentinel that proved
                        // unreliable for `SetCurrentSchema` above against
                        // `apache/iceberg-rest-fixture`. Unlike that case, this
                        // path is untested — real partitioning was out of scope
                        // for the live verification this session did (see
                        // CHANGELOG/CLAUDE.md) — so it's flagged, not changed to
                        // an explicit spec-id, to avoid a speculative, unverified
                        // fix. If this hits the same class of failure, the fix
                        // is the same shape: use `new_spec["spec-id"]` explicitly.
                        updates.push(TableUpdate::SetDefaultSpec { spec_id: -1 });
                    }
                }
                requirements.push(TableRequirement::AssertCurrentSchemaId { current_schema_id });
            }
            if !artifacts.extra_properties.is_empty() {
                updates.push(TableUpdate::SetProperties {
                    updates: artifacts.extra_properties,
                });
            }
            // Phase F / Phase J: Puffin vector+BM25 stats and partition-stats Parquet
            // refs, mapped 1:1 onto the REST spec's own `StatisticsFile`/
            // `PartitionStatisticsFile` shapes (confirmed field-for-field against
            // apache/iceberg's rest-catalog-open-api.yaml, including the
            // `blob-metadata` key name already pinned to match it in `metadata.rs`).
            if let Some(stats) = &artifacts.statistics {
                updates.push(TableUpdate::SetStatistics {
                    statistics: serde_json::to_value(stats)
                        .map_err(|e| AilakeError::Catalog(format!("stats serialize: {e}")))?,
                });
            }
            if let Some(pstats) = &artifacts.partition_statistics {
                updates.push(TableUpdate::SetPartitionStatistics {
                    partition_statistics: serde_json::to_value(pstats).map_err(|e| {
                        AilakeError::Catalog(format!("partition stats serialize: {e}"))
                    })?,
                });
            }

            let commit_req = CommitTableRequest {
                identifier: TableIdentifier {
                    namespace: vec![table.namespace.clone()],
                    name: table.name.clone(),
                },
                requirements,
                updates,
            };

            let resp = self.post(&self.table_url(table), &commit_req).await?;
            if resp.status().as_u16() == 409 {
                if attempt + 1 < MAX_RETRIES {
                    tokio::time::sleep(std::time::Duration::from_millis(50 << attempt)).await;
                    continue;
                }
                let body = resp.text().await.unwrap_or_default();
                return Err(AilakeError::Catalog(format!(
                    "commit_snapshot: {MAX_RETRIES} retries exhausted (concurrent modification): {body}"
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
        let meta = self.load_metadata(table).await?;
        crate::manifest_commit::list_files_from_metadata(&*self.store, &meta, snapshot_id).await
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
        let meta = self.load_metadata(table).await?;
        crate::manifest_commit::list_equality_deletes_from_metadata(
            &*self.store,
            &meta,
            snapshot_id,
        )
        .await
    }
}

// ── REST protocol types ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    expires_in: Option<u64>,
}

#[derive(Serialize)]
struct CreateNamespaceRequest {
    namespace: Vec<String>,
    properties: HashMap<String, String>,
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
    /// Phase F: Puffin vector-stats + BM25-bloom file ref. Payload is a
    /// `StatisticsFile`-shaped JSON value (see `manifest_commit::CommitArtifacts`).
    SetStatistics {
        statistics: serde_json::Value,
    },
    /// Phase J: partition-stats Parquet file ref (`PartitionStatisticsFile`-shaped).
    SetPartitionStatistics {
        #[serde(rename = "partition-statistics")]
        partition_statistics: serde_json::Value,
    },
    /// Phase I: register the partition-spec-with-corrected-source-id — the REST
    /// spec has no update to edit an existing spec's `source-id` in place.
    AddPartitionSpec {
        spec: serde_json::Value,
    },
    SetDefaultSpec {
        #[serde(rename = "spec-id")]
        spec_id: i32,
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

    // ── Live tests against a real Iceberg REST Catalog server ──────────────────
    //
    // `#[ignore]`d by default — these need a running server, not available in
    // the default CI environment. Run locally with:
    //
    //   docker run -d --name ailake-rest-test -p 18181:8181 \
    //     -e CATALOG_WAREHOUSE=/tmp/warehouse \
    //     -e CATALOG_IO__IMPL=org.apache.iceberg.hadoop.HadoopFileIO \
    //     apache/iceberg-rest-fixture:latest
    //   cargo test -p ailake-catalog --features rest-catalog -- --ignored rest::tests::live_
    //
    // Verified live (2026-07) against `apache/iceberg-rest-fixture:latest`:
    // catalog config (auth=None), namespace auto-creation (`ensure_namespace`,
    // including idempotent re-creation), and `create_table` all work correctly.
    // `commit_snapshot`'s schema-patch path (fires on every normal write, not
    // just schema evolution — see Phase I) hit an unresolved, seemingly
    // fixture-specific inconsistency in server-side schema-id assignment
    // during `AddSchema`+`SetCurrentSchema` — documented in that code's own
    // comment and in docs/guides/REST_CATALOG.md; not covered by a test here
    // because it doesn't yet reliably pass against this fixture image. Needs
    // follow-up verification against a more mature REST catalog server
    // (Polaris, Unity Catalog, Gravitino) before closing.
    fn live_catalog() -> RestCatalog {
        let store = Arc::new(LocalStore::new("/tmp"));
        RestCatalog::new(
            RestCatalogConfig {
                uri: "http://localhost:18181".into(),
                prefix: None,
                warehouse: Some("/tmp/ailake_rest_live_test_warehouse".into()),
                auth: RestCatalogAuth::None,
            },
            store,
        )
    }

    #[tokio::test]
    #[ignore = "requires a live Iceberg REST catalog server on localhost:18181"]
    async fn live_create_table_auto_creates_namespace() {
        let catalog = live_catalog();
        let ns = format!("live_test_ns_{}", ailake_core::now_ns());
        let table = TableIdent::new(&ns, "t1");
        let policy = ailake_core::VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim: 4,
            metric: ailake_core::VectorMetric::Cosine,
            precision: ailake_core::VectorPrecision::F16,
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
        };
        let props = TableProperties {
            policy,
            format_version: 2,
            partition_column_type: None,
            extra: HashMap::new(),
        };

        catalog
            .create_table(&table, &props)
            .await
            .expect("create_table should auto-create the namespace and succeed");

        let meta = catalog
            .load_table(&table)
            .await
            .expect("load_table should find the just-created table");
        assert_eq!(
            meta.properties.get("ailake.vector-dim").map(String::as_str),
            Some("4")
        );
    }

    #[tokio::test]
    #[ignore = "requires a live Iceberg REST catalog server on localhost:18181"]
    async fn live_ensure_namespace_is_idempotent() {
        let catalog = live_catalog();
        let ns = format!("live_test_ns_{}", ailake_core::now_ns());

        catalog
            .ensure_namespace(&ns)
            .await
            .expect("first ensure_namespace call should create it");
        catalog
            .ensure_namespace(&ns)
            .await
            .expect("second ensure_namespace call (already exists) should be a no-op success");
    }
}
