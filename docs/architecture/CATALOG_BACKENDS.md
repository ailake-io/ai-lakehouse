# CATALOG_BACKENDS.md — Catalog Backend Implementation Guide

## Overview

`ailake-catalog` implements `CatalogProvider` for every supported Iceberg catalog. All backend logic is confined to this crate. The rest of the SDK uses only `Arc<dyn CatalogProvider>`.

---

## `CatalogProvider` trait contract

```rust
// ailake-catalog/src/provider.rs

#[async_trait]
pub trait CatalogProvider: Send + Sync {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()>;
    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata>;
    async fn commit_snapshot(&self, table: &TableIdent, snapshot: NewSnapshot) -> AilakeResult<SnapshotId>;
    async fn list_files(&self, table: &TableIdent, snapshot_id: Option<SnapshotId>) -> AilakeResult<Vec<DataFileEntry>>;
    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()>;
}
```

Key types:

```rust
pub struct TableIdent {
    pub namespace: String,   // e.g. "default" or "my_schema"
    pub name: String,
}

pub struct DataFileEntry {
    pub path: String,
    pub record_count: u64,
    pub file_size_bytes: u64,
    pub centroid_b64: Option<String>,   // base64-encoded F32 centroid
    pub radius: Option<f32>,
    pub hnsw_offset: Option<u64>,
    pub hnsw_len: Option<u64>,
    pub vector_column: Option<String>,
    pub vector_dim: Option<u32>,
}

pub struct NewSnapshot {
    pub snapshot_id: SnapshotId,
    pub parent_snapshot_id: Option<SnapshotId>,
    pub files: Vec<DataFileEntry>,
    pub operation: SnapshotOperation,   // Append | Overwrite | Delete | Replace
}
```

All vector statistics (centroid, radius, HNSW offsets) live in `DataFileEntry` fields, which are stored in `custom_properties` of each Iceberg DataFile entry — a spec-defined extension point ignored by unknown readers.

---

## Crate layout

```
ailake-catalog/src/
├── lib.rs          # re-exports, module declarations
├── provider.rs     # CatalogProvider trait, TableIdent, DataFileEntry, NewSnapshot
├── metadata.rs     # metadata.json read/write (Iceberg Spec v2)
├── snapshot.rs     # manifest JSON builder (Phase 1/2 — JSON, not Avro)
├── hadoop.rs       # HadoopCatalog — filesystem / any Store backend
├── rest.rs         # RestCatalog — Iceberg REST Catalog spec
├── databricks.rs   # DatabricksAuth + convenience builders for Azure/AWS/GCP
├── glue.rs         # GlueCatalog — AWS Glue (feature = "catalog-glue")
├── nessie.rs       # NessieCatalog — Nessie branching extensions (feature = "catalog-nessie")
└── jdbc.rs         # JdbcCatalog — PostgreSQL/MySQL (feature = "catalog-jdbc")
```

---

## Backend: `HadoopCatalog` (filesystem)

No external service required. Suitable for local dev, CI, and single-writer S3/GCS/Azure deployments.

```rust
pub struct HadoopCatalog {
    store: Arc<dyn Store>,
    warehouse: String,
}

impl HadoopCatalog {
    pub fn new(store: Arc<dyn Store>, warehouse: &str) -> Self
}
```

**Table layout**:
```
{warehouse}/{namespace}.db/{table}/
  metadata/
    current.json          ← IcebergMetadata (replaces metadata/vN.metadata.json for simplicity)
    snap-{id}.json        ← manifest JSON (list of DataFileEntry)
  data/
    part-NNNNN.parquet    ← Parquet + AILK footer
```

**Commit**: overwrites `current.json` atomically per-object (S3 `PUT` semantics). Safe for single-writer workloads.

```rust
// Local dev
let store = Arc::new(LocalStore::new("/tmp/warehouse"));
let catalog = HadoopCatalog::new(store, "/tmp/warehouse");

// S3
let s3 = AmazonS3Builder::new()
    .with_bucket_name("my-bucket")
    .with_region("us-east-1")
    .build()?;
let store = Arc::new(ObjectStoreBackend::new(Arc::new(s3), "warehouse/"));
let catalog = HadoopCatalog::new(store, "s3://my-bucket/warehouse");

// Azure Blob
let azure = MicrosoftAzureBuilder::new()
    .with_account("myaccount")
    .with_container("mycontainer")
    .build()?;
let store = Arc::new(ObjectStoreBackend::new(Arc::new(azure), "warehouse/"));
let catalog = HadoopCatalog::new(store, "abfss://mycontainer@myaccount.dfs.core.windows.net/warehouse");
```

---

## Backend: `RestCatalog`

Implements the [Iceberg REST Catalog spec](https://iceberg.apache.org/spec/#rest-catalog). Works with:
- Apache Polaris (ASF)
- AWS S3 Tables
- GCP BigLake Metastore
- Azure Databricks Unity Catalog
- Project Nessie (REST mode)
- Gravitino
- Any spec-compliant server

```rust
pub struct RestCatalog {
    config: RestCatalogConfig,
    client: reqwest::Client,
    token_cache: Mutex<Option<CachedToken>>,
    store: Arc<dyn Store>,   // for writing manifest files
}

pub struct RestCatalogConfig {
    pub uri: String,              // base URL (no trailing slash)
    pub prefix: Option<String>,   // path prefix between /v1 and /namespaces
    pub warehouse: Option<String>, // storage root for new tables
    pub auth: RestCatalogAuth,
}

pub enum RestCatalogAuth {
    None,
    Bearer(String),
    OAuth2 {
        token_endpoint: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    },
}
```

**URL layout**: `{uri}/v1/{prefix}/namespaces/{namespace}/tables/{table}`

**REST operations used**:

| Operation | Method | Path |
|---|---|---|
| `create_table` | POST | `/v1/{prefix}/namespaces/{ns}/tables` |
| `load_table` | GET | `/v1/{prefix}/namespaces/{ns}/tables/{table}` |
| `commit_snapshot` | POST | `/v1/{prefix}/namespaces/{ns}/tables/{table}` |
| `drop_table` | DELETE | `/v1/{prefix}/namespaces/{ns}/tables/{table}` |

**Commit payload** (`CommitTableRequest`):
```json
{
  "identifier": {"namespace": ["default"], "name": "my_table"},
  "requirements": [],
  "updates": [
    {
      "action": "add-snapshot",
      "snapshot": {
        "snapshot-id": 1234,
        "timestamp-ms": 1700000000000,
        "manifest-list": "s3://bucket/warehouse/default/my_table/metadata/snap-1234.json",
        "summary": {"operation": "append", "added-data-files": "1"},
        "schema-id": 0
      }
    },
    {
      "action": "set-snapshot-ref",
      "ref-name": "main",
      "type": "branch",
      "snapshot-id": 1234
    }
  ]
}
```

Manifests (JSON, not Avro) are written to object storage before the REST commit. The REST server only updates the metadata pointer.

**OAuth2 token caching**: tokens are cached until `expires_in - 30s` to avoid clock-edge failures. Thread-safe via `tokio::sync::Mutex`.

### Generic REST configuration

```rust
// Apache Polaris
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://my-account.snowflakecomputing.com/polaris/api/catalog".into(),
        prefix: Some("my_polaris_catalog".into()),
        warehouse: Some("s3://my-bucket/warehouse".into()),
        auth: RestCatalogAuth::OAuth2 {
            token_endpoint: "https://my-account.snowflakecomputing.com/polaris/api/catalog/v1/oauth/tokens".into(),
            client_id: "my_client_id".into(),
            client_secret: "my_client_secret".into(),
            scope: Some("PRINCIPAL_ROLE:ALL".into()),
        },
    },
    store,
);

// Nessie (no auth, local dev)
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "http://localhost:19120/api".into(),
        prefix: Some("main".into()),
        warehouse: Some("s3://my-bucket/warehouse".into()),
        auth: RestCatalogAuth::None,
    },
    store,
);
```

---

## Databricks Unity Catalog

`ailake-catalog/src/databricks.rs` provides convenience builders that wire up the correct URI, prefix, and auth for each Databricks cloud. Internally creates a `RestCatalogConfig`.

```
TableIdent { namespace: "my_schema", name: "my_table" }
→ https://{workspace}/api/2.1/unity-catalog/iceberg/v1/{unity_catalog}/namespaces/my_schema/tables/my_table
```

### Auth variants

```rust
pub enum DatabricksAuth {
    Pat(String),                        // Personal Access Token — all clouds, dev/CI
    AzureServicePrincipal {             // Azure AD service principal
        tenant_id: String,
        client_id: String,
        client_secret: String,
    },
    AwsOAuth2 {                         // Databricks M2M OAuth (AWS)
        client_id: String,
        client_secret: String,
    },
    GcpBearer(String),                  // pre-obtained GCP/Databricks access token
}
```

### Azure (Unity Catalog)

Token endpoint: `https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token`
Scope: `2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default` (Databricks resource in Azure AD)

```rust
use ailake_catalog::{databricks_azure, DatabricksAuth, RestCatalog};

let catalog = RestCatalog::new(
    databricks_azure(
        "myworkspace.azuredatabricks.net",
        "my_unity_catalog",            // catalog name = REST prefix
        "abfss://container@account.dfs.core.windows.net/warehouse",
        DatabricksAuth::AzureServicePrincipal {
            tenant_id: "00000000-0000-0000-0000-000000000000".into(),
            client_id: "app-client-id".into(),
            client_secret: "app-client-secret".into(),
        },
    ),
    azure_store,
);

// Or with PAT (dev)
let catalog = RestCatalog::new(
    databricks_azure(
        "myworkspace.azuredatabricks.net",
        "my_unity_catalog",
        "abfss://container@account.dfs.core.windows.net/warehouse",
        DatabricksAuth::Pat("dapi...".into()),
    ),
    azure_store,
);
```

### AWS (Unity Catalog)

Token endpoint: `https://{workspace_host}/oidc/v1/token`, scope `all-apis`

```rust
let catalog = RestCatalog::new(
    databricks_aws(
        "myworkspace.cloud.databricks.com",
        "my_unity_catalog",
        "s3://my-bucket/warehouse",
        DatabricksAuth::AwsOAuth2 {
            client_id: "sp-client-id".into(),
            client_secret: "sp-client-secret".into(),
        },
    ),
    s3_store,
);
```

### GCP (Unity Catalog)

```rust
// Obtain token via Workload Identity or gcloud:
// gcloud auth print-access-token
let catalog = RestCatalog::new(
    databricks_gcp(
        "myworkspace.gcp.databricks.com",
        "my_unity_catalog",
        "gs://my-bucket/warehouse",
        DatabricksAuth::GcpBearer(access_token),
    ),
    gcs_store,
);
```

### Namespace model for Unity Catalog

Unity Catalog has a 3-level hierarchy: `catalog.schema.table`.

- `RestCatalogConfig.prefix` → Unity Catalog name (e.g. `"main"`)
- `TableIdent.namespace` → schema name (e.g. `"my_schema"`)
- `TableIdent.name` → table name

```rust
// Table: main.prod_schema.embeddings
let table = TableIdent::new("prod_schema", "embeddings");
// catalog prefix = "main" (set in databricks_azure/aws/gcp)
```

---

## Backend: `GlueCatalog` (feature = `catalog-glue`)

AWS-native. Stores Iceberg metadata pointers in Glue Data Catalog. Tables are visible in Athena, EMR, Redshift Spectrum, and Glue ETL.

Stub implementation — Phase 3. Enable with:
```toml
ailake-catalog = { path = "...", features = ["catalog-glue"] }
```

---

## Backend: `NessieCatalog` (feature = `catalog-nessie`)

Extends `RestCatalog` with Nessie-specific branching operations. Stub — Phase 3.

```toml
ailake-catalog = { path = "...", features = ["catalog-nessie"] }
```

---

## Backend: `JdbcCatalog` (feature = `catalog-jdbc`)

Self-hosted PostgreSQL/MySQL. Stub — Phase 3.

```toml
ailake-catalog = { path = "...", features = ["catalog-jdbc"] }
```

---

## Selecting a catalog at runtime

The `ailake-query` layer depends only on `Arc<dyn CatalogProvider>`. Pass any backend:

```rust
// Local dev
let catalog: Arc<dyn CatalogProvider> = Arc::new(
    HadoopCatalog::new(local_store, "/tmp/warehouse")
);

// REST (Polaris / Nessie / S3 Tables)
let catalog: Arc<dyn CatalogProvider> = Arc::new(
    RestCatalog::new(rest_config, store)
);

// Databricks Unity Catalog
let catalog: Arc<dyn CatalogProvider> = Arc::new(
    RestCatalog::new(databricks_azure(...), azure_store)
);

// Same search() call regardless of catalog backend
let results = search(&table, &query, config, "embedding", dim, catalog, store).await?;
```

---

## Phase status

| Backend | Status | Phase |
|---|---|---|
| `HadoopCatalog` | ✅ Implemented | 1 |
| `RestCatalog` | ✅ Implemented | 2 |
| Databricks helpers | ✅ Implemented | 2 |
| `GlueCatalog` | Stub (compile-only) | 3 |
| `NessieCatalog` | Stub (compile-only) | 3 |
| `JdbcCatalog` | Stub (compile-only) | 3 |

---

## Testing catalog backends

```bash
# HadoopCatalog — no external service needed
cargo test -p ailake-catalog

# RestCatalog — requires a running REST catalog server
# Local Nessie:
docker run -p 19120:19120 ghcr.io/projectnessie/nessie:latest
cargo test -p tests --test rest_nessie -- --ignored

# Local Polaris:
docker run -p 8181:8181 apache/polaris:latest
cargo test -p tests --test rest_polaris -- --ignored
```

Integration test pattern (same for all REST-based backends):
1. Create table
2. Write 2 batches → assert 2 `DataFileEntry` with centroid/radius
3. Search with pruning → assert correct file pruned
4. Drop table
