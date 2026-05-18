# CATALOG_BACKENDS.md — Catalog Backend Implementation Guide

## Overview

`ailake-catalog` implements the `CatalogProvider` trait for every supported Iceberg catalog. All backend-specific logic is confined to this crate. The rest of the SDK uses only `dyn CatalogProvider`.

---

## `CatalogProvider` trait contract

```rust
// ailake-catalog/src/lib.rs

use async_trait::async_trait;

#[async_trait]
pub trait CatalogProvider: Send + Sync {
    /// Returns current table metadata (schema, properties, current snapshot id).
    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata>;

    /// Creates a new table. Fails if table already exists.
    async fn create_table(
        &self,
        name: &TableIdent,
        schema: &Schema,
        props: &TableProperties,
    ) -> AilakeResult<()>;

    /// Returns the DataFile entries for the given snapshot (or current if None).
    /// Includes custom_properties for each file (centroid, radius, HNSW offsets).
    async fn list_files(
        &self,
        name: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>>;

    /// Atomically appends new DataFiles and creates a new snapshot.
    /// This is the APPEND path (used by write_batch).
    async fn append_files(
        &self,
        name: &TableIdent,
        new_files: Vec<DataFileEntry>,
    ) -> AilakeResult<SnapshotId>;

    /// Atomically replaces old DataFiles with new ones (used by compaction).
    async fn replace_files(
        &self,
        name: &TableIdent,
        old_files: Vec<DataFileEntry>,
        new_files: Vec<DataFileEntry>,
    ) -> AilakeResult<SnapshotId>;

    /// Returns files reachable from snapshots older than retention_ms.
    /// Used by vacuum to identify files safe to delete.
    async fn expired_files(
        &self,
        name: &TableIdent,
        retention_ms: u64,
    ) -> AilakeResult<Vec<String>>;
}

/// A fully-qualified table name: catalog.database.table
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableIdent {
    pub namespace: Vec<String>,  // e.g. ["my_database"] or ["db", "schema"]
    pub name: String,
}

/// One data file entry from an Iceberg manifest, enriched with AI-Lake custom_properties
#[derive(Debug, Clone)]
pub struct DataFileEntry {
    pub file_path: String,
    pub record_count: u64,
    pub file_size_bytes: u64,
    pub partition: serde_json::Value,
    // AI-Lake extensions (from custom_properties)
    pub centroid: Option<Vec<f32>>,
    pub radius: Option<f32>,
    pub hnsw_offset: Option<u64>,
    pub hnsw_len: Option<u64>,
}
```

---

## Backend: `HadoopCatalog` (filesystem)

Used for: local development, CI, HDFS, and any object storage where the catalog lives directly in the filesystem as `metadata.json` files.

No external service required.

```rust
// ailake-catalog/src/hadoop.rs

pub struct HadoopCatalog {
    warehouse: String,       // local path or s3://... or gs://...
    store: Arc<dyn Store>,
}

impl HadoopCatalog {
    pub fn new(warehouse: &str, store: Arc<dyn Store>) -> Self { ... }
}
```

### How it works

- Table location: `{warehouse}/{namespace}/{table_name}/`
- Metadata: `{table_location}/metadata/v{N}.metadata.json`
- Current version tracked by `{table_location}/metadata/version-hint.text` (single file with the current version number)
- Commit: write `v{N+1}.metadata.json`, then atomically overwrite `version-hint.text`

### Limitations

- No true atomic commit on S3 (S3 `PUT` is atomic per-object, but `version-hint.text` update is not atomic with metadata file write — race condition on concurrent writers). Use `RestCatalog` or `GlueCatalog` for multi-writer production.
- Suitable for single-writer workloads and all Phase 1 development.

### Configuration

```rust
let catalog = HadoopCatalog::new(
    "/tmp/warehouse",
    Arc::new(LocalStore::new("/tmp/warehouse"))
);

// For S3:
let catalog = HadoopCatalog::new(
    "s3://my-bucket/warehouse",
    Arc::new(S3Store::new(s3_config))
);
```

---

## Backend: `RestCatalog`

Used for: Apache Polaris, Project Nessie (via REST API), Unity Catalog, AWS S3 Tables, BigLake Metastore, Tabular — any server implementing the [Iceberg REST Catalog spec](https://github.com/apache/iceberg/blob/main/open-api/rest-catalog-open-api.yaml).

```rust
// ailake-catalog/src/rest.rs

pub struct RestCatalog {
    base_uri: String,
    warehouse: String,
    http: reqwest::Client,
    auth: RestAuth,
}

pub enum RestAuth {
    None,
    Bearer { token: String },
    OAuth2 {
        server_uri: String,
        credential: String,   // "client_id:client_secret"
        scope: String,
    },
}
```

### REST endpoints used

| Operation | Method | Path |
|---|---|---|
| Load table | GET | `/v1/{prefix}/namespaces/{ns}/tables/{table}` |
| Create table | POST | `/v1/{prefix}/namespaces/{ns}/tables` |
| Commit transaction | POST | `/v1/{prefix}/namespaces/{ns}/tables/{table}` |
| List tables | GET | `/v1/{prefix}/namespaces/{ns}/tables` |

The commit endpoint uses the Iceberg `UpdateTableRequest` with `requirements` (for optimistic concurrency) and `updates` (list of `TableUpdate` operations).

### Commit with vector stats

AI-Lake stores per-file vector stats by adding a `SetPropertiesUpdate` to the commit that sets `custom-properties` on each new DataFile. The REST catalog server stores this in its metadata store transparently.

```json
{
  "identifier": {"namespace": ["db"], "name": "my_table"},
  "requirements": [{"type": "assert-current-snapshot-id", "snapshot-id": 1234}],
  "updates": [
    {
      "action": "add-snapshot",
      "snapshot": {
        "snapshot-id": 5678,
        "parent-snapshot-id": 1234,
        "manifest-list": "s3://bucket/table/metadata/snap-5678.avro",
        "summary": {"operation": "append"}
      }
    },
    {
      "action": "set-snapshot-ref",
      "ref-name": "main",
      "type": "branch",
      "snapshot-id": 5678
    }
  ]
}
```

The manifest Avro is written by `ailake-catalog` directly to object storage before calling the REST commit. The REST server only needs to update the metadata pointer.

### Configuration

```rust
let catalog = RestCatalog::new(RestCatalogConfig {
    uri: "https://my-catalog.example.com".to_string(),
    warehouse: "my_warehouse".to_string(),
    prefix: None,    // optional URL prefix
    auth: RestAuth::OAuth2 {
        server_uri: "https://auth.example.com/token".to_string(),
        credential: "my_client_id:my_client_secret".to_string(),
        scope: "PRINCIPAL_ROLE:ALL".to_string(),
    },
});
```

### Provider-specific notes

**Apache Polaris (ASF)**:
- OAuth2 auth required
- `scope = "PRINCIPAL_ROLE:ALL"` or role-specific

**Project Nessie**:
- Uses standard REST spec, adds `X-Iceberg-Access-Delegation` header for S3 credentials
- See `NessieCatalog` (below) for branching extensions

**Unity Catalog (Databricks)**:
- OAuth2 via Databricks token
- `warehouse = <unity_catalog_external_location>`

**AWS S3 Tables**:
- Uses IAM auth (AWS SigV4) — handled by `reqwest` middleware
- `uri = https://s3tables.<region>.amazonaws.com/iceberg`
- `warehouse = arn:aws:s3tables:<region>:<account>:bucket/<bucket_name>`

**BigLake Metastore (GCP)**:
- OAuth2 via Google service account
- `uri = https://biglake.googleapis.com/iceberg/v1beta`

---

## Backend: `NessieCatalog`

Extends `RestCatalog` with Nessie-specific branch and tag operations. Nessie implements the Iceberg REST spec fully, so `NessieCatalog` internally wraps `RestCatalog` and adds branch management.

```rust
// ailake-catalog/src/nessie.rs

pub struct NessieCatalog {
    inner: RestCatalog,   // delegates all Iceberg ops to RestCatalog
    ref_name: String,     // current branch/tag
    nessie_uri: String,
    http: reqwest::Client,
}

impl NessieCatalog {
    pub async fn create_branch(&self, new_branch: &str, from: &str) -> AilakeResult<()>;
    pub async fn merge_branch(&self, source: &str, target: &str) -> AilakeResult<()>;
    pub async fn list_branches(&self) -> AilakeResult<Vec<String>>;
    pub async fn delete_branch(&self, branch: &str) -> AilakeResult<()>;
}
```

`NessieCatalog` forwards all `CatalogProvider` calls to `inner` (which uses the correct Nessie REST endpoint with the `ref` header set). The branch operations use the Nessie native API (`/api/v2/trees`).

### Configuration

```rust
let catalog = NessieCatalog::new(NessieCatalogConfig {
    nessie_uri: "http://localhost:19120".to_string(),
    ref_name: "main".to_string(),
    auth: NessieAuth::None,
    warehouse: "s3://my-bucket/warehouse".to_string(),
    s3_store: Arc::new(S3Store::new(s3_config)),
});
```

---

## Backend: `GlueCatalog`

Used for: AWS-native deployments. Stores Iceberg metadata in AWS Glue Data Catalog. Tables are visible in Athena, EMR, Redshift Spectrum, and Glue ETL without additional configuration.

```rust
// ailake-catalog/src/glue.rs

pub struct GlueCatalog {
    database: String,
    warehouse: String,
    glue: aws_sdk_glue::Client,
    s3: Arc<dyn Store>,
}
```

### How Glue stores Iceberg metadata

Glue stores Iceberg tables with:
- Table type: `ICEBERG`
- Table property `metadata_location`: points to the current `metadata.json` on S3
- Schema registered in Glue (for Athena compatibility)

The actual `metadata.json`, manifest Avro files, and Parquet data files all live in S3. Glue only stores the pointer.

Commit sequence:
1. Write new `v{N+1}.metadata.json` to S3
2. Call `glue.update_table(metadata_location = new_metadata_path)` — atomic from Glue's perspective
3. Glue handles concurrent write detection via its optimistic concurrency (ETag-based)

### Required IAM permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "glue:GetTable",
    "glue:CreateTable",
    "glue:UpdateTable",
    "glue:GetDatabase",
    "glue:CreateDatabase"
  ],
  "Resource": [
    "arn:aws:glue:*:*:catalog",
    "arn:aws:glue:*:*:database/my_database",
    "arn:aws:glue:*:*:table/my_database/*"
  ]
}
```

Plus S3 permissions on the warehouse bucket.

### Configuration

```rust
// Credentials from environment (recommended for EC2/ECS/Lambda)
let glue_client = aws_sdk_glue::Client::new(
    &aws_config::load_from_env().await
);

let catalog = GlueCatalog::new(GlueCatalogConfig {
    database: "my_database".to_string(),
    warehouse: "s3://my-bucket/warehouse".to_string(),
    glue: glue_client,
    s3: Arc::new(S3Store::new(s3_config)),
});
```

---

## Backend: `JdbcCatalog`

Used for: self-hosted deployments where a PostgreSQL or MySQL database is already available. Stores Iceberg metadata pointers in a relational table.

```rust
// ailake-catalog/src/jdbc.rs

pub struct JdbcCatalog {
    pool: sqlx::AnyPool,    // PostgreSQL or MySQL
    warehouse: String,
    store: Arc<dyn Store>,
}
```

### Database schema

```sql
-- Created automatically on first use
CREATE TABLE IF NOT EXISTS iceberg_tables (
    catalog_name   VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name     VARCHAR(255) NOT NULL,
    metadata_location VARCHAR(1000) NOT NULL,
    previous_metadata_location VARCHAR(1000),
    PRIMARY KEY (catalog_name, table_namespace, table_name)
);
```

### Commit (optimistic concurrency via SQL)

```sql
-- Atomic commit: only succeeds if metadata_location hasn't changed since we read it
UPDATE iceberg_tables
SET metadata_location = $new_location,
    previous_metadata_location = $old_location
WHERE catalog_name = $catalog
  AND table_namespace = $namespace
  AND table_name = $table
  AND metadata_location = $old_location;
-- If rows_affected == 0 → another writer committed first → retry with new metadata
```

### Configuration

```rust
let pool = sqlx::PgPool::connect("postgresql://user:password@localhost/mydb").await?;

let catalog = JdbcCatalog::new(JdbcCatalogConfig {
    pool,
    catalog_name: "ailake".to_string(),
    warehouse: "s3://my-bucket/warehouse".to_string(),
    store: Arc::new(S3Store::new(s3_config)),
});
```

---

## Selecting a catalog at runtime

The `ailake-py` Python bindings expose a unified config-based selection:

```python
import ailake

# Filesystem (local dev, no external service)
catalog = ailake.HadoopCatalog(warehouse="/tmp/warehouse")

# REST (Polaris, Nessie, Unity Catalog, BigLake, S3 Tables)
catalog = ailake.RestCatalog(
    uri="https://my-catalog.example.com",
    warehouse="my_warehouse",
    credential="client_id:client_secret",    # optional OAuth2
    scope="PRINCIPAL_ROLE:ALL"               # optional
)

# AWS Glue
catalog = ailake.GlueCatalog(
    database="my_database",
    warehouse="s3://my-bucket/warehouse",
    region="us-east-1"                       # uses env credentials
)

# JDBC (PostgreSQL)
catalog = ailake.JdbcCatalog(
    connection_string="postgresql://user:password@localhost/mydb",
    warehouse="s3://my-bucket/warehouse"
)

# Use with TableWriter
writer = ailake.TableWriter(
    table_name="db.my_table",
    catalog=catalog,
    storage=ailake.S3Store(bucket="my-bucket", region="us-east-1")
)
```

The Rust SDK mirrors this pattern via the `CatalogProvider` trait — pass `Arc<dyn CatalogProvider>` to any `TableWriter` or `VectorScanner`.

---

## Testing catalog backends

Each backend has an integration test that requires the corresponding service running (via Docker):

```
tests/compat/
├── hadoop_catalog.rs     # Phase 1 — no Docker needed (local FS)
├── rest_nessie.rs        # Phase 2 — requires Nessie Docker
├── rest_polaris.rs       # Phase 2 — requires Polaris Docker
├── glue_catalog.rs       # Phase 2 — requires Localstack Docker
└── jdbc_catalog.rs       # Phase 2 — requires PostgreSQL Docker
```

All integration tests follow the same script:

1. Create a table
2. Write 3 batches
3. List files — assert 3 DataFileEntry items with correct custom_properties
4. Write a 4th batch
5. Replace 2 old files with 1 compacted file
6. List files — assert 3 items (2 old replaced by 1 + 2 remaining)
7. Assert snapshot history shows 5 snapshots (create + 3 writes + 1 compact)
