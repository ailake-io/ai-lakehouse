# INTEGRATIONS.md — Engine and Cloud Provider Compatibility

## Overview

AI-Lake tables are read-compatible with any engine that supports Apache Iceberg without any plugin. This document covers:

1. **Tabular compatibility** (read/write without AI-Lake plugin) — works today, zero effort
2. **Vector-scan compatibility** (search via AI-Lake plugin) — requires the `ailake-jni` connector
3. **Cloud provider setup** — catalog and storage configuration per cloud
4. **Engine-specific notes** — version requirements, known limitations, configuration snippets

---

## Compatibility matrix

| Engine / Platform | Tabular read | Tabular write | Vector scan | Streaming ingest |
|---|---|---|---|---|
| **Apache Spark 3.5** | ✅ Native Iceberg | ✅ Native Iceberg | ✅ `spark-plugin/` | ✅ Structured Streaming |
| **Apache Spark 4.0** | ✅ Native Iceberg | ✅ Native Iceberg | ✅ `spark-plugin/` (untested) | ✅ Structured Streaming |
| **Trino 430** | ✅ Native Iceberg | ✅ Native Iceberg | ✅ `trino-plugin/` | — |
| **Apache Flink 1.18+** | ✅ Iceberg connector | ✅ `ailake-flink` sink | ✅ `ailake-flink` source | ✅ `AilakeSinkFunction` |
| **Apache Beam 2.56+** | ✅ Managed IcebergIO | ✅ Managed IcebergIO | via SDK direct | ✅ Streaming read/write |
| **DuckDB 0.10+** | ✅ Iceberg extension | Read-only | — | — |
| **PyIceberg 0.6+** | ✅ | ✅ | via SDK direct | — |
| **AWS Athena** | ✅ Glue catalog | Limited | — | — |
| **AWS EMR** | ✅ Spark/Trino on EMR | ✅ | Phase 3 | ✅ |
| **AWS Glue ETL** | ✅ | ✅ | via SDK direct | ✅ |
| **Azure Synapse** | ✅ Spark pool | ✅ | Phase 3 | ✅ |
| **Azure Databricks** | ✅ | ✅ | Phase 3 | ✅ |
| **GCP Dataproc** | ✅ Spark/Trino | ✅ | Phase 3 | ✅ |
| **GCP Dataflow** | ✅ Beam IcebergIO | ✅ Beam IcebergIO | via SDK direct | ✅ |
| **Snowflake** | ✅ Iceberg tables | Limited | — | — |
| **Databricks (general)** | ✅ | ✅ | Phase 3 | ✅ |
| **Python (`ailake-py`)** | ✅ PyArrow | ✅ `open_table` + `Table.insert` | ✅ `SearchQuery` fluent chain | ✅ `write_batch_idempotent`, async API |
| **Go (`ailake-go`)** | ✅ AilakeReader | ✅ AilakeWriter | ✅ VectorSearch | — |
| **C++17 (`ailake-cpp`)** | ✅ header-only | ✅ header-only | ✅ header-only | — |

**SDK direct** = use the `ailake-py` Python SDK, `ailake-go` Go SDK, `ailake-cpp` C++ SDK, or `ailake-jni` JVM SDK to run vector search directly, outside of the engine's SQL planner.

---

## 1. Apache Spark

### What works today (no AI-Lake plugin)

Spark reads AI-Lake tables as standard Iceberg tables. The vector column appears as `BinaryType`. SQL analytics work on all non-vector columns.

### Version support

| Spark version | Iceberg runtime jar | Scala |
|---|---|---|
| 3.3 | `iceberg-spark-runtime-3.3_2.12` | 2.12 |
| 3.4 | `iceberg-spark-runtime-3.4_2.12` | 2.12 |
| 3.5 | `iceberg-spark-runtime-3.5_2.12` | 2.12 / 2.13 |
| 4.0 | `iceberg-spark-runtime-4.0_2.13` | 2.13 |

Always use the runtime jar only — do not include `iceberg-core` or `iceberg-parquet` in the uberjar as they cause dependency conflicts.

### Configuration snippet (Spark 3.5, S3, Glue catalog)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ailake-app") \
    .config("spark.jars.packages",
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0,"
        "org.apache.iceberg:iceberg-aws-bundle:1.5.0") \
    .config("spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.glue_catalog",
        "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.glue_catalog.catalog-impl",
        "org.apache.iceberg.aws.glue.GlueCatalog") \
    .config("spark.sql.catalog.glue_catalog.warehouse",
        "s3://my-bucket/warehouse") \
    .config("spark.sql.catalog.glue_catalog.io-impl",
        "org.apache.iceberg.aws.s3.S3FileIO") \
    .getOrCreate()

# Read AI-Lake table (tabular, no plugin)
df = spark.read.format("iceberg").load("glue_catalog.db.my_ailake_table")
df.filter("category = 'finance'").show()

# The embedding column is BinaryType — valid SQL, not vector-searchable here
df.select("chunk_text", "embedding").printSchema()
```

### Phase 3: Vector-scan plugin (Spark) ✅ Implemented

Source: `spark-plugin/` · Build: `cd spark-plugin && ./gradlew shadowJar`

Key classes:
- `io.ailake.spark.AilakeSparkExtensions` — Spark extensions entry point
- `io.ailake.spark.VectorScanStrategy` — Catalyst `SparkStrategy` (converts `VectorSearchPlan` → `VectorScanExec`)
- `io.ailake.spark.VectorScanExec` — physical `LeafExecNode`; calls `libailake_jni.so` via JNA
- `io.ailake.spark.implicits.AilakeSession` — implicit `spark.ailakeSearch(uri, query, topK)`

```scala
import io.ailake.spark.implicits._

val spark = SparkSession.builder()
  .config("spark.jars", "/path/to/spark-plugin-0.1.0-plugin.jar")
  .config("spark.sql.extensions",
    "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions," +
    "io.ailake.spark.AilakeSparkExtensions")
  .config("spark.driver.extraJavaOptions",
    "-Djava.library.path=/path/to/target/release")
  .getOrCreate()

// Returns DataFrame: row_id (Long), distance (Double), file_path (String)
val results = spark.ailakeSearch(
  tableUri    = "s3://my-lake/docs/",
  queryVector = Array(0.021f, -0.043f, 0.118f /* ... 1536 dims */),
  topK        = 100,
)
results.orderBy("distance").show(10)
```

See `SETUP.md §16` for a complete walkthrough including demo table generation and cluster submission.

### Structured Streaming (ingest)

AI-Lake tables accept streaming writes via Iceberg's standard streaming sink. The HNSW index is built on compaction (separate job), not inline during streaming:

```python
# Streaming ingest: data lands in Parquet, HNSW built on compaction
query = df_stream \
    .writeStream \
    .format("iceberg") \
    .outputMode("append") \
    .option("path", "glue_catalog.db.my_ailake_table") \
    .option("checkpointLocation", "s3://my-bucket/checkpoints/") \
    .start()

# Separately, run the compaction job periodically:
spark.sql("CALL glue_catalog.system.rewrite_data_files("
          "table => 'db.my_ailake_table')")
# After rewrite_data_files, run AI-Lake HNSW rebuild:
ailake_compact("s3://my-bucket/warehouse/db/my_ailake_table/")
```

**Important**: during streaming, new files produced by the stream writer have no HNSW index in their AI-Lake footer. They are still readable as standard Iceberg files. Vector search returns only chunks from files that have been compacted (with an index). Run compaction frequently to minimize the "blind window."

---

## 2. Apache Trino

### What works today (no AI-Lake plugin)

Trino reads AI-Lake tables as standard Iceberg. The vector column is `VARBINARY`. Full SQL support on non-vector columns.

### Version support

Trino embeds Iceberg directly (not as a runtime jar). Supported versions:

| Trino version | Iceberg spec | Notes |
|---|---|---|
| 400–431 | v2 | Stable |
| 432+ | v2 + experimental v3 | Use v2 |

### Configuration snippet (`etc/catalog/ailake.properties`)

```properties
# Trino catalog config for AI-Lake tables
connector.name=iceberg
iceberg.catalog.type=glue

# AWS
hive.metastore.glue.region=us-east-1
hive.metastore.glue.default-warehouse-dir=s3://my-bucket/warehouse

# For REST catalog (cloud-agnostic)
# iceberg.catalog.type=rest
# iceberg.rest-catalog.uri=https://my-rest-catalog.example.com

# Custom properties — Trino exposes ailake.* as table properties
# readable via SHOW CREATE TABLE — these are informational for operators
```

### Querying AI-Lake tables in Trino

```sql
-- Standard Iceberg query — works without plugin
SELECT chunk_id, chunk_text, document_title
FROM ailake.db.my_ailake_table
WHERE category = 'finance'
  AND created_at > TIMESTAMP '2024-01-01 00:00:00 UTC';

-- The embedding column appears as VARBINARY
-- DESCRIBE to see the full schema including ailake.* field properties
DESCRIBE ailake.db.my_ailake_table;
```

### Phase 3: Vector-scan plugin (Trino) ✅ Implemented

Source: `trino-plugin/` · Build: `cd trino-plugin && ./gradlew shadowJar`
Install: copy fat-jar to `$TRINO_HOME/plugin/ailake/`, native lib to `java.library.path`.

Key classes:
- `io.ailake.trino.AilakePlugin` — `Plugin` entry point (ServiceLoader)
- `io.ailake.trino.VectorScanConnectorFactory` — factory; reads `ailake.table-uri`, `ailake.vector-column`, `ailake.vector-dim` from catalog properties
- `io.ailake.trino.VectorScanConnector` — `Connector`; exposes session properties `query_vector` (CSV floats) and `top_k`
- `io.ailake.trino.VectorScanMetadata` — schema `default`, table `search` with columns `row_id / distance / file_path`
- `io.ailake.trino.AilakeNative` — JNA bridge to `libailake_jni.so`; graceful degradation when lib absent

**Catalog config** (`etc/catalog/ailake.properties`):
```properties
connector.name=ailake
ailake.table-uri=s3://my-lake/docs/
ailake.vector-column=embedding
ailake.vector-dim=1536
```

**Query**:
```sql
SET SESSION ailake.query_vector = '0.021,-0.043,0.118,...';  -- 1536 CSV floats
SET SESSION ailake.top_k = 100;

SELECT row_id, distance, file_path
FROM ailake.default.search
ORDER BY distance;
```

See `SETUP.md §15` for a complete walkthrough including demo table generation.

### Trino + REST catalog (cloud-agnostic)

```properties
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=https://my-catalog-api.example.com
iceberg.rest-catalog.security=OAUTH2
iceberg.rest-catalog.oauth2.token=<token>
```

---

## 3. Apache Beam

### How Beam reads AI-Lake tables

Beam uses its `Managed IcebergIO` (Beam 2.56+), which is the recommended path. It wraps the standard Iceberg Java library and reads/writes AI-Lake tables as standard Iceberg — the vector column comes through as `bytes`.

The `Managed.ICEBERG` transform exclusively uses `Beam Rows`. The vector column maps to `BYTES` type in the Beam schema.

### Version support

| Beam version | IcebergIO | Notes |
|---|---|---|
| 2.56.0+ | `Managed.ICEBERG` | Recommended |
| 2.59.0+ | CDC streaming support | Use for streaming reads |
| < 2.56.0 | No native Iceberg support | Do not use |

### Java SDK (batch)

```java
import org.apache.beam.sdk.managed.Managed;
import java.util.Map;

Map<String, Object> config = Map.of(
    "table", "db.my_ailake_table",
    "catalog_properties", Map.of(
        "type", "glue",
        "warehouse", "s3://my-bucket/warehouse",
        "io-impl", "org.apache.iceberg.aws.s3.S3FileIO",
        "client.region", "us-east-1"
    )
);

// Read tabular data from AI-Lake table
PCollection<Row> rows = pipeline
    .apply(Managed.read(Managed.ICEBERG).withConfig(config))
    .getSinglePCollection();

// The embedding column is Row field of type BYTES
// Vector search is not available via Beam IcebergIO — use ailake-py directly
```

### Java SDK (streaming write — ingest pipeline)

```java
Map<String, Object> writeConfig = Map.of(
    "table", "db.my_ailake_table",
    "catalog_properties", Map.of(
        "type", "rest",
        "uri", "https://my-catalog-api.example.com"
    )
);

// Stream Rows into AI-Lake table (data lands in Parquet, HNSW built on compaction)
inputRows.apply(Managed.write(Managed.ICEBERG).withConfig(writeConfig));
```

### Python SDK (batch)

```python
import apache_beam as beam
from apache_beam.transforms.managed import Managed

with beam.Pipeline() as p:
    rows = (
        p
        | Managed.read(Managed.ICEBERG, config={
            "table": "db.my_ailake_table",
            "catalog_properties": {
                "type": "rest",
                "uri": "https://my-catalog-api.example.com"
            }
        })
    )
    # rows["output"] is a PCollection[Row]
    # embedding column is bytes
```

### Vector search in Beam pipelines

Beam does not support vector search via `Managed.ICEBERG` — that IO is tabular only. For vector search within a Beam pipeline, use the AI-Lake Python/Java SDK directly inside a `DoFn`:

```python
import ailake

class VectorSearchFn(beam.DoFn):
    def setup(self):
        # initialize SDK once per worker (not per element)
        self.searcher = ailake.TableSearcher("s3://my-bucket/warehouse/db/my_table/")

    def process(self, query_embedding):
        results = self.searcher.search(
            query=query_embedding,
            top_k=10,
            filter="category = 'finance'"
        )
        yield from results

results = (
    query_embeddings
    | beam.ParDo(VectorSearchFn())
)
```

### Beam runners compatibility

| Runner | Tabular I/O | Vector search (DoFn) | Notes |
|---|---|---|---|
| **Direct** | ✅ | ✅ | Dev/testing |
| **Google Cloud Dataflow** | ✅ | ✅ | Managed upgrades automatic |
| **Apache Flink** | ✅ | ✅ | See §4 for native `ailake-flink` connector (preferred over Beam on Flink) |
| **Apache Spark** | ✅ | ✅ | Uses Spark Iceberg reader |
| **Apache Samza** | ✅ | Untested | |

---

## 4. Apache Flink

### Architecture

`ailake-flink` is a Kotlin/Gradle module (`ailake-flink/`) that implements the Flink Table API connector. It bridges to the Rust SDK via JNA (Java Native Access) loading `libailake_jni.so`.

```
Flink SQL DDL
  └─ AilakeVectorConnectorFactory  (connector = 'ailake')
       ├─ AilakeVectorTableSource  →  AilakeInputFormat  →  AilakeNativeLoader.search()
       └─ AilakeVectorTableSink   →  AilakeSinkFunction  →  AilakeNativeLoader.writeBatch()

AilakeCatalogFactory  →  AilakeCatalog  (Flink catalog API, delegates to ailake-catalog Rust)
```

### Version support

| Dependency | Version |
|---|---|
| Apache Flink | 1.18.1 |
| Kotlin | 1.9.23 |
| JVM | 11+ |
| JNA | 5.14.0 |

### Build

```bash
cd ailake-flink
./gradlew shadowJar
# Output: build/libs/ailake-flink-0.1.0-plugin.jar
```

The shadow jar bundles JNA and Jackson. Flink dependencies are `compileOnly` — provided by the cluster.

### SQL DDL

```sql
-- Create AI-Lake table source + sink
CREATE TABLE docs (
  id        BIGINT,
  text      STRING,
  embedding BYTES,
  _distance FLOAT   -- populated by vector search, ignored on writes
) WITH (
  'connector'        = 'ailake',
  'warehouse'        = 's3://my-lake/',
  'namespace'        = 'default',
  'table-name'       = 'docs',
  'vector.column'    = 'embedding',
  'vector.dim'       = '1536',
  'vector.metric'    = 'cosine',
  'vector.precision' = 'f16',
  'search.top-k'     = '10',
  'search.ef'        = '50'
);

-- Write (streaming ingest)
INSERT INTO docs SELECT id, text, embedding FROM upstream_source;

-- Read (vector search — query vector passed as job parameter)
SELECT id, text, _distance FROM docs;
```

### Query vector at runtime

The query vector is passed via Flink job parameter `ailake.query.vector` (comma-separated floats):

```bash
flink run \
  -p 4 \
  -D "pipeline.global-job-parameters=ailake.query.vector=0.021,-0.043,0.118,..." \
  my-pipeline.jar
```

Or programmatically:

```kotlin
val env = StreamExecutionEnvironment.getExecutionEnvironment()
env.config.setGlobalJobParameters(
    mapOf("ailake.query.vector" to floatVector.joinToString(","))
)
```

### Catalog registration

```sql
CREATE CATALOG ailake_catalog WITH (
  'type'      = 'ailake',
  'warehouse' = 's3://my-lake/'
);

USE CATALOG ailake_catalog;
SHOW DATABASES;
```

### Deploy to Flink cluster

1. Build the plugin jar: `./gradlew shadowJar`
2. Copy `ailake-flink-0.1.0-plugin.jar` to `$FLINK_HOME/lib/`
3. Copy `libailake_jni.so` to a path in `java.library.path` on all TaskManagers:

```yaml
# flink-conf.yaml
env.java.opts.taskmanager: -Djava.library.path=/opt/ailake/lib
```

Or set `ailake.native.lib` system property or `AILAKE_NATIVE_LIB` env var to point directly to the `.so`.

### Connector options

| Option | Required | Default | Description |
|---|---|---|---|
| `warehouse` | ✅ | — | S3/GCS/Azure/local root path |
| `table-name` | ✅ | — | AI-Lake table name |
| `vector.dim` | ✅ | — | Embedding dimension |
| `namespace` | | `default` | Iceberg namespace |
| `vector.column` | | `embedding` | Vector column name |
| `vector.metric` | | `euclidean` | `cosine` / `euclidean` / `dot_product` |
| `vector.precision` | | `f16` | `f16` / `f32` / `i8` |
| `search.top-k` | | `10` | Results per query |
| `search.ef` | | `50` | HNSW ef_search parameter (ignored for IVF-PQ index type) |
| `search.rerank-factor` | | `1` | Rerank multiplier for IVF-PQ (fetches `top_k × factor` candidates, reranks with exact distances). Ignored for HNSW. |

---

## 5. Cloud Providers

### 4A — Amazon Web Services (AWS)

#### Storage: Amazon S3

AI-Lake tables on S3 use the `object_store` crate's S3 backend.

```rust
// ailake-store: S3 configuration
let store = S3Store::new(S3Config {
    bucket: "my-bucket".to_string(),
    region: "us-east-1".to_string(),
    prefix: "warehouse/db/my_table/".to_string(),
    credentials: S3Credentials::FromEnv,  // AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
    // or: S3Credentials::InstanceProfile for EC2/ECS/Lambda
    // or: S3Credentials::WebIdentity for EKS (IRSA)
});
```

Environment variables recognized:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- `AWS_REGION` or `AWS_DEFAULT_REGION`
- `AWS_ROLE_ARN` + `AWS_WEB_IDENTITY_TOKEN_FILE` (IRSA on EKS)

S3-compatible endpoints (Minio, Localstack):
```rust
S3Config {
    endpoint: Some("http://localhost:9000".to_string()),
    path_style: true,   // required for Minio
    ...
}
```

#### Catalog: AWS Glue Data Catalog

The standard Iceberg catalog for AWS. AI-Lake `ailake-catalog` uses `iceberg-rust`'s Glue catalog implementation.

```rust
let catalog = GlueCatalog::new(GlueConfig {
    region: "us-east-1".to_string(),
    database: "my_database".to_string(),
    warehouse: "s3://my-bucket/warehouse".to_string(),
});
```

Glue creates a native Glue table entry with the Iceberg metadata location. Athena, EMR, Glue ETL, and Redshift Spectrum can all read the table directly.

#### Catalog: AWS S3 Tables (managed Iceberg)

AWS S3 Tables (GA 2024) provides fully-managed Iceberg tables stored in S3:

```python
# Python SDK — writing to S3 Tables
import ailake

writer = ailake.TableWriter(
    table_uri="arn:aws:s3tables:us-east-1:123456789:bucket/my-bucket/namespace/my_table",
    catalog_type="s3tables",
    region="us-east-1"
)
```

S3 Tables uses its own Iceberg REST catalog API. Configure via:
```
catalog_type = s3tables
s3tables.region = us-east-1
s3tables.warehouse = arn:aws:s3tables:...
```

#### Query engines on AWS

| Service | Config |
|---|---|
| **Athena** | Glue catalog, reads Iceberg natively. No vector scan. |
| **EMR on EC2** | Spark 3.3–3.5 + `iceberg-spark-runtime`. Full Spark support. |
| **EMR Serverless** | Same as EMR on EC2. |
| **AWS Glue ETL** | Beam IcebergIO or direct Glue Iceberg support (Glue 4.0+). |
| **Redshift Spectrum** | Reads Parquet files from Iceberg table manifest. Vector column as bytes. |
| **SageMaker** | Use `ailake-py` directly in notebooks/training jobs. |

---

### 4B — Google Cloud Platform (GCP)

#### Storage: Google Cloud Storage (GCS)

```rust
let store = GcsStore::new(GcsConfig {
    bucket: "my-bucket".to_string(),
    prefix: "warehouse/db/my_table/".to_string(),
    credentials: GcsCredentials::ApplicationDefault,
    // or: GcsCredentials::ServiceAccount { key_file: "/path/sa.json" }
    // or: GcsCredentials::WorkloadIdentity  (GKE)
});
```

Environment:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`
- GKE Workload Identity (recommended for production)

#### Catalog: Google BigLake Metastore (BigLake + Iceberg)

GCP's managed Iceberg catalog via BigLake:

```python
import ailake

writer = ailake.TableWriter(
    table_uri="bigquery://project.dataset.my_table",
    catalog_type="biglake",
    project="my-gcp-project",
    region="us-central1",
    warehouse_gcs="gs://my-bucket/warehouse"
)
```

BigLake Metastore provides the REST catalog API — use `catalog_type=rest` with the BigLake endpoint as a generic alternative.

#### Query engines on GCP

| Service | Config |
|---|---|
| **BigQuery (Iceberg tables)** | Native Iceberg read via BigLake Metastore. Vector column as bytes. AILK footer compat validated in CI (`compat-bigquery`). |
| **Dataproc (Spark)** | Spark 3.3–3.5, standard Iceberg runtime jar. |
| **Dataproc (Trino)** | Standard Iceberg connector. |
| **Cloud Dataflow** | Beam `Managed.ICEBERG` for tabular. `ailake-py` DoFn for vector search. |
| **Vertex AI** | Use `ailake-py` directly in training pipelines / notebooks. |

---

### 4C — Microsoft Azure

#### Storage: Azure Blob Storage / ADLS Gen2

```rust
let store = AzureStore::new(AzureConfig {
    account: "mystorageaccount".to_string(),
    container: "warehouse".to_string(),
    prefix: "db/my_table/".to_string(),
    credentials: AzureCredentials::ClientSecret {
        tenant_id: std::env::var("AZURE_TENANT_ID")?,
        client_id: std::env::var("AZURE_CLIENT_ID")?,
        client_secret: std::env::var("AZURE_CLIENT_SECRET")?,
    },
    // or: AzureCredentials::WorkloadIdentity  (AKS)
    // or: AzureCredentials::ConnectionString(std::env::var("AZURE_STORAGE_CONNECTION_STRING")?)
});
```

ADLS Gen2 path format: `abfss://container@account.dfs.core.windows.net/prefix`

#### Catalog: Azure Purview / Unity Catalog

For Azure, the recommended catalog is an Iceberg REST catalog (Unity Catalog on Databricks or self-hosted Polaris/Nessie):

```
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=https://my-catalog.azurewebsites.net
iceberg.rest-catalog.security=OAUTH2
iceberg.rest-catalog.oauth2.credential=<client_id>:<client_secret>
iceberg.rest-catalog.oauth2.scope=PRINCIPAL_ROLE:ALL
```

#### Query engines on Azure

| Service | Config |
|---|---|
| **Azure Synapse Analytics** | Spark pool 3.3+, standard Iceberg runtime jar. |
| **Azure Databricks** | Unity Catalog + Iceberg. Vector column as bytes without plugin. |
| **HDInsight** | Spark/Hive on HDInsight. Standard Iceberg support. |
| **Azure Machine Learning** | Use `ailake-py` in compute clusters and pipelines. |

---

### 4D — Multi-cloud / cloud-agnostic

#### Iceberg REST Catalog

The most portable option. Any implementation of the Iceberg REST Catalog spec works:

| Implementation | Hosting | Notes |
|---|---|---|
| **Apache Polaris** (ASF) | Self-hosted or managed | Reference REST catalog implementation |
| **Project Nessie** | Self-hosted / Dremio | Git-like branching semantics |
| **Tabular** | SaaS | Managed Iceberg catalog |
| **Lakeformation** | AWS-managed | Via Glue REST API |
| **Unity Catalog** | Databricks | Also exposes REST API |

AI-Lake `ailake-catalog` uses `iceberg-rust`'s REST catalog client:

```rust
let catalog = RestCatalog::new(RestCatalogConfig {
    uri: "https://my-catalog.example.com".to_string(),
    warehouse: "my_warehouse".to_string(),
    oauth2_server_uri: Some("https://auth.example.com/token".to_string()),
    credential: Some("client_id:client_secret".to_string()),
    scope: Some("PRINCIPAL_ROLE:ALL".to_string()),
});
```

#### Nessie catalog (Git-like branching)

For environments requiring branching and versioning of the catalog:

```rust
let catalog = NessieCatalog::new(NessieConfig {
    uri: "http://localhost:19120/api/v1".to_string(),
    ref_name: "main".to_string(),
    auth: NessieAuth::None,  // or Bearer, OAuth2
    warehouse: "s3://my-bucket/warehouse".to_string(),
});
```

Nessie supports creating branches per feature/experiment:

```python
# Python — write to a feature branch, merge to main when ready
writer = ailake.TableWriter(
    table_uri="my_table",
    catalog_type="nessie",
    nessie_uri="http://localhost:19120",
    nessie_ref="feature/new-embeddings"   # writes go to this branch
)
# merge via Nessie API when ready
```

---

## 6. Catalog compatibility summary

| Catalog | Protocol | AWS | GCP | Azure | Self-hosted |
|---|---|---|---|---|---|
| **AWS Glue** | Glue API | ✅ Native | ❌ | ❌ | ❌ |
| **AWS S3 Tables** | REST | ✅ Native | ❌ | ❌ | ❌ |
| **BigLake Metastore** | REST | ❌ | ✅ Native | ❌ | ❌ |
| **Apache Polaris** | REST | ✅ | ✅ | ✅ | ✅ |
| **Project Nessie** | Nessie API | ✅ | ✅ | ✅ | ✅ |
| **Unity Catalog** | REST | ✅ | ✅ | ✅ | Limited |
| **Hive Metastore** | Thrift | ✅ (EMR) | ✅ (Dataproc) | ✅ (HDInsight) | ✅ |
| **Hadoop FS** | Filesystem | ✅ (S3) | ✅ (GCS) | ✅ (ADLS) | ✅ (HDFS) |
| **JDBC** | PostgreSQL/MySQL | ✅ | ✅ | ✅ | ✅ |

---

## 7. `ailake-catalog` crate — catalog abstraction

The `ailake-catalog` crate provides a `CatalogProvider` trait that all catalog backends implement. The `ailake-query` layer uses only this trait — switching catalogs requires only a config change, not code changes.

```rust
// ailake-catalog/src/lib.rs

#[async_trait]
pub trait CatalogProvider: Send + Sync {
    /// Load the current table metadata (Iceberg metadata.json)
    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata>;

    /// Commit a new snapshot (atomic)
    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> AilakeResult<SnapshotId>;

    /// List DataFile entries with their custom_properties (centroids, HNSW offsets)
    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>>;
}

// Implementations
pub struct GlueCatalog { ... }
pub struct RestCatalog { ... }
pub struct NessieCatalog { ... }
pub struct HadoopCatalog { ... }   // filesystem-based, no external service
pub struct JdbcCatalog { ... }     // stores metadata in PostgreSQL/MySQL

impl CatalogProvider for GlueCatalog { ... }
impl CatalogProvider for RestCatalog { ... }
// etc.
```

### Python API — catalog selection

```python
import ailake

# AWS Glue
writer = ailake.TableWriter(
    table_uri="s3://my-bucket/warehouse/db/my_table",
    catalog=ailake.GlueCatalog(region="us-east-1", database="db")
)

# REST catalog (Polaris / Nessie / Unity Catalog)
writer = ailake.TableWriter(
    table_uri="s3://my-bucket/warehouse/db/my_table",
    catalog=ailake.RestCatalog(
        uri="https://my-catalog.example.com",
        warehouse="my_warehouse",
        credential="client_id:client_secret"
    )
)

# Filesystem catalog (local dev, no external service)
writer = ailake.TableWriter(
    table_uri="/tmp/warehouse/db/my_table",
    catalog=ailake.HadoopCatalog(warehouse="/tmp/warehouse")
)
```

---

## 8. AI-Lake-specific SQL UDF (Phase 3)

When the AI-Lake plugin is loaded in Spark or Trino, the following SQL surface is exposed:

```sql
-- Spark SQL
SELECT *
FROM ailake_search(
    'catalog.database.table',       -- fully-qualified table name
    ailake_embed('my query text'),  -- optional: text → embedding via configured model
    top_k => 100,
    filter => "category = 'finance' AND created_at > '2024-01-01'",
    metric => 'cosine',             -- optional: cosine (default), euclidean, dot_product
    embedding_column => 'context_embedding'  -- optional: which column to search
);

-- Trino SQL
SELECT *
FROM TABLE(ailake.system.vector_search(
    table_name => 'catalog.schema.table',
    query_vector => ailake.system.embed('my query text'),
    top_k => BIGINT '100',
    filter => 'category = ''finance'''
));
```

The `ailake_embed` / `ailake.system.embed` function calls a configured embedding model (env var `AILAKE_EMBED_MODEL_URI`). This is optional — callers can pass a pre-computed vector as a hex blob.

---

## 9. Compatibility tests (CI)

All integration tests use Docker Compose to spin up required services.

### Phase 1 tests (local, no external services)

```bash
cargo test --workspace
# includes: positional_invariant, vector_pruning, parquet_trailing_bytes, write_read_roundtrip
```

### Phase 2 tests (requires Docker)

```bash
docker compose -f tests/docker/compose.yml up -d   # MinIO, Nessie, mock Glue
cargo test --workspace --features integration
```

Services spun up:
- **MinIO** (`localhost:9000`) — S3-compatible storage
- **Project Nessie** (`localhost:19120`) — REST catalog
- **Localstack** (`localhost:4566`) — mock AWS Glue + S3

### Phase 3+ tests (requires Docker + JVM — triggered via `compat-heavy.yml` manual dispatch)

```bash
docker compose -f tests/docker/compose-engines.yml up -d  # Spark, Trino
./tests/compat/run_spark_compat.sh
./tests/compat/run_trino_compat.sh
./tests/compat/run_beam_compat.sh
```

Each test:
1. Writes an AI-Lake table via Rust SDK
2. Reads it via the target engine (Spark/Trino/Beam) without AI-Lake plugin
3. Asserts: correct row count, correct schema, vector column as bytes
4. Asserts: no errors or warnings about unrecognized file format

Failure of any compat test is a release blocker.
