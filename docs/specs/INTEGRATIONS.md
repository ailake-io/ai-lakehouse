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
| **DuckDB 0.10+** | ✅ Iceberg extension | ✅ `duckdb-ailake/` extension | ✅ `ailake_search()` + `ailake_write_batch()` | — |
| **PyIceberg 0.6+** | ✅ | ✅ | via SDK direct | — |
| **AWS Athena** | ✅ Glue catalog | Limited | — | — |
| **AWS EMR** | ✅ Spark/Trino on EMR | ✅ | ✅ | ✅ |
| **AWS Glue ETL** | ✅ | ✅ | via SDK direct | ✅ |
| **Azure Synapse** | ✅ Spark pool | ✅ | ✅ | ✅ |
| **Azure Databricks** | ✅ | ✅ | ✅ | ✅ |
| **GCP Dataproc** | ✅ Spark/Trino | ✅ | ✅ | ✅ |
| **GCP Dataflow** | ✅ Beam IcebergIO | ✅ Beam IcebergIO | via SDK direct | ✅ |
| **Snowflake** | ✅ Iceberg tables | Limited | — | — |
| **Databricks (general)** | ✅ | ✅ | ✅ | ✅ |
| **Python (`ailake-py`)** | ✅ PyArrow | ✅ `open_table` + `Table.insert` / `write_batch_auto_deferred` | ✅ `SearchQuery` fluent chain | ✅ `write_batch_auto_deferred`, `write_batch_idempotent`, async API |
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

`AilakeNative.createTable(...)` (wraps `ailake_create_table_json` — schema-only, no data) exists as a
direct Scala method call, but is not yet wired into `CREATE TABLE ... USING ailake` SQL DDL —
`AilakeCatalog.createTable` currently just builds an in-memory `Table` object without touching the
native library; the physical table is created lazily by the first write.

```scala
import io.ailake.spark.implicits._

val spark = SparkSession.builder()
  .config("spark.jars", "/path/to/spark-plugin-0.1.7-plugin.jar")
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

`topK` (like every plugin's `top_k`/`top-k`) is capped at `ailake_core::MAX_TOP_K` = 100,000 — the underlying `ailake_search_json` call returns an error above that rather than proceeding, enforced uniformly in `ailake-query::scanner` for every binding (CLI, Python, JNI).

See `SETUP.md §16` for a complete walkthrough including demo table generation and cluster submission.

### Hybrid search and full-text search (Spark)

```scala
import io.ailake.spark.implicits._

// Hybrid BM25+vector
val hybrid = spark.ailakeSearch(
    tableUri    = "s3://my-lake/docs/",
    queryVector = myEmbedding,
    topK        = 100,
    hybridText  = Some("rust programming"),
    bm25Weight  = 0.5f,
    textColumn  = "chunk_text",
)

// Full-text search only (no vector)
val textResults = spark.ailakeSearchText(
    tableUri    = "s3://my-lake/docs/",
    queryText   = "rust programming async",
    topK        = 20,
)
// returns DataFrame: row_id, score (BM25, higher=more relevant), file_path
```

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

`AilakeNative.createTable(...)` (wraps `ailake_create_table_json`) exists as a direct Kotlin method
call with no other caller in the plugin today — there is no `CREATE TABLE`/`CALL` surface wired to it
in Trino yet.

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

`ailake.top_k` is capped at 100,000 (rejected above that — same `MAX_TOP_K` enforced in `ailake-query::scanner` for every plugin/binding).

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
    def process(self, query_embedding):
        # ailake.search() is a plain function, not a class to instantiate in setup() —
        # note the same storage-binding caveat as elsewhere in this doc: ailake-py only
        # constructs a LocalStore, so table_path must be a local/mounted path, not s3://
        results = ailake.search(
            "/local/path/to/warehouse/db/my_table/",
            query_embedding,
            top_k=10,
            partition_filter="finance",  # optional — Iceberg identity-partition value, not a SQL predicate
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

`AilakeInputFormat.open()` degrades gracefully when the native lib can't be loaded — same contract as the Spark/Trino/DuckDB bridges — returning an empty result set instead of failing the Flink task. The write path (`AilakeVectorTableSink`) intentionally does not degrade: a missing native lib fails the sink loudly rather than silently dropping a write batch.

`AilakeNativeLoader.createTable(...)` (wraps `ailake_create_table_json`) exists but has no caller anywhere in this module today — `AilakeCatalog.createTable` (the real `CREATE TABLE` DDL entry point) only registers an in-memory `CatalogTableImpl`; the physical AI-Lake table is created lazily by the first write, same as Spark.

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
# Output: build/libs/ailake-flink-0.1.7-plugin.jar
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
  'search.ef'        = '50',
  -- Optional: model name stored in Iceberg metadata on every INSERT via this table
  'embedding.model'  = 'text-embedding-3-small@v1'
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
2. Copy `ailake-flink-0.1.7-plugin.jar` to `$FLINK_HOME/lib/`
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
| `search.top-k` | | `10` | Results per query. Capped at 100,000 (`MAX_TOP_K`, rejected above that) |
| `search.ef` | | `50` | HNSW ef_search parameter (ignored for IVF-PQ index type) |
| `search.rerank-factor` | | `1` | Rerank multiplier for IVF-PQ (fetches `top_k × factor` candidates, reranks with exact distances). Ignored for HNSW. |
| `partition.by` | | `` | Iceberg identity partition column (e.g. `agent_id`). Enables manifest-level per-agent pruning (Phase 9). |
| `partition.value` | | `` | Partition value for this table source/sink instance. |
| `search.partition-filter` | | `` | Restrict search to files with this partition_value (Phase 9). |
| `fts.columns` | | `` | Comma-separated text columns to build Tantivy FTS index (Phase T). E.g. `chunk_text,document_title`. |
| `fts.tokenizer` | | `default` | Tantivy tokenizer for FTS index. |
| `search.hybrid-text` | | `` | Query text for BM25 hybrid RRF fusion (Phase 9). |
| `search.bm25-weight` | | `0.5` | BM25 weight in RRF fusion. |
| `search.mode` | | `search` | `search` (fixed 3-column shape) or `full` (Phase 11 — search + full-row fetch via `AilakeScanTableSource`, dynamic columns from the DDL, no JOIN needed; last declared column must be `_distance`). |
| `hnsw.m` | | native default | HNSW graph degree *M*. |
| `hnsw.ef-construction` | | native default | HNSW build-time beam width. |
| `pre-normalize` | | `false` | Normalize vectors to unit L2 at write time. |
| `deferred` | | `false` | Write Parquet immediately, build the index asynchronously. |
| `vector.columns` | | `[]` | JSON array `[{"column","dim","metric"?,"precision"?,"modality"?}]` for multi-column (Phase 8 multimodal) writes — one `ARRAY<FLOAT>` per entry instead of the single `vector.column`, via `ailake_write_batch_multi_json`. |

### FTS and hybrid search (Flink)

```kotlin
// Full-text search via AilakeNativeLoader
val hits = AilakeNativeLoader.searchText(
    warehouse = "s3://my-lake/",
    namespace = "default",
    table     = "docs",
    queryText = "rust programming async",
    topK      = 20,
)
// hits: List<SearchTextResult>(rowId, score, filePath)
```

```scala
// Hybrid BM25+vector search via AilakeNative (Scala)
val results = AilakeNative.search(
    tableUri  = "s3://my-lake/docs/",
    query     = floatArray,
    topK      = 100,
    hybridText   = Some("rust async"),
    bm25Weight   = 0.5f,
    textColumn   = "chunk_text",
    tableName    = "table",
)
```

---

## 5. DuckDB Extension (`duckdb-ailake`)

### What it does

`duckdb-ailake/` is a C++ DuckDB community extension that bridges DuckDB SQL to `libailake_jni.so` via `dlopen`. It exposes these table/scalar functions (top_k-bearing functions are capped at 100,000, same `MAX_TOP_K` as every other binding):

| Function | Signature | Description |
|---|---|---|
| `ailake_search` | `(table_path VARCHAR, query FLOAT[], top_k INTEGER [, vec_col VARCHAR, ef_search INTEGER, table_name VARCHAR, namespace VARCHAR, partition_filter VARCHAR, hybrid_text VARCHAR, text_column VARCHAR, bm25_weight FLOAT]) → TABLE(row_id BIGINT, distance FLOAT, file_path VARCHAR)` | Vector nearest-neighbor search (with optional BM25 hybrid) |
| `ailake_search_multimodal` | `(table_path VARCHAR, queries LIST(STRUCT(...)), top_k INTEGER [, partition_filter VARCHAR, table_name VARCHAR, namespace VARCHAR]) → TABLE(row_id BIGINT, rrf_score FLOAT, file_path VARCHAR)` | Cross-modal RRF search |
| `ailake_search_text` | `(table_path VARCHAR, query_text VARCHAR, top_k INTEGER [, text_columns VARCHAR[], text_column VARCHAR, partition_filter VARCHAR, table_name VARCHAR, namespace VARCHAR]) → TABLE(row_id BIGINT, score FLOAT, file_path VARCHAR)` | Pure BM25 full-text search (Tantivy O(log N) when FTS index present; brute-force fallback) |
| `ailake_scan` | `(table_path VARCHAR, query FLOAT[], top_k INTEGER [, vec_col VARCHAR, ef_search INTEGER, table_name VARCHAR, namespace VARCHAR]) → TABLE(<all Parquet columns>, _distance FLOAT)` | Vector search + full row fetch in one call (no JOIN required) |
| `ailake_write_batch` | `(table_path VARCHAR, ids BIGINT[], embeddings FLOAT[][] [, vec_col, metric, precision, partition_by, partition_value, partition_fields, format_version, fts_columns, fts_tokenizer, hnsw_m, hnsw_ef_construction, pre_normalize, deferred, namespace, table_name]) → BIGINT` | Write a batch; returns snapshot ID or -1 on error |
| `ailake_write_batch_multi` | `(table_path VARCHAR, ids BIGINT[], vector_columns LIST(STRUCT(col VARCHAR, dim INTEGER, embeddings FLOAT[][], metric VARCHAR, precision VARCHAR, modality VARCHAR))) → BIGINT` | Write a batch with N independent vector columns (Phase 8 multimodal); returns snapshot ID or -1 |
| `ailake_delete_where` | `(table_path VARCHAR, column VARCHAR, values VARCHAR[] [, namespace VARCHAR, table_name VARCHAR]) → BOOLEAN` | Equality-delete matching rows |
| `ailake_evolve_schema` | `(table_path VARCHAR, add_columns_json VARCHAR, rename_columns_json VARCHAR [, namespace VARCHAR, table_name VARCHAR]) → INTEGER` | Metadata-only schema evolution; returns new schema_id |
| `ailake_compact` | `(table_path VARCHAR [, min_files BIGINT, target_size_bytes BIGINT, max_files_per_pass BIGINT, deferred BOOLEAN, namespace VARCHAR, table_name VARCHAR]) → BIGINT` | Compacts small files into a larger merged file; returns files compacted (0 = nothing eligible), -1 on error |
| `ailake_create_table` | `(table_path VARCHAR, dim INTEGER [, vector_column, metric, precision, format_version, hnsw_m, hnsw_ef_construction, pre_normalize, modality, partition_by, partition_value, partition_column_type, partition_fields_json, fts_columns, fts_tokenizer, embedding_model, namespace, table_name]) → BOOLEAN` | Creates an empty AI-Lake/Iceberg table (schema only, no data) |

`namespace` (default `'default'`) and `table_name` (default `'table'`) are optional trailing parameters on every function above — pass them (as named params on the table functions, or positionally on the scalar functions) to address a table other than the warehouse root's default. `partition_filter` (search) and `partition_by`/`partition_value` (write, single-column identity) are optional named parameters for per-agent/per-tenant file pruning (Phase 9). `partition_fields` accepts a JSON array (`[{"column":"topic_id","transform":"identity","column_type":"int"}]`) for multi-column Iceberg partition specs with any transform (identity, bucket, truncate, year, month, day, hour) — Phase L/R. `format_version` (default 2) enables Iceberg v3 when set to `3`. `hnsw_m`/`hnsw_ef_construction` (default -1 = use table default) tune the HNSW index; `pre_normalize` (default false) normalizes vectors to unit L2 at write time; `deferred` (default false) builds the index asynchronously. All functions degrade gracefully when `libailake_jni.so` is not loaded — `ailake_search`/`ailake_search_text`/`ailake_search_multimodal`/`ailake_scan` return 0 rows, `ailake_write_batch` returns -1, `ailake_delete_where` returns false, `ailake_evolve_schema` returns -1.

The extension uses the same JSON-envelope C-ABI protocol as the Spark and Trino plugins — no additional Rust code required.

### Build

```bash
# Prerequisites: CMake ≥ 3.28, C++17 compiler
cargo build --release -p ailake-jni          # build native lib first

cmake -S duckdb-ailake -B duckdb-ailake/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDUCKDB_VERSION=v1.5.3
cmake --build duckdb-ailake/build --parallel

# No network access to git protocol? Fetch the source tarball directly and
# point FetchContent at the extracted directory instead of letting it git-clone:
#   curl -sL https://codeload.github.com/duckdb/duckdb/tar.gz/refs/tags/v1.5.3 -o duckdb.tar.gz
#   tar xzf duckdb.tar.gz
#   cmake -S duckdb-ailake -B duckdb-ailake/build -DCMAKE_BUILD_TYPE=Release \
#     -DFETCHCONTENT_SOURCE_DIR_DUCKDB=$(pwd)/duckdb-1.5.3 \
#     -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$(pwd)/nlohmann_json_src/json

# Artifact: duckdb-ailake/build/ailake.duckdb_extension
```

### Load and use

```python
import ctypes, duckdb

# Pre-load native lib (RTLD_GLOBAL so DuckDB's dlopen finds symbols)
ctypes.CDLL("./target/release/libailake_jni.so", ctypes.RTLD_GLOBAL)

conn = duckdb.connect()
conn.execute("LOAD './duckdb-ailake/build/ailake.duckdb_extension'")

# Vector search
rows = conn.execute("""
    SELECT row_id, distance, file_path
    FROM ailake_search(
        '/path/to/table',
        [0.021, -0.043, 0.118, ...]::FLOAT[],
        10
    )
    ORDER BY distance
""").fetchall()

# Write a batch
snap_id = conn.execute("""
    SELECT ailake_write_batch(
        'file:///path/to/table',
        [0, 1, 2]::BIGINT[],
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )
""").fetchone()[0]
```

### Full-text search and hybrid search (DuckDB)

```sql
-- Pure BM25 full-text search
SELECT row_id, score, file_path
FROM ailake_search_text(
    '/path/to/table',
    'rust programming async',
    10
)
ORDER BY score DESC;

-- Hybrid BM25+vector via named params on ailake_search
SELECT * FROM ailake_search(
    '/path/to/table',
    [0.021, -0.043, 0.118]::FLOAT[],
    10,
    hybrid_text='rust async programming',
    bm25_weight=0.5
);
```

### Named parameters

```sql
-- explicit vector column and ef_search
SELECT * FROM ailake_search(
    '/path/to/table',
    [0.1, 0.2, 0.3]::FLOAT[],
    5,
    vec_col='context_embedding',
    ef_search=100
);

-- explicit metric and precision
SELECT ailake_write_batch(
    'file:///path/to/table',
    [0, 1]::BIGINT[],
    [[0.1, 0.2], [0.3, 0.4]],
    'embedding',   -- vec_col
    'cosine',      -- metric
    'f16'          -- precision
);

-- non-default namespace/table_name (table functions: named params)
SELECT * FROM ailake_search(
    '/path/to/warehouse',
    [0.1, 0.2, 0.3]::FLOAT[],
    5,
    namespace='analytics',
    table_name='embeddings'
);

-- non-default namespace/table_name (ailake_write_batch: trailing positional args,
-- arity 18 — every arg from vec_col onward must be supplied up to this point)
SELECT ailake_write_batch(
    '/path/to/warehouse',
    [0, 1]::BIGINT[],
    [[0.1, 0.2], [0.3, 0.4]],
    'embedding', 'cosine', 'f16',
    '', '', '', 2, '', '', -1, -1, false, false,
    'analytics', 'embeddings'
);
```

### Graceful degradation

When `libailake_jni.so` is not found or not pre-loaded, `ailake_search` returns 0 rows instead of raising an error — the same behaviour as the Spark and Trino plugins. `ailake_write_batch` returns -1.

### CI

`ci-duckdb.yml` (`workflow_dispatch`) builds the extension, generates a fixture via `tests/fixtures/write_fixture.py`, then runs `duckdb-ailake/test/test_write.py` and `duckdb-ailake/test/test_search.py`.

---

## 6. Cloud Providers

### 4A — Amazon Web Services (AWS)

#### Storage: Amazon S3

AI-Lake tables on S3 use the `object_store` crate's S3 backend, behind `ailake-store`'s
`store-s3` Cargo feature. Only `ailake-cli` enables this feature today — `ailake-jni`
(Spark/Trino/Flink) and `ailake-py` link `ailake-store` without `store-s3`/`store-gcs`/
`store-azure` and only ever construct `LocalStore`, so the object-storage builders below
are reachable from the CLI binary and Rust callers, not from the Python or JVM bindings
(see `CLOUD_DEPLOY.md`'s storage-backend caveat).

```rust
// ailake-store: S3 configuration (ailake_store::s3::{S3Config, S3Credentials, s3_store})
let store = s3_store(
    S3Config {
        bucket: "my-bucket".to_string(),
        region: "us-east-1".to_string(),
        endpoint: None,       // Some("http://localhost:9000") for MinIO/LocalStack
        allow_http: false,    // true required for MinIO without TLS
        credentials: S3Credentials::Default,  // env → ~/.aws → IMDSv2 → WebIdentity chain
        // or: S3Credentials::Static { access_key_id, secret_access_key, session_token }
        // or: S3Credentials::InstanceProfile for EC2/ECS/Lambda
        // or: S3Credentials::WebIdentity for EKS (IRSA)
    },
    "warehouse/db/my_table/",  // prefix — separate argument, not a config field
)?;
```

Or the zero-config dispatcher: `ailake_store::store_from_url("s3://my-bucket/warehouse/db/my_table/")`.

Environment variables recognized:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- `AWS_REGION` or `AWS_DEFAULT_REGION`
- `AWS_ROLE_ARN` + `AWS_WEB_IDENTITY_TOKEN_FILE` (IRSA on EKS)

#### Catalog: AWS Glue Data Catalog

`ailake-catalog::GlueCatalog` — its own AWS SDK-based implementation (not backed by the
`iceberg` crate, which is not a dependency of this workspace; see `CLAUDE.md` §11).

```rust
// GlueCatalog::from_client(aws_sdk_glue::Client, GlueCatalogConfig, Arc<dyn Store>) -> Self
// (there is also an env-credential constructor; see ailake-catalog/src/glue.rs)
```

Glue creates a native Glue table entry with the Iceberg metadata location. Athena, EMR, Glue ETL, and Redshift Spectrum can all read the table directly. **Not wired into `ailake-cli`, `ailake-py`, or `ailake-jni` today** — usable only as a direct Rust dependency (`ailake-catalog::GlueCatalog`), same status as `JdbcCatalog`/`NessieCatalog` (see §8).

#### Catalog: AWS S3 Tables (managed Iceberg)

AWS S3 Tables (GA 2024) provides fully-managed Iceberg tables stored in S3, exposed through
its own Iceberg REST catalog API. `ailake-catalog::RestCatalog` (feature `rest-catalog`,
wired into `ailake-cli --catalog rest`, `ailake-py`'s `catalog_opts={"catalog": "rest", ...}`,
and `ailake-jni`'s `catalog_opts` — see `docs/guides/REST_CATALOG.md`) can target any
spec-compliant REST catalog server, including S3 Tables' endpoint, but this combination is
not specifically documented or tested against S3 Tables yet.

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

Same binding-scope caveat as S3 above — only `ailake-cli` (`store-gcs` feature) reaches GCS directly; `ailake-py`/`ailake-jni` do not.

```rust
// ailake_store::gcs::{GcsConfig, GcsCredentials, gcs_store}
let store = gcs_store(
    GcsConfig {
        bucket: "my-bucket".to_string(),
        credentials: GcsCredentials::ApplicationDefault,
        // reads GOOGLE_APPLICATION_CREDENTIALS, falls back to the GCE metadata
        // server automatically — covers GKE Workload Identity and Cloud Run
        // or: GcsCredentials::ServiceAccountFile("/path/sa.json".into())
        // or: GcsCredentials::ServiceAccountJson(inline_json)
    },
    "warehouse/db/my_table/",  // prefix — separate argument
)?;
```

Environment:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`
- GKE Workload Identity (recommended for production)

#### Catalog: Google BigLake Metastore (BigLake + Iceberg)

BigLake Metastore provides the Iceberg REST catalog API. Reach it the same way as any other
REST catalog — via `ailake-catalog::RestCatalog` (`ailake-cli --catalog rest`, `ailake-py`'s
`catalog_opts={"catalog": "rest", "rest_uri": "https://<biglake-endpoint>", ...}`, or
`ailake-jni`'s `catalog_opts`; see `docs/guides/REST_CATALOG.md`). There is no
BigLake-specific `catalog_type` — it is a REST catalog like any other, and the combination is
not specifically documented or tested against BigLake yet.

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

Same binding-scope caveat as S3/GCS above — only `ailake-cli` (`store-azure` feature) reaches Azure directly; `ailake-py`/`ailake-jni` do not.

```rust
// ailake_store::azure::{AzureConfig, AzureCredentials, azure_store}
let store = azure_store(
    AzureConfig {
        account_name: "mystorageaccount".to_string(),
        container: "warehouse".to_string(),
        credentials: AzureCredentials::ClientSecret {
            tenant_id: std::env::var("AZURE_TENANT_ID")?,
            client_id: std::env::var("AZURE_CLIENT_ID")?,
            client_secret: std::env::var("AZURE_CLIENT_SECRET")?,
        },
        // or: AzureCredentials::ManagedIdentity { client_id: None }  (system-assigned, AKS)
        // or: AzureCredentials::AccessKey(key) / AzureCredentials::SasToken(token) / AzureCredentials::AzureCli
    },
    "db/my_table/",  // prefix — separate argument
)?;
```

ADLS Gen2 path format: `abfss://container@account.dfs.core.windows.net/prefix`

#### Catalog: Azure Purview / Unity Catalog

For Azure, the recommended catalog is an Iceberg REST catalog (Unity Catalog on Databricks or self-hosted Polaris/Nessie), reached the same way as any other REST catalog — via `ailake-catalog::RestCatalog` (`ailake-cli --catalog rest --rest-uri ... --rest-auth oauth2 ...`, `ailake-py`'s `catalog_opts`, or `ailake-jni`'s `catalog_opts`; see `docs/guides/REST_CATALOG.md` for the full flag/kwarg set, including OAuth2).

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

`ailake-catalog::RestCatalog` is AI-Lake's own implementation (not backed by the `iceberg` crate, which is not a workspace dependency) — this is the catalog backend wired into `ailake-cli --catalog rest`, `ailake-py`'s `catalog_opts={"catalog": "rest", ...}`, and `ailake-jni`'s `catalog_opts` (see `docs/guides/REST_CATALOG.md` for the full CLI-flag/kwarg reference). Direct Rust construction:

```rust
// ailake_catalog::rest::{RestCatalog, RestCatalogConfig, RestCatalogAuth}
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://my-catalog.example.com".to_string(),
        prefix: None,          // catalog name (Polaris) / branch (Nessie), if required
        warehouse: Some("my_warehouse".to_string()),
        auth: RestCatalogAuth::OAuth2 {
            token_endpoint: "https://auth.example.com/token".to_string(),
            client_id: "client_id".to_string(),
            client_secret: "client_secret".to_string(),
            scope: Some("PRINCIPAL_ROLE:ALL".to_string()),
        },
    },
    store,
);
```

#### Nessie catalog (Git-like branching)

`ailake-catalog::NessieCatalog` wraps a `RestCatalog` internally and adds Nessie-specific branch/tag operations (`/api/v2/trees/*`). **Not wired into `ailake-cli`, `ailake-py`, or `ailake-jni` today** — usable only as a direct Rust dependency, same status as `GlueCatalog`/`JdbcCatalog` (see §8). Generic Nessie access over the plain Iceberg REST protocol (no branch operations) works today through `RestCatalog` the same way as any other REST catalog server.

```rust
// ailake_catalog::nessie::{NessieCatalog, NessieCatalogConfig}
let catalog = NessieCatalog::new(
    NessieCatalogConfig {
        uri: "http://localhost:19120/api".to_string(),
        default_branch: "main".to_string(),
        warehouse: Some("s3://my-bucket/warehouse".to_string()),
        auth: RestCatalogAuth::None,  // or Bearer, OAuth2 — shared with RestCatalog
    },
    store,
);
```

---

## 7. Catalog compatibility summary

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

## 8. `ailake-catalog` crate — catalog abstraction

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

Only two of the five `CatalogProvider` implementations are wired into `ailake-py` today:
`HadoopCatalog` (default) and `RestCatalog` (Polaris / Nessie / Unity Catalog / S3 Tables /
BigLake / any Iceberg REST Catalog spec server). There are no `ailake.GlueCatalog`,
`ailake.RestCatalog`, or `ailake.HadoopCatalog` Python classes — selection is a plain
`catalog_opts: dict[str, str]` kwarg accepted by `TableWriter`, `Table`/`open_table`,
`search()`, `compact()`, and the other catalog-touching functions (see `docs/guides/REST_CATALOG.md`):

```python
import ailake

# Hadoop-style filesystem catalog — local dev/CLI default, no catalog_opts needed
writer = ailake.TableWriter("/tmp/warehouse/db/my_table")

# REST catalog (Polaris / Nessie / Unity Catalog / S3 Tables / BigLake / Gravitino)
writer = ailake.TableWriter(
    "s3://my-bucket/warehouse/db/my_table",
    catalog_opts={
        "catalog": "rest",
        "rest_uri": "https://my-catalog.example.com",
        "rest_warehouse": "my_warehouse",
        "rest_auth": "oauth2",
        "rest_oauth_token_endpoint": "https://auth.example.com/token",
        "rest_oauth_client_id": "client_id",
        "rest_oauth_client_secret": "client_secret",
    },
)
```

`GlueCatalog`/`NessieCatalog`/`JdbcCatalog` remain Rust-only (`ailake-catalog`), not reachable from `ailake-py`, `ailake-jni`, or `ailake-cli`.

---

## 9. AI-Lake SQL / DataFrame surface (Phase 3, 11, 12)

There is no `ailake_search(...)` SQL function or `ailake_embed(...)` text-to-embedding
function in either plugin — Spark's own `VectorSearchPlan.scala` documents this explicitly
("Not currently reachable via SQL — no parser/function is registered for an
`ailake_vector_search(...)` syntax"), and no `AILAKE_EMBED_MODEL_URI`-style embedding
function exists in either plugin's source. Embedding text into a query vector is the
caller's responsibility (e.g. `ailake_embed()` from your own model client) before calling
search. The actual surfaces:

```scala
// Spark — DataFrame API, not raw SQL (io.ailake.spark.implicits.AilakeSession)
val results = spark.ailakeSearch(
    tableUri    = "s3://my-lake/docs/",
    queryVector = myEmbedding,   // pre-computed Array[Float]
    topK        = 100,
)
results.filter("distance < 0.3").orderBy("distance").show(10)
// Full-row fetch (no JOIN): spark.ailakeSearchWithData(...) — see §1
```

```sql
-- Trino — query the connector's virtual tables directly
SET SESSION ailake.query_vector = '0.021,-0.043,0.118,...';
SET SESSION ailake.top_k = 100;

SELECT row_id, distance, file_path
FROM ailake.default.search       -- or search_full (full row data), search_multimodal
ORDER BY distance;

-- Trino stored procedure for compaction
CALL ailake.system.compact();
```

---

## 10. Compatibility tests (CI)

All integration tests use Docker Compose to spin up required services.

### Phase 1 tests (local, no external services)

```bash
cargo test --workspace
# includes: positional_invariant, vector_pruning, parquet_trailing_bytes, write_read_roundtrip
```

### Manual local testing against Docker services (not run by CI)

`tests/docker/compose.yml` (MinIO, Project Nessie, Localstack) and `tests/docker/compose-engines.yml`
(Spark, Trino) exist for local developer use — spinning up real S3/REST-catalog/engine backends to
test against manually. Neither file is referenced by any GitHub Actions workflow; there is no
`--features integration` Cargo feature. The handful of tests that hit a live service (e.g.
`ailake-catalog::rest`'s `live_create_table_auto_creates_namespace`) are `#[ignore]`-by-default and
must be run explicitly (`cargo test -- --ignored`) against a manually-started server.

```bash
docker compose -f tests/docker/compose.yml up -d
# then run whichever #[ignore]-marked tests target that service, e.g.:
cargo test -p ailake-catalog -- --ignored
```

### CI compat testing (`ci.yml` always-on + `compat-heavy.yml` manual/scheduled)

The real CI compat jobs spin up services directly as inline `services:`/container steps within the
workflow YAML (not via the `tests/docker/*.yml` compose files above). See `ICEBERG_COMPAT.md` §"Verifying
compatibility in CI" for the authoritative list: `compat-pyarrow`/`compat-duckdb`/`compat-pyiceberg`/
`compat-ailake-py` run on every PR; `compat-spark`/`compat-trino`/`compat-jvm-plugins`/`compat-fts`/
`compat-bigquery` run via `compat-heavy.yml` (manual dispatch + scheduled/push to `main`). There is no
automated Beam compat CI job today — Beam's `Managed.ICEBERG` read/write path (§3) is exercised
manually, not asserted in CI.

Each engine compat test follows the same pattern: writes an AI-Lake table via the Rust SDK, reads it
via the target engine without the AI-Lake plugin, and asserts correct row count, schema, and that the
vector column round-trips as bytes with no errors about unrecognized file format.

Failure of the always-on PR jobs is a PR blocker; failure of the `compat-heavy.yml` jobs is a release blocker.
