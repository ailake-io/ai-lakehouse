# JVM_PLUGINS.md — Trino VectorScanConnector + Spark VectorScanStrategy

Reference guide for the two JVM query-engine plugins that expose AI-Lake vector search to Trino and Spark.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Query Engine (Trino / Spark / Flink)                           │
│                                                                 │
│  SQL: SELECT * FROM ailake.default.search ORDER BY distance     │
│  Scala: spark.ailakeSearch(uri, query, topK)                    │
│                         │                                       │
│         ┌───────────────▼──────────────────┐                   │
│         │  JVM Plugin (Kotlin / Scala)      │                   │
│         │  VectorScanConnector (Trino)       │                   │
│         │  VectorScanStrategy  (Spark)       │                   │
│         │  AilakeNativeLoader  (Flink)       │                   │
│         │  AilakeNative — JNA bridge         │                   │
│         └───────────────┬──────────────────┘                   │
└─────────────────────────┼───────────────────────────────────────┘
                          │  JNA (System.loadLibrary)
                          ▼
             ┌────────────────────────────────────────┐
             │  libailake_jni.so  (Rust cdylib)        │
             │                                         │
             │  ailake_search_json()        ← search   │
             │  ailake_write_batch_json()   ← write    │
             │  ailake_delete_where_json()  ← delete   │
             │  ailake_evolve_schema_json() ← schema   │
             │  ailake_compact_json()       ← compact  │
             │  ailake_scan_json()          ← full-read│
             │  ailake_search_text_json()   ← BM25 FTS │
             │  ailake_search_multimodal_json() ← RRF  │
             │  ailake_version()            ← version  │
             │  ailake_free_string()        ← free ptr │
             │          │                              │
             │  do_search() ← ailake-query             │
             │  HNSW + pruning                         │
             └────────────────────────────────────────┘
```

**Key invariant**: the search logic lives entirely in Rust (`ailake-jni` → `ailake-query` → `ailake-index`). The JVM plugins are thin adapters that translate engine-specific SPI calls into native library calls and parse the JSON response.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Rust + Cargo | 1.75+ stable | `curl https://sh.rustup.rs -sSf \| sh` |
| JDK | 17+ | `sudo apt install openjdk-17-jdk` |
| Gradle | 8+ | `sdk install gradle` or Gradle wrapper |
| Trino server | 430+ | [trino.io/download](https://trino.io/download.html) |
| Spark | 3.5.x | [spark.apache.org](https://spark.apache.org/downloads.html) |

---

## Download pre-built JARs (recommended)

Each GitHub Release includes pre-built artifacts uploaded by the `publish-jvm.yml` workflow. No Rust toolchain or Gradle required.

```bash
VERSION=0.0.25   # replace with desired release

# Spark plugin
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/spark-plugin-${VERSION}-plugin.jar

# Trino plugin
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/trino-plugin-${VERSION}-plugin.jar

# Flink connector
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/ailake-flink-${VERSION}-plugin.jar

# Native library (required by all three)
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/libailake_jni.so
```

Place `libailake_jni.so` in a directory accessible to the JVM (see [Native library deployment](#native-library-deployment)).

---

## Step 0 — Build the native library

Both plugins share the same `libailake_jni.so`. Build once:

```bash
# From the project root
cargo build --release -p ailake-jni

# Outputs:
#   Linux:  target/release/libailake_jni.so
#   macOS:  target/release/libailake_jni.dylib
#   Windows: target/release/ailake_jni.dll

NATIVE_LIB_DIR=$(pwd)/target/release
```

The library exports C-ABI symbols consumed by JNA. All three plugins use the JSON-envelope API:

> **Note**: JVM bindings use JNA + C-ABI (`#[no_mangle]` Rust exports). `uniffi` was removed from the workspace in Phase 4 — there is no generated bindings layer.

```c
// Vector search
// request_json: {"warehouse":"...","namespace":"default","table":"...","vec_col":"embedding",
//                "dim":1536,"query":[...],"top_k":10,
//                "partition_filter":"agent-42",  ← optional
//                "hybrid_text":"rust programming", ← optional (BM25)
//                "text_column":"chunk_text",        ← optional, default "chunk_text"
//                "bm25_weight":0.5}                 ← optional, default 0.5
// Returns: {"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}
char* ailake_search_json(const char* request_json);

// Write batch (auto-selects IVF-PQ or HNSW based on HardwareProfile::detect())
// request_json: {"warehouse":"...","namespace":"default","table":"...",
//                "dim":1536,"ids":[...],"embeddings":[[...],...],
//                "metric":"cosine","precision":"f16",
//                "embedding_model":"text-embedding-3-small@v1",  ← optional
//                "partition_by":"agent_id",     ← optional (single-col identity partition)
//                "partition_value":"agent-42",  ← optional
//                "partition_fields":[{"column":"topic_id","transform":"identity","column_type":"int"}],
//                "format_version":3}            ← optional, default 2
// Returns: {"ok":true,"snapshot_id":N}
char* ailake_write_batch_json(const char* request_json);

// Pure BM25 full-text search (no HNSW required, O(N) scan)
// request_json: {"warehouse":"...","namespace":"default","table":"...",
//                "query_text":"rust programming","top_k":10,
//                "text_column":"chunk_text","partition_filter":"agent-42"}
// Returns: {"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}
// distance = negated BM25 score (lower = more relevant).
char* ailake_search_text_json(const char* request_json);

// Equality delete — writes Iceberg delete file and commits Delete snapshot
// request_json: {"warehouse":"...","namespace":"default","table":"...",
//                "column":"id","values":["0","1","2"]}
// Returns: {"ok":true} or {"ok":false,"error":"..."}
// Empty values array is a no-op (returns ok=true without writing a file).
char* ailake_delete_where_json(const char* request_json);

// Schema evolution — metadata-only, no data files rewritten
// request_json: {"warehouse":"...","namespace":"default","table":"...",
//                "add_columns":[{"name":"source_url","type":"string","initial_default":""}],
//                "rename_columns":[{"from":"source_url","to":"url"}]}
// Returns: {"ok":true,"new_schema_id":N}
// Empty arrays = no-op, returns new_schema_id=current schema id.
char* ailake_evolve_schema_json(const char* request_json);

// Multimodal search (search across multiple HNSW columns, fused via RRF)
char* ailake_search_multimodal_json(const char* request_json);

void ailake_free_string(char* ptr);

// Static version string — do NOT free.
const char* ailake_version();
```

### JSON-envelope field reference

All optional fields are `#[serde(default)]` — absent fields deserialize to empty / `None`, backward-compatible with callers that omit them.

| Field | Function | Type | Description |
|---|---|---|---|
| `partition_filter` | search, search_text, search_multimodal | string | Restrict to files with matching `partition_value`. Pruned before centroid check. |
| `partition_by` | write_batch | string | Single-column identity partition column (legacy; prefer `partition_fields`). |
| `partition_value` | write_batch | string | Value for `partition_by` column. |
| `partition_fields` | write_batch | array | Multi-column partition spec: `[{column, transform, column_type}]`. Supports all Iceberg transforms (identity, year, month, day, hour, bucket[N], truncate[N]). |
| `format_version` | write_batch | int | Iceberg format version (2 or 3). Default: 2. |
| `hybrid_text` | search | string | BM25 query text. When set, retrieves 10×top_k HNSW candidates, scores by BM25, fuses via RRF. |
| `text_column` | search, search_text | string | Parquet column for BM25 scoring. Default: `"chunk_text"`. |
| `bm25_weight` | search | float | BM25 weight in RRF fusion (0.0 = pure vector, 1.0 = pure BM25). Default: `0.5`. |

---

## Trino VectorScanConnector

### Build

```bash
cd trino-plugin
gradle wrapper       # creates ./gradlew — run once
./gradlew shadowJar  # builds fat-jar with JNA bundled

# Output
ls -lh build/libs/trino-plugin-0.1.0-plugin.jar
```

### Install

```bash
TRINO_HOME=/opt/trino   # adjust to your installation

# 1. Plugin jar
mkdir -p $TRINO_HOME/plugin/ailake
cp build/libs/trino-plugin-0.1.0-plugin.jar $TRINO_HOME/plugin/ailake/

# 2. Native library — add to Trino's JVM library path
echo "-Djava.library.path=$NATIVE_LIB_DIR" >> $TRINO_HOME/etc/jvm.config
```

### Catalog configuration

Create `$TRINO_HOME/etc/catalog/ailake.properties`:

```properties
# connector.name must be "ailake" — matches VectorScanConnectorFactory.getName()
connector.name=ailake

# Required: absolute or s3:// URI of the AI-Lake table root
ailake.table-uri=s3://my-lake/docs/

# Optional: defaults match typical schema
ailake.vector-column=embedding
ailake.vector-dim=1536
ailake.metric=cosine
ailake.precision=f16

# Optional: embedding model tracking — written to ailake.embedding-model in Iceberg props
# and to per-file key_metadata on every INSERT via this catalog.
# Format: "name" or "name@version"
ailake.embedding-model=text-embedding-3-small@v1
```

Multiple AI-Lake tables → multiple catalog files with different names and `table-uri` values.

### Session properties

| Property | Type | Default | Description |
|---|---|---|---|
| `query_vector` | `varchar` | `""` | Comma-separated f32 values: `"0.1,-0.2,0.3,..."` |
| `top_k` | `integer` | `10` | Nearest neighbors to return |

### Schema

The connector exposes a single table `ailake.default.search` with columns:

| Column | Trino type | Description |
|---|---|---|
| `row_id` | `bigint` | HNSW node ID (maps to Parquet row position) |
| `distance` | `double` | Distance from query vector (lower = more similar) |
| `file_path` | `varchar` | Relative path of the Parquet file within the table |

### Step-by-step walkthrough

**1. Generate a demo table**

```bash
cargo run --example demo -p ailake-query 2>&1 | grep Workspace
# Workspace: /tmp/ailakeABCDEF
```

Update `ailake.table-uri` in the catalog properties to point to that path.

**2. Start Trino**

```bash
$TRINO_HOME/bin/launcher start
$TRINO_HOME/bin/trino   # connect
```

**3. Verify the connector**

```sql
SHOW SCHEMAS FROM ailake;
-- default

SHOW TABLES FROM ailake.default;
-- search

DESCRIBE ailake.default.search;
-- Column    | Type    | Extra | Comment
-- ----------+---------+-------+--------
-- row_id    | bigint  |       |
-- distance  | double  |       |
-- file_path | varchar |       |
```

**4. Set the query vector and search**

The demo table uses `dim=64`. Generate a 64-float CSV:

```sql
-- Set session properties
SET SESSION ailake.query_vector =
  '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,
   0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,
   0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
   0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4';

SET SESSION ailake.top_k = 5;

-- Run vector search
SELECT row_id, ROUND(distance, 6) AS distance, file_path
FROM ailake.default.search
ORDER BY distance;
```

**5. Join with tabular data (via the Iceberg connector)**

```sql
-- Assuming you also have the Iceberg connector pointing to the same table:
SELECT s.row_id, s.distance, i.chunk_text, i.document_title
FROM ailake.default.search s
JOIN iceberg.default.demo_table i ON CAST(s.row_id AS BIGINT) = i.id
ORDER BY s.distance
LIMIT 10;
```

### Running the Trino plugin tests

No running Trino server required:

```bash
cd trino-plugin
./gradlew test --info

# Test classes:
#   VectorScanMetadataTest   — schema discovery (7 tests)
#   VectorScanConnectorTest  — session properties, transaction handle (7 tests)
#   VectorScanSplitManagerTest — split creation from session (5 tests)
#   VectorScanRecordSetTest  — cursor iteration, column types (9 tests)
#   AilakeNativeTest         — graceful degradation, CSV parsing (5 tests)
```

---

## Spark VectorScanStrategy

### Build

```bash
cd spark-plugin
gradle wrapper
./gradlew shadowJar

ls -lh build/libs/spark-plugin-0.1.0-plugin.jar
```

### How the strategy works

```
spark.ailakeSearch(uri, query, topK)
        │
        ▼
  VectorSearchPlan (LogicalPlan LeafNode)
        │
        ▼  VectorScanStrategy.apply()
  VectorScanExec (physical LeafExecNode)
        │
        ▼  doExecute()
  AilakeNative.search()
        │
        ▼  JNA → libailake_jni.so
  Vec<SearchResult>  →  RDD[InternalRow]
        │
        ▼
  DataFrame: (row_id Long, distance Double, file_path String)
```

### Launching Spark

```bash
PLUGIN_JAR=$(pwd)/spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar

# Interactive (spark-shell)
$SPARK_HOME/bin/spark-shell \
  --jars $PLUGIN_JAR \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$NATIVE_LIB_DIR" \
  --conf spark.ui.enabled=false

# PySpark
$SPARK_HOME/bin/pyspark \
  --jars $PLUGIN_JAR \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$NATIVE_LIB_DIR"

# spark-submit (cluster)
spark-submit \
  --jars $PLUGIN_JAR \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  my-job.jar
```

### Scala API (recommended)

```scala
import io.ailake.spark.implicits._

// -- Basic search --
val query: Array[Float] = Array.fill(1536)(0.0f)  // your real embedding here

val results = spark.ailakeSearch(
  tableUri    = "s3://my-lake/docs/",
  queryVector = query,
  topK        = 100,
)
// DataFrame columns: row_id (Long), distance (Double), file_path (String)

results.orderBy("distance").show(10)

// -- Join with Iceberg data to get chunk text --
val iceberg = spark.read.format("iceberg").load("glue.db.my_ailake_table")

results
  .join(iceberg, results("row_id") === iceberg("id"))
  .select("row_id", "distance", "chunk_text", "document_title")
  .orderBy("distance")
  .limit(20)
  .show(truncate = false)

// -- Save top-100 results to Parquet --
spark.ailakeSearch("s3://my-lake/docs/", query, topK = 100)
  .write.parquet("s3://results/rag-candidates/")

// -- Multi-query batch (parallelize queries) --
val queries: Seq[Array[Float]] = loadQueriesFromFile(...)
val allResults = queries.map(q => spark.ailakeSearch("s3://my-lake/docs/", q, 10))
allResults.reduce(_ union _).distinct().write.parquet("s3://results/batch/")
```

### Step-by-step walkthrough with the demo table

```bash
# 1. Generate demo table (dim=64, 1000 rows)
cargo run --example demo -p ailake-query 2>&1 | grep Workspace
# Workspace: /tmp/ailakeXXXXXX
export AILAKE_TABLE=/tmp/ailakeXXXXXX/warehouse/default/demo_table

# 2. Start spark-shell with plugin
$SPARK_HOME/bin/spark-shell \
  --jars $(pwd)/spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$(pwd)/target/release" \
  --conf spark.ui.enabled=false
```

In the Scala prompt:

```scala
// 3. Import implicit
import io.ailake.spark.implicits._

// 4. Build query vector (dim=64 for demo table)
val query = Array.fill(64)(0.5f)

// 5. Search
val results = spark.ailakeSearch(
  tableUri    = sys.env("AILAKE_TABLE"),
  queryVector = query,
  topK        = 10,
)

// 6. Inspect
results.printSchema()
// root
//  |-- row_id: long (nullable = false)
//  |-- distance: double (nullable = false)
//  |-- file_path: string (nullable = false)

results.show()
// +------+--------------------+-----------------------------+
// |row_id|distance            |file_path                    |
// +------+--------------------+-----------------------------+
// |0     |0.0                 |data/part-00000.parquet      |
// |12    |0.031456...         |data/part-00000.parquet      |
// ...

// 7. Verify strategy ran
results.queryExecution.executedPlan   // should show VectorScanExec
```

### PySpark via py4j

For Python workflows, prefer `ailake-py` (the native Python SDK in `ailake-py/`). If you must call the JVM plugin from PySpark:

```python
# Access JVM via py4j gateway
jvm = spark._jvm

# Build float array
query_java = jvm.Array(jvm.Float.TYPE, 64)
for i, v in enumerate([0.5] * 64):
    query_java[i] = v

# Call native search directly (bypasses Spark planner — for scripting only)
native = jvm.io.ailake.spark.AilakeNative
rows = native.search(table_uri, query_java, 10)

for r in rows:
    print(f"row_id={r.rowId()}  distance={r.distance():.6f}  file={r.filePath()}")
```

### Running the Spark plugin tests

```bash
cd spark-plugin
./gradlew test

# Test classes:
#   VectorSearchPlanTest       — output schema, equals/hashCode (8 tests)
#   VectorScanStrategyTest     — plan→exec conversion (6 tests)
#   AilakeNativeTest           — graceful degradation (4 tests)
#   AilakeSparkExtensionsTest  — local SparkSession, end-to-end (5 tests)
#                                ↑ takes ~15s — starts embedded SparkSession
```

---

## Flink connector (`ailake-flink`)

`ailake-flink` implements `DynamicTableSink` via `VectorScanTableFactory` — a Flink Table API / SQL connector that writes AI-Lake tables as part of Flink streaming or batch jobs.

### Build

```bash
cd ailake-flink
./gradlew shadowJar
# Output: build/libs/ailake-flink-<version>-plugin.jar
```

### Add to Flink job classpath

```bash
flink run \
  --jarfile my-pipeline.jar \
  --classpath ailake-flink-<version>-plugin.jar \
  -Dtaskmanager.extraLibFolders=/opt/ailake/lib  # directory containing libailake_jni.so
```

### SQL DDL

```sql
-- Create sink table (Flink SQL)
CREATE TABLE ailake_sink (
    id         STRING,
    text       STRING,
    embedding  ARRAY<FLOAT>
) WITH (
    'connector'        = 'ailake',
    'table.uri'        = 's3://my-lake/docs/',
    'vector.column'    = 'embedding',
    'dim'              = '1536',
    'metric'           = 'cosine',
    'precision'        = 'f16',
    'partition.fields' = '[{"column":"topic_id","transform":"identity","column_type":"int"}]',
    'format.version'   = '3'
);

-- Insert from another Flink table
INSERT INTO ailake_sink SELECT id, text, embedding FROM source_table;
```

### Kotlin API (low-level)

```kotlin
import io.ailake.flink.internal.AilakeNativeLoader

val loader = AilakeNativeLoader()

// Write batch (auto-selects IVF-PQ or HNSW based on hardware detection)
loader.writeBatch(
    tableUri       = "s3://my-lake/docs/",
    namespace      = "default",
    tableName      = "docs",
    dim            = 1536,
    ids            = listOf("doc-1", "doc-2"),
    embeddings     = listOf(floatArrayOf(...), floatArrayOf(...)),
    metric         = "cosine",
    partitionFields = listOf(
        AilakeNativeLoader.PartitionFieldDef("topic_id", "identity", "int"),
    ),
    formatVersion  = 3,
)

// Equality delete (writes Iceberg delete file)
loader.deleteWhere(tableUri, "default", "docs", "id", listOf("doc-1", "doc-2"))

// Schema evolution (metadata-only)
loader.evolveSchema(
    tableUri, "default", "docs",
    addCols    = listOf(AilakeNativeLoader.AddColReq("source_url", "string")),
    renameCols = listOf(AilakeNativeLoader.RenameColReq("source_url", "url")),
)
```

### Supported options

| Option | Default | Description |
|---|---|---|
| `connector` | — | Must be `"ailake"` |
| `table.uri` | required | AI-Lake table root URI (S3, GCS, Azure, local) |
| `vector.column` | `"embedding"` | Arrow column containing `ARRAY<FLOAT>` values |
| `dim` | required | Embedding dimension |
| `metric` | `"cosine"` | Distance metric (`cosine`, `euclidean`, `dot_product`) |
| `precision` | `"f16"` | Storage precision (`f32`, `f16`, `i8`) |
| `partition.fields` | `"[]"` | JSON array of `{column, transform, column_type}` objects |
| `format.version` | `"2"` | Iceberg format version (`2` or `3`) |

---

## Delete and schema evolution (all plugins)

All three JVM plugins expose `deleteWhere()` and `evolveSchema()` via the JNA bridge.

### Spark (Scala)

```scala
import io.ailake.spark.AilakeNative

// Equality delete — returns true on success, false on empty values or absent lib
val ok = AilakeNative.deleteWhere(
  tableUri  = "s3://my-lake/docs/",
  namespace = "default",
  tableName = "docs",
  column    = "id",
  values    = Seq("doc-1", "doc-2"),
)

// Schema evolution — returns new schema_id, 0 for no-op, -1 on error
val schemaId = AilakeNative.evolveSchema(
  tableUri   = "s3://my-lake/docs/",
  namespace  = "default",
  tableName  = "docs",
  addCols    = Seq(AilakeNative.AddColReq("source_url", "string", None)),
  renameCols = Seq(AilakeNative.RenameColReq("source_url", "url")),
)
```

### Trino (Kotlin)

```kotlin
import io.ailake.trino.AilakeNative

AilakeNative.deleteWhere("s3://my-lake/docs/", "default", "docs", "id", listOf("doc-1"))

val schemaId = AilakeNative.evolveSchema(
    "s3://my-lake/docs/", "default", "docs",
    addCols    = listOf(AilakeNative.AddColReq("source_url", "string", null)),
    renameCols = listOf(AilakeNative.RenameColReq("source_url", "url")),
)
```

### Flink (Kotlin)

```kotlin
import io.ailake.flink.internal.AilakeNativeLoader

val loader = AilakeNativeLoader()
loader.deleteWhere(tableUri, "default", "docs", "id", listOf("doc-1"))
loader.evolveSchema(tableUri, "default", "docs", addCols = listOf(...), renameCols = listOf(...))
```

---

## Cross-modal search (Phase 8)

All three JVM plugins expose `searchMultimodal()` backed by `ailake_search_multimodal_json` in `libailake_jni.so`.

### C-ABI entry point

```
ailake_search_multimodal_json(request_json: *const c_char) -> *mut c_char
ailake_free_string(ptr: *mut c_char)
```

**Request JSON**:
```json
{
  "warehouse": "s3://my-lake/",
  "namespace": "default",
  "table": "docs",
  "queries": [
    { "col": "embedding",         "query": [0.1, 0.2, ...], "weight": 0.6, "dim": 0 },
    { "col": "context_embedding", "query": [0.3, 0.1, ...], "weight": 0.4, "dim": 0 }
  ],
  "top_k": 20
}
```

`dim: 0` = infer from stored column metadata.

**Response JSON**:
```json
{
  "ok": true,
  "results": [
    { "row_id": 42, "rrf_score": 0.0312, "file_path": "data/part-00001.parquet" }
  ]
}
```

`rrf_score = Σ weight_i / (60 + rank_i)` — higher is better. Rank is 0-indexed.

### Spark (Scala)

```scala
import io.ailake.spark.AilakeNative
import io.ailake.spark.MultimodalSearchRow

val queries = Array(
  (0.6f, "embedding",         queryVec1),
  (0.4f, "context_embedding", queryVec2),
)
val rows: Array[MultimodalSearchRow] = AilakeNative.searchMultimodal(tableUri, queries, topK = 20)
rows.foreach(r => println(s"row_id=${r.rowId}  rrf=${r.rrfScore:.4f}  file=${r.filePath}"))
```

### Trino (Kotlin)

```kotlin
import io.ailake.trino.AilakeNative
import io.ailake.trino.MultimodalSearchRow

val queries = listOf(
    Triple(0.6f, "embedding",         queryVec1),
    Triple(0.4f, "context_embedding", queryVec2),
)
val rows: List<MultimodalSearchRow> = AilakeNative.searchMultimodal(tableUri, queries, topK = 20)
```

### Flink (Kotlin)

```kotlin
import io.ailake.flink.internal.AilakeNativeLoader

val loader = AilakeNativeLoader()

// Cross-modal RRF search
val queries = listOf(
    Triple(0.6f, "embedding",         queryVec1),
    Triple(0.4f, "context_embedding", queryVec2),
)
val rows = loader.searchMultimodal(tableUri, queries, topK = 20)

// Vector search with optional BM25 hybrid scoring (Phase 9)
val hybridRows = loader.search(
    tableUri, queryVec, topK = 10,
    hybridText  = "rust programming",   // optional — null = pure vector
    textColumn  = "chunk_text",         // optional, default "chunk_text"
    bm25Weight  = 0.5f,                 // optional, default 0.5
)

// Pure BM25 full-text search (Phase 9)
val textRows = loader.searchText(
    tableUri,
    queryText       = "rust programming async",
    topK            = 10,
    textColumn      = "chunk_text",
    partitionFilter = "agent-42",       // optional
)
```

### Architecture diagram (multimodal path)

```
JVM caller
  └─ searchMultimodal(uri, queries, topK)
       └─ AilakeNative.searchMultimodal()         [Spark/Trino/Flink]
            └─ JNA: ailake_search_multimodal_json  [libailake_jni.so]
                 └─ search_multimodal()             [ailake-query, Rust]
                      ├─ Per-column HNSW search     [ailake-index]
                      └─ Reciprocal Rank Fusion      [score = Σ w/(60+rank)]
```

---

## Native library deployment

### Local / development

```bash
# Add to shell profile
export LD_LIBRARY_PATH=/path/to/target/release:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=/path/to/target/release:$LD_LIBRARY_PATH # macOS
```

### Trino server

```
# etc/jvm.config
-Djava.library.path=/opt/ailake/lib
```

### Spark cluster (YARN / Kubernetes)

```bash
# Ship the native lib with the job
spark-submit \
  --files /path/to/libailake_jni.so#libailake_jni.so \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=." \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=." \
  ...
```

For Kubernetes, bake `libailake_jni.so` into the Spark executor Docker image:
```dockerfile
COPY target/release/libailake_jni.so /opt/ailake/lib/
ENV LD_LIBRARY_PATH=/opt/ailake/lib
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `UnsatisfiedLinkError: libailake_jni` | native lib not on `java.library.path` | Add `-Djava.library.path=...` to JVM config |
| Trino returns 0 rows | `query_vector` session prop empty | `SET SESSION ailake.query_vector = '...'` |
| `ailake.table-uri is required` | Catalog properties file missing required key | Add `ailake.table-uri=...` to properties file |
| `ClassNotFoundException: AilakeSparkExtensions` | Plugin jar not on Spark classpath | Pass `--jars /path/to/spark-plugin-...jar` |
| `spark.ailakeSearch` not found | Missing import | Add `import io.ailake.spark.implicits._` |
| Spark returns empty DataFrame | Native lib absent (expected in tests) | Ensure `java.library.path` points to `target/release/` |
| `dim mismatch` | `ailake.vector-dim` in catalog props ≠ actual table dim | Match the value used when writing the table |
| `query dim=N does not match table dim=M (table model: ...)` | Query vector dimension differs from the table's stored dim | Use the correct embedding model for this table; the error message names the expected model |
