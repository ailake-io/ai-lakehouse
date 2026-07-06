# JVM Integration Guide — Spark, Databricks, Trino, Flink

All three engines share one native library (`libailake_jni.so`) and one
JSON-envelope C-ABI. Each engine gets a thin JVM adapter that translates its
SPI calls into JNA calls against the Rust core.

```
Engine (Spark / Trino / Flink)
   └─ JVM plugin (Scala / Kotlin)
        └─ JNA: ailake_*_json()   ← libailake_jni.so (Rust cdylib)
                                        └─ ailake-query (HNSW + pruning)
```

---

## 1. Prerequisites

| Tool | Version |
|---|---|
| JDK | 17+ |
| Gradle | 8+ (or use `./gradlew` wrapper) |
| Rust + Cargo | 1.75+ stable (only for source build) |
| Spark | 3.5.x |
| Trino | 430+ |
| Flink | 1.18+ |

---

## 2. Native library and JARs

### 2A — Download pre-built (recommended)

```bash
TAG=v0.1.1          # GitHub release tag — replace with target release (Rust/PyPI version)
JAR_VERSION=0.1.0   # JVM plugin version — gradle, versioned independently of TAG; check the release page

# Native library (required by all three engines)
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/${TAG}/libailake_jni.so

# Engine JARs (download the ones you need)
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/${TAG}/spark-plugin-${JAR_VERSION}-plugin.jar
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/${TAG}/trino-plugin-${JAR_VERSION}-plugin.jar
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/${TAG}/ailake-flink-${JAR_VERSION}-plugin.jar
```

### 2B — Build from source

```bash
# 1. Native library (shared by all engines)
cargo build --release -p ailake-jni
# → target/release/libailake_jni.so  (Linux)
#   target/release/libailake_jni.dylib (macOS)
#   target/release/ailake_jni.dll     (Windows)

# 2. Spark plugin
cd spark-plugin && ./gradlew shadowJar
# → build/libs/spark-plugin-<version>-plugin.jar

# 3. Trino plugin
cd ../trino-plugin && ./gradlew shadowJar
# → build/libs/trino-plugin-<version>-plugin.jar

# 4. Flink connector
cd ../ailake-flink && ./gradlew shadowJar
# → build/libs/ailake-flink-<version>-plugin.jar
```

### 2C — Native library deployment

The JVM plugin loads the library via JNA in this order:
1. System property `-Dailake.native.lib=/full/path/to/libailake_jni.so`
2. Env var `AILAKE_NATIVE_LIB=/full/path/to/libailake_jni.so`
3. Standard JNA search: `java.library.path`, `jna.library.path`, classpath resources

---

## 3. Spark

### 3A — Starting Spark with the plugin

**spark-shell (interactive):**

```bash
PLUGIN_JAR=/opt/ailake/spark-plugin-0.1.0-plugin.jar
LIB_DIR=/opt/ailake/lib

$SPARK_HOME/bin/spark-shell \
  --jars $PLUGIN_JAR \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$LIB_DIR" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=$LIB_DIR"
```

**spark-submit (cluster):**

```bash
spark-submit \
  --jars $PLUGIN_JAR \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  my-job.jar
```

**Kubernetes — bake into executor image:**

```dockerfile
FROM apache/spark:3.5.1
COPY libailake_jni.so /opt/ailake/lib/
ENV LD_LIBRARY_PATH=/opt/ailake/lib
```

**SparkSession (programmatic):**

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .config("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
  .config("spark.driver.extraJavaOptions", "-Djava.library.path=/opt/ailake/lib")
  .getOrCreate()
```

### 3B — Vector search (Scala)

```scala
import io.ailake.spark.implicits._

val query: Array[Float] = myEmbeddingModel.embed("What is geometric pruning?")

// Basic search → DataFrame(row_id: Long, distance: Double, file_path: String)
val results = spark.ailakeSearch(
  tableUri    = "s3://my-lake/docs/",
  queryVector = query,
  topK        = 100,
)
results.orderBy("distance").show(10)

// Hybrid BM25+vector search
val hybridRows = AilakeNative.search(
  tableUri    = "s3://my-lake/docs/",
  query       = query,
  topK        = 20,
  hybridText  = Some("geometric pruning"),
  textColumn  = "chunk_text",
  bm25Weight  = 0.4f,
)

// Join with Iceberg to get full row data
val iceberg = spark.read.format("iceberg").load("s3://my-lake/docs/")

results
  .join(iceberg, results("row_id") === iceberg("id"))
  .select("row_id", "distance", "chunk_text", "document_title")
  .orderBy("distance")
  .limit(20)
  .show(truncate = false)

// Multi-query batch
val queries: Seq[Array[Float]] = loadQueries(...)
queries
  .map(q => spark.ailakeSearch("s3://my-lake/docs/", q, 10))
  .reduce(_ union _)
  .distinct()
  .write.parquet("s3://results/batch/")
```

### 3C — Writing data (Scala)

```scala
import io.ailake.spark.implicits._

// Using ailakeWrite() implicit
spark.ailakeWrite(
  tableUri     = "s3://my-lake/docs/",
  df           = chunksDF,            // must have: id (Long), embedding (Array[Double])
  vectorColumn = "embedding",
  idColumn     = "id",
  metric       = "cosine",
  precision    = "f16",
)

// Using DataSource V2 directly (more options)
chunksDF.write
  .format("io.ailake.spark.AilakeDataSource")   // or short alias "ailake"
  .option("tableUri",       "s3://my-lake/docs/")
  .option("vectorColumn",   "embedding")
  .option("dim",            "1536")
  .option("metric",         "cosine")
  .option("precision",      "f16")
  .option("namespace",      "default")
  .option("tableName",      "docs")
  .option("format-version", "2")
  // optional Iceberg partition fields (JSON array)
  .option("partition-fields",
    """[{"column":"topic_id","transform":"identity","column_type":"int"}]""")
  .save()

// Low-level via AilakeNative.writeBatch
import io.ailake.spark.AilakeNative

val snapshotId = AilakeNative.writeBatch(
  tableUri        = "s3://my-lake/docs/",
  namespace       = "default",
  tableName       = "docs",
  vectorColumn    = "embedding",
  dim             = 1536,
  metric          = "cosine",
  precision       = "f16",
  ids             = Seq(1L, 2L, 3L),
  embeddings      = Seq(vec1, vec2, vec3),
  embeddingModel  = Some("text-embedding-3-small@v1"),
  formatVersion   = 2,
  ftsColumns      = Seq("chunk_text"),
  deferred        = false,             // true = Parquet now, index async
)
```

### 3D — FTS and multimodal search (Scala)

```scala
import io.ailake.spark.AilakeNative

// Full-text search (Tantivy O(log N) or BM25 fallback)
val ftsRows = AilakeNative.searchText(
  tableUri    = "s3://my-lake/docs/",
  namespace   = "default",
  tableName   = "docs",
  queryText   = "machine learning embeddings",
  textColumns = Seq("chunk_text"),
  topK        = 10,
)
ftsRows.foreach(r => println(s"row=${r.rowId}  score=${r.distance}"))

// Cross-modal RRF (text + image, Phase 8)
val mmRows = AilakeNative.searchMultimodal(
  tableUri = "s3://my-lake/media/",
  queries  = Seq(
    ("embedding",       textVec,  0.7f),
    ("image_embedding", imageVec, 0.3f),
  ),
  topK = 20,
)
mmRows.foreach(r => println(s"row=${r.rowId}  rrf=${r.rrfScore}"))
```

### 3E — Delete, schema evolution, compact (Scala)

```scala
// Equality delete — writes Iceberg delete file, no data rewrite
AilakeNative.deleteWhere(
  tableUri  = "s3://my-lake/docs/",
  namespace = "default",
  tableName = "docs",
  column    = "id",
  values    = Seq("doc-1", "doc-2"),
)

// Schema evolution — metadata-only
val newSchemaId = AilakeNative.evolveSchema(
  tableUri   = "s3://my-lake/docs/",
  namespace  = "default",
  tableName  = "docs",
  addCols    = Seq(AilakeNative.AddColReq("language", "string", Some(""""en""""))),
  renameCols = Seq(AilakeNative.RenameColReq("old_text", "chunk_text")),
)

// Compact small files
AilakeNative.compact(
  tableUri        = "s3://my-lake/docs/",
  namespace       = "default",
  tableName       = "docs",
  minFiles        = 4,
  targetSizeBytes = 128L * 1024 * 1024,
)
```

### 3F — Running tests

```bash
cd spark-plugin
./gradlew test

# Test classes:
#   VectorSearchPlanTest       — output schema, equals/hashCode
#   VectorScanStrategyTest     — plan→exec conversion
#   AilakeNativeTest           — graceful degradation when lib absent
#   AilakeSparkExtensionsTest  — end-to-end with embedded SparkSession (~15 s)
```

---

## 4. Databricks

Databricks Runtime (DBR) runs Spark 3.5 under the hood. The Spark plugin works
without changes. The native library requires a cluster init script to be present
on all nodes before the Spark executor JVM starts.

### 4A — Cluster setup

**Step 1 — Upload artifacts to DBFS:**

```bash
databricks fs cp libailake_jni.so         dbfs:/FileStore/ailake/libailake_jni.so
databricks fs cp spark-plugin-0.1.0-plugin.jar \
                                          dbfs:/FileStore/ailake/spark-plugin.jar
```

**Step 2 — Init script** (`/dbfs/databricks/scripts/install_ailake.sh`):

```bash
#!/bin/bash
# Install libailake_jni.so on every node before JVM starts.
set -e
mkdir -p /opt/ailake/lib
cp /dbfs/FileStore/ailake/libailake_jni.so /opt/ailake/lib/
chmod 755 /opt/ailake/lib/libailake_jni.so
```

Add this script in the cluster **Init Scripts** setting (the path must be a DBFS
path or a workspace path visible to all nodes).

**Step 3 — Cluster Spark config:**

```ini
spark.sql.extensions     io.ailake.spark.AilakeSparkExtensions
spark.driver.extraJavaOptions  -Djava.library.path=/opt/ailake/lib
spark.executor.extraJavaOptions -Djava.library.path=/opt/ailake/lib
```

**Step 4 — Attach the JAR as a cluster library:**

In the Databricks UI: **Cluster → Libraries → Install → DBFS/S3**  
Path: `dbfs:/FileStore/ailake/spark-plugin.jar`

Or via Terraform:

```hcl
resource "databricks_library" "ailake" {
  cluster_id = databricks_cluster.my_cluster.id
  jar        = "dbfs:/FileStore/ailake/spark-plugin.jar"
}
```

### 4B — Scala notebook

```scala
%scala
import io.ailake.spark.implicits._

val query: Array[Float] = /* your embedding */ Array.fill(1536)(0.0f)

val results = spark.ailakeSearch(
  tableUri    = "s3://my-lake/docs/",
  queryVector = query,
  topK        = 20,
)
display(results.orderBy("distance"))
```

### 4C — Python notebook (PySpark + ailake-py)

Install `ailake` on the cluster (**Libraries → PyPI → ailake**), then:

```python
%pip install ailake   # or install via cluster UI

import ailake
import numpy as np

query_vec = np.random.rand(1536).astype(np.float32)

# Pure Python SDK — does not require the Spark plugin
results = ailake.search(
    "s3://my-lake/docs/",
    query_vec.tolist(),
    top_k=20,
    fetch_data=True,
)
df = results.to_pandas()
display(df)
```

For large-scale batch queries that need Spark parallelism, use the JVM plugin
via py4j:

```python
jvm = spark._jvm

# Build float[]
query_java = jvm.Array(jvm.Float.TYPE, 1536)
for i, v in enumerate(query_vec.tolist()):
    query_java[i] = v

native = jvm.io.ailake.spark.AilakeNative
rows = native.search("s3://my-lake/docs/", query_java, 20)

for r in rows:
    print(f"row={r.rowId()}  dist={r.distance():.4f}  file={r.filePath()}")
```

### 4D — Unity Catalog — read AI-Lake tables as Iceberg

AI-Lake tables are standard Iceberg — Unity Catalog can register and query them
without any plugin for the tabular columns:

```sql
-- Register external Iceberg table in Unity Catalog
CREATE TABLE my_catalog.my_schema.docs
USING iceberg
LOCATION 's3://my-lake/docs/';

-- Query tabular data normally (embedding column appears as BINARY)
SELECT id, chunk_text, created_at
FROM my_catalog.my_schema.docs
WHERE language = 'en'
LIMIT 100;
```

For vector search, use `ailake.search()` (Python SDK) or `spark.ailakeSearch()`
(Scala plugin) and join back to the Unity Catalog table:

```python
import ailake

results = ailake.search("s3://my-lake/docs/", query_vec, top_k=20)
hits_df = results.to_pandas()

# Join with Unity Catalog table via Spark
hits_spark = spark.createDataFrame(hits_df)
uc_df      = spark.table("my_catalog.my_schema.docs")

final = hits_spark.join(
    uc_df.select("id", "chunk_text", "document_title"),
    hits_spark["row_id"] == uc_df["id"],
).orderBy("distance")
display(final)
```

### 4E — Delta + AI-Lake hybrid

Keep structured data in Delta, vectors in AI-Lake:

```python
# Write structured data to Delta
chunks_df.write.format("delta").mode("append").saveAsTable("my_catalog.db.chunks")

# Write embeddings to AI-Lake
writer = ailake.TableWriter("s3://my-lake/chunks/", dim=1536)
writer.write_batch(texts, embeddings)
writer.commit()

# At query time: search AI-Lake → row_ids → fetch from Delta
results = ailake.search("s3://my-lake/chunks/", query_vec, top_k=20)
row_ids = [r["row_id"] for r in results]

spark.sql(f"""
    SELECT id, chunk_text, title
    FROM my_catalog.db.chunks
    WHERE id IN ({','.join(map(str, row_ids))})
""")
```

---

## 5. Trino

### 5A — Install

```bash
TRINO_HOME=/opt/trino

# 1. Plugin jar
mkdir -p $TRINO_HOME/plugin/ailake
cp trino-plugin-0.1.0-plugin.jar $TRINO_HOME/plugin/ailake/

# 2. Native library
mkdir -p /opt/ailake/lib
cp libailake_jni.so /opt/ailake/lib/

# 3. JVM config — add to etc/jvm.config
echo "-Djava.library.path=/opt/ailake/lib" >> $TRINO_HOME/etc/jvm.config
```

### 5B — Catalog configuration

Create `$TRINO_HOME/etc/catalog/ailake.properties`:

```properties
# The connector name must be exactly "ailake"
connector.name=ailake

# Required: table root URI
ailake.table-uri=s3://my-lake/docs/

# Optional — defaults shown
ailake.vector-column=embedding
ailake.vector-dim=1536
ailake.metric=cosine
ailake.precision=f16
ailake.embedding-model=text-embedding-3-small@v1
```

Multiple tables → multiple catalog files with different names:

```bash
# catalog/ailake_docs.properties  →  ailake_docs.default.search
# catalog/ailake_media.properties →  ailake_media.default.search
```

### 5C — Querying

```sql
-- Inspect available catalogs / schemas
SHOW SCHEMAS FROM ailake;
SHOW TABLES  FROM ailake.default;

-- Schema: row_id bigint, distance double, file_path varchar
DESCRIBE ailake.default.search;

-- Set session properties then query
SET SESSION ailake.query_vector =
    '0.1,0.2,0.3,...';   -- comma-separated f32 values (dim must match table)
SET SESSION ailake.top_k = 10;

SELECT row_id, ROUND(distance, 6) AS dist, file_path
FROM   ailake.default.search
ORDER  BY distance;

-- Join with Iceberg data (requires Iceberg connector pointing to same table)
SELECT s.row_id, s.distance, i.chunk_text, i.document_title
FROM   ailake.default.search s
JOIN   iceberg.default.docs  i ON CAST(s.row_id AS BIGINT) = i.id
ORDER  BY s.distance
LIMIT  10;
```

**Session properties:**

| Property | Type | Default | Description |
|---|---|---|---|
| `query_vector` | `varchar` | `""` | Comma-separated f32 values |
| `top_k` | `integer` | `10` | Nearest neighbors to return |

### 5D — Nessie catalog (demo stack)

When using the demo Docker stack, Trino reads AI-Lake tables via the Nessie
catalog (`ailake.properties` in `tests/docker/demo/trino-catalog/`):

```properties
connector.name=iceberg
iceberg.catalog.type=nessie
iceberg.nessie-catalog.uri=http://nessie:19120/api/v1
iceberg.nessie-catalog.ref=main
iceberg.nessie-catalog.default-warehouse-dir=file:///data/ailake_demo
```

```sql
-- Query tabular columns from a Nessie-registered AI-Lake table
SELECT text, length(embedding) AS emb_bytes
FROM   ailake.default.table
LIMIT  10;
```

### 5E — Running Trino plugin tests

No running Trino server required:

```bash
cd trino-plugin
./gradlew test --info

# Test classes:
#   VectorScanMetadataTest     — schema discovery
#   VectorScanConnectorTest    — session properties, transaction handle
#   VectorScanSplitManagerTest — split creation from session
#   VectorScanRecordSetTest    — cursor iteration, column types
#   AilakeNativeTest           — graceful degradation, CSV parsing
```

---

## 6. Flink

### 6A — Add to job classpath

```bash
flink run \
  --jar my-pipeline.jar \
  --classpath ailake-flink-0.1.0-plugin.jar \
  -Dtaskmanager.extraLibFolders=/opt/ailake/lib
```

Or add to `$FLINK_HOME/lib/` so all jobs on the cluster pick it up:

```bash
cp ailake-flink-0.1.0-plugin.jar /opt/flink/lib/
cp libailake_jni.so               /opt/ailake/lib/
echo 'env.java.opts.taskmanager: -Djava.library.path=/opt/ailake/lib' \
    >> /opt/flink/conf/flink-conf.yaml
```

### 6B — SQL DDL (sink)

```sql
-- Create AI-Lake sink table in Flink SQL
CREATE TABLE ailake_docs (
    id        STRING,
    text      STRING,
    embedding ARRAY<FLOAT>
) WITH (
    'connector'        = 'ailake',
    'table.uri'        = 's3://my-lake/docs/',
    'vector.column'    = 'embedding',
    'dim'              = '1536',
    'metric'           = 'cosine',
    'precision'        = 'f16',
    'format.version'   = '2',
    -- Optional Iceberg partition fields (JSON array)
    'partition.fields' = '[{"column":"topic_id","transform":"identity","column_type":"int"}]'
);

-- Stream embeddings from Kafka source table to AI-Lake
INSERT INTO ailake_docs
SELECT id, text, embedding
FROM kafka_embeddings_source;

-- Batch ingest from a Hive table
INSERT INTO ailake_docs
SELECT id, text, embedding FROM hive_catalog.default.chunks;
```

### 6C — Kotlin API (low-level)

```kotlin
import io.ailake.flink.internal.AilakeNativeLoader

val loader = AilakeNativeLoader

// Vector search
val hits = loader.search(
    warehouse = "s3://my-lake/",
    namespace = "default",
    table     = "docs",
    vecCol    = "embedding",
    dim       = 1536,
    query     = myEmbedding,
    topK      = 10,
    hybridText     = "machine learning",  // optional — BM25+vector RRF
    textColumn     = "chunk_text",
    bm25Weight     = 0.5f,
    partitionFilter = "agent-001",       // optional
)
hits.forEach { println("row=${it.row_id}  dist=${it.distance}  file=${it.file_path}") }

// BM25 full-text search
val ftsHits = loader.searchText(
    warehouse       = "s3://my-lake/",
    namespace       = "default",
    table           = "docs",
    queryText       = "rust async patterns",
    topK            = 10,
    textColumns     = listOf("chunk_text"),
    partitionFilter = null,
)

// Cross-modal RRF (Phase 8)
val mmHits = loader.searchMultimodal(
    warehouse = "s3://my-lake/",
    namespace = "default",
    table     = "media",
    queries   = listOf(
        Triple(0.7f, "embedding",       textVec),
        Triple(0.3f, "image_embedding", imageVec),
    ),
    topK = 20,
)

// Write batch
loader.writeBatch(
    warehouse        = "s3://my-lake/",
    namespace        = "default",
    table            = "docs",
    vecCol           = "embedding",
    dim              = 1536,
    metric           = "cosine",
    precision        = "f16",
    ids              = listOf("doc-1", "doc-2"),
    embeddings       = listOf(vec1, vec2),
    embeddingModel   = "text-embedding-3-small@v1",
    formatVersion    = 2,
    ftsColumns       = listOf("chunk_text"),
    deferred         = false,
)

// Equality delete
loader.deleteWhere(
    warehouse = "s3://my-lake/",
    namespace = "default",
    table     = "docs",
    column    = "id",
    values    = listOf("doc-1", "doc-2"),
)

// Schema evolution
loader.evolveSchema(
    warehouse  = "s3://my-lake/",
    namespace  = "default",
    table      = "docs",
    addCols    = listOf(
        AilakeNativeLoader.AddColReq("language", "string", initialDefault = """"en""""),
    ),
    renameCols = listOf(
        AilakeNativeLoader.RenameColReq("old_text", "chunk_text"),
    ),
)
```

### 6D — Supported DDL options

| Option | Default | Description |
|---|---|---|
| `connector` | required | Must be `"ailake"` |
| `table.uri` | required | AI-Lake table root URI |
| `vector.column` | `"embedding"` | Column containing `ARRAY<FLOAT>` |
| `dim` | required | Embedding dimension |
| `metric` | `"cosine"` | `cosine` \| `euclidean` \| `dot_product` |
| `precision` | `"f16"` | `f32` \| `f16` \| `i8` |
| `partition.fields` | `"[]"` | JSON array of `{column, transform, column_type}` |
| `format.version` | `"2"` | Iceberg format version (`2` or `3`) |

### 6E — Running Flink tests

```bash
cd ailake-flink
./gradlew test

# Test classes:
#   AilakeVectorConnectorFactoryTest — DDL option parsing, factory lookup
#   AilakeNativeLoaderTest           — data classes, JSON payload shape (no native lib needed)
#   AilakeVectorTableSourceTest      — AilakeInputFormat.open() degrades to an empty
#                                      result set when the native lib can't be loaded,
#                                      instead of failing the Flink task
#   AilakeJniIntegrationTest         — end-to-end when AILAKE_JNI_TEST=1
```

---

## 7. Cross-engine reference — delete and schema evolution

All three JVM plugins expose the same operations via the JSON-envelope ABI.

### Delete (equality delete — no data rewrite)

| Engine | Call |
|---|---|
| Spark | `AilakeNative.deleteWhere(tableUri, ns, table, col, values)` |
| Trino | `AilakeNative.deleteWhere(tableUri, ns, table, col, values)` (Kotlin) |
| Flink | `AilakeNativeLoader.deleteWhere(warehouse, ns, table, col, values)` |

### Schema evolution (metadata-only)

| Engine | Call |
|---|---|
| Spark | `AilakeNative.evolveSchema(tableUri, ns, table, addCols, renameCols)` |
| Trino | `AilakeNative.evolveSchema(...)` (Kotlin) |
| Flink | `AilakeNativeLoader.evolveSchema(...)` |

### Compact

| Engine | Call |
|---|---|
| Spark | `AilakeNative.compact(tableUri, ns, table, minFiles, targetSizeBytes)` |
| Flink | `AilakeNativeLoader.compact(...)` |
| Python | `ailake.compact(path, min_files=4, target_size_bytes=128*1024*1024)` |

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `UnsatisfiedLinkError: libailake_jni` | Native lib not on JNA search path | Add `-Djava.library.path=/opt/ailake/lib` to JVM opts, or set `AILAKE_NATIVE_LIB` |
| Spark: `spark.ailakeSearch` not found | Missing import | `import io.ailake.spark.implicits._` |
| Spark: `ClassNotFoundException: AilakeSparkExtensions` | JAR not on classpath | Pass `--jars /path/to/spark-plugin.jar` |
| Spark: empty DataFrame | Native lib absent (graceful degradation) | Verify `java.library.path` points to `libailake_jni.so` |
| Trino: 0 rows | `query_vector` session prop empty | `SET SESSION ailake.query_vector = '...'` |
| Trino: `ailake.table-uri is required` | Missing catalog property | Add `ailake.table-uri=...` to `ailake.properties` |
| Flink: `dim mismatch` | `dim` DDL option ≠ table dim | Match the value used when writing |
| All: `query dim=N does not match table dim=M` | Wrong embedding model | Use the model named in the error; it matches `ailake.embedding-model` in Iceberg metadata |
| Databricks: lib not found after init script | Init script not run yet | Restart cluster after adding init script; verify DBFS path |
| Databricks: `ailake.native.lib` mismatch | Init script path differs from JVM opt | Set both to `/opt/ailake/lib/libailake_jni.so` |

---

## Related docs

- [JVM Plugins Spec](../specs/JVM_PLUGINS.md) — C-ABI JSON-envelope protocol, full field reference
- [File Format Spec](../specs/FILE_FORMAT.md) — AILK section layout
- [Python Integration](PYTHON_INTEGRATION.md) — PyO3 SDK (preferred for Python workloads on Databricks)
- [Go Integration](GO_INTEGRATION.md) — pure-Go client
- [C++ Integration](CPP_INTEGRATION.md) — C++17 header-only client
- [DBT Integration](DBT_INTEGRATION.md) — dbt pipelines on Spark / Trino / DuckDB
