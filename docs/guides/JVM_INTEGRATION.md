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
| Trino | **430** (pinned — see §5A note; Trino 460 breaks connector construction) |
| Flink | 1.18+ |

---

## 2. Native library and JARs

### 2A — Download pre-built (recommended)

```bash
TAG=v0.1.8          # GitHub release tag — replace with target release (Rust/PyPI version)
JAR_VERSION=0.1.8   # JVM plugin version — gradle, versioned independently of TAG; check the release page

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
PLUGIN_JAR=/opt/ailake/spark-plugin-0.1.8-plugin.jar
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

// Hybrid BM25+vector search — DataFrame-level (ailakeSearch's hybridText/textColumn/
// bm25Weight params); low-level AilakeNative.search(hybridText=...) also still available.
val hybridDf = spark.ailakeSearch(
  tableUri    = "s3://my-lake/docs/",
  queryVector = query,
  topK        = 20,
  hybridText  = Some("geometric pruning"),
  textColumn  = "chunk_text",
  bm25Weight  = 0.4f,
)

// Join with Iceberg to get full row data — or skip the JOIN entirely with
// spark.ailakeSearchWithData(...) (Fase 11, see §3D), which fetches real
// columns in the same native call as the search itself.
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

> Embeddings containing `NaN`/`Infinity` are rejected at write time with a clear error
> (both here and in Trino's `INSERT`) rather than being silently accepted — the exception
> propagates through Spark's `commit()`/Trino's `finish()` and fails the job visibly.

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

// Multi-column (Phase 8 multimodal) write — e.g. text + image embeddings on the same
// row, each with its own HNSW index (Trino's equivalent: catalog property
// `ailake.vector-columns`; Flink's: DDL option `vector.columns`, see §5B/§6B).
// Collects `df` to the driver and issues a single native `writeBatchMulti` call — every
// non-id, non-vector column in `df` must be StringType (written as AI-Lake extra metadata).
import io.ailake.spark.implicits._
import io.ailake.spark.AilakeNative.VectorColSpec

val snapshotId2: Option[Long] = spark.ailakeWriteMulti(
  tableUri      = "s3://my-lake/media/",
  df            = mediaDF,          // columns: id (Long), embedding (Array), image_embedding (Array)
  vectorColumns = Seq(
    VectorColSpec("embedding",       dim = 1536, metric = "cosine"),
    VectorColSpec("image_embedding", dim = 512,  metric = "cosine", modality = Some("image")),
  ),
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

// DataFrame-level equivalent (`spark.ailakeSearchText`) — columns:
// row_id (Long), distance (Double), file_path (String)
import io.ailake.spark.implicits._
val ftsDf = spark.ailakeSearchText(
  tableUri    = "s3://my-lake/docs/",
  queryText   = "machine learning embeddings",
  textColumns = Seq("chunk_text"),
  topK        = 10,
)
ftsDf.orderBy("distance").show()

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

// DataFrame-level equivalent (`spark.ailakeSearchMultimodal`) — columns:
// row_id (Long), rrf_score (Double), file_path (String)
import io.ailake.spark.implicits._
val mmDf = spark.ailakeSearchMultimodal(
  tableUri = "s3://my-lake/media/",
  queries  = Seq(
    ("embedding",       textVec,  0.7f),
    ("image_embedding", imageVec, 0.3f),
  ),
  topK = 20,
)
mmDf.orderBy(mmDf("rrf_score").desc).show()

// Search + full-row fetch in one call (Fase 11) — no manual JOIN against a
// separately-registered Iceberg table needed to get chunk_text/document_title/etc
// back. Schema is dynamic, built from the response: every stored column comes
// back (vector column as ArrayType(FloatType)), plus a trailing _distance column.
val fullDf = spark.ailakeSearchWithData(
  tableUri    = "s3://my-lake/docs/",
  queryVector = query,
  topK        = 20,
)
fullDf.orderBy("_distance").show(truncate = false)
```

### 3E — Delete, schema evolution, compact, create table (Scala)

```scala
// Create an empty table (schema only, no data written) — raw AilakeNative call, same
// pattern as compact/deleteWhere before they grew a SparkSession-level wrapper. No SQL
// CREATE TABLE DDL surface yet — this is the only way to reach it from Spark today.
AilakeNative.createTable(
  warehouse    = "s3://my-lake/docs/",
  namespace    = "default",
  table        = "docs",
  vectorColumn = "embedding",
  dim          = 1536,
  metric       = "cosine",
)

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

// DataFrame-level equivalent (`spark.ailakeCompact`) — Spark has no native CALL-procedure
// syntax outside a full catalog stored-procedure API, so this is a plain SparkSession
// method (same shape as ailakeWrite) rather than Trino's CALL/Flink's scalar UDF.
val filesCompacted: Option[Int] = spark.ailakeCompact("s3://my-lake/docs/")

// SQL-level DELETE and ALTER TABLE, via the io.ailake.spark.AilakeCatalog catalog plugin
// (spark.sql.catalog.ailake = io.ailake.spark.AilakeCatalog — see §5's Trino catalog
// registration pattern; Spark's is per-table-uri, set as spark.sql.catalog.ailake.table-uri).
// Equality/IN pushdown only — no row-level scan-and-delete.
spark.sql("DELETE FROM ailake.default.docs WHERE id = 5")
spark.sql("DELETE FROM ailake.default.docs WHERE id IN (1, 2, 3)")

// Adds/renames the column in the table's Iceberg schema on disk immediately. This
// catalog resolves its schema per-call from spark.sql.catalog.ailake.* options and the
// current DataFrame, not from any tracked state — same limitation Trino/Flink document.
spark.sql("ALTER TABLE ailake.default.docs ADD COLUMN source STRING")
spark.sql("ALTER TABLE ailake.default.docs RENAME COLUMN source TO doc_source")
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
databricks fs cp spark-plugin-0.1.8-plugin.jar \
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
cp trino-plugin-0.1.8-plugin.jar $TRINO_HOME/plugin/ailake/

# 2. Native library
mkdir -p /opt/ailake/lib
cp libailake_jni.so /opt/ailake/lib/

# 3. JVM config — add to etc/jvm.config
echo "-Djava.library.path=/opt/ailake/lib" >> $TRINO_HOME/etc/jvm.config
```

> **Pin the Trino server to 430**, matching `trino-plugin/build.gradle.kts`'s `trinoVersion`
> compileOnly target. Trino 460 breaks connector construction outright
> (`ConnectorMetadata getTableHandle() is not implemented`) — a real Trino SPI signature
> change between the two versions not yet accounted for in this plugin.
>
> **`SELECT` execution works end-to-end** (verified live against a real Trino 430 server,
> 2026-07-13). This was previously blocked by two distinct Jackson serialization bugs in
> Trino's internal `TaskUpdateRequest` codec (coordinator → worker HTTP call, exercised even
> in single-node mode) — a table handle that rendered correctly in `EXPLAIN` plan text came
> back with `tableUri` (and every other field) `null` on the worker side, then a second bug
> (`IllegalAccessException` reflecting on `VectorScanTransactionHandle`'s private Kotlin
> `object` constructor) surfaced once the first was fixed. Both are fixed in the current
> plugin (`VectorScanHandles.kt`, `AilakeIngestTableHandle.kt`, `AilakeNative.kt`'s
> `PartitionFieldDef`/`VectorColSpec`, and `VectorScanHandles.kt`'s
> `VectorScanTransactionHandle`) — `search`, `search_full`, and `search_multimodal` all
> execute real `SELECT` queries, not just `EXPLAIN`/planning. Full root-cause writeup in
> `docs/specs/JVM_PLUGINS.md`'s Trino section and `CHANGELOG.md` ("Fixed (cont. 4)").

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
ailake.namespace=default
ailake.table-name=docs
ailake.text-columns=chunk_text,source     # extra VARCHAR columns on the ingest table
ailake.fts-columns=chunk_text             # subset of text-columns to Tantivy-index (default: none)
ailake.fts-tokenizer=default
ailake.hnsw-m=                            # unset = table/HnswConfig default
ailake.hnsw-ef-construction=
ailake.pre-normalize=false
ailake.deferred=false

# Multi-column (Phase 8 multimodal) ingest — e.g. text + image embeddings on the same
# row, each with its own HNSW index. When set, INSERT INTO ailake.default.ingest expects
# one ARRAY<DOUBLE> column per entry (by name) instead of the single ailake.vector-column,
# and writes go through ailake_write_batch_multi_json. Leave unset ([]) for single-column
# ingest (the default, unchanged).
ailake.vector-columns=[{"column":"embedding","dim":1536,"metric":"cosine","precision":"f16"},{"column":"image_embedding","dim":512,"metric":"cosine","precision":"f16","modality":"image"}]
```

With `ailake.vector-columns` set as above:

```sql
-- Schema: id bigint, embedding array(double), image_embedding array(double)
DESCRIBE ailake.default.ingest;

INSERT INTO ailake.default.ingest
VALUES (1, ARRAY[0.1, 0.2, ...], ARRAY[0.4, 0.5, ...]);
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

-- Cross-modal RRF search table — row_id bigint, rrf_score double, file_path varchar
DESCRIBE ailake.default.search_multimodal;

-- Search + full-row fetch, no JOIN needed (Fase 11) — id bigint, embedding varchar
-- (JSON-encoded, see below), ...ailake.text-columns varchar, _distance double
DESCRIBE ailake.default.search_full;

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

-- Same result without the JOIN (Fase 11) — real columns come back directly.
-- The vector column is JSON-encoded text here (e.g. '[0.1,-0.2]'), not ARRAY<DOUBLE>
-- — see VectorScanMetadata.scanColumns()'s KDoc for why.
SET SESSION ailake.query_vector = '0.1,0.2,0.3,...';
SET SESSION ailake.top_k = 10;

SELECT id, chunk_text, ROUND(_distance, 6) AS dist
FROM   ailake.default.search_full
ORDER  BY _distance
LIMIT  10;
```

**Session properties:**

| Property | Type | Default | Description |
|---|---|---|---|
| `query_vector` | `varchar` | `""` | Comma-separated f32 values |
| `top_k` | `integer` | `10` | Nearest neighbors to return. Capped at `ailake_core::MAX_TOP_K` (100,000) — a value above that is rejected with an error rather than silently proceeding (unbounded `top_k` used to risk an out-of-memory abort) |
| `query_text` | `varchar` | `""` | Query text. Alone → pure full-text search (Tantivy O(log N) if `ailake.fts-columns` indexed, else O(N) BM25). With `query_vector` → hybrid BM25+vector RRF fusion |
| `hybrid_weight` | `double` | `0.5` | BM25 weight in RRF fusion when both `query_vector` and `query_text` are set (`0.0` = pure vector, `1.0` = pure BM25) |
| `multimodal_queries` | `varchar` | `""` | JSON array of `{col, query (csv f32), weight}` for cross-modal RRF search of `ailake.default.search_multimodal` (see below) |

```sql
-- Pure full-text search
SET SESSION ailake.query_text = 'rust programming';
SELECT row_id, file_path FROM ailake.default.search ORDER BY distance LIMIT 10;

-- Hybrid BM25+vector
SET SESSION ailake.query_vector = '0.1,0.2,...';
SET SESSION ailake.query_text = 'rust programming';
SET SESSION ailake.hybrid_weight = 0.3;

-- Cross-modal RRF search (e.g. text + image embeddings on the same row).
-- Schema: row_id bigint, rrf_score double, file_path varchar
SET SESSION ailake.multimodal_queries =
    '[{"col":"embedding","query":"0.1,-0.2","weight":0.7},
      {"col":"image_embedding","query":"0.4,0.5","weight":0.3}]';
SET SESSION ailake.top_k = 20;

SELECT row_id, rrf_score, file_path
FROM   ailake.default.search_multimodal
ORDER  BY rrf_score DESC;
```

**DELETE, ALTER TABLE, and maintenance:**

```sql
-- Equality/IN deletes only — no row-level scan-and-delete
DELETE FROM ailake.default.ingest WHERE id = 5;
DELETE FROM ailake.default.ingest WHERE id IN (1, 2, 3);

-- Adds/renames the column in the table's Iceberg schema on disk immediately.
-- The running Trino worker's own schema (ailake.text-columns) is fixed at
-- catalog startup — add the new column to ailake.text-columns and restart
-- Trino (or reload the catalog) before INSERT/SELECT can see it.
ALTER TABLE ailake.default.ingest ADD COLUMN source VARCHAR;
ALTER TABLE ailake.default.ingest RENAME COLUMN source TO doc_source;

-- Compacts small files in the catalog's configured table
CALL ailake.system.compact();
```

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
#   VectorScanMetadataTest     — schema discovery, DELETE pushdown, ADD/RENAME COLUMN
#   VectorScanConnectorTest    — session properties, transaction handle
#   VectorScanSplitManagerTest — split creation from session
#   VectorScanRecordSetTest    — cursor iteration, column types
#   AilakeNativeTest           — graceful degradation, CSV parsing
#   AilakeProceduresTest       — CALL ailake.system.compact()
```

---

## 6. Flink

### 6A — Add to job classpath

```bash
flink run \
  --jar my-pipeline.jar \
  --classpath ailake-flink-0.1.8-plugin.jar \
  -Dtaskmanager.extraLibFolders=/opt/ailake/lib
```

Or add to `$FLINK_HOME/lib/` so all jobs on the cluster pick it up:

```bash
cp ailake-flink-0.1.8-plugin.jar /opt/flink/lib/
cp libailake_jni.so               /opt/ailake/lib/
echo 'env.java.opts.taskmanager: -Djava.library.path=/opt/ailake/lib' \
    >> /opt/flink/conf/flink-conf.yaml
```

### 6B — SQL DDL (sink and source)

The `ailake` connector serves two DIFFERENT DDL shapes depending on direction —
exactly like Spark/Trino's separate `ingest`/`search` tables, just modeled here
as two Flink `CREATE TABLE` statements sharing the same `warehouse`/`namespace`/
`table-name` (i.e. the same physical AI-Lake table). Mixing them up (e.g. trying
to `SELECT` from a table declared with the write-shaped schema) fails at
DDL-resolution time with a clear `ValidationException`, not a runtime crash.

> Same as Spark/Trino: a `NaN`/`Infinity` value in `embedding` is rejected at write time
> with a clear error; `search.top-k` above 100,000 is rejected the same way. Flink's
> `AilakeNativeLoader` already throws `RuntimeException` on either, failing the job
> visibly (the reference behavior the Spark/Trino write path was brought in line with).

```sql
-- Write (sink): id + vector + any number of extra STRING columns
CREATE TABLE ailake_docs_ingest (
    id        BIGINT,
    text      STRING,
    embedding ARRAY<FLOAT>
) WITH (
    'connector'        = 'ailake',
    'warehouse'        = 's3://my-lake/',
    'namespace'        = 'default',
    'table-name'       = 'docs',
    'vector.column'    = 'embedding',
    'vector.dim'       = '1536',
    'vector.metric'    = 'cosine',
    'vector.precision' = 'f16',
    'format.version'   = '2',
    -- Optional Iceberg partition fields (JSON array)
    'partition.fields' = '[{"column":"topic_id","transform":"identity","column_type":"int"}]',
    -- Optional write-tuning knobs
    'hnsw.m'                = '32',
    'hnsw.ef-construction'  = '200',
    'pre-normalize'         = 'false',
    'deferred'              = 'false'
);

-- Stream embeddings from Kafka source table to AI-Lake
INSERT INTO ailake_docs_ingest
SELECT id, text, embedding
FROM kafka_embeddings_source;

-- Batch ingest from a Hive table
INSERT INTO ailake_docs_ingest
SELECT id, text, embedding FROM hive_catalog.default.chunks;

-- Equality/IN deletes only — no row-level scan-and-delete
DELETE FROM ailake_docs_ingest WHERE id = 5;
DELETE FROM ailake_docs_ingest WHERE id IN (1, 2, 3);

-- Adds/renames the column in the table's Iceberg schema on disk immediately.
-- This connector's own in-memory schema is fixed at table-creation time —
-- restart the job (or reissue CREATE TABLE) before INSERT/SELECT can see it.
ALTER TABLE ailake_docs_ingest ADD COLUMN source STRING;
ALTER TABLE ailake_docs_ingest RENAME COLUMN source TO doc_source;

-- Multi-column (Phase 8 multimodal) ingest — e.g. text + image embeddings on the same
-- row, each with its own HNSW index. One ARRAY<FLOAT> column per vector.columns entry
-- (resolved by name against the declared schema) instead of the single vector.column.
CREATE TABLE ailake_docs_multimodal_ingest (
    id              BIGINT,
    text            STRING,
    embedding       ARRAY<FLOAT>,
    image_embedding ARRAY<FLOAT>
) WITH (
    'connector'      = 'ailake',
    'warehouse'      = 's3://my-lake/',
    'namespace'      = 'default',
    'table-name'     = 'media',
    'vector.dim'     = '1536',  -- required option, unused in multi-column mode
    'vector.columns' =
      '[{"column":"embedding","dim":1536,"metric":"cosine","precision":"f16"},
        {"column":"image_embedding","dim":512,"metric":"cosine","precision":"f16","modality":"image"}]'
);

INSERT INTO ailake_docs_multimodal_ingest
SELECT id, text, embedding, image_embedding FROM hive_catalog.default.media_chunks;

-- Read (source): fixed 3-column search-result shape
CREATE TABLE ailake_docs_search (
    row_id    BIGINT,
    distance  FLOAT,
    file_path STRING
) WITH (
    'connector'     = 'ailake',
    'warehouse'     = 's3://my-lake/',
    'namespace'     = 'default',
    'table-name'    = 'docs',
    'vector.column' = 'embedding',
    'vector.dim'    = '1536',
    'search.top-k'  = '10',
    'search.ef'     = '50'
);

-- Query vector passed via job parameters (Flink SQL has no per-query SET SESSION):
--   flink run -pyfs ... -Dailake.query.vector='0.1,0.2,...' -Dailake.top-k=10
-- or programmatically via ExecutionConfig.setGlobalJobParameters(...).
--
-- ailake.query.text alone -> pure full-text search (Tantivy O(log N) when the
-- table has an FTS index via vector.column's fts.columns, else O(N) BM25).
-- ailake.query.vector + ailake.query.text together -> hybrid BM25+vector RRF
-- (weight via ailake.hybrid.weight, default 0.5).
SELECT row_id, distance, file_path FROM ailake_docs_search ORDER BY distance;

-- Cross-modal RRF search (e.g. text + image embeddings on the same row) instead
-- selected via ailake.multimodal.queries — JSON array of {col, query (csv f32), weight}:
--   flink run ... -Dailake.multimodal.queries=
--     '[{"col":"embedding","query":"0.1,-0.2","weight":0.7},
--       {"col":"image_embedding","query":"0.4,0.5","weight":0.3}]'
-- Same physical (row_id, distance, file_path) schema/table as above — the
-- "distance" slot carries the fused RRF score in this mode.
SELECT row_id, distance AS rrf_score, file_path FROM ailake_docs_search ORDER BY distance DESC;

-- Read (source), search + full-row fetch, no JOIN needed (Fase 11) — columns
-- come straight from the DDL (schema-on-read), not a fixed 3-column shape.
-- Only requirement: the last declared column must be _distance (FLOAT or DOUBLE).
CREATE TABLE ailake_docs_search_full (
    id        BIGINT,
    text      STRING,
    embedding ARRAY<FLOAT>,
    _distance FLOAT
) WITH (
    'connector'     = 'ailake',
    'warehouse'     = 's3://my-lake/',
    'namespace'     = 'default',
    'table-name'    = 'docs',
    'vector.column' = 'embedding',
    'vector.dim'    = '1536',
    'search.top-k'  = '10',
    'search.mode'   = 'full'
);

-- flink run ... -Dailake.query.vector='0.1,0.2,...'
SELECT id, text, ROUND(_distance, 6) AS dist
FROM   ailake_docs_search_full
ORDER  BY _distance;

-- Compact small files — no CALL-equivalent for connectors in Flink SQL, exposed
-- as a plain scalar function instead:
CREATE TEMPORARY FUNCTION ailake_compact AS 'io.ailake.flink.AilakeCompactFunction';
SELECT ailake_compact('s3://my-lake/', 'default', 'docs');
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

// Cross-modal RRF (Phase 8) — Triple is (column, query vector, weight), in that order
val mmHits = loader.searchMultimodal(
    warehouse = "s3://my-lake/",
    namespace = "default",
    table     = "media",
    queries   = listOf(
        Triple("embedding",       textVec,  0.7f),
        Triple("image_embedding", imageVec, 0.3f),
    ),
    topK = 20,
)

// Write batch — ids: LongArray, embeddings: Array<FloatArray>
loader.writeBatch(
    warehouse        = "s3://my-lake/",
    namespace        = "default",
    table            = "docs",
    vecCol           = "embedding",
    dim              = 1536,
    metric           = "cosine",
    precision        = "f16",
    ids              = longArrayOf(1L, 2L),
    embeddings       = arrayOf(vec1, vec2),
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
    values    = listOf("1", "2"),
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
| `warehouse` | required | AI-Lake warehouse root URI |
| `table-name` | required | Table name within the warehouse |
| `vector.dim` | required | Embedding dimension |
| `namespace` | `"default"` | Iceberg namespace |
| `vector.column` | `"embedding"` | Column containing `ARRAY<FLOAT>` |
| `vector.metric` | `"euclidean"` | `cosine` \| `euclidean` \| `dot_product` |
| `vector.precision` | `"f16"` | `f32` \| `f16` \| `i8` |
| `search.top-k` | `10` | Nearest neighbors to return (source tables). Capped at 100,000 — a higher value fails the job with a clear error |
| `search.ef` | `50` | HNSW `ef_search` (source tables) |
| `embedding.model` | unset | Stored in `ailake.embedding-model` Iceberg property |
| `partition.fields` | `"[]"` | JSON array of `{column, transform, column_type}` |
| `format.version` | `2` | Iceberg format version (`2` or `3`) |
| `fts.columns` | `""` | Comma-separated extra text columns, persisted as metadata (write path) |
| `fts.tokenizer` | `"default"` | Tantivy tokenizer for `fts.columns` |
| `hnsw.m` | unset | HNSW graph connectivity (M); unset = table default |
| `hnsw.ef-construction` | unset | HNSW `ef_construction`; unset = table default |
| `pre-normalize` | `false` | Normalize vectors to unit L2 at write time (recommended for cosine) |
| `deferred` | `false` | Build index asynchronously; Parquet committed immediately |

### 6E — Running Flink tests

```bash
cd ailake-flink
./gradlew test

# Test classes:
#   AilakeVectorConnectorFactoryTest — DDL option parsing, factory lookup, source schema validation
#   AilakeNativeLoaderTest           — data classes, JSON payload shape (no native lib needed)
#   AilakeVectorTableSourceTest      — AilakeInputFormat.open() degrades to an empty
#                                      result set when the native lib can't be loaded,
#                                      instead of failing the Flink task; query.vector/
#                                      query.text job-parameter combinations
#   AilakeVectorTableSinkTest        — extra-column capture, null id/vector guards,
#                                      type validation, DELETE pushdown
#   AilakeCatalogTest                — table-name/namespace injection, ALTER TABLE wiring
#   AilakeCompactFunctionTest        — CALL-equivalent compact scalar function
#   AilakeJniIntegrationTest         — end-to-end against a real libailake_jni.so when
#                                      AILAKE_NATIVE_LIB is set (write+search, delete,
#                                      schema evolution, DELETE/ALTER TABLE/compact/
#                                      hybrid-search SQL-surface wiring)
```

---

## 7. Cross-engine reference — delete, schema evolution, compact

All three JVM plugins expose the same operations via the JSON-envelope ABI,
and (as of this section's last update) all three also wire them into a real
SQL surface, not just a Kotlin/Scala API — equality/IN pushdown only for
delete, matching the native equality-delete-file mechanism.

### Delete (equality delete — no data rewrite)

| Engine | SQL surface | Underlying call |
|---|---|---|
| Spark | `DELETE FROM ailake.default.ingest WHERE id = 5` (via catalog) | `AilakeNative.deleteWhere(tableUri, ns, table, col, values)` |
| Trino | `DELETE FROM ailake.default.ingest WHERE id IN (1,2,3)` | `AilakeNative.deleteWhere(...)` (Kotlin) |
| Flink | `DELETE FROM ailake_docs_ingest WHERE id = 5` (`SupportsDeletePushDown`) | `AilakeNativeLoader.deleteWhere(warehouse, ns, table, col, values)` |

### Schema evolution (metadata-only)

| Engine | SQL surface | Underlying call |
|---|---|---|
| Spark | `ALTER TABLE ailake.default.docs ADD COLUMN`/`RENAME COLUMN` (via catalog) | `AilakeNative.evolveSchema(tableUri, ns, table, addCols, renameCols)` |
| Trino | `ALTER TABLE ailake.default.ingest ADD COLUMN`/`RENAME COLUMN` | `AilakeNative.evolveSchema(...)` (Kotlin) |
| Flink | `ALTER TABLE ailake_docs_ingest ADD COLUMN`/`RENAME COLUMN` | `AilakeNativeLoader.evolveSchema(...)` |

### Compact

| Engine | SQL surface | Underlying call |
|---|---|---|
| Spark | `spark.ailakeCompact(tableUri, ...)` (Spark has no native CALL-procedure syntax outside a full catalog stored-procedure API) | `AilakeNative.compact(tableUri, ns, table, minFiles, targetSizeBytes)` |
| Trino | `CALL ailake.system.compact()` | `AilakeNative.compact(...)` (Kotlin) |
| Flink | `SELECT ailake_compact(warehouse, ns, table)` (scalar function — Flink has no `CALL`-equivalent for connectors) | `AilakeNativeLoader.compact(...)` |
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
| Trino: `ConnectorMetadata getTableHandle() is not implemented` at catalog startup | Running Trino 460 (or another version past the plugin's SPI target) | Pin the server to Trino **430**, matching `trino-plugin/build.gradle.kts`'s `trinoVersion` |
| Trino: `NullPointerException` / `IllegalAccessException` on `SELECT` | Stale plugin JAR predating the Jackson serialization fix | Rebuild/redownload the plugin — fixed in current `trino-plugin` (see §5A note) |
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
