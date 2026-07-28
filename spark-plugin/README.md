# ailake-spark

Spark plugin exposing AI-Lake vector search, hybrid BM25+vector search,
cross-modal RRF search, `INSERT INTO`/`DELETE`/`ALTER TABLE`, compaction, and
Phase 9 agent-memory operations (decay/migrate/backfill) as both a
`SparkSession` Scala API and a V2 catalog for SQL `INSERT INTO`/`DELETE` —
backed by `libailake_jni.so` via JNA (same C-ABI Trino and Flink use).

Full reference: [`docs/guides/JVM_INTEGRATION.md`](../docs/guides/JVM_INTEGRATION.md) §3
(build/install/Scala API/catalog config/cross-engine tables) and
[`docs/specs/JVM_PLUGINS.md`](../docs/specs/JVM_PLUGINS.md) (architecture, C-ABI reference).

## Build

```bash
cargo build --release -p ailake-jni   # from the repo root — builds libailake_jni.so
cd spark-plugin
./gradlew shadowJar
# → build/libs/spark-plugin-<version>-plugin.jar
```

## Install

```bash
$SPARK_HOME/bin/spark-shell \
  --jars build/libs/spark-plugin-*-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib"
```

`AilakeCatalog` (for `INSERT INTO`/`DELETE`/`ALTER TABLE` SQL) is registered separately:

```
spark.sql.catalog.ailake     = io.ailake.spark.AilakeCatalog
spark.sql.catalog.ailake.table-uri     = s3://my-lake/docs/
spark.sql.catalog.ailake.vector-column = embedding   (default: embedding)
spark.sql.catalog.ailake.vector-dim    = 1536        (default: 1536)
spark.sql.catalog.ailake.metric        = cosine      (default: cosine)
spark.sql.catalog.ailake.precision     = f16         (default: f16)
```

## Quickstart

```scala
import io.ailake.spark.implicits._

val query: Array[Float] = myEmbeddingModel.embed("What is geometric pruning?")
val results = spark.ailakeSearch("s3://my-lake/docs/", query, topK = 100)
results.orderBy("distance").show(10)

spark.ailakeWrite("s3://my-lake/docs/", chunksDF, vectorColumn = "embedding", idColumn = "id")
spark.ailakeCompact("s3://my-lake/docs/")
spark.ailakeDecayMemories("s3://my-lake/agent-memory/", lambda = 0.1f)
```

```sql
INSERT INTO ailake.default.docs VALUES (1, array(0.1, 0.2, 0.3))
```

## Test

```bash
./gradlew test   # 9 test classes, no running Spark cluster required (some integration
                 # tests skip automatically without AILAKE_BIN/AILAKE_FIXTURE)
                 # AilakeSparkExtensionsTest starts an embedded SparkSession (~15s)
```

## License

MIT OR Apache-2.0 — same as the rest of the AI-Lake SDK.
