// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.{DataFrame, Row, SparkSession, SparkSessionExtensions}
import org.apache.spark.sql.types.{ArrayType, BooleanType, DoubleType, FloatType, LongType, StringType, StructField, StructType}

/**
 * Spark extensions entry point. Register via:
 *
 *   spark.conf.set("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
 *
 * or in spark-defaults.conf / SparkSession builder:
 *
 *   SparkSession.builder()
 *     .config("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
 *     .getOrCreate()
 */
class AilakeSparkExtensions extends (SparkSessionExtensions => Unit) {
  def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectPlannerStrategy(_ => new VectorScanStrategy)
  }
}

/**
 * DataFrame extension methods for AI-Lake vector search.
 *
 * Usage:
 *   import io.ailake.spark.implicits._
 *
 *   val results: DataFrame = spark.ailakeSearch(
 *     tableUri     = "s3://my-lake/docs/",
 *     queryVector  = embeddingArray,   // Array[Float]
 *     topK         = 20,
 *   )
 *   results.show()
 *
 * Returns a DataFrame with columns: row_id (Long), distance (Double), file_path (String).
 *
 * Cross-modal search: `spark.ailakeSearchMultimodal(tableUri, queries, topK)` returns
 * columns: row_id (Long), rrf_score (Double), file_path (String).
 *
 * REST Catalog: every method below takes a `catalogOpts: Map[String, String]` (default
 * empty = Hadoop catalog), e.g. `Map("catalog" -> "rest", "rest_uri" -> "...", "rest_auth"
 * -> "bearer", "rest_token" -> "...")` — see docs/guides/REST_CATALOG.md. `ailakeWrite`
 * (the DataSourceV2 path) instead takes `.option("catalog", "rest").option("rest-uri", ...)`
 * on the DataFrameWriter, same options `spark.sql.catalog.<name>.rest-*` uses.
 */
object implicits {
  implicit class AilakeSession(private val spark: SparkSession) extends AnyVal {
    /**
     * @param namespace   Iceberg namespace the table was written under (default: "default").
     *                    Must match the `namespace` passed to [[ailakeWrite]] — otherwise this
     *                    silently searches a different (likely empty) table location.
     * @param tableName   table name; defaults to the last segment of `tableUri`, same
     *                    resolution rule [[ailakeWrite]] uses.
     * @param hybridText  when set, enables hybrid BM25+vector RRF fusion — was already threaded
     *                    through by `AilakeNative.search` but unreachable from this, the only
     *                    DataFrame entry point, same "dead capability" gap `ailakeSearchMultimodal`
     *                    and `ailakeSearchWithData` had before being wired.
     * @param textColumn  Parquet column for BM25 scoring when `hybridText` is set (default "chunk_text")
     * @param bm25Weight  BM25 weight in RRF fusion when `hybridText` is set (0.0 = pure vector, 1.0 = pure BM25)
     * @param efSearch    HNSW ef_search — higher recall at the cost of latency (server default: 50)
     * @param pruningThreshold geometric pruning cutoff (server default: no pruning)
     */
    def ailakeSearch(
      tableUri:         String,
      queryVector:      Array[Float],
      topK:             Int,
      namespace:        String = "default",
      tableName:        String = "",
      hybridText:       Option[String] = None,
      textColumn:       String = "chunk_text",
      bm25Weight:       Float = 0.5f,
      catalogOpts:      Map[String, String] = Map.empty,
      efSearch:         Option[Int] = None,
      pruningThreshold: Option[Float] = None,
    ): DataFrame = {
      val plan = VectorSearchPlan(tableUri, queryVector, topK)
      val rows = AilakeNative.search(
        tableUri, queryVector, topK, namespace = namespace, tableName = tableName,
        hybridText = hybridText, textColumn = textColumn, bm25Weight = bm25Weight,
        catalogOpts = catalogOpts, efSearch = efSearch, pruningThreshold = pruningThreshold)
      val sparkRows = rows.map(r => Row(r.rowId, r.distance.toDouble, r.filePath))
      spark.createDataFrame(spark.sparkContext.parallelize(sparkRows, numSlices = 1), plan.schema)
    }

    /**
     * Pure full-text search via Tantivy (fast path when the table has an FTS index) or BM25
     * brute-force fallback. `AilakeNative.searchText` was already fully implemented but had no
     * DataFrame caller anywhere in this plugin — same "dead capability" gap as the others.
     *
     * @param textColumns  columns to search; defaults to `["chunk_text"]`
     */
    def ailakeSearchText(
      tableUri:        String,
      queryText:       String,
      namespace:       String = "default",
      tableName:       String = "",
      textColumns:     Seq[String] = Seq("chunk_text"),
      topK:            Int = 10,
      partitionFilter: Option[String] = None,
      catalogOpts:     Map[String, String] = Map.empty,
    ): DataFrame = {
      val schema = StructType(Seq(
        StructField("row_id", LongType, nullable = false),
        StructField("distance", DoubleType, nullable = false),
        StructField("file_path", StringType, nullable = false),
      ))
      val rows = AilakeNative.searchText(
        tableUri, namespace, tableName, queryText, textColumns, topK, partitionFilter, catalogOpts)
      val sparkRows = rows.map(r => Row(r.rowId, r.distance.toDouble, r.filePath))
      spark.createDataFrame(spark.sparkContext.parallelize(sparkRows, numSlices = 1), schema)
    }

    /**
     * Cross-modal vector search via Reciprocal Rank Fusion (Phase 8 multimodal —
     * e.g. text + image embeddings on the same row). Was implemented natively
     * (`AilakeNative.searchMultimodal`) but never exposed as a DataFrame call —
     * same "dead capability" gap as `ailakeSearch` before it, closed the same way.
     *
     * @param queries  one or more (column, query vector, weight) triples
     * @param topK     number of fused results to return
     */
    def ailakeSearchMultimodal(
      tableUri:        String,
      queries:         Seq[(String, Array[Float], Float)],
      topK:            Int,
      namespace:       String = "default",
      tableName:       String = "",
      partitionFilter: Option[String] = None,
      catalogOpts:     Map[String, String] = Map.empty,
    ): DataFrame = {
      val schema = StructType(Seq(
        StructField("row_id", LongType, nullable = false),
        StructField("rrf_score", DoubleType, nullable = false),
        StructField("file_path", StringType, nullable = false),
      ))
      val rows = AilakeNative.searchMultimodal(
        tableUri, queries, topK, partitionFilter, namespace, tableName, catalogOpts)
      val sparkRows = rows.map(r => Row(r.rowId, r.rrfScore.toDouble, r.filePath))
      spark.createDataFrame(spark.sparkContext.parallelize(sparkRows, numSlices = 1), schema)
    }

    /**
     * Vector search + full-row fetch in one DataFrame — closes the "SQL search only returns
     * row_id/distance/file_path" gap (Fase 11): previously getting real columns
     * (chunk_text, document_title, ...) back from a search required a manual `JOIN` against
     * a separately-registered Iceberg table pointing at the same physical location (see
     * `docs/guides/JVM_INTEGRATION.md` §4D). Schema is built dynamically from
     * `AilakeNative.scan`'s response — every stored column comes back (no column-subset
     * filter on the native side), vector column as `ArrayType(FloatType)`, plus a trailing
     * `_distance` column.
     *
     * @param vectorColumn vector column to search (default "embedding")
     */
    def ailakeSearchWithData(
      tableUri:        String,
      queryVector:     Array[Float],
      topK:            Int,
      vectorColumn:    String = "embedding",
      namespace:       String = "default",
      tableName:       String = "",
      partitionFilter: Option[String] = None,
      catalogOpts:     Map[String, String] = Map.empty,
    ): DataFrame = {
      val scanResult = AilakeNative.scan(
        tableUri, queryVector, topK, vectorColumn, partitionFilter, namespace, tableName, catalogOpts)
      val fields = scanResult.schema.map { col =>
        val sparkType = col.dataType match {
          case "int64"       => LongType
          case "float32"     => FloatType
          case "float64"     => DoubleType
          case "bool"        => BooleanType
          case "list_float32" => ArrayType(FloatType)
          case _             => StringType
        }
        StructField(col.name, sparkType, nullable = true)
      }
      val schema = StructType(fields)
      val rows = (0 until scanResult.numRows).map { i =>
        Row.fromSeq(scanResult.schema.map(col => scanResult.columns(col.name)(i)))
      }
      spark.createDataFrame(spark.sparkContext.parallelize(rows, numSlices = 1), schema)
    }

    /**
     * Compact small files in an AI-Lake table. `AilakeNative.compact` was already fully
     * implemented but had no DataFrame/SQL entry point anywhere in this plugin — same "dead
     * capability" gap DELETE and ALTER TABLE had before being wired (Trino has this as
     * `CALL ailake.system.compact()`, Flink as a scalar `ailake_compact(...)` UDF; Spark has
     * no native CALL-procedure syntax outside a full catalog stored-procedure API, so this
     * is a plain SparkSession method instead, matching [[ailakeWrite]]'s existing shape).
     *
     * @return number of files compacted (0 = nothing to compact), or `None` when the native
     *         library is absent.
     */
    def ailakeCompact(
      tableUri:        String,
      namespace:       String = "default",
      tableName:       String = "",
      minFiles:        Int = 4,
      targetSizeBytes: Long = 128L * 1024 * 1024,
      maxFilesPerPass: Int = 20,
      deferred:        Boolean = false,
      catalogOpts:     Map[String, String] = Map.empty,
    ): Option[Int] = {
      val resolvedName = if (tableName.nonEmpty) tableName
                         else tableUri.stripSuffix("/").split("/").last
      AilakeNative.compact(
        tableUri, namespace, resolvedName, minFiles, targetSizeBytes, maxFilesPerPass, deferred, catalogOpts)
    }

    // ── decay_memories / migrate / delete_rows / add_vector_column /
    // backfill_vector_column / estimate — same reasoning as ailakeCompact
    // above (Spark has no CALL-procedure syntax outside a full catalog
    // stored-procedure API): plain SparkSession methods. Closes a gap found
    // auditing trino-plugin — these 6 had zero C-ABI path at all in
    // ailake-jni until now (unlike create_table, which existed but was
    // simply unwired), so all three JVM plugins were equally affected.

    /** Recomputes recency_weight for every row (Phase 9 agent memory). Returns files_updated, or None on failure. */
    def ailakeDecayMemories(
      tableUri:    String,
      lambda:      Float,
      namespace:   String = "default",
      tableName:   String = "",
      catalogOpts: Map[String, String] = Map.empty,
    ): Option[Int] = {
      val resolvedName = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
      AilakeNative.decayMemories(tableUri, namespace, resolvedName, lambda, catalogOpts)
    }

    /** Re-embeds oldColumn → newColumn via an external embed command. Throws on failure. */
    def ailakeMigrate(
      tableUri:     String,
      oldColumn:    String,
      newColumn:    String,
      textColumn:   String,
      embedCmd:     String,
      namespace:    String = "default",
      tableName:    String = "",
      strategy:     String = "atomic-replace",
      batchSize:    Int = 10000,
      modelName:    Option[String] = None,
      modelVersion: Option[String] = None,
      catalogOpts:  Map[String, String] = Map.empty,
    ): Unit = {
      val resolvedName = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
      AilakeNative.migrate(
        tableUri, namespace, resolvedName, oldColumn, newColumn, textColumn, embedCmd,
        strategy, batchSize, modelName, modelVersion, catalogOpts)
    }

    /** Deletes row positions via Iceberg V3 Deletion Vectors — different from `DELETE FROM` (equality predicate). Throws on failure. */
    def ailakeDeleteRows(
      tableUri:    String,
      file:        String,
      rowIds:      Seq[Int],
      namespace:   String = "default",
      tableName:   String = "",
      catalogOpts: Map[String, String] = Map.empty,
    ): Unit = {
      val resolvedName = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
      AilakeNative.deleteRows(tableUri, namespace, resolvedName, file, rowIds, catalogOpts)
    }

    /** Adds a new vector column to the schema (metadata-only, no backfill). Returns the new schema_id, or None on failure. */
    def ailakeAddVectorColumn(
      tableUri:           String,
      column:             String,
      dim:                Int,
      namespace:          String = "default",
      tableName:          String = "",
      metric:             String = "cosine",
      precision:          String = "f16",
      preNormalize:       Boolean = false,
      hnswM:              Option[Int] = None,
      hnswEfConstruction: Option[Int] = None,
      catalogOpts:        Map[String, String] = Map.empty,
    ): Option[Int] = {
      val resolvedName = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
      AilakeNative.addVectorColumn(
        tableUri, namespace, resolvedName, column, dim, metric, precision,
        preNormalize, hnswM, hnswEfConstruction, catalogOpts)
    }

    /** Backfills embeddings for a column added via `ailakeAddVectorColumn`. Throws on failure. */
    def ailakeBackfillVectorColumn(
      tableUri:    String,
      column:      String,
      textColumn:  String,
      embedCmd:    String,
      namespace:   String = "default",
      tableName:   String = "",
      batchSize:   Int = 512,
      catalogOpts: Map[String, String] = Map.empty,
    ): Unit = {
      val resolvedName = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
      AilakeNative.backfillVectorColumn(tableUri, namespace, resolvedName, column, textColumn, embedCmd, batchSize, catalogOpts)
    }

    /** Storage/index size estimates for a hypothetical table — pure math, no table needed. Returns the raw JSON response text, or None on failure. */
    def ailakeEstimate(rows: Long, dim: Int, hnswM: Int = 16, pqM: Option[Int] = None): Option[String] =
      AilakeNative.estimate(rows, dim, hnswM, pqM)

    /**
     * Table metadata: current snapshot, file/row/size counts, index status
     * breakdown, and "foreign" files (written by a generic Iceberg engine —
     * no AI-Lake centroid/HNSW). Mirrors `ailake info --format json`.
     * Found in a later audit pass: zero binding coverage anywhere outside
     * the CLI and the Airflow provider before this. Returns the raw JSON
     * response text, or None on failure.
     */
    def ailakeInfo(
      tableUri:    String,
      namespace:   String = "default",
      tableName:   String = "",
      catalogOpts: Map[String, String] = Map.empty,
    ): Option[String] = {
      val resolvedName = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
      AilakeNative.info(tableUri, namespace, resolvedName, catalogOpts)
    }

    /**
     * Write a DataFrame to an AI-Lake table via the native library.
     *
     * The DataFrame must have columns: id (Long), embedding (Array[Double]).
     *
     * @param tableUri     AI-Lake table root URI
     * @param df           DataFrame with (id, embedding) schema
     * @param vectorColumn embedding column name in the DataFrame (default: "embedding")
     * @param idColumn     id column name in the DataFrame (default: "id")
     * @param metric       distance metric (default: "cosine")
     * @param precision    storage precision (default: "f16")
     * @param namespace    Iceberg namespace (default: "default")
     * @param tableName    table name derived from tableUri by default
     */
    /**
     * Write a DataFrame to an AI-Lake table via the native library.
     *
     * The DataFrame must have columns matching `idColumn` (Long) and
     * `vectorColumn` (Array[Double]).
     *
     * @param tableUri     AI-Lake table root URI
     * @param df           DataFrame with id + embedding columns
     * @param vectorColumn embedding column name (default: "embedding")
     * @param idColumn     id column name (default: "id")
     * @param metric       distance metric (default: "cosine")
     * @param precision    storage precision (default: "f16")
     * @param namespace    Iceberg namespace (default: "default")
     * @param tableName    table name; defaults to last segment of tableUri
     */
    def ailakeWrite(
      tableUri:        String,
      df:              DataFrame,
      vectorColumn:    String = "embedding",
      idColumn:        String = "id",
      metric:          String = "cosine",
      precision:       String = "f16",
      namespace:       String = "default",
      tableName:       String = "",
      partitionFields: Seq[AilakeNative.PartitionFieldDef] = Seq.empty,
      formatVersion:   Int = 2,
    ): Unit = {
      val resolvedName = if (tableName.nonEmpty) tableName
                         else tableUri.stripSuffix("/").split("/").last
      df.schema.find(_.name == vectorColumn).foreach(_.dataType match {
        case _: org.apache.spark.sql.types.ArrayType => // ok
        case _ => throw new IllegalArgumentException(s"Column $vectorColumn must be ArrayType")
      })
      val pfJson = partitionFields.map(pf =>
        s"""{"column":"${pf.column}","transform":"${pf.transform}","column_type":"${pf.columnType}"}"""
      ).mkString("[", ",", "]")
      df.write
        .format("io.ailake.spark.AilakeDataSource")
        .option("tableUri",          tableUri)
        .option("namespace",         namespace)
        .option("tableName",         resolvedName)
        .option("vectorColumn",      vectorColumn)
        .option("idColumn",          idColumn)
        .option("metric",            metric)
        .option("precision",         precision)
        .option("partition-fields",  pfJson)
        .option("format-version",    formatVersion.toString)
        .save()
    }

    /**
     * Write a DataFrame with N independent vector columns to an AI-Lake table
     * (Phase 8 multimodal — e.g. text + image embeddings on the same row,
     * searchable via [[ailakeSearchMultimodal]]-style calls to
     * `AilakeNative.searchMultimodal`). Each column gets its own HNSW index.
     *
     * Unlike [[ailakeWrite]] (a distributed DataSourceV2 writer — one native
     * call per Spark partition), this collects `df` to the driver and issues a
     * single native `writeBatchMulti` call: the JNI multi-column write contract
     * produces one AI-Lake file per call with N HNSW sections built together,
     * so there's no partitioning strategy available without either a shared
     * cross-executor HNSW build (not supported) or a later multi-file merge
     * across partial column sets (not implemented). Suitable for batch sizes
     * that fit in driver memory; for larger multimodal ingest, write from the
     * Python SDK (`TableWriter.write_batch_multi`) instead.
     *
     * @param vectorColumns  one or more specs; `column` must name an ArrayType
     *                       column in `df`. First entry is primary (used for
     *                       geometric pruning in the manifest).
     * @param idColumn       id column name (default: "id"), must be LongType.
     *                       Every other column not named by `idColumn` or a
     *                       `vectorColumns` entry must be StringType (written
     *                       as AI-Lake extra metadata, same rule as [[ailakeWrite]]).
     * @return the snapshot id on success, None if the native library is absent
     *         or the write failed (see driver logs for the reason).
     */
    def ailakeWriteMulti(
      tableUri:       String,
      df:             DataFrame,
      vectorColumns:  Seq[AilakeNative.VectorColSpec],
      idColumn:       String = "id",
      namespace:      String = "default",
      tableName:      String = "",
      embeddingModel: Option[String] = None,
      formatVersion:  Int = 2,
      ftsColumns:     Seq[String] = Seq.empty,
      ftsTokenizer:   String = "default",
      deferred:       Boolean = false,
      catalogOpts:    Map[String, String] = Map.empty,
    ): Option[Long] = {
      import org.apache.spark.sql.types.{ArrayType, LongType, StringType}
      require(vectorColumns.nonEmpty, "ailakeWriteMulti requires at least one VectorColSpec")

      val resolvedName = if (tableName.nonEmpty) tableName
                         else tableUri.stripSuffix("/").split("/").last
      val schema = df.schema

      val idField = schema.find(_.name == idColumn).getOrElse(
        throw new IllegalArgumentException(s"Column '$idColumn' not found in DataFrame"))
      if (idField.dataType != LongType)
        throw new IllegalArgumentException(
          s"Column '$idColumn' must be LongType, got ${idField.dataType.simpleString}")

      val vecColNames = vectorColumns.map(_.column).toSet
      vectorColumns.foreach { spec =>
        schema.find(_.name == spec.column) match {
          case Some(f) if f.dataType.isInstanceOf[ArrayType] => // ok
          case Some(f) =>
            throw new IllegalArgumentException(
              s"Vector column '${spec.column}' must be ArrayType, got ${f.dataType.simpleString}")
          case None =>
            throw new IllegalArgumentException(s"Vector column '${spec.column}' not found in DataFrame")
        }
      }

      val extraFields = schema.fields.filter(f => f.name != idColumn && !vecColNames.contains(f.name))
      extraFields.foreach { f =>
        if (f.dataType != StringType)
          throw new IllegalArgumentException(
            s"Column '${f.name}' must be StringType to be written as AI-Lake extra metadata " +
            s"(got ${f.dataType.simpleString}). Cast other columns to string first, " +
            s"e.g. col('${f.name}').cast('string').")
      }

      val rows = df.collect()
      val ids  = rows.map(_.getAs[Long](idColumn)).toSeq

      def toFloat(v: Any): Float = v match {
        case d: java.lang.Double => d.floatValue()
        case f: java.lang.Float  => f.floatValue()
        case n: Number           => n.floatValue()
      }

      val vectorColumnData: Seq[(AilakeNative.VectorColSpec, Seq[Seq[Float]])] = vectorColumns.map { spec =>
        val embs = rows.map { row =>
          row.getAs[Seq[Any]](spec.column).map(toFloat)
        }.toSeq
        spec -> embs
      }

      val extraColumnData: Map[String, Seq[String]] = extraFields.map { f =>
        val idx = df.schema.fieldIndex(f.name)
        f.name -> rows.map(row => if (row.isNullAt(idx)) "" else row.getAs[String](f.name)).toSeq
      }.toMap

      AilakeNative.writeBatchMulti(
        tableUri       = tableUri,
        namespace      = namespace,
        tableName      = resolvedName,
        ids            = ids,
        vectorColumns  = vectorColumnData,
        embeddingModel = embeddingModel,
        formatVersion  = formatVersion,
        ftsColumns     = ftsColumns,
        ftsTokenizer   = ftsTokenizer,
        deferred       = deferred,
        columns        = extraColumnData,
        catalogOpts    = catalogOpts,
      )
    }
  }
}
