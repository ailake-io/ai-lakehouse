// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.{DataFrame, Row, SparkSession, SparkSessionExtensions}

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
 */
object implicits {
  implicit class AilakeSession(private val spark: SparkSession) extends AnyVal {
    /**
     * @param namespace  Iceberg namespace the table was written under (default: "default").
     *                   Must match the `namespace` passed to [[ailakeWrite]] — otherwise this
     *                   silently searches a different (likely empty) table location.
     * @param tableName  table name; defaults to the last segment of `tableUri`, same
     *                   resolution rule [[ailakeWrite]] uses.
     */
    def ailakeSearch(
      tableUri:    String,
      queryVector: Array[Float],
      topK:        Int,
      namespace:   String = "default",
      tableName:   String = "",
    ): DataFrame = {
      val plan = VectorSearchPlan(tableUri, queryVector, topK)
      val rows = AilakeNative.search(tableUri, queryVector, topK, namespace = namespace, tableName = tableName)
      val sparkRows = rows.map(r => Row(r.rowId, r.distance.toDouble, r.filePath))
      spark.createDataFrame(spark.sparkContext.parallelize(sparkRows, numSlices = 1), plan.schema)
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
      )
    }
  }
}
