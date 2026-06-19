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
    def ailakeSearch(tableUri: String, queryVector: Array[Float], topK: Int): DataFrame = {
      val plan = VectorSearchPlan(tableUri, queryVector, topK)
      val rows = AilakeNative.search(tableUri, queryVector, topK)
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
  }
}
