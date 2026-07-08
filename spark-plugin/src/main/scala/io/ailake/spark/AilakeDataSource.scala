// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.spark.sql.connector.catalog.Table
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.connector.catalog.TableProvider
import java.util

/**
 * DataSourceV2 TableProvider for AI-Lake write path.
 *
 * Enables:
 *   df.write.format("io.ailake.spark.AilakeDataSource")
 *     .option("tableUri", "s3://my-lake/docs/")
 *     .option("idColumn", "id")              // default: id
 *     .option("vectorColumn", "embedding")   // default: embedding
 *     .option("dim", "1536")                 // default: inferred from schema
 *     .option("metric", "cosine")            // default: cosine
 *     .option("precision", "f16")            // default: f16
 *     .option("namespace", "default")        // default: default
 *     .option("tableName", "docs")           // default: last path segment of tableUri
 *     .option("textColumns", "text,source,page") // default: none — extra StringType columns to keep
 *     .save()
 *
 * `textColumns` must be declared up front via this option: `TableProvider.inferSchema`
 * only receives write options, never the DataFrame's own schema, so there's no
 * other point at which Spark's V2 write-validation ("table columns" vs. "data
 * columns" arity check) can learn about extra columns before comparing schemas.
 * Declared columns are written as AI-Lake extra metadata via
 * `AilakeNative.writeBatch`'s `columns` map (same capability the Flink
 * connector already exposes) — must be StringType in the DataFrame; cast
 * other types first (`col("page").cast("string")`).
 *
 * Short alias "ailake" available if registered in META-INF/services.
 */
class AilakeDataSource extends TableProvider with DataSourceRegister {

  override def shortName(): String = "ailake"

  override def inferSchema(options: CaseInsensitiveStringMap): StructType =
    AilakeDataSource.buildSchema(options)

  override def getTable(
    schema: StructType,
    partitioning: Array[Transform],
    properties: util.Map[String, String],
  ): Table = {
    val opts = new CaseInsensitiveStringMap(properties)
    val tableUri     = requireOpt(opts, "tableUri", "table-uri")
    val vectorColumn = opts.getOrDefault("vectorColumn", opts.getOrDefault("vector-column", "embedding"))
    val dim          = opts.getOrDefault("dim", opts.getOrDefault("vector-dim", "1536")).toInt
    val metric       = opts.getOrDefault("metric", "cosine")
    val precision    = opts.getOrDefault("precision", "f16")
    val namespace    = opts.getOrDefault("namespace", "default")
    val tableName    = opts.getOrDefault("tableName",
      opts.getOrDefault("table-name",
        tableUri.stripSuffix("/").split("/").last))
    val pfJson       = opts.getOrDefault("partition-fields", opts.getOrDefault("partitionFields", "[]"))
    val partitionFields: Seq[AilakeNative.PartitionFieldDef] = if (pfJson == "[]" || pfJson.isEmpty) Seq.empty else {
      val node = new ObjectMapper().readTree(pfJson)
      (0 until node.size()).map { i =>
        val n = node.get(i)
        AilakeNative.PartitionFieldDef(n.get("column").asText(), n.get("transform").asText(), n.get("column_type").asText())
      }.toSeq
    }
    val formatVersion = opts.getOrDefault("format-version", opts.getOrDefault("formatVersion", "2")).toInt
    // Resolved from options, not the `schema` param: TableProvider.getTable's
    // `schema` argument reflects whatever inferSchema() returned earlier in
    // this same options-driven flow, not the caller's DataFrame — deriving it
    // again here (rather than trusting `schema`) keeps inferSchema/getTable
    // guaranteed consistent regardless of how Spark plumbs the argument.
    val idColumn      = opts.getOrDefault("idColumn", opts.getOrDefault("id-column", "id"))
    val resolvedSchema = AilakeDataSource.buildSchema(opts)
    val (idIdx, vecIdx, textCols) = AilakeWriteHandle.resolveColumns(resolvedSchema, vectorColumn, idColumn)
    new AilakeTable(
      AilakeWriteHandle(tableUri, namespace, tableName, vectorColumn, dim, metric, precision,
        idColIndex = idIdx, vecColIndex = vecIdx, textColIndices = textCols,
        partitionFields = partitionFields, formatVersion = formatVersion),
      tableSchema = resolvedSchema,
    )
  }

  private def requireOpt(opts: CaseInsensitiveStringMap, keys: String*): String =
    keys.collectFirst { case k if opts.containsKey(k) => opts.get(k) }
      .getOrElse(throw new IllegalArgumentException(
        s"One of [${keys.mkString(", ")}] is required for AilakeDataSource"))
}

object AilakeDataSource {

  /** Builds (idColumn, vectorColumn, ...textColumns) from options alone — see class doc. */
  def buildSchema(opts: CaseInsensitiveStringMap): StructType = {
    val idColumn      = opts.getOrDefault("idColumn", opts.getOrDefault("id-column", "id"))
    val vectorColumn = opts.getOrDefault("vectorColumn", opts.getOrDefault("vector-column", "embedding"))
    val textColumns = opts.getOrDefault("textColumns", opts.getOrDefault("text-columns", ""))
      .split(",").map(_.trim).filter(_.nonEmpty).toSeq
    StructType(
      Seq(
        StructField(idColumn, LongType, nullable = true),
        StructField(vectorColumn, ArrayType(DoubleType), nullable = false),
      ) ++ textColumns.map(name => StructField(name, StringType, nullable = true))
    )
  }
}
