// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.connector.catalog.Table
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.connector.catalog.TableProvider
import java.util

/**
 * DataSourceV2 TableProvider for AI-Lake write path.
 *
 * Enables:
 *   df.write.format("io.ailake.spark.AilakeDataSource")
 *     .option("tableUri", "s3://my-lake/docs/")
 *     .option("vectorColumn", "embedding")   // default: embedding
 *     .option("dim", "1536")                 // default: inferred from schema
 *     .option("metric", "cosine")            // default: cosine
 *     .option("precision", "f16")            // default: f16
 *     .option("namespace", "default")        // default: default
 *     .option("tableName", "docs")           // default: last path segment of tableUri
 *     .save()
 *
 * Short alias "ailake" available if registered in META-INF/services.
 */
class AilakeDataSource extends TableProvider with DataSourceRegister {

  override def shortName(): String = "ailake"

  override def inferSchema(options: CaseInsensitiveStringMap): StructType =
    AilakeTable.WRITE_SCHEMA

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
    new AilakeTable(AilakeWriteHandle(tableUri, namespace, tableName, vectorColumn, dim, metric, precision))
  }

  private def requireOpt(opts: CaseInsensitiveStringMap, keys: String*): String =
    keys.collectFirst { case k if opts.containsKey(k) => opts.get(k) }
      .getOrElse(throw new IllegalArgumentException(
        s"One of [${keys.mkString(", ")}] is required for AilakeDataSource"))
}
