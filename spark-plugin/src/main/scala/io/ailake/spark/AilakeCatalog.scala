// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.catalyst.analysis.NoSuchTableException
import org.apache.spark.sql.connector.catalog._
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import java.util

/**
 * V2 catalog plugin that enables SQL INSERT INTO for AI-Lake tables.
 *
 * Register in Spark configuration:
 *   spark.sql.catalog.ailake     = io.ailake.spark.AilakeCatalog
 *   spark.sql.catalog.ailake.table-uri      = s3://my-lake/docs/
 *   spark.sql.catalog.ailake.vector-column  = embedding     (default: embedding)
 *   spark.sql.catalog.ailake.vector-dim     = 1536          (default: 1536)
 *   spark.sql.catalog.ailake.metric         = cosine        (default: cosine)
 *   spark.sql.catalog.ailake.precision      = f16           (default: f16)
 *
 * Usage:
 *   INSERT INTO ailake.default.docs VALUES (1, array(0.1, 0.2, ...))
 *
 * The catalog derives `namespace` and `tableName` from the Identifier passed
 * by Spark. `table-uri` serves as the warehouse root; the JNI layer receives
 * it as `warehouse` along with the resolved namespace and table name.
 */
class AilakeCatalog extends CatalogPlugin with TableCatalog {

  private var catalogName_ : String = _
  private var opts: CaseInsensitiveStringMap = _

  override def initialize(name: String, options: CaseInsensitiveStringMap): Unit = {
    catalogName_ = name
    opts = options
  }

  override def name(): String = catalogName_

  // ── TableCatalog ──────────────────────────────────────────────────────────

  override def loadTable(ident: Identifier): Table = {
    val tableUri       = requireOpt("table-uri")
    val vectorColumn   = opts.getOrDefault("vector-column", "embedding")
    val dim            = opts.getOrDefault("vector-dim", "1536").toInt
    val metric         = opts.getOrDefault("metric", "cosine")
    val precision      = opts.getOrDefault("precision", "f16")
    val namespace      = if (ident.namespace().nonEmpty) ident.namespace()(0) else "default"
    val tableName      = ident.name()
    val embeddingModel = Option(opts.get("embedding-model")).filter(_.nonEmpty)

    new AilakeTable(AilakeWriteHandle(tableUri, namespace, tableName, vectorColumn, dim, metric, precision, embeddingModel = embeddingModel))
  }

  override def listTables(namespace: Array[String]): Array[Identifier] = Array.empty

  override def createTable(
    ident: Identifier,
    schema: StructType,
    partitions: Array[Transform],
    properties: util.Map[String, String],
  ): Table = loadTable(ident)

  override def alterTable(ident: Identifier, changes: TableChange*): Table =
    throw new UnsupportedOperationException("ALTER TABLE not supported by AI-Lake catalog")

  override def dropTable(ident: Identifier): Boolean = false

  override def renameTable(oldIdent: Identifier, newIdent: Identifier): Unit =
    throw new UnsupportedOperationException("RENAME TABLE not supported by AI-Lake catalog")

  // ── helpers ───────────────────────────────────────────────────────────────

  private def requireOpt(key: String): String =
    Option(opts.get(key)).getOrElse(
      throw new IllegalArgumentException(
        s"spark.sql.catalog.$catalogName_.$key is required"))
}
