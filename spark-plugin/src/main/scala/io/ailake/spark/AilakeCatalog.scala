// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.catalyst.analysis.NoSuchTableException
import org.apache.spark.sql.connector.catalog._
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.slf4j.LoggerFactory
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

  private val log = LoggerFactory.getLogger(getClass.getName)

  private var catalogName_ : String = _
  private var opts: CaseInsensitiveStringMap = _

  override def initialize(name: String, options: CaseInsensitiveStringMap): Unit = {
    catalogName_ = name
    opts = options
  }

  override def name(): String = catalogName_

  // ── TableCatalog ──────────────────────────────────────────────────────────

  override def loadTable(ident: Identifier): Table =
    buildTable(ident, AilakeTable.defaultSchema(opts.getOrDefault("vector-column", "embedding")))

  override def listTables(namespace: Array[String]): Array[Identifier] = Array.empty

  override def createTable(
    ident: Identifier,
    schema: StructType,
    partitions: Array[Transform],
    properties: util.Map[String, String],
  ): Table = buildTable(ident, schema)

  /**
   * `AilakeNative.evolveSchema` was already fully implemented and tested but had no SQL
   * surface — this used to unconditionally throw. Same "dead capability" gap Trino/Flink
   * already closed, closed the same way: ADD COLUMN / RENAME COLUMN only (metadata-only,
   * Iceberg primitive types only — matches `AilakeWriteHandle.resolveColumns`' own minimal
   * column type surface).
   *
   * IMPORTANT limitation, same as Trino/Flink: this catalog resolves its schema per-call
   * from `spark.sql.catalog.<name>.*` options (see `buildTable`) — a column added here is
   * genuinely persisted to the AI-Lake table's Iceberg schema on disk (evolveSchema is a
   * real metadata-only operation, not a no-op), but nothing here tracks it, so subsequent
   * `INSERT`/`SELECT` against this catalog still won't see it without a DataFrame schema
   * that already includes the new column (`AilakeWriteHandle.resolveColumns` derives extra
   * text columns from whatever schema Spark resolves per-call, not from catalog state).
   */
  override def alterTable(ident: Identifier, changes: TableChange*): Table = {
    val tableUri  = requireOpt("table-uri")
    val namespace = if (ident.namespace().nonEmpty) ident.namespace()(0) else "default"
    val tableName = ident.name()

    val addCols    = scala.collection.mutable.ArrayBuffer.empty[AilakeNative.AddColReq]
    val renameCols = scala.collection.mutable.ArrayBuffer.empty[AilakeNative.RenameColReq]

    changes.foreach {
      case ac: TableChange.AddColumn =>
        if (ac.fieldNames().length != 1)
          throw new UnsupportedOperationException(
            "AI-Lake catalog's ADD COLUMN only supports top-level columns, not nested paths")
        addCols += AilakeNative.AddColReq(ac.fieldNames()(0), sparkTypeToIcebergType(ac.dataType()))
      case rc: TableChange.RenameColumn =>
        if (rc.fieldNames().length != 1)
          throw new UnsupportedOperationException(
            "AI-Lake catalog's RENAME COLUMN only supports top-level columns, not nested paths")
        renameCols += AilakeNative.RenameColReq(rc.fieldNames()(0), rc.newName())
      case other =>
        throw new UnsupportedOperationException(
          s"AI-Lake catalog only supports ADD COLUMN / RENAME COLUMN, got: ${other.getClass.getSimpleName}")
    }

    val schemaId = AilakeNative.evolveSchema(tableUri, namespace, tableName, addCols.toSeq, renameCols.toSeq)
    if (schemaId < 0)
      throw new RuntimeException(s"ailake ALTER TABLE failed for $namespace.$tableName — see logs")
    log.warn(
      s"[ailake] ALTER TABLE applied to $namespace.$tableName on disk (new_schema_id=$schemaId) — " +
      "this catalog resolves its schema per-call from spark.sql.catalog.*.* options and the current " +
      "DataFrame, not from any tracked state, so subsequent INSERT/SELECT only see the new column " +
      "if their own DataFrame schema already includes it.")
    buildTable(ident, AilakeTable.defaultSchema(opts.getOrDefault("vector-column", "embedding")))
  }

  /**
   * Common Iceberg primitive types only — matches this catalog's own minimal column type
   * surface (id BIGINT, embedding ARRAY<DOUBLE>, text columns STRING). Complex types
   * (ARRAY/MAP/STRUCT) and other timestamp/decimal variants are rejected rather than
   * silently mis-mapped — see ailake-catalog's `schema_evolution.rs` for the full set of
   * Iceberg type strings the native side accepts.
   */
  private def sparkTypeToIcebergType(dataType: DataType): String = dataType match {
    case LongType    => "long"
    case IntegerType => "int"
    case DoubleType  => "double"
    case FloatType   => "float"
    case BooleanType => "boolean"
    case DateType    => "date"
    case _: StringType => "string"
    case other => throw new UnsupportedOperationException(
      s"Column type $other is not supported by ALTER TABLE ADD COLUMN for AI-Lake tables — " +
      "supported types: bigint, integer, double, float, boolean, date, string",
    )
  }

  override def dropTable(ident: Identifier): Boolean = false

  override def renameTable(oldIdent: Identifier, newIdent: Identifier): Unit =
    throw new UnsupportedOperationException("RENAME TABLE not supported by AI-Lake catalog")

  // ── helpers ───────────────────────────────────────────────────────────────

  /**
   * `loadTable` (called for bare `INSERT INTO ailake.ns.table VALUES (...)`
   * with no known DataFrame schema) falls back to the bare `id, <vector-column>`
   * schema, named after the configured `vector-column` option (not hardcoded
   * to `"embedding"` — a table configured with a different vector column name
   * would otherwise fail `resolveColumns`' `fieldIndex` lookup). `createTable`
   * (called with a `CREATE TABLE ... AS SELECT`-shaped schema, or any other
   * schema Spark already resolved) passes it through, so extra string columns
   * (chunk text, source, page, ...) get written as AI-Lake metadata instead of
   * silently dropped — see [[AilakeWriteHandle.resolveColumns]].
   */
  private def buildTable(ident: Identifier, schema: StructType): Table = {
    val tableUri       = requireOpt("table-uri")
    val vectorColumn   = opts.getOrDefault("vector-column", "embedding")
    val dim            = opts.getOrDefault("vector-dim", "1536").toInt
    val metric         = opts.getOrDefault("metric", "cosine")
    val precision      = opts.getOrDefault("precision", "f16")
    val namespace      = if (ident.namespace().nonEmpty) ident.namespace()(0) else "default"
    val tableName      = ident.name()
    val embeddingModel = Option(opts.get("embedding-model")).filter(_.nonEmpty)
    val (idIdx, vecIdx, textCols) = AilakeWriteHandle.resolveColumns(schema, vectorColumn)

    new AilakeTable(
      AilakeWriteHandle(tableUri, namespace, tableName, vectorColumn, dim, metric, precision,
        idColIndex = idIdx, vecColIndex = vecIdx, textColIndices = textCols,
        embeddingModel = embeddingModel),
      tableSchema = schema,
    )
  }

  private def requireOpt(key: String): String =
    Option(opts.get(key)).getOrElse(
      throw new IllegalArgumentException(
        s"spark.sql.catalog.$catalogName_.$key is required"))
}
