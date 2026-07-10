// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.connector.catalog.{SupportsDelete, SupportsWrite, Table, TableCapability}
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, WriteBuilder}
import org.apache.spark.sql.sources.{EqualTo, Filter, In}
import org.apache.spark.sql.types._
import java.util

object AilakeTable {
  val WRITE_SCHEMA: StructType = defaultSchema("embedding")

  /** Bare (id, vectorColumn) schema, named after whatever `vector-column` the catalog is configured with. */
  def defaultSchema(vectorColumn: String): StructType = StructType(Array(
    StructField("id",          LongType,              nullable = true),
    StructField(vectorColumn,  ArrayType(DoubleType), nullable = false),
  ))
}

/**
 * AI-Lake write table exposed by [[AilakeCatalog]] and [[AilakeDataSource]].
 *
 * Minimum schema: (id BIGINT, embedding ARRAY<DOUBLE>). Any other columns in
 * `tableSchema` are extra string metadata (see [[AilakeWriteHandle.resolveColumns]])
 * — `tableSchema` defaults to the bare `WRITE_SCHEMA` for callers that don't
 * resolve a real DataFrame schema (e.g. [[AilakeCatalog.loadTable]] today).
 *
 * Supports BATCH_WRITE and equality/IN-pushdown DELETE. Reads are handled by
 * the Iceberg connector or standard Parquet reader — AI-Lake files are valid
 * Iceberg/Parquet.
 */
class AilakeTable(val handle: AilakeWriteHandle, tableSchema: StructType = AilakeTable.WRITE_SCHEMA)
    extends Table with SupportsWrite with SupportsDelete {

  override def name(): String = handle.tableName

  override def schema(): StructType = tableSchema

  override def capabilities(): util.Set[TableCapability] =
    util.EnumSet.of(TableCapability.BATCH_WRITE, TableCapability.TRUNCATE)

  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder =
    new AilakeWriteBuilder(handle)

  // ── DELETE (equality/IN pushdown only) ─────────────────────────────────────
  //
  // AilakeNative.deleteWhere was already fully implemented and tested but had
  // no SQL/DataFrame surface anywhere in this plugin — same "dead capability"
  // gap Trino/Flink already closed via equality/IN predicate pushdown (the
  // only shape AilakeNative.deleteWhere's equality-delete-file mechanism
  // supports — no row-level scan-and-delete exists). Anything else (multi-
  // column predicates, ranges, no WHERE clause) makes canDeleteWhere return
  // false, which Spark surfaces as "DELETE not supported for this table"
  // rather than a silently partial or wrong delete.

  override def canDeleteWhere(filters: Array[Filter]): Boolean =
    filters.length == 1 && (filters(0) match {
      case _: EqualTo => true
      case _: In      => true
      case _          => false
    })

  override def deleteWhere(filters: Array[Filter]): Unit = {
    val (column, values) = filters match {
      case Array(EqualTo(attr, value)) => (attr, Seq(value.toString))
      case Array(In(attr, values))     => (attr, values.toSeq.map(_.toString))
      case _ =>
        throw new UnsupportedOperationException(
          "DELETE requires a WHERE clause that reduces to a single-column equality or IN " +
          "predicate (e.g. WHERE id = 5 or WHERE id IN (1,2,3)) — AI-Lake only supports " +
          "equality deletes, no row-level scan-and-delete is available for this table",
        )
    }
    val ok = AilakeNative.deleteWhere(handle.tableUri, handle.namespace, handle.tableName, column, values)
    if (!ok) {
      throw new RuntimeException(
        s"ailake DELETE WHERE $column IN (...) failed for ${handle.namespace}.${handle.tableName} — see logs")
    }
  }
}
