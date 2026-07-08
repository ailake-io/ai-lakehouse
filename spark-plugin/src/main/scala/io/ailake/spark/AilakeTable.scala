// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.connector.catalog.{SupportsWrite, Table, TableCapability}
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, WriteBuilder}
import org.apache.spark.sql.types._
import java.util

object AilakeTable {
  val WRITE_SCHEMA: StructType = StructType(Array(
    StructField("id",        LongType,              nullable = true),
    StructField("embedding", ArrayType(DoubleType), nullable = false),
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
 * Supports BATCH_WRITE only. Reads are handled by the Iceberg connector or
 * standard Parquet reader — AI-Lake files are valid Iceberg/Parquet.
 */
class AilakeTable(val handle: AilakeWriteHandle, tableSchema: StructType = AilakeTable.WRITE_SCHEMA)
    extends Table with SupportsWrite {

  override def name(): String = handle.tableName

  override def schema(): StructType = tableSchema

  override def capabilities(): util.Set[TableCapability] =
    util.EnumSet.of(TableCapability.BATCH_WRITE, TableCapability.TRUNCATE)

  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder =
    new AilakeWriteBuilder(handle)
}
