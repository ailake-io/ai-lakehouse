// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.types.{DoubleType, LongType, StringType}
import org.apache.spark.unsafe.types.UTF8String

/**
 * Physical execution node for AI-Lake vector search.
 *
 * Calls the native library synchronously on the driver (single partition).
 * Large-scale production deployments should shard by table partition; for Phase 3
 * this single-driver approach is sufficient to validate the integration path.
 *
 * @param tableUri  AI-Lake table root URI
 * @param query     f32 query embedding
 * @param topK      number of nearest neighbors
 */
case class VectorScanExec(
  tableUri: String,
  query: Array[Float],
  topK: Int,
  override val output: Seq[Attribute],
) extends LeafExecNode {

  override protected def doExecute(): RDD[InternalRow] = {
    val rows = AilakeNative.search(tableUri, query, topK)
    val internalRows: Seq[InternalRow] = rows.map { r =>
      InternalRow(r.rowId, r.distance.toDouble, UTF8String.fromString(r.filePath))
    }
    sparkContext.parallelize(internalRows, numSlices = 1)
  }

  // Array equality for plan identity.
  override def equals(other: Any): Boolean = other match {
    case o: VectorScanExec =>
      tableUri == o.tableUri && java.util.Arrays.equals(query, o.query) && topK == o.topK
    case _ => false
  }

  override def hashCode(): Int =
    31 * (31 * tableUri.hashCode + java.util.Arrays.hashCode(query)) + topK
}
