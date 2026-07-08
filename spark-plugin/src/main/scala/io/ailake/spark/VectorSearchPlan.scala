// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.types.{DoubleType, LongType, StringType}

/**
 * Logical plan node for AI-Lake vector search.
 *
 * Not currently reachable via SQL — no parser/function is registered for an
 * `ailake_vector_search(...)` syntax, only [[VectorScanStrategy]]'s planner
 * injection exists ([[AilakeSparkExtensions]]). The only production caller,
 * `implicits.AilakeSession.ailakeSearch`, builds this plan solely to borrow
 * its `output` schema and calls [[AilakeNative.search]] directly rather than
 * executing the plan through Spark's planner — see [[VectorScanStrategy]] and
 * [[VectorScanExec]] for the (currently untriggered) strategy-conversion path.
 *
 * @param tableUri    AI-Lake table root URI
 * @param query       f32 query embedding
 * @param topK        number of nearest neighbors
 */
case class VectorSearchPlan(
  tableUri: String,
  query: Array[Float],
  topK: Int,
) extends LeafNode {

  override val output: Seq[Attribute] = Seq(
    AttributeReference("row_id", LongType, nullable = false)(),
    AttributeReference("distance", DoubleType, nullable = false)(),
    AttributeReference("file_path", StringType, nullable = false)(),
  )

  // Array equality is reference-based by default; override for plan dedup.
  override def equals(other: Any): Boolean = other match {
    case o: VectorSearchPlan =>
      tableUri == o.tableUri && java.util.Arrays.equals(query, o.query) && topK == o.topK
    case _ => false
  }

  override def hashCode(): Int =
    31 * (31 * tableUri.hashCode + java.util.Arrays.hashCode(query)) + topK
}
