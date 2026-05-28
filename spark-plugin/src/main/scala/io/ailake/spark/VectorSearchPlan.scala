// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.spark

import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.types.{DoubleType, LongType, StringType}

/**
 * Logical plan node for AI-Lake vector search.
 *
 * Created by [[AilakeImplicits.VectorSearchDataFrame.vectorSearch]] or by
 * `ailake_vector_search(tableUri, queryVector, topK)` SQL syntax.
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
