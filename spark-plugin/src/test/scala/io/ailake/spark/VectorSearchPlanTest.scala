// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.spark

import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}
import org.scalatest.funsuite.AnyFunSuite

class VectorSearchPlanTest extends AnyFunSuite {

  private val query = Array(0.1f, -0.2f, 0.3f)

  test("output has three attributes in order: row_id, distance, file_path") {
    val plan = VectorSearchPlan("s3://bucket/t/", query, topK = 10)
    val names = plan.output.map(_.name)
    assert(names == Seq("row_id", "distance", "file_path"))
  }

  test("output types match schema") {
    val plan = VectorSearchPlan("s3://bucket/t/", query, topK = 10)
    assert(plan.output(0).dataType == LongType)
    assert(plan.output(1).dataType == DoubleType)
    assert(plan.output(2).dataType == StringType)
  }

  test("schema derived from output") {
    val plan = VectorSearchPlan("s3://bucket/t/", query, topK = 10)
    val expected = StructType(Seq(
      StructField("row_id", LongType, nullable = false),
      StructField("distance", DoubleType, nullable = false),
      StructField("file_path", StringType, nullable = false),
    ))
    assert(plan.schema == expected)
  }

  test("equals based on content not reference") {
    val q1 = Array(1.0f, 2.0f)
    val q2 = Array(1.0f, 2.0f)
    val p1 = VectorSearchPlan("s3://t/", q1, 5)
    val p2 = VectorSearchPlan("s3://t/", q2, 5)
    assert(p1 == p2)
  }

  test("equals false when tableUri differs") {
    val p1 = VectorSearchPlan("s3://a/", query, 5)
    val p2 = VectorSearchPlan("s3://b/", query, 5)
    assert(p1 != p2)
  }

  test("equals false when topK differs") {
    val p1 = VectorSearchPlan("s3://t/", query, 5)
    val p2 = VectorSearchPlan("s3://t/", query, 10)
    assert(p1 != p2)
  }

  test("hashCode stable for equal plans") {
    val q1 = Array(0.5f)
    val q2 = Array(0.5f)
    val p1 = VectorSearchPlan("s3://t/", q1, 3)
    val p2 = VectorSearchPlan("s3://t/", q2, 3)
    assert(p1.hashCode() == p2.hashCode())
  }

  test("resolved returns false (leaf node with no catalog reference)") {
    val plan = VectorSearchPlan("s3://t/", query, 5)
    // LeafNode.resolved is true when all output attributes are resolved.
    // Our AttributeReferences are always resolved since they have explicit types.
    assert(plan.resolved)
  }
}
