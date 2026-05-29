// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.catalyst.plans.logical.{Filter, LocalRelation}
import org.scalatest.funsuite.AnyFunSuite

class VectorScanStrategyTest extends AnyFunSuite {

  private val strategy = new VectorScanStrategy
  private val query = Array(0.1f, 0.2f, 0.3f)

  test("converts VectorSearchPlan to VectorScanExec") {
    val plan = VectorSearchPlan("s3://bucket/t/", query, topK = 10)
    val physical = strategy.apply(plan)
    assert(physical.size == 1)
    assert(physical.head.isInstanceOf[VectorScanExec])
  }

  test("VectorScanExec carries correct tableUri") {
    val plan = VectorSearchPlan("s3://my-lake/docs/", query, topK = 5)
    val exec = strategy.apply(plan).head.asInstanceOf[VectorScanExec]
    assert(exec.tableUri == "s3://my-lake/docs/")
  }

  test("VectorScanExec carries correct topK") {
    val plan = VectorSearchPlan("s3://t/", query, topK = 42)
    val exec = strategy.apply(plan).head.asInstanceOf[VectorScanExec]
    assert(exec.topK == 42)
  }

  test("VectorScanExec output matches VectorSearchPlan output") {
    val plan = VectorSearchPlan("s3://t/", query, topK = 10)
    val exec = strategy.apply(plan).head.asInstanceOf[VectorScanExec]
    assert(exec.output == plan.output)
  }

  test("returns empty for unrecognised logical plan") {
    val other = LocalRelation()
    assert(strategy.apply(other).isEmpty)
  }

  test("VectorScanExec equals based on content") {
    val p1 = VectorSearchPlan("s3://t/", Array(1.0f), 5)
    val p2 = VectorSearchPlan("s3://t/", Array(1.0f), 5)
    val e1 = strategy.apply(p1).head.asInstanceOf[VectorScanExec]
    val e2 = strategy.apply(p2).head.asInstanceOf[VectorScanExec]
    assert(e1 == e2)
  }
}
