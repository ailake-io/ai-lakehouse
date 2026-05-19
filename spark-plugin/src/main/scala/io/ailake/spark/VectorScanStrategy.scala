package io.ailake.spark

import org.apache.spark.sql.Strategy
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.SparkPlan

/**
 * Spark planner strategy that converts [[VectorSearchPlan]] logical nodes into
 * [[VectorScanExec]] physical nodes.
 *
 * Registered via [[AilakeSparkExtensions]] when the user sets:
 *   spark.sql.extensions = io.ailake.spark.AilakeSparkExtensions
 */
class VectorScanStrategy extends Strategy {
  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case vsp: VectorSearchPlan =>
      VectorScanExec(vsp.tableUri, vsp.query, vsp.topK, vsp.output) :: Nil
    case _ =>
      Nil
  }
}
