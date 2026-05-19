package io.ailake.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class AilakeSparkExtensionsTest extends AnyFunSuite with BeforeAndAfterAll {

  @transient private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[1]")
      .appName("AilakeSparkExtensionsTest")
      .config("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .getOrCreate()
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
      SparkSession.clearActiveSession()
    }
  }

  test("VectorScanStrategy is registered in planner") {
    val strategies = spark.sessionState.planner.strategies
    assert(strategies.exists(_.isInstanceOf[VectorScanStrategy]))
  }

  test("ailakeSearch returns DataFrame with correct schema") {
    import io.ailake.spark.implicits._
    val query = Array(0.1f, -0.2f, 0.3f)
    val df = spark.ailakeSearch("s3://test-bucket/table/", query, topK = 5)

    val expectedSchema = StructType(Seq(
      StructField("row_id", LongType, nullable = false),
      StructField("distance", DoubleType, nullable = false),
      StructField("file_path", StringType, nullable = false),
    ))
    assert(df.schema == expectedSchema)
  }

  test("ailakeSearch returns empty DataFrame when native library absent") {
    import io.ailake.spark.implicits._
    val query = Array(0.1f, 0.2f, 0.3f)
    val df = spark.ailakeSearch("s3://test-bucket/table/", query, topK = 10)
    assert(df.count() == 0)
  }

  test("VectorSearchPlan is converted to VectorScanExec by planner") {
    val plan = VectorSearchPlan("s3://t/", Array(1.0f), topK = 3)
    import org.apache.spark.sql.{Dataset, Row}
    import org.apache.spark.sql.catalyst.encoders.RowEncoder
    val df = Dataset[Row](spark, plan)(RowEncoder(plan.schema))
    // executedPlan traversal verifies the strategy ran
    val execPlan = df.queryExecution.executedPlan
    assert(execPlan.toString.contains("VectorScanExec"))
  }

  test("ailakeSearch with dimension 1536 produces valid empty result") {
    import io.ailake.spark.implicits._
    val query = Array.fill(1536)(0.0f)
    val df = spark.ailakeSearch("s3://lake/docs/", query, topK = 20)
    assert(df.schema.fieldNames sameElements Array("row_id", "distance", "file_path"))
    assert(df.count() == 0)
  }
}
