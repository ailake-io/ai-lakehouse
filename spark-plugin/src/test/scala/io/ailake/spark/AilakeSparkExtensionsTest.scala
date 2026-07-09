// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.junit.runner.RunWith
import org.scalatestplus.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
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
    val execPlan = spark.sessionState.executePlan(plan).executedPlan
    assert(execPlan.isInstanceOf[VectorScanExec], s"expected VectorScanExec, got: ${execPlan.getClass.getSimpleName}")
  }

  test("ailakeSearch with dimension 1536 produces valid empty result") {
    import io.ailake.spark.implicits._
    val query = Array.fill(1536)(0.0f)
    val df = spark.ailakeSearch("s3://lake/docs/", query, topK = 20)
    assert(df.schema.fieldNames sameElements Array("row_id", "distance", "file_path"))
    assert(df.count() == 0)
  }

  // Regression: ailakeSearch used to have no namespace/tableName parameters at
  // all, always searching AilakeNative.search's hardcoded defaults
  // (namespace="default") regardless of what ailakeWrite actually wrote to —
  // a write to a non-default namespace was unfindable via search, silently
  // returning empty results. These params now exist and are threaded through
  // to AilakeNative.search; with no native lib present this still degrades to
  // an empty result, but the signature/plumbing itself is what's under test.
  test("ailakeSearch accepts namespace and tableName parameters") {
    import io.ailake.spark.implicits._
    val query = Array(0.1f, 0.2f, 0.3f)
    val df = spark.ailakeSearch("s3://test-bucket/table/", query, topK = 5, namespace = "prod", tableName = "docs")
    assert(df.schema.fieldNames sameElements Array("row_id", "distance", "file_path"))
    assert(df.count() == 0)
  }

  // ── ailakeSearchMultimodal (cross-modal RRF search) ───────────────────────
  //
  // Regression: AilakeNative.searchMultimodal was fully implemented but never
  // exposed as a DataFrame call — same "dead capability" gap as ailakeSearch
  // before it, closed the same way.

  test("ailakeSearchMultimodal returns DataFrame with correct schema") {
    import io.ailake.spark.implicits._
    val queries = Seq(("embedding", Array(0.1f, -0.2f), 1.0f))
    val df = spark.ailakeSearchMultimodal("s3://test-bucket/table/", queries, topK = 5)

    val expectedSchema = StructType(Seq(
      StructField("row_id", LongType, nullable = false),
      StructField("rrf_score", DoubleType, nullable = false),
      StructField("file_path", StringType, nullable = false),
    ))
    assert(df.schema == expectedSchema)
  }

  test("ailakeSearchMultimodal returns empty DataFrame when native library absent") {
    import io.ailake.spark.implicits._
    val queries = Seq(
      ("embedding", Array(0.1f, 0.2f), 1.0f),
      ("image_embedding", Array(0.3f, 0.4f), 0.5f),
    )
    val df = spark.ailakeSearchMultimodal("s3://test-bucket/table/", queries, topK = 10)
    assert(df.count() == 0)
  }

  // ── ailakeWriteMulti (Phase 8 multimodal write) ───────────────────────────
  //
  // Closes the "searchMultimodal has no write path from Spark" gap: previously
  // only the Python SDK (TableWriter.write_batch_multi via PyO3) could write a
  // table with 2+ vector columns; there was no ailake-jni C-ABI export for it
  // at all, so searchMultimodal was reachable from Spark but never
  // self-sufficient. ailakeWriteMulti + AilakeNative.writeBatchMulti (backed
  // by the new ailake_write_batch_multi_json JNI export) close that gap.

  import org.apache.spark.sql.Row
  import org.apache.spark.sql.types.ArrayType

  private def multiModalRow(id: Long, text: Seq[Double], image: Seq[Double]): Row =
    Row(id, text, image)

  private val multiModalSchema = StructType(Seq(
    StructField("id",              LongType,                          nullable = true),
    StructField("embedding",       ArrayType(DoubleType), nullable = false),
    StructField("image_embedding", ArrayType(DoubleType), nullable = false),
  ))

  test("ailakeWriteMulti returns None when native library absent") {
    // Guarded like AilakeNativeTest's lib-absent tests: with the real lib on
    // the classpath (CI's test-jvm job, or this session's manual verification
    // run), "s3://test-bucket/multimodal/" resolves to a real local path via
    // LocalStore (no scheme validation) and the write actually succeeds.
    assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
    import io.ailake.spark.implicits._
    val rows = Seq(
      multiModalRow(1L, Seq(0.1, 0.2), Seq(0.5, 0.6)),
      multiModalRow(2L, Seq(0.3, 0.4), Seq(0.7, 0.8)),
    )
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows), multiModalSchema)
    val result = spark.ailakeWriteMulti(
      tableUri      = "s3://test-bucket/multimodal/",
      df            = df,
      vectorColumns = Seq(
        AilakeNative.VectorColSpec("embedding", dim = 2),
        AilakeNative.VectorColSpec("image_embedding", dim = 2),
      ),
    )
    assert(result.isEmpty)
  }

  test("ailakeWriteMulti requires at least one VectorColSpec") {
    import io.ailake.spark.implicits._
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(Seq(multiModalRow(1L, Seq(0.1, 0.2), Seq(0.5, 0.6)))),
      multiModalSchema,
    )
    intercept[IllegalArgumentException] {
      spark.ailakeWriteMulti("s3://test-bucket/multimodal/", df, vectorColumns = Seq.empty)
    }
  }

  test("ailakeWriteMulti rejects missing id column") {
    import io.ailake.spark.implicits._
    val schema = StructType(Seq(StructField("embedding", ArrayType(DoubleType), nullable = false)))
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(Seq(Row(Seq(0.1, 0.2)))),
      schema,
    )
    val ex = intercept[IllegalArgumentException] {
      spark.ailakeWriteMulti("s3://test-bucket/multimodal/", df,
        vectorColumns = Seq(AilakeNative.VectorColSpec("embedding", dim = 2)))
    }
    assert(ex.getMessage.contains("id"))
  }

  test("ailakeWriteMulti rejects unknown vector column") {
    import io.ailake.spark.implicits._
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(Seq(multiModalRow(1L, Seq(0.1, 0.2), Seq(0.5, 0.6)))),
      multiModalSchema,
    )
    val ex = intercept[IllegalArgumentException] {
      spark.ailakeWriteMulti("s3://test-bucket/multimodal/", df,
        vectorColumns = Seq(AilakeNative.VectorColSpec("nonexistent_col", dim = 2)))
    }
    assert(ex.getMessage.contains("nonexistent_col"))
  }

  test("ailakeWriteMulti rejects non-string extra column") {
    import io.ailake.spark.implicits._
    val schema = StructType(Seq(
      StructField("id",        LongType,               nullable = true),
      StructField("embedding", ArrayType(DoubleType), nullable = false),
      StructField("page",      org.apache.spark.sql.types.IntegerType, nullable = true),
    ))
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(Seq(Row(1L, Seq(0.1, 0.2), 7))),
      schema,
    )
    val ex = intercept[IllegalArgumentException] {
      spark.ailakeWriteMulti("s3://test-bucket/multimodal/", df,
        vectorColumns = Seq(AilakeNative.VectorColSpec("embedding", dim = 2)))
    }
    assert(ex.getMessage.contains("page"))
    assert(ex.getMessage.contains("StringType"))
  }
}
