package io.ailake.spark

import org.apache.spark.sql.{Dataset, DataFrame, Row, SparkSession, SparkSessionExtensions}
import org.apache.spark.sql.catalyst.encoders.RowEncoder

/**
 * Spark extensions entry point. Register via:
 *
 *   spark.conf.set("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
 *
 * or in spark-defaults.conf / SparkSession builder:
 *
 *   SparkSession.builder()
 *     .config("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
 *     .getOrCreate()
 */
class AilakeSparkExtensions extends (SparkSessionExtensions => Unit) {
  def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectPlannerStrategy(_ => new VectorScanStrategy)
  }
}

/**
 * DataFrame extension methods for AI-Lake vector search.
 *
 * Usage:
 *   import io.ailake.spark.implicits._
 *
 *   val results: DataFrame = spark.ailakeSearch(
 *     tableUri     = "s3://my-lake/docs/",
 *     queryVector  = embeddingArray,   // Array[Float]
 *     topK         = 20,
 *   )
 *   results.show()
 *
 * Returns a DataFrame with columns: row_id (Long), distance (Double), file_path (String).
 */
object implicits {
  implicit class AilakeSession(private val spark: SparkSession) extends AnyVal {
    def ailakeSearch(tableUri: String, queryVector: Array[Float], topK: Int): DataFrame = {
      val plan = VectorSearchPlan(tableUri, queryVector, topK)
      Dataset[Row](spark, plan)(RowEncoder(plan.schema))
    }
  }
}
