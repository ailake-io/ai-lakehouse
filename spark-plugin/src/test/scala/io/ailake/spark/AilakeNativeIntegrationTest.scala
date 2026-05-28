// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.spark

import org.scalatest.funsuite.AnyFunSuite

/**
 * End-to-end integration test for the Spark JNA bridge.
 *
 * Required env vars:
 *   AILAKE_SPARK_TRINO_FIXTURE — path to a warehouse directory that contains
 *                                 table "default.table" (written by check_jni_cabi.py)
 *   AILAKE_LIB_PATH            — directory containing libailake_jni.so
 *
 * Skipped automatically when either env var is absent.
 */
class AilakeNativeIntegrationTest extends AnyFunSuite {

  private val fixturePath = sys.env.get("AILAKE_SPARK_TRINO_FIXTURE")
  private val libPath = sys.env.get("AILAKE_LIB_PATH")

  private def libPresent: Boolean =
    libPath.exists(dir => new java.io.File(dir, "libailake_jni.so").exists())

  test("AilakeNative.search with real libailake_jni") {
    assume(fixturePath.nonEmpty, "AILAKE_SPARK_TRINO_FIXTURE not set — skipping")
    assume(libPath.nonEmpty, "AILAKE_LIB_PATH not set — skipping")
    assume(libPresent, s"libailake_jni.so not found in ${libPath.getOrElse("?")} — skipping")

    val dim = 8
    val queryIdx = 7
    val v = Array.tabulate(dim)(j => (queryIdx * dim + j + 1).toFloat)
    val norm = math.sqrt(v.map(x => x.toDouble * x).sum).toFloat
    val query = v.map(_ / norm)

    val results = AilakeNative.search(fixturePath.get, query, topK = 5)
    assert(results.nonEmpty, "search returned empty results — check fixture and native lib")

    val best = results.minBy(_.distance)
    assert(best.rowId == queryIdx.toLong,
      s"nearest row_id=${best.rowId}, expected $queryIdx")

    println(s"PASS (Spark): row_id=${best.rowId} distance=${best.distance}")
    println()
    println("PASS: Spark AilakeNative.search — JNA bridge functional with real library.")
  }
}
