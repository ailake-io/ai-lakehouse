// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.sqrt

/**
 * End-to-end integration test for the Trino JNA bridge.
 *
 * Required env vars:
 *   AILAKE_SPARK_TRINO_FIXTURE — warehouse directory containing table "default.table"
 *                                 (written by check_jni_cabi.py)
 *   AILAKE_LIB_PATH            — directory containing libailake_jni.so
 *
 * Skipped automatically when either env var is absent.
 */
class AilakeNativeIntegrationTest {

    private val fixturePath = System.getenv("AILAKE_SPARK_TRINO_FIXTURE")
    private val libPath = System.getenv("AILAKE_LIB_PATH")
    private val libPresent get() =
        libPath != null && File(libPath, "libailake_jni.so").exists()

    @Test
    fun searchWithNativeLibrary() {
        assumeTrue(fixturePath != null) { "AILAKE_SPARK_TRINO_FIXTURE not set — skipping" }
        assumeTrue(libPath != null) { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(libPresent) { "libailake_jni.so not found in $libPath — skipping" }

        val dim = 8
        val queryIdx = 7
        val v = FloatArray(dim) { j -> (queryIdx * dim + j + 1).toFloat() }
        val norm = sqrt(v.fold(0f) { acc, x -> acc + x * x }.toDouble()).toFloat()
        val queryCsv = v.joinToString(",") { (it / norm).toString() }
        val queryBytes = VectorScanSplitManager.csvFloatsToBase64(queryCsv)

        val results = AilakeNative.search(fixturePath!!, queryBytes, topK = 5, tableName = "table")
        check(results.isNotEmpty()) { "search returned empty results — check fixture and native lib" }

        val best = results.minByOrNull { it.distance }!!
        check(best.rowId == queryIdx.toLong()) {
            "nearest rowId=${best.rowId}, expected $queryIdx"
        }
        println("PASS (Trino): rowId=${best.rowId} distance=${best.distance}")
        println()
        println("PASS: Trino AilakeNative.search — JNA bridge functional with real library.")
    }
}
