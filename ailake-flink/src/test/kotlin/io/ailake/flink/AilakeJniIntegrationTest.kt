package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.sqrt

/**
 * End-to-end integration test for the Flink JNI bridge.
 * Requires AILAKE_NATIVE_LIB to point to libailake_jni.so.
 *
 * Skipped automatically when the env var is absent (unit-test runs on CI).
 */
class AilakeJniIntegrationTest {

    @Test
    fun writeAndSearch() {
        val nativeLib = System.getenv("AILAKE_NATIVE_LIB")
            ?: System.getProperty("ailake.native.lib")

        assumeTrue(nativeLib != null && File(nativeLib).exists()) {
            "AILAKE_NATIVE_LIB not set or file absent — skipping integration test"
        }

        val dim = 8
        val n = 10
        val embeddings = Array(n) { i ->
            val v = FloatArray(dim) { j -> (i * dim + j + 1).toFloat() }
            val norm = sqrt(v.fold(0f) { acc, x -> acc + x * x }.toDouble()).toFloat()
            FloatArray(dim) { j -> v[j] / norm }
        }
        val ids = LongArray(n) { it.toLong() }

        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-it-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            val snapId = AilakeNativeLoader.writeBatch(
                warehouse = tmp.absolutePath,
                namespace = "default",
                table = "flink_it",
                vecCol = "embedding",
                dim = dim,
                metric = "cosine",
                ids = ids,
                embeddings = embeddings,
            )
            check(snapId >= 0) { "writeBatch returned snapshot_id=$snapId" }
            println("PASS (write): snapshot_id=$snapId")

            val queryIdx = 4
            val results = AilakeNativeLoader.search(
                warehouse = tmp.absolutePath,
                namespace = "default",
                table = "flink_it",
                vecCol = "embedding",
                dim = dim,
                query = embeddings[queryIdx],
                topK = 3,
            )
            check(results.isNotEmpty()) { "search returned empty results" }

            val best = results.minByOrNull { it.distance }!!
            check(best.row_id == queryIdx.toLong()) {
                "nearest row_id=${best.row_id}, expected $queryIdx"
            }
            println("PASS (search): row_id=${best.row_id} distance=${best.distance}")
            println()
            println("PASS: Flink JNI integration — write + search via AilakeNativeLoader.")
        } finally {
            tmp.deleteRecursively()
        }
    }
}
