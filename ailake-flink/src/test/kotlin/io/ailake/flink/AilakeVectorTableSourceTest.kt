// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.apache.flink.api.common.ExecutionConfig
import org.apache.flink.api.common.functions.RuntimeContext
import org.apache.flink.configuration.Configuration
import org.apache.flink.core.io.GenericInputSplit
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever

/**
 * Regression: `AilakeInputFormat.open()` used to call `AilakeNativeLoader.search()`
 * unguarded — `AilakeNativeLoader.lib` is `by lazy { ... }.getOrThrow()`, so a missing
 * `libailake_jni.so` (guaranteed in this unit test environment, which never sets
 * `AILAKE_NATIVE_LIB`/has the library on the path) threw an uncaught error out of
 * `open()`, failing the whole Flink task instead of degrading to zero rows like
 * Spark/Trino/DuckDB do. `open()` now catches native-load failures and leaves the
 * result iterator empty.
 */
class AilakeVectorTableSourceTest {

    private fun runtimeContextWithQueryVector(vector: String): RuntimeContext {
        val params = Configuration()
        params.setString("ailake.query.vector", vector)
        return runtimeContextWithParams(params)
    }

    private fun runtimeContextWithParams(params: Configuration): RuntimeContext {
        val executionConfig = ExecutionConfig()
        executionConfig.globalJobParameters = params

        val ctx = mock<RuntimeContext>()
        whenever(ctx.executionConfig).thenReturn(executionConfig)
        return ctx
    }

    @Test
    fun openDoesNotThrowWhenNativeLibMissing() {
        val format = AilakeInputFormat(
            warehouse = "file:///tmp/ailake-flink-test-does-not-need-to-exist",
            namespace = "default",
            tableName = "table",
            vecCol = "embedding",
            dim = 4,
            topK = 5,
            efSearch = 50,
        )
        format.runtimeContext = runtimeContextWithQueryVector("1.0,0.0,0.0,0.0")

        assertDoesNotThrow {
            format.open(GenericInputSplit(0, 1))
        }
    }

    @Test
    fun openWithoutNativeLibLeavesResultSetEmpty() {
        val format = AilakeInputFormat(
            warehouse = "file:///tmp/ailake-flink-test-does-not-need-to-exist",
            namespace = "default",
            tableName = "table",
            vecCol = "embedding",
            dim = 4,
            topK = 5,
            efSearch = 50,
        )
        format.runtimeContext = runtimeContextWithQueryVector("1.0,0.0,0.0,0.0")
        format.open(GenericInputSplit(0, 1))

        assertTrue(
            format.reachedEnd(),
            "expected empty result set (reachedEnd()==true) when the native lib can't be loaded"
        )
    }

    // ── query.vector / query.text job parameter combinations ──────────────────
    //
    // Regression: `open()` used to throw if `ailake.query.vector` wasn't set, with
    // no way to do a pure full-text search (`AilakeNativeLoader.searchText`) or a
    // hybrid BM25+vector search (`AilakeNativeLoader.search`'s `hybridText` path) —
    // both were already fully implemented but unreachable from any Flink source.

    @Test
    fun openThrowsWhenNeitherQueryVectorNorQueryTextSet() {
        val format = AilakeInputFormat(
            warehouse = "file:///tmp/x", namespace = "default", tableName = "table",
            vecCol = "embedding", dim = 4, topK = 5, efSearch = 50,
        )
        format.runtimeContext = runtimeContextWithParams(Configuration())
        assertThrows(IllegalStateException::class.java) { format.open(GenericInputSplit(0, 1)) }
    }

    @Test
    fun openWithOnlyQueryTextDoesNotThrow() {
        val params = Configuration()
        params.setString("ailake.query.text", "rust programming")
        val format = AilakeInputFormat(
            warehouse = "file:///tmp/x", namespace = "default", tableName = "table",
            vecCol = "embedding", dim = 4, topK = 5, efSearch = 50,
        )
        format.runtimeContext = runtimeContextWithParams(params)
        assertDoesNotThrow { format.open(GenericInputSplit(0, 1)) }
    }

    @Test
    fun openWithQueryVectorAndQueryTextDoesNotThrow() {
        val params = Configuration()
        params.setString("ailake.query.vector", "1.0,0.0,0.0,0.0")
        params.setString("ailake.query.text", "rust programming")
        params.setString("ailake.hybrid.weight", "0.3")
        val format = AilakeInputFormat(
            warehouse = "file:///tmp/x", namespace = "default", tableName = "table",
            vecCol = "embedding", dim = 4, topK = 5, efSearch = 50,
        )
        format.runtimeContext = runtimeContextWithParams(params)
        assertDoesNotThrow { format.open(GenericInputSplit(0, 1)) }
    }

    // ── ailake.multimodal.queries job parameter ────────────────────────────────
    //
    // Regression: AilakeNativeLoader.searchMultimodal was fully implemented but
    // unreachable from any Flink source — same "dead capability" gap as
    // searchText was, closed the same way.

    @Test
    fun openWithOnlyMultimodalQueriesDoesNotThrow() {
        val params = Configuration()
        params.setString(
            "ailake.multimodal.queries",
            """[{"col":"embedding","query":"0.1,0.2","weight":1.0},""" +
            """{"col":"image_embedding","query":"0.3,0.4","weight":0.5}]""",
        )
        val format = AilakeInputFormat(
            warehouse = "file:///tmp/x", namespace = "default", tableName = "table",
            vecCol = "embedding", dim = 4, topK = 5, efSearch = 50,
        )
        format.runtimeContext = runtimeContextWithParams(params)
        assertDoesNotThrow { format.open(GenericInputSplit(0, 1)) }
        assertTrue(format.reachedEnd(), "expected empty result set when the native lib can't be loaded")
    }

    @Test
    fun parseMultimodalQueriesParsesColQueryAndWeight() {
        val parsed = AilakeInputFormat.parseMultimodalQueries(
            """[{"col":"embedding","query":"0.1,0.2,0.3","weight":0.7}]"""
        )
        assertEquals(1, parsed.size)
        val (col, query, weight) = parsed[0]
        assertEquals("embedding", col)
        assertArrayEquals(floatArrayOf(0.1f, 0.2f, 0.3f), query, 1e-6f)
        assertEquals(0.7f, weight, 1e-6f)
    }

    @Test
    fun parseMultimodalQueriesDefaultsWeightToOne() {
        val parsed = AilakeInputFormat.parseMultimodalQueries(
            """[{"col":"embedding","query":"0.1,0.2"}]"""
        )
        assertEquals(1.0f, parsed[0].third, 1e-6f)
    }
}
