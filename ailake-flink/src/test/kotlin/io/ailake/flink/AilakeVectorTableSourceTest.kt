// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.apache.flink.api.common.ExecutionConfig
import org.apache.flink.api.common.functions.RuntimeContext
import org.apache.flink.configuration.Configuration
import org.apache.flink.core.io.GenericInputSplit
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
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
}
