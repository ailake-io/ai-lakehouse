// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.apache.flink.api.common.ExecutionConfig
import org.apache.flink.api.common.functions.RuntimeContext
import org.apache.flink.configuration.Configuration
import org.apache.flink.core.io.GenericInputSplit
import org.apache.flink.table.api.DataTypes
import org.apache.flink.table.catalog.Column
import org.apache.flink.table.catalog.ResolvedSchema
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever

/**
 * Regression: AilakeNativeLoader.scan (backed by ailake_scan_json) had no wrapper or table
 * source in any of the three JVM plugins — SQL search always returned only
 * row_id/distance/file_path, forcing a manual JOIN against a separately-registered Iceberg
 * table to get real columns. Mirrors AilakeVectorTableSourceTest's graceful-degradation
 * coverage for the new `search.mode = 'full'` source.
 */
class AilakeScanTableSourceTest {

    private val schema = ResolvedSchema.of(
        Column.physical("id", DataTypes.BIGINT()),
        Column.physical("text", DataTypes.STRING()),
        Column.physical("_distance", DataTypes.FLOAT()),
    )

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
        val format = AilakeScanInputFormat(
            warehouse = "file:///tmp/ailake-flink-test-does-not-need-to-exist",
            namespace = "default", tableName = "table",
            vecCol = "embedding", dim = 4, topK = 5, schema = schema,
        )
        format.runtimeContext = runtimeContextWithQueryVector("1.0,0.0,0.0,0.0")

        assertDoesNotThrow { format.open(GenericInputSplit(0, 1)) }
    }

    @Test
    fun openWithoutNativeLibLeavesResultSetEmpty() {
        val format = AilakeScanInputFormat(
            warehouse = "file:///tmp/ailake-flink-test-does-not-need-to-exist",
            namespace = "default", tableName = "table",
            vecCol = "embedding", dim = 4, topK = 5, schema = schema,
        )
        format.runtimeContext = runtimeContextWithQueryVector("1.0,0.0,0.0,0.0")
        format.open(GenericInputSplit(0, 1))

        assertTrue(
            format.reachedEnd(),
            "expected empty result set (reachedEnd()==true) when the native lib can't be loaded"
        )
    }

    @Test
    fun openThrowsWhenQueryVectorNotSet() {
        val format = AilakeScanInputFormat(
            warehouse = "file:///tmp/x", namespace = "default", tableName = "table",
            vecCol = "embedding", dim = 4, topK = 5, schema = schema,
        )
        val ctx = mock<RuntimeContext>()
        val executionConfig = ExecutionConfig()
        executionConfig.globalJobParameters = Configuration()
        whenever(ctx.executionConfig).thenReturn(executionConfig)
        format.runtimeContext = ctx

        assertThrows(IllegalStateException::class.java) { format.open(GenericInputSplit(0, 1)) }
    }
}
