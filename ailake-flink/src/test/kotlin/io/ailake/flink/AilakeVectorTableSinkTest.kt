// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.apache.flink.streaming.api.functions.sink.SinkFunction
import org.apache.flink.table.api.DataTypes
import org.apache.flink.table.catalog.Column
import org.apache.flink.table.catalog.ResolvedSchema
import org.apache.flink.table.data.GenericArrayData
import org.apache.flink.table.data.GenericRowData
import org.apache.flink.table.data.StringData
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.lang.reflect.Field

/**
 * Regression: AilakeSinkFunction only captured columns listed in
 * `fts.columns`, silently dropping any other declared STRING column even
 * though `columns=` on the native side is a general persisted-metadata
 * mechanism, not FTS-only. `computeExtraColumnIndices` now includes every
 * declared STRING column except id/vector, and `AilakeSinkFunction` (tested
 * directly here) accumulates real per-row values for all of them.
 */
class AilakeVectorTableSinkTest {

    private val noopContext = object : SinkFunction.Context {
        override fun currentProcessingTime() = 0L
        override fun currentWatermark() = 0L
        override fun timestamp(): Long? = null
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T> privateField(target: Any, name: String): T {
        val f: Field = target.javaClass.getDeclaredField(name)
        f.isAccessible = true
        return f.get(target) as T
    }

    // ── computeExtraColumnIndices ─────────────────────────────────────────────

    @Test
    fun computeExtraColumnIndicesIncludesEveryStringColumnNotJustFtsColumns() {
        val schema = ResolvedSchema.of(
            Column.physical("id", DataTypes.BIGINT()),
            Column.physical("embedding", DataTypes.ARRAY(DataTypes.FLOAT())),
            Column.physical("text", DataTypes.STRING()),
            Column.physical("source", DataTypes.STRING()),
        )
        // Neither "text" nor "source" is in fts.columns — both must still be captured.
        val result = AilakeVectorTableSink.computeExtraColumnIndices(schema, idIdx = 0, vecIdx = 1)
        assertEquals(mapOf("text" to 2, "source" to 3), result)
    }

    @Test
    fun computeExtraColumnIndicesSkipsNonStringColumns() {
        // e.g. a shared source+sink table declaring "_distance FLOAT" (search-only, ignored on write).
        val schema = ResolvedSchema.of(
            Column.physical("id", DataTypes.BIGINT()),
            Column.physical("embedding", DataTypes.ARRAY(DataTypes.FLOAT())),
            Column.physical("text", DataTypes.STRING()),
            Column.physical("_distance", DataTypes.FLOAT()),
        )
        val result = AilakeVectorTableSink.computeExtraColumnIndices(schema, idIdx = 0, vecIdx = 1)
        assertEquals(mapOf("text" to 2), result)
    }

    @Test
    fun computeExtraColumnIndicesEmptyWhenOnlyIdAndEmbeddingDeclared() {
        val schema = ResolvedSchema.of(
            Column.physical("id", DataTypes.BIGINT()),
            Column.physical("embedding", DataTypes.ARRAY(DataTypes.FLOAT())),
        )
        val result = AilakeVectorTableSink.computeExtraColumnIndices(schema, idIdx = 0, vecIdx = 1)
        assertTrue(result.isEmpty())
    }

    // ── AilakeSinkFunction accumulation ────────────────────────────────────────

    @Test
    fun invokeAccumulatesRealPerRowValuesForEveryExtraColumn() {
        val sink = AilakeSinkFunction(
            warehouse = "file:///tmp/ailake-flink-test-does-not-need-to-exist",
            namespace = "default",
            tableName = "docs",
            vecCol    = "embedding",
            dim       = 2,
            metric    = "cosine",
            precision = "f16",
            idIdx     = 0,
            vecIdx    = 1,
            extraColumnIndices = mapOf("text" to 2, "source" to 3),
        )

        val row1 = GenericRowData(4)
        row1.setField(0, 1L)
        row1.setField(1, GenericArrayData(floatArrayOf(0.1f, 0.2f)))
        row1.setField(2, StringData.fromString("hello world"))
        row1.setField(3, StringData.fromString("doc-a"))

        val row2 = GenericRowData(4)
        row2.setField(0, 2L)
        row2.setField(1, GenericArrayData(floatArrayOf(0.3f, 0.4f)))
        row2.setField(2, StringData.fromString("second row"))
        row2.setField(3, StringData.fromString("doc-b"))

        sink.invoke(row1, noopContext)
        sink.invoke(row2, noopContext)

        val textBuffers: Map<String, List<String>> = privateField(sink, "textBuffers")
        assertEquals(listOf("hello world", "second row"), textBuffers.getValue("text"))
        assertEquals(listOf("doc-a", "doc-b"), textBuffers.getValue("source"))
    }

    @Test
    fun invokeWithNoExtraColumnsConfiguredOnlyReadsIdAndEmbedding() {
        val sink = AilakeSinkFunction(
            warehouse = "file:///tmp/ailake-flink-test-does-not-need-to-exist",
            namespace = "default",
            tableName = "docs",
            vecCol    = "embedding",
            dim       = 2,
            metric    = "cosine",
            precision = "f16",
            idIdx     = 0,
            vecIdx    = 1,
        )
        val row = GenericRowData(2)
        row.setField(0, 1L)
        row.setField(1, GenericArrayData(floatArrayOf(0.1f, 0.2f)))

        assertDoesNotThrow { sink.invoke(row, noopContext) }
        val textBuffers: Map<String, List<String>> = privateField(sink, "textBuffers")
        assertTrue(textBuffers.isEmpty())
    }

    @Test
    fun invokeTreatsNullExtraColumnAsEmptyString() {
        val sink = AilakeSinkFunction(
            warehouse = "file:///tmp/ailake-flink-test-does-not-need-to-exist",
            namespace = "default",
            tableName = "docs",
            vecCol    = "embedding",
            dim       = 2,
            metric    = "cosine",
            precision = "f16",
            idIdx     = 0,
            vecIdx    = 1,
            extraColumnIndices = mapOf("text" to 2),
        )
        val row = GenericRowData(3)
        row.setField(0, 1L)
        row.setField(1, GenericArrayData(floatArrayOf(0.1f, 0.2f)))
        row.setField(2, null)

        sink.invoke(row, noopContext)
        val textBuffers: Map<String, List<String>> = privateField(sink, "textBuffers")
        assertEquals(listOf(""), textBuffers.getValue("text"))
    }
}
