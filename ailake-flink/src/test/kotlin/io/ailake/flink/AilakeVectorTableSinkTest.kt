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
import org.apache.flink.table.expressions.CallExpression
import org.apache.flink.table.expressions.FieldReferenceExpression
import org.apache.flink.table.expressions.ResolvedExpression
import org.apache.flink.table.expressions.ValueLiteralExpression
import org.apache.flink.table.functions.BuiltInFunctionDefinitions
import org.apache.flink.table.types.logical.LogicalTypeRoot
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
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

    // ── null id/vector guard ────────────────────────────────────────────────────
    //
    // Regression: extra text columns were null-checked (empty-string fallback) but
    // id/vector were not — row.getLong/getArray on a null field threw an opaque NPE
    // instead of a clear validation error.

    @Test
    fun invokeThrowsClearErrorOnNullId() {
        val sink = AilakeSinkFunction(
            warehouse = "file:///tmp/x", namespace = "default", tableName = "docs",
            vecCol = "embedding", dim = 2, metric = "cosine", precision = "f16", idIdx = 0, vecIdx = 1,
        )
        val row = GenericRowData(2)
        row.setField(0, null)
        row.setField(1, GenericArrayData(floatArrayOf(0.1f, 0.2f)))
        val ex = assertThrows(IllegalStateException::class.java) { sink.invoke(row, noopContext) }
        assertTrue(ex.message!!.contains("id"))
    }

    @Test
    fun invokeThrowsClearErrorOnNullEmbedding() {
        val sink = AilakeSinkFunction(
            warehouse = "file:///tmp/x", namespace = "default", tableName = "docs",
            vecCol = "embedding", dim = 2, metric = "cosine", precision = "f16", idIdx = 0, vecIdx = 1,
        )
        val row = GenericRowData(2)
        row.setField(0, 1L)
        row.setField(1, null)
        val ex = assertThrows(IllegalStateException::class.java) { sink.invoke(row, noopContext) }
        assertTrue(ex.message!!.contains("embedding"))
    }

    // ── id/vector column type validation ────────────────────────────────────────
    //
    // Regression: a type mismatch (e.g. id STRING instead of BIGINT) only surfaced
    // as an opaque ClassCastException deep in RowData extraction on the first row.

    @Test
    fun validateColumnTypeAcceptsMatchingType() {
        assertDoesNotThrow {
            AilakeVectorTableSink.validateColumnType("id", LogicalTypeRoot.BIGINT, setOf(LogicalTypeRoot.BIGINT), "BIGINT")
        }
    }

    @Test
    fun validateColumnTypeRejectsMismatchedType() {
        val ex = assertThrows(IllegalArgumentException::class.java) {
            AilakeVectorTableSink.validateColumnType("id", LogicalTypeRoot.VARCHAR, setOf(LogicalTypeRoot.BIGINT), "BIGINT")
        }
        assertTrue(ex.message!!.contains("id"))
        assertTrue(ex.message!!.contains("BIGINT"))
    }

    // ── DELETE pushdown (applyDeleteFilters / executeDeletion) ─────────────────
    //
    // Regression: AilakeNativeLoader.deleteWhere was fully implemented but had no
    // SQL surface — DELETE FROM did nothing. Now wired via SupportsDeletePushDown,
    // equality/IN pushdown only.

    private fun sink() = AilakeVectorTableSink(
        warehouse = "file:///tmp/x", namespace = "default", tableName = "docs",
        vecCol = "embedding", dim = 4, metric = "cosine", precision = "f16",
        schema = ResolvedSchema.of(
            Column.physical("id", DataTypes.BIGINT()),
            Column.physical("embedding", DataTypes.ARRAY(DataTypes.FLOAT())),
        ),
    )

    private fun equalsExpr(field: String, value: Any): CallExpression {
        val fieldRef = FieldReferenceExpression(field, DataTypes.BIGINT(), 0, 0)
        val literal = ValueLiteralExpression(value)
        return CallExpression.anonymous(BuiltInFunctionDefinitions.EQUALS, listOf(fieldRef, literal), DataTypes.BOOLEAN())
    }

    private fun inExpr(field: String, values: List<Any>): CallExpression {
        val fieldRef = FieldReferenceExpression(field, DataTypes.BIGINT(), 0, 0)
        val children: List<ResolvedExpression> = listOf(fieldRef) + values.map { ValueLiteralExpression(it) }
        return CallExpression.anonymous(BuiltInFunctionDefinitions.IN, children, DataTypes.BOOLEAN())
    }

    @Test
    fun applyDeleteFiltersAcceptsSingleEquality() {
        val s = sink()
        assertTrue(s.applyDeleteFilters(listOf(equalsExpr("id", 5L))))
        assertEquals("id", privateField<String?>(s, "deleteColumn"))
        assertEquals(listOf("5"), privateField<List<String>?>(s, "deleteValues"))
    }

    @Test
    fun applyDeleteFiltersAcceptsIn() {
        val s = sink()
        assertTrue(s.applyDeleteFilters(listOf(inExpr("id", listOf(1L, 2L, 3L)))))
        assertEquals("id", privateField<String?>(s, "deleteColumn"))
        assertEquals(listOf("1", "2", "3"), privateField<List<String>?>(s, "deleteValues"))
    }

    @Test
    fun applyDeleteFiltersRejectsMultipleFilters() {
        val s = sink()
        assertFalse(s.applyDeleteFilters(listOf(equalsExpr("id", 5L), equalsExpr("id", 6L))))
    }

    @Test
    fun applyDeleteFiltersRejectsNonEqualsNonIn() {
        val s = sink()
        val fieldRef = FieldReferenceExpression("id", DataTypes.BIGINT(), 0, 0)
        val gt = CallExpression.anonymous(
            BuiltInFunctionDefinitions.GREATER_THAN,
            listOf(fieldRef, ValueLiteralExpression(5L)),
            DataTypes.BOOLEAN(),
        )
        assertFalse(s.applyDeleteFilters(listOf(gt)))
    }

    @Test
    fun executeDeletionReturnsEmptyWithoutCapturedPredicate() {
        val s = sink()
        assertTrue(s.executeDeletion().isEmpty)
    }

    @Test
    fun executeDeletionFailsClearlyWhenNativeLibraryAbsent() {
        assumeTrue(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        val s = sink()
        s.applyDeleteFilters(listOf(equalsExpr("id", 5L)))
        // AilakeNativeLoader.lib throws (via getOrThrow()) when the native lib isn't on
        // the library path — surfaces as UnsatisfiedLinkError (a JVM Error), not a
        // RuntimeException; see AilakeInputFormat.open()'s comment on this same quirk.
        assertThrows(Throwable::class.java) { s.executeDeletion() }
    }
}
