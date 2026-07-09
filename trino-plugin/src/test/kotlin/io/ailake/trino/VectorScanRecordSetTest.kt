// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.DoubleType.DOUBLE
import io.trino.spi.type.VarcharType.VARCHAR
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class VectorScanRecordSetTest {

    private val rows = listOf(
        AilakeNative.SearchRow(rowId = 1L, distance = 0.12f, filePath = "part-001.parquet"),
        AilakeNative.SearchRow(rowId = 2L, distance = 0.34f, filePath = "part-002.parquet"),
    )

    private val allColumns = listOf(
        VectorScanColumnHandle("row_id", 0),
        VectorScanColumnHandle("distance", 1),
        VectorScanColumnHandle("file_path", 2),
    )

    @Test
    fun columnTypesMatchSchema() {
        val rs = VectorScanRecordSet(rows, allColumns)
        val types = rs.getColumnTypes()
        assertEquals(BIGINT, types[0])
        assertEquals(DOUBLE, types[1])
        assertEquals(VARCHAR, types[2])
    }

    @Test
    fun cursorIteratesAllRows() {
        val cursor = VectorScanRecordSet(rows, allColumns).cursor()
        var count = 0
        while (cursor.advanceNextPosition()) count++
        assertEquals(2, count)
    }

    @Test
    fun cursorReturnsCorrectRowId() {
        val cursor = VectorScanRecordSet(rows, allColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals(1L, cursor.getLong(0))
        cursor.advanceNextPosition()
        assertEquals(2L, cursor.getLong(0))
    }

    @Test
    fun cursorReturnsCorrectDistance() {
        val cursor = VectorScanRecordSet(rows, allColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals(0.12, cursor.getDouble(1), 0.001)
    }

    @Test
    fun cursorReturnsCorrectFilePath() {
        val cursor = VectorScanRecordSet(rows, allColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals("part-001.parquet", cursor.getSlice(2).toStringUtf8())
    }

    @Test
    fun cursorReturnsFalseWhenExhausted() {
        val cursor = VectorScanRecordSet(rows, allColumns).cursor()
        repeat(rows.size) { cursor.advanceNextPosition() }
        assertFalse(cursor.advanceNextPosition())
    }

    @Test
    fun isNullAlwaysFalse() {
        val cursor = VectorScanRecordSet(rows, allColumns).cursor()
        cursor.advanceNextPosition()
        assertFalse(cursor.isNull(0))
        assertFalse(cursor.isNull(1))
        assertFalse(cursor.isNull(2))
    }

    @Test
    fun emptyRowsProducesEmptyCursor() {
        val cursor = VectorScanRecordSet(emptyList(), allColumns).cursor()
        assertEquals(0L, cursor.getCompletedBytes())
        assertFalse(cursor.advanceNextPosition())
    }

    @Test
    fun projectedColumnsOnlyDistance() {
        val distanceOnly = listOf(VectorScanColumnHandle("distance", 1))
        val cursor = VectorScanRecordSet(rows, distanceOnly).cursor()
        cursor.advanceNextPosition()
        assertEquals(0.12, cursor.getDouble(0), 0.001)
    }

    // ── MultimodalScanRecordSet ────────────────────────────────────────────────

    private val multimodalRows = listOf(
        AilakeNative.MultimodalSearchRow(rowId = 1L, rrfScore = 0.9f, filePath = "part-001.parquet"),
        AilakeNative.MultimodalSearchRow(rowId = 2L, rrfScore = 0.5f, filePath = "part-002.parquet"),
    )

    private val multimodalColumns = listOf(
        VectorScanColumnHandle("row_id", 0),
        VectorScanColumnHandle("rrf_score", 1),
        VectorScanColumnHandle("file_path", 2),
    )

    @Test
    fun multimodalColumnTypesMatchSchema() {
        val rs = MultimodalScanRecordSet(multimodalRows, multimodalColumns)
        val types = rs.getColumnTypes()
        assertEquals(BIGINT, types[0])
        assertEquals(DOUBLE, types[1])
        assertEquals(VARCHAR, types[2])
    }

    @Test
    fun multimodalCursorReturnsCorrectRrfScore() {
        val cursor = MultimodalScanRecordSet(multimodalRows, multimodalColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals(0.9, cursor.getDouble(1), 0.001)
    }

    @Test
    fun multimodalCursorReturnsCorrectFilePath() {
        val cursor = MultimodalScanRecordSet(multimodalRows, multimodalColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals("part-001.parquet", cursor.getSlice(2).toStringUtf8())
    }

    // ── ScanRecordSet (Fase 11 — search_full, dynamic per-catalog columns) ────

    private val scanColumns = listOf(
        VectorScanColumnHandle("id", 0),
        VectorScanColumnHandle("embedding", 1),
        VectorScanColumnHandle("text", 2),
        VectorScanColumnHandle("_distance", 3),
    )

    private val scanResult = AilakeNative.ScanResult(
        schema = listOf(
            AilakeNative.ScanColumn("id", "int64"),
            AilakeNative.ScanColumn("embedding", "list_float32"),
            AilakeNative.ScanColumn("text", "utf8"),
            AilakeNative.ScanColumn("_distance", "float32"),
        ),
        numRows = 2,
        columns = mapOf(
            "id" to listOf(1L, 2L),
            "embedding" to listOf(listOf(0.1, 0.2), listOf(0.3, 0.4)),
            "text" to listOf("hello", null),
            "_distance" to listOf(0.12, 0.34),
        ),
    )

    @Test
    fun scanColumnTypesMatchSchema() {
        val rs = ScanRecordSet(scanResult, scanColumns)
        val types = rs.getColumnTypes()
        assertEquals(BIGINT, types[0])
        assertEquals(VARCHAR, types[1]) // embedding — JSON-encoded, see VectorScanMetadata.scanColumns() KDoc
        assertEquals(VARCHAR, types[2])
        assertEquals(DOUBLE, types[3])
    }

    @Test
    fun scanCursorReturnsCorrectId() {
        val cursor = ScanRecordSet(scanResult, scanColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals(1L, cursor.getLong(0))
        cursor.advanceNextPosition()
        assertEquals(2L, cursor.getLong(0))
    }

    @Test
    fun scanCursorReturnsVectorColumnAsJsonArrayString() {
        val cursor = ScanRecordSet(scanResult, scanColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals("[0.1,0.2]", cursor.getSlice(1).toStringUtf8())
    }

    @Test
    fun scanCursorReturnsCorrectTextColumn() {
        val cursor = ScanRecordSet(scanResult, scanColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals("hello", cursor.getSlice(2).toStringUtf8())
    }

    @Test
    fun scanCursorReturnsCorrectDistance() {
        val cursor = ScanRecordSet(scanResult, scanColumns).cursor()
        cursor.advanceNextPosition()
        assertEquals(0.12, cursor.getDouble(3), 0.001)
    }

    @Test
    fun scanCursorIsNullTrueForMissingValue() {
        val cursor = ScanRecordSet(scanResult, scanColumns).cursor()
        cursor.advanceNextPosition()
        cursor.advanceNextPosition() // second row — text is null
        assertTrue(cursor.isNull(2))
        assertEquals("", cursor.getSlice(2).toStringUtf8())
    }

    @Test
    fun scanCursorIteratesAllRows() {
        val cursor = ScanRecordSet(scanResult, scanColumns).cursor()
        var count = 0
        while (cursor.advanceNextPosition()) count++
        assertEquals(2, count)
    }
}
