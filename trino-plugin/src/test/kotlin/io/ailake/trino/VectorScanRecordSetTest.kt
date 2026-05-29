// SPDX-License-Identifier: MIT OR Apache-2.0
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
}
