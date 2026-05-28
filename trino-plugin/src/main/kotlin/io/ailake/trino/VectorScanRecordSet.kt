package io.ailake.trino

import io.airlift.slice.Slices
import io.trino.spi.connector.ColumnHandle
import io.trino.spi.connector.ConnectorRecordSetProvider
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorSplit
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTransactionHandle
import io.trino.spi.connector.RecordCursor
import io.trino.spi.connector.RecordSet
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.DoubleType.DOUBLE
import io.trino.spi.type.Type
import io.trino.spi.type.VarcharType.VARCHAR

class VectorScanRecordSetProvider : ConnectorRecordSetProvider {
    override fun getRecordSet(
        transaction: ConnectorTransactionHandle,
        session: ConnectorSession,
        split: ConnectorSplit,
        table: ConnectorTableHandle,
        columns: List<ColumnHandle>,
    ): RecordSet {
        val s = split as VectorScanSplit
        val rows = AilakeNative.search(s.tableUri, s.queryBytes, s.topK)
        val cols = columns.map { it as VectorScanColumnHandle }
        return VectorScanRecordSet(rows, cols)
    }
}

internal class VectorScanRecordSet(
    private val rows: List<AilakeNative.SearchRow>,
    private val columns: List<VectorScanColumnHandle>,
) : RecordSet {
    override fun getColumnTypes(): List<Type> = columns.map { col ->
        when (col.name) {
            "row_id" -> BIGINT
            "distance" -> DOUBLE
            else -> VARCHAR
        }
    }
    override fun cursor(): RecordCursor = VectorScanRecordCursor(rows, columns)
}

internal class VectorScanRecordCursor(
    private val rows: List<AilakeNative.SearchRow>,
    private val columns: List<VectorScanColumnHandle>,
) : RecordCursor {
    private var position = -1

    override fun getCompletedBytes(): Long = rows.size.toLong() * 64L
    override fun getReadTimeNanos(): Long = 0L
    override fun advanceNextPosition(): Boolean = ++position < rows.size
    override fun getType(field: Int): Type = when (columns[field].name) {
        "row_id" -> BIGINT
        "distance" -> DOUBLE
        else -> VARCHAR
    }

    override fun getBoolean(field: Int): Boolean =
        throw UnsupportedOperationException("no boolean columns")

    override fun getLong(field: Int): Long = when (columns[field].name) {
        "row_id" -> rows[position].rowId
        else -> throw IllegalArgumentException("getLong not applicable for ${columns[field].name}")
    }

    override fun getDouble(field: Int): Double = when (columns[field].name) {
        "distance" -> rows[position].distance.toDouble()
        else -> throw IllegalArgumentException("getDouble not applicable for ${columns[field].name}")
    }

    override fun getSlice(field: Int): io.airlift.slice.Slice = when (columns[field].name) {
        "file_path" -> Slices.utf8Slice(rows[position].filePath)
        else -> throw IllegalArgumentException("getSlice not applicable for ${columns[field].name}")
    }

    override fun getObject(field: Int): Any? = null
    override fun isNull(field: Int): Boolean = false
    override fun close() {}
}
