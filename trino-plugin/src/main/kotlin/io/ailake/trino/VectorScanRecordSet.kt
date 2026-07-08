// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
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
        // Three modes, selected by which session properties are set (see
        // VectorScanConnector.getSessionProperties): pure vector search
        // (query_vector only), hybrid BM25+vector RRF fusion (both set —
        // AilakeNative.search's hybridText path), or pure text search
        // (query_text only, no query_vector — AilakeNative.searchText,
        // O(log N) via Tantivy when the table has an FTS index, see
        // ailake.fts-columns).
        val rows = when {
            s.queryBytes.isBlank() && s.queryText.isNotBlank() ->
                AilakeNative.searchText(s.tableUri, s.namespace, s.tableName, s.queryText, topK = s.topK)
            s.queryText.isNotBlank() ->
                AilakeNative.search(
                    s.tableUri, s.queryBytes, s.topK,
                    hybridText = s.queryText, bm25Weight = s.hybridWeight,
                    namespace = s.namespace, tableName = s.tableName, vectorColumn = s.vectorColumn,
                )
            else ->
                AilakeNative.search(
                    s.tableUri, s.queryBytes, s.topK,
                    namespace = s.namespace, tableName = s.tableName, vectorColumn = s.vectorColumn,
                )
        }
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
