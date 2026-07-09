// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.databind.ObjectMapper
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
    private val mapper = ObjectMapper()

    override fun getRecordSet(
        transaction: ConnectorTransactionHandle,
        session: ConnectorSession,
        split: ConnectorSplit,
        table: ConnectorTableHandle,
        columns: List<ColumnHandle>,
    ): RecordSet {
        val cols = columns.map { it as VectorScanColumnHandle }
        if (split is MultimodalScanSplit) {
            val queries = parseMultimodalQueries(split.queriesJson)
            val rows = AilakeNative.searchMultimodal(
                split.tableUri, queries, split.topK,
                namespace = split.namespace, tableName = split.tableName,
            )
            return MultimodalScanRecordSet(rows, cols)
        }
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
        return VectorScanRecordSet(rows, cols)
    }

    /** Parses `SET SESSION ailake.multimodal_queries` — see VectorScanConnector's KDoc for the JSON shape. */
    private fun parseMultimodalQueries(json: String): List<Triple<String, List<Float>, Float>> {
        if (json.isBlank()) return emptyList()
        val node = mapper.readTree(json)
        return (0 until node.size()).map { i ->
            val n = node.get(i)
            val col = n.get("col").asText()
            val query = n.get("query").asText().split(',').mapNotNull { it.trim().toFloatOrNull() }
            val weight = if (n.has("weight")) n.get("weight").floatValue() else 1.0f
            Triple(col, query, weight)
        }
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

internal class MultimodalScanRecordSet(
    private val rows: List<AilakeNative.MultimodalSearchRow>,
    private val columns: List<VectorScanColumnHandle>,
) : RecordSet {
    override fun getColumnTypes(): List<Type> = columns.map { col ->
        when (col.name) {
            "row_id" -> BIGINT
            "rrf_score" -> DOUBLE
            else -> VARCHAR
        }
    }
    override fun cursor(): RecordCursor = MultimodalScanRecordCursor(rows, columns)
}

internal class MultimodalScanRecordCursor(
    private val rows: List<AilakeNative.MultimodalSearchRow>,
    private val columns: List<VectorScanColumnHandle>,
) : RecordCursor {
    private var position = -1

    override fun getCompletedBytes(): Long = rows.size.toLong() * 64L
    override fun getReadTimeNanos(): Long = 0L
    override fun advanceNextPosition(): Boolean = ++position < rows.size
    override fun getType(field: Int): Type = when (columns[field].name) {
        "row_id" -> BIGINT
        "rrf_score" -> DOUBLE
        else -> VARCHAR
    }

    override fun getBoolean(field: Int): Boolean =
        throw UnsupportedOperationException("no boolean columns")

    override fun getLong(field: Int): Long = when (columns[field].name) {
        "row_id" -> rows[position].rowId
        else -> throw IllegalArgumentException("getLong not applicable for ${columns[field].name}")
    }

    override fun getDouble(field: Int): Double = when (columns[field].name) {
        "rrf_score" -> rows[position].rrfScore.toDouble()
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
