// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorSplitManager
import io.trino.spi.connector.ConnectorSplitSource
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTransactionHandle
import io.trino.spi.connector.Constraint
import io.trino.spi.connector.DynamicFilter
import io.trino.spi.connector.FixedSplitSource
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

class VectorScanSplitManager : ConnectorSplitManager {
    override fun getSplits(
        transaction: ConnectorTransactionHandle,
        session: ConnectorSession,
        table: ConnectorTableHandle,
        dynamicFilter: DynamicFilter,
        constraint: Constraint,
    ): ConnectorSplitSource {
        val topK = session.getProperty("top_k", Int::class.java) ?: 10
        if (table is MultimodalScanTableHandle) {
            val queriesJson = session.getProperty("multimodal_queries", String::class.java) ?: ""
            return FixedSplitSource(
                MultimodalScanSplit(
                    tableUri    = table.tableUri,
                    namespace   = table.namespace,
                    tableName   = table.tableName,
                    queriesJson = queriesJson,
                    topK        = topK,
                )
            )
        }
        val handle = table as VectorScanTableHandle
        val queryVectorCsv = session.getProperty("query_vector", String::class.java) ?: ""
        val queryText = session.getProperty("query_text", String::class.java) ?: ""
        val hybridWeight = session.getProperty("hybrid_weight", Double::class.java)?.toFloat() ?: 0.5f
        // Parse CSV→bytes once at planning; split carries compact Base64 binary.
        val queryBytes = csvFloatsToBase64(queryVectorCsv)
        return FixedSplitSource(
            VectorScanSplit(
                tableUri     = handle.tableUri,
                queryBytes   = queryBytes,
                topK         = topK,
                namespace    = handle.namespace,
                tableName    = handle.tableName,
                vectorColumn = handle.vectorColumn,
                queryText    = queryText,
                hybridWeight = hybridWeight,
            )
        )
    }

    companion object {
        /** Converts a comma-separated f32 string to Base64-encoded little-endian bytes. */
        fun csvFloatsToBase64(csv: String): String {
            if (csv.isBlank()) return ""
            val floats = csv.split(',').mapNotNull { it.trim().toFloatOrNull() }.toFloatArray()
            if (floats.isEmpty()) return ""
            val buf = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            floats.forEach { buf.putFloat(it) }
            return Base64.getEncoder().encodeToString(buf.array())
        }
    }
}
