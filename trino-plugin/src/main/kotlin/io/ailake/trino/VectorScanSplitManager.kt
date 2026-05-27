package io.ailake.trino

import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorSplitManager
import io.trino.spi.connector.ConnectorSplitSource
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTransactionHandle
import io.trino.spi.connector.Constraint
import io.trino.spi.connector.DynamicFilter
import io.trino.spi.connector.FixedSplitSource

class VectorScanSplitManager : ConnectorSplitManager {
    override fun getSplits(
        transaction: ConnectorTransactionHandle,
        session: ConnectorSession,
        table: ConnectorTableHandle,
        dynamicFilter: DynamicFilter,
        constraint: Constraint,
    ): ConnectorSplitSource {
        val handle = table as VectorScanTableHandle
        val queryVector = session.getProperty("query_vector", String::class.java) ?: ""
        val topK = session.getProperty("top_k", Int::class.java) ?: 10
        return FixedSplitSource(VectorScanSplit(handle.tableUri, queryVector, topK))
    }
}
