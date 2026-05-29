// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.trino

import io.trino.spi.connector.Connector
import io.trino.spi.connector.ConnectorMetadata
import io.trino.spi.connector.ConnectorRecordSetProvider
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorSplitManager
import io.trino.spi.connector.ConnectorTransactionHandle
import io.trino.spi.session.PropertyMetadata
import io.trino.spi.transaction.IsolationLevel

class VectorScanConnector(
    private val tableUri: String,
    private val vectorColumn: String,
    private val dim: Int,
) : Connector {

    private val metadata = VectorScanMetadata(tableUri, vectorColumn, dim)
    private val splitManager = VectorScanSplitManager()
    private val recordSetProvider = VectorScanRecordSetProvider()

    override fun beginTransaction(
        isolationLevel: IsolationLevel,
        readOnly: Boolean,
        autoCommit: Boolean,
    ): ConnectorTransactionHandle = VectorScanTransactionHandle

    override fun getMetadata(
        session: ConnectorSession,
        transactionHandle: ConnectorTransactionHandle,
    ): ConnectorMetadata = metadata

    override fun getSplitManager(): ConnectorSplitManager = splitManager

    override fun getRecordSetProvider(): ConnectorRecordSetProvider = recordSetProvider

    /**
     * Session properties consumed by this connector:
     *
     *   SET SESSION ailake.query_vector = '0.1,-0.2,0.3,...';
     *   SET SESSION ailake.top_k = 10;
     *   SELECT * FROM ailake.default.search ORDER BY distance;
     */
    override fun getSessionProperties(): List<PropertyMetadata<*>> = listOf(
        PropertyMetadata.stringProperty(
            "query_vector",
            "Comma-separated f32 query vector, e.g. '0.1,-0.2,0.3'",
            "",
            false,
        ),
        PropertyMetadata.integerProperty(
            "top_k",
            "Number of nearest-neighbor results to return",
            10,
            false,
        ),
    )
}
