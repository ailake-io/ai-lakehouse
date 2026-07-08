// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.Connector
import io.trino.spi.connector.ConnectorMetadata
import io.trino.spi.connector.ConnectorPageSinkProvider
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
    private val metric: String,
    private val precision: String,
    private val namespace: String,
    private val tableName: String,
    private val embeddingModel: String? = null,
    private val partitionFields: List<AilakeNative.PartitionFieldDef> = emptyList(),
    private val formatVersion: Int = 2,
    private val textColumns: List<String> = emptyList(),
) : Connector {

    private val metadata = VectorScanMetadata(tableUri, vectorColumn, dim, metric, precision, namespace, tableName, embeddingModel, partitionFields, formatVersion, textColumns)
    private val splitManager = VectorScanSplitManager()
    private val recordSetProvider = VectorScanRecordSetProvider()
    private val pageSinkProvider = AilakePageSinkProvider()

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

    override fun getPageSinkProvider(): ConnectorPageSinkProvider = pageSinkProvider

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
