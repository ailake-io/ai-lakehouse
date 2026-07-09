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
import io.trino.spi.procedure.Procedure
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
    private val hnswM: Int? = null,
    private val hnswEfConstruction: Int? = null,
    private val preNormalize: Boolean = false,
    private val deferred: Boolean = false,
    private val ftsColumns: List<String> = emptyList(),
    private val ftsTokenizer: String = "default",
) : Connector {

    private val metadata = VectorScanMetadata(
        tableUri, vectorColumn, dim, metric, precision, namespace, tableName, embeddingModel,
        partitionFields, formatVersion, textColumns,
        hnswM, hnswEfConstruction, preNormalize, deferred, ftsColumns, ftsTokenizer,
    )
    private val splitManager = VectorScanSplitManager()
    private val recordSetProvider = VectorScanRecordSetProvider()
    private val pageSinkProvider = AilakePageSinkProvider()
    private val procedures = AilakeProcedures(tableUri, namespace, tableName)

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
     * `CALL ailake.system.compact()` — compacts small files in the configured
     * ingest table. See [AilakeProcedures].
     */
    override fun getProcedures(): Set<Procedure> = procedures.getProcedures()

    /**
     * Session properties consumed by this connector:
     *
     *   -- pure vector search
     *   SET SESSION ailake.query_vector = '0.1,-0.2,0.3,...';
     *   SET SESSION ailake.top_k = 10;
     *   SELECT * FROM ailake.default.search ORDER BY distance;
     *
     *   -- hybrid BM25+vector RRF fusion (both query_vector and query_text set)
     *   SET SESSION ailake.query_text = 'rust programming';
     *   SET SESSION ailake.hybrid_weight = 0.5;  -- 0.0 = pure vector, 1.0 = pure BM25
     *
     *   -- pure full-text search (query_text set, query_vector left unset) —
     *   -- O(log N) via Tantivy when the table has an FTS index (see
     *   -- ailake.fts-columns), falls back to O(N) BM25 brute-force otherwise
     *   SET SESSION ailake.query_text = 'rust programming';
     *
     *   -- cross-modal RRF search (e.g. text + image embeddings on the same row)
     *   SET SESSION ailake.multimodal_queries =
     *     '[{"col":"embedding","query":"0.1,-0.2","weight":1.0},
     *       {"col":"image_embedding","query":"0.4,0.5","weight":0.5}]';
     *   SET SESSION ailake.top_k = 10;
     *   SELECT * FROM ailake.default.search_multimodal ORDER BY rrf_score DESC;
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
        PropertyMetadata.stringProperty(
            "query_text",
            "Query text for hybrid BM25+vector search (with query_vector set) or pure full-text search (without)",
            "",
            false,
        ),
        PropertyMetadata.doubleProperty(
            "hybrid_weight",
            "BM25 weight in RRF fusion when both query_vector and query_text are set (0.0 = pure vector, 1.0 = pure BM25)",
            0.5,
            false,
        ),
        PropertyMetadata.stringProperty(
            "multimodal_queries",
            "JSON array of {col, query (csv f32), weight} for cross-modal RRF search of " +
            "ailake.default.search_multimodal, e.g. '[{\"col\":\"embedding\",\"query\":\"0.1,-0.2\",\"weight\":1.0}]'",
            "",
            false,
        ),
    )
}
