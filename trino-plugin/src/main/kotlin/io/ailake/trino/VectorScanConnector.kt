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
    private val vectorColumns: List<AilakeNative.VectorColSpec> = emptyList(),
    // REST Catalog (Fase 17/19) config from ailake.catalog/ailake.rest-* properties
    // (VectorScanConnectorFactory). Only threaded to `procedures` (CALL ailake.system.compact())
    // today — search/scan/multimodal/insert go through JSON-serialized Trino handle classes
    // (VectorScanHandles.kt) that this project has a documented history of subtle Jackson
    // serialization bugs in (see that file's NB comment); extending them needs live-server
    // verification this sandbox can't do, so it's deferred — see docs/guides/REST_CATALOG.md.
    private val catalogOpts: Map<String, String> = emptyMap(),
) : Connector {

    private val metadata = VectorScanMetadata(
        tableUri, vectorColumn, dim, metric, precision, namespace, tableName, embeddingModel,
        partitionFields, formatVersion, textColumns,
        hnswM, hnswEfConstruction, preNormalize, deferred, ftsColumns, ftsTokenizer,
        vectorColumns,
    )
    private val splitManager = VectorScanSplitManager()
    private val recordSetProvider = VectorScanRecordSetProvider()
    private val pageSinkProvider = AilakePageSinkProvider()
    private val procedures = AilakeProcedures(
        tableUri, namespace, tableName, catalogOpts,
        vectorColumn, dim, metric, precision, formatVersion,
    )

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
     *   -- HNSW recall/latency tuning (0 / -1 = server defaults: ef_search=50, no pruning)
     *   SET SESSION ailake.ef_search = 100;
     *   SET SESSION ailake.pruning_threshold = 0.8;
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
        // ailake_search_json (ailake-jni) has always accepted both — never
        // forwarded from here (or Spark's AilakeNative.search — same gap,
        // same fix) until now. 0 / -1.0 = "not set, use the server's own
        // default (ef_search=50, pruning_threshold=infinity/no pruning)".
        PropertyMetadata.integerProperty(
            "ef_search",
            "HNSW ef_search — higher recall at the cost of latency (0 = server default, 50)",
            0,
            false,
        ),
        PropertyMetadata.doubleProperty(
            "pruning_threshold",
            "Geometric pruning cutoff — files whose centroid is farther than this from the " +
            "query are skipped entirely (-1 = server default, no pruning)",
            -1.0,
            false,
        ),
        // Session properties consumed by AilakeProcedures' CALL ailake.system.*
        // maintenance procedures below (decay_memories/migrate/delete_rows/
        // add_vector_column/backfill_vector_column/estimate) — same
        // JSON-string-session-property carrier already used by
        // `multimodal_queries` above for operations with more fields than fit
        // comfortably as individual typed session properties.
        PropertyMetadata.doubleProperty(
            "decay_lambda",
            "Exponential decay rate for CALL ailake.system.decay_memories() (Phase 9 agent memory)",
            0.1,
            false,
        ),
        PropertyMetadata.stringProperty(
            "migrate_config",
            "JSON config for CALL ailake.system.migrate(): " +
            "{\"old_column\",\"new_column\",\"text_column\",\"embed_cmd\",\"strategy\"?,\"batch_size\"?,\"model_name\"?,\"model_version\"?}",
            "",
            false,
        ),
        PropertyMetadata.stringProperty(
            "delete_rows_config",
            "JSON config for CALL ailake.system.delete_rows(): {\"file\",\"row_ids\":[...]}",
            "",
            false,
        ),
        PropertyMetadata.stringProperty(
            "add_vector_column_config",
            "JSON config for CALL ailake.system.add_vector_column(): " +
            "{\"column\",\"dim\",\"metric\"?,\"precision\"?,\"pre_normalize\"?,\"hnsw_m\"?,\"hnsw_ef_construction\"?}",
            "",
            false,
        ),
        PropertyMetadata.stringProperty(
            "backfill_vector_column_config",
            "JSON config for CALL ailake.system.backfill_vector_column(): " +
            "{\"column\",\"text_column\",\"embed_cmd\",\"batch_size\"?}",
            "",
            false,
        ),
        PropertyMetadata.stringProperty(
            "estimate_config",
            "JSON config for CALL ailake.system.estimate(): {\"rows\",\"dim\",\"hnsw_m\"?,\"pq_m\"?} — " +
            "result is logged server-side (INFO), CALL procedures have no result set",
            "",
            false,
        ),
    )
}
