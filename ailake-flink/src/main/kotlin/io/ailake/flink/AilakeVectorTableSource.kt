// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import com.fasterxml.jackson.databind.ObjectMapper
import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.api.common.io.GenericInputFormat
import org.apache.flink.api.common.io.statistics.BaseStatistics
import org.apache.flink.core.io.GenericInputSplit
import org.apache.flink.table.catalog.ResolvedSchema
import org.apache.flink.table.connector.ChangelogMode
import org.apache.flink.table.connector.source.DynamicTableSource
import org.apache.flink.table.connector.source.InputFormatProvider
import org.apache.flink.table.connector.source.ScanTableSource
import org.apache.flink.table.connector.source.abilities.SupportsFilterPushDown
import org.apache.flink.table.data.GenericRowData
import org.apache.flink.table.data.RowData
import org.apache.flink.table.data.StringData
import org.apache.flink.table.expressions.ResolvedExpression
import org.slf4j.LoggerFactory

/**
 * AI-Lake Flink table source.  Executes ANN search and streams results as [RowData].
 *
 * The query vector is passed at runtime via a dynamic source parameter
 * (`ailake.query.vector` job parameter — raw f32 little-endian bytes as base64).
 *
 * Result schema must have columns matching those returned by the native search:
 *   row_id (BIGINT), distance (FLOAT), file_path (STRING)
 * plus any additional columns fetched from Parquet via predicate pushdown (future work).
 *
 * Cross-modal RRF search (`AilakeNativeLoader.searchMultimodal` — e.g. text + image
 * embeddings on the same row) is selected instead by setting `ailake.multimodal.queries`
 * to a JSON array of `{"col", "query" (csv f32), "weight"}` objects, e.g.:
 *   '[{"col":"embedding","query":"0.1,-0.2","weight":1.0},
 *     {"col":"image_embedding","query":"0.4,0.5","weight":0.5}]'
 * The result still lands in the fixed 3-column shape above — the "distance" slot carries
 * the fused RRF score in this mode, same physical (BIGINT, FLOAT, STRING) row.
 */
class AilakeVectorTableSource(
    private val warehouse: String,
    private val namespace: String,
    private val tableName: String,
    private val vecCol: String,
    private val dim: Int,
    private val topK: Int,
    private val efSearch: Int,
    private val schema: ResolvedSchema,
    private val partitionFilter: String? = null,
) : ScanTableSource {

    override fun getChangelogMode(): ChangelogMode = ChangelogMode.insertOnly()

    override fun getScanRuntimeProvider(context: ScanTableSource.ScanContext): ScanTableSource.ScanRuntimeProvider {
        val format = AilakeInputFormat(
            warehouse       = warehouse,
            namespace       = namespace,
            tableName       = tableName,
            vecCol          = vecCol,
            dim             = dim,
            topK            = topK,
            efSearch        = efSearch,
            partitionFilter = partitionFilter,
        )
        return InputFormatProvider.of(format)
    }

    override fun copy(): DynamicTableSource = AilakeVectorTableSource(
        warehouse, namespace, tableName, vecCol, dim, topK, efSearch, schema, partitionFilter
    )

    override fun asSummaryString(): String = "AI-Lake[$namespace.$tableName]"
}

/**
 * Flink InputFormat that calls the native ailake_search_json via [AilakeNativeLoader].
 *
 * The query vector is read from the Flink runtime configuration key `ailake.query.vector`
 * (comma-separated float values) set at job submission time.
 */
class AilakeInputFormat(
    private val warehouse: String,
    private val namespace: String,
    private val tableName: String,
    private val vecCol: String,
    private val dim: Int,
    private val topK: Int,
    private val efSearch: Int,
    private val partitionFilter: String? = null,
) : GenericInputFormat<RowData>() {

    @Transient private var results: Iterator<AilakeNativeLoader.SearchResultItem>? = null

    override fun open(split: GenericInputSplit) {
        val params = runtimeContext.executionConfig.globalJobParameters.toMap()
        val queryVectorParam = params["ailake.query.vector"]
        // Hybrid BM25+vector RRF fusion (both set) or pure full-text search
        // (query.text set, query.vector unset) — AilakeNativeLoader.search's
        // hybridText path and .searchText were already fully implemented but
        // unreachable from any Flink source before this.
        val queryTextParam = params["ailake.query.text"]
        val hybridWeight = params["ailake.hybrid.weight"]?.toFloatOrNull() ?: 0.5f
        // Cross-modal RRF fusion (AilakeNativeLoader.searchMultimodal) was already fully
        // implemented but unreachable from any Flink source before this — same "dead
        // capability" gap as searchText was, closed the same way.
        val multimodalQueriesParam = params["ailake.multimodal.queries"]
        if (queryVectorParam == null && queryTextParam == null && multimodalQueriesParam == null) {
            throw IllegalStateException(
                "Job parameter 'ailake.query.vector', 'ailake.query.text' and/or " +
                "'ailake.multimodal.queries' must be set — 'ailake.query.vector': comma-separated " +
                "f32 values for vector/hybrid search; 'ailake.query.text' alone: pure full-text " +
                "search; 'ailake.multimodal.queries': JSON array of {col, query, weight} for " +
                "cross-modal RRF search"
            )
        }
        // partition_filter from job params overrides constructor value (constructor wins if both set)
        val effectivePartition = partitionFilter ?: params["ailake.partition.filter"]

        // AilakeNativeLoader.lib throws (via getOrThrow()) when libailake_jni.so isn't on
        // the library path, and every AilakeNativeLoader method reads it eagerly — unlike
        // Spark/Trino/DuckDB, which all resolve the native handle to a nullable/Optional
        // and degrade to empty results. Catch here so a missing native lib fails this
        // source with empty results instead of crashing the whole Flink task.
        results = try {
            when {
                multimodalQueriesParam != null -> AilakeNativeLoader.searchMultimodal(
                    warehouse = warehouse, namespace = namespace, table = tableName,
                    queries = parseMultimodalQueries(multimodalQueriesParam),
                    topK = topK, partitionFilter = effectivePartition,
                ).map {
                    AilakeNativeLoader.SearchResultItem(it.row_id, it.rrf_score, it.file_path)
                }.iterator()
                queryVectorParam == null -> AilakeNativeLoader.searchText(
                    warehouse = warehouse, namespace = namespace, table = tableName,
                    queryText = queryTextParam!!, topK = topK, partitionFilter = effectivePartition,
                ).iterator()
                queryTextParam != null -> AilakeNativeLoader.search(
                    warehouse = warehouse, namespace = namespace, table = tableName,
                    vecCol = vecCol, dim = dim,
                    query = queryVectorParam.split(",").map { it.trim().toFloat() }.toFloatArray(),
                    topK = topK, efSearch = efSearch, partitionFilter = effectivePartition,
                    hybridText = queryTextParam, bm25Weight = hybridWeight,
                ).iterator()
                else -> AilakeNativeLoader.search(
                    warehouse = warehouse, namespace = namespace, table = tableName,
                    vecCol = vecCol, dim = dim,
                    query = queryVectorParam.split(",").map { it.trim().toFloat() }.toFloatArray(),
                    topK = topK, efSearch = efSearch, partitionFilter = effectivePartition,
                ).iterator()
            }
        } catch (e: Throwable) {
            // Broad by design: `Native.load()` failure surfaces as `UnsatisfiedLinkError`
            // (a JVM Error, not Exception) and the exact wrapping can vary by JVM/JNA
            // version — the contract this restores is "never crash the task", so every
            // failure mode from the native call must degrade the same way.
            log.warn("[ailake] native library unavailable — table={}.{} returns no rows: {}",
                namespace, tableName, e.message)
            emptyList<AilakeNativeLoader.SearchResultItem>().iterator()
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(AilakeInputFormat::class.java)
        private val mapper = ObjectMapper()

        /** Parses the `ailake.multimodal.queries` job parameter — see this class's KDoc for the JSON shape. */
        fun parseMultimodalQueries(json: String): List<Triple<String, FloatArray, Float>> {
            val node = mapper.readTree(json)
            return (0 until node.size()).map { i ->
                val n = node.get(i)
                val col = n.get("col").asText()
                val query = n.get("query").asText().split(',').mapNotNull { it.trim().toFloatOrNull() }.toFloatArray()
                val weight = if (n.has("weight")) n.get("weight").floatValue() else 1.0f
                Triple(col, query, weight)
            }
        }
    }

    override fun reachedEnd(): Boolean = results?.hasNext()?.not() ?: true

    override fun nextRecord(reuse: RowData?): RowData {
        val r = results!!.next()
        val row = GenericRowData(3)
        row.setField(0, r.row_id)
        row.setField(1, r.distance)
        row.setField(2, StringData.fromString(r.file_path))
        return row
    }

    override fun getStatistics(cachedStatistics: BaseStatistics?): BaseStatistics? = null
    override fun createInputSplits(minNumSplits: Int) = arrayOf(GenericInputSplit(0, 1))
    override fun getInputSplitAssigner(inputSplits: Array<out GenericInputSplit>) =
        org.apache.flink.api.common.io.DefaultInputSplitAssigner(inputSplits)
}
