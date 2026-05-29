// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.flink

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

/**
 * AI-Lake Flink table source.  Executes ANN search and streams results as [RowData].
 *
 * The query vector is passed at runtime via a dynamic source parameter
 * (`ailake.query.vector` job parameter — raw f32 little-endian bytes as base64).
 *
 * Result schema must have columns matching those returned by the native search:
 *   row_id (BIGINT), distance (FLOAT), file_path (STRING)
 * plus any additional columns fetched from Parquet via predicate pushdown (future work).
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
) : ScanTableSource {

    override fun getChangelogMode(): ChangelogMode = ChangelogMode.insertOnly()

    override fun getScanRuntimeProvider(context: ScanTableSource.ScanContext): ScanTableSource.ScanRuntimeProvider {
        val format = AilakeInputFormat(
            warehouse = warehouse,
            namespace = namespace,
            tableName = tableName,
            vecCol    = vecCol,
            dim       = dim,
            topK      = topK,
            efSearch  = efSearch,
        )
        return InputFormatProvider.of(format)
    }

    override fun copy(): DynamicTableSource = AilakeVectorTableSource(
        warehouse, namespace, tableName, vecCol, dim, topK, efSearch, schema
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
) : GenericInputFormat<RowData>() {

    @Transient private var results: Iterator<AilakeNativeLoader.SearchResultItem>? = null

    override fun open(split: GenericInputSplit) {
        val queryParam = runtimeContext.executionConfig
            .globalJobParameters
            .toMap()["ailake.query.vector"]
            ?: throw IllegalStateException(
                "Job parameter 'ailake.query.vector' not set — " +
                "provide comma-separated f32 values for the query vector"
            )
        val query = queryParam.split(",").map { it.trim().toFloat() }.toFloatArray()
        results = AilakeNativeLoader.search(
            warehouse = warehouse,
            namespace = namespace,
            table     = tableName,
            vecCol    = vecCol,
            dim       = dim,
            query     = query,
            topK      = topK,
            efSearch  = efSearch,
        ).iterator()
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
