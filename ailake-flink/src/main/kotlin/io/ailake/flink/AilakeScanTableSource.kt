// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
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
import org.apache.flink.table.data.GenericArrayData
import org.apache.flink.table.data.GenericRowData
import org.apache.flink.table.data.RowData
import org.apache.flink.table.data.StringData
import org.apache.flink.table.types.logical.LogicalTypeRoot
import org.slf4j.LoggerFactory

/**
 * AI-Lake Flink table source for `search.mode = 'full'` (Fase 11 — search + full-row fetch,
 * no JOIN needed). Backed by `ailake_scan_json` (one native call, every stored column comes
 * back) instead of `ailake_search_json` (fixed `row_id`/`distance`/`file_path` triple, see
 * [AilakeVectorTableSource]). Columns are read straight from the DDL — whatever the user
 * declares in `CREATE TABLE` is looked up by name in the scan response, in declared order —
 * see [AilakeVectorConnectorFactory.validateScanResultSchema] for the one constraint
 * (`_distance` must be the last declared column).
 *
 * Same runtime contract as [AilakeVectorTableSource]: query vector via the
 * `ailake.query.vector` job parameter (comma-separated f32 values).
 */
class AilakeScanTableSource(
    private val warehouse: String,
    private val namespace: String,
    private val tableName: String,
    private val vecCol: String,
    private val dim: Int,
    private val topK: Int,
    private val schema: ResolvedSchema,
    private val partitionFilter: String? = null,
) : ScanTableSource {

    override fun getChangelogMode(): ChangelogMode = ChangelogMode.insertOnly()

    override fun getScanRuntimeProvider(context: ScanTableSource.ScanContext): ScanTableSource.ScanRuntimeProvider {
        val format = AilakeScanInputFormat(
            warehouse       = warehouse,
            namespace       = namespace,
            tableName       = tableName,
            vecCol          = vecCol,
            dim             = dim,
            topK            = topK,
            schema          = schema,
            partitionFilter = partitionFilter,
        )
        return InputFormatProvider.of(format)
    }

    override fun copy(): DynamicTableSource = AilakeScanTableSource(
        warehouse, namespace, tableName, vecCol, dim, topK, schema, partitionFilter
    )

    override fun asSummaryString(): String = "AI-Lake-Scan[$namespace.$tableName]"
}

/**
 * Flink InputFormat that calls the native `ailake_scan_json` via [AilakeNativeLoader.scan]
 * and projects the declared DDL columns out of the columnar response, by name, per row.
 */
class AilakeScanInputFormat(
    private val warehouse: String,
    private val namespace: String,
    private val tableName: String,
    private val vecCol: String,
    private val dim: Int,
    private val topK: Int,
    private val schema: ResolvedSchema,
    private val partitionFilter: String? = null,
) : GenericInputFormat<RowData>() {

    @Transient private var response: AilakeNativeLoader.ScanResponse = AilakeNativeLoader.ScanResponse(ok = true)
    @Transient private var position: Int = -1

    override fun open(split: GenericInputSplit) {
        val params = runtimeContext.executionConfig.globalJobParameters.toMap()
        val queryVectorParam = params["ailake.query.vector"]
            ?: throw IllegalStateException(
                "Job parameter 'ailake.query.vector' must be set — comma-separated f32 values"
            )
        val effectivePartition = partitionFilter ?: params["ailake.partition.filter"]
        val query = queryVectorParam.split(",").map { it.trim().toFloat() }.toFloatArray()

        // Same graceful-degradation contract as AilakeInputFormat — missing native lib fails
        // this source with an empty result set instead of crashing the whole Flink task.
        response = try {
            AilakeNativeLoader.scan(
                warehouse = warehouse, namespace = namespace, table = tableName,
                vecCol = vecCol, dim = dim, query = query, topK = topK, partitionFilter = effectivePartition,
            )
        } catch (e: Throwable) {
            log.warn("[ailake] native library unavailable — table={}.{} scan returns no rows: {}",
                namespace, tableName, e.message)
            AilakeNativeLoader.ScanResponse(ok = true)
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(AilakeScanInputFormat::class.java)
    }

    override fun reachedEnd(): Boolean = position + 1 >= response.num_rows

    override fun nextRecord(reuse: RowData?): RowData {
        position++
        val columns = schema.columns
        val row = GenericRowData(columns.size)
        columns.forEachIndexed { i, col ->
            val raw = response.columns[col.name]?.getOrNull(position)
            row.setField(i, convert(raw, col.dataType.logicalType.typeRoot))
        }
        return row
    }

    /** Converts a scan-response value (Jackson's generic JSON representation) to Flink's internal row representation. */
    private fun convert(raw: Any?, root: LogicalTypeRoot): Any? {
        if (raw == null) return null
        return when (root) {
            LogicalTypeRoot.BIGINT -> (raw as Number).toLong()
            LogicalTypeRoot.INTEGER -> (raw as Number).toInt()
            LogicalTypeRoot.FLOAT -> (raw as Number).toFloat()
            LogicalTypeRoot.DOUBLE -> (raw as Number).toDouble()
            LogicalTypeRoot.BOOLEAN -> raw as Boolean
            LogicalTypeRoot.VARCHAR, LogicalTypeRoot.CHAR -> StringData.fromString(raw.toString())
            LogicalTypeRoot.ARRAY -> {
                @Suppress("UNCHECKED_CAST")
                val values = (raw as List<Any?>).map { (it as? Number)?.toFloat() ?: 0f }.toFloatArray()
                GenericArrayData(values)
            }
            else -> StringData.fromString(raw.toString())
        }
    }

    override fun getStatistics(cachedStatistics: BaseStatistics?): BaseStatistics? = null
    override fun createInputSplits(minNumSplits: Int) = arrayOf(GenericInputSplit(0, 1))
    override fun getInputSplitAssigner(inputSplits: Array<out GenericInputSplit>) =
        org.apache.flink.api.common.io.DefaultInputSplitAssigner(inputSplits)
}
