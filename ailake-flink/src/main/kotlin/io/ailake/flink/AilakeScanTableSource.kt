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
import java.io.Serializable

/** (name, type root) pair for one declared DDL column — both fields are actually
 *  Serializable (String, enum), unlike [ResolvedSchema] itself (see below). */
data class ScanColumnSpec(val name: String, val typeRoot: LogicalTypeRoot) : Serializable

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
 *
 * Takes [columns] (a plain `List<ScanColumnSpec>`), not `ResolvedSchema` directly —
 * `ResolvedSchema` is not `Serializable`, and Flink serializes the `InputFormat` this class
 * builds to ship it to TaskManagers; holding onto the full schema object failed every
 * `search.mode=full` query with `NotSerializableException:
 * org.apache.flink.table.catalog.ResolvedSchema` on a real (non-local-only) cluster —
 * confirmed live, not caught by any test in this repo. Callers extract [columns] from
 * `ResolvedSchema` once, in [AilakeVectorConnectorFactory] (planning-time, not distributed).
 */
class AilakeScanTableSource(
    private val warehouse: String,
    private val namespace: String,
    private val tableName: String,
    private val vecCol: String,
    private val dim: Int,
    private val topK: Int,
    private val columns: List<ScanColumnSpec>,
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
            columns         = columns,
            partitionFilter = partitionFilter,
        )
        return InputFormatProvider.of(format)
    }

    override fun copy(): DynamicTableSource = AilakeScanTableSource(
        warehouse, namespace, tableName, vecCol, dim, topK, columns, partitionFilter
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
    private val columns: List<ScanColumnSpec>,
    private val partitionFilter: String? = null,
) : GenericInputFormat<RowData>() {

    // Iterator over pre-built rows, not a manual position index into `response` — see [open].
    @Transient private var rows: Iterator<RowData> = emptyList<RowData>().iterator()

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
        val response = try {
            AilakeNativeLoader.scan(
                warehouse = warehouse, namespace = namespace, table = tableName,
                vecCol = vecCol, dim = dim, query = query, topK = topK, partitionFilter = effectivePartition,
            )
        } catch (e: Throwable) {
            log.warn("[ailake] native library unavailable — table={}.{} scan returns no rows: {}",
                namespace, tableName, e.message)
            AilakeNativeLoader.ScanResponse(ok = true)
        }
        // Materialize every row up front (response is already fully in memory — same
        // native call already returned it as one JSON blob, no extra I/O here) and hand
        // out a plain Iterator, exactly like AilakeInputFormat's *working* pattern.
        //
        // Regression: a manual `position: Int` field + `reachedEnd() = position + 1 >=
        // response.num_rows` looks equivalent on paper but silently dropped row 0 on a
        // real (non-local) Flink cluster — `search.top-k=1` returned 0 rows, `top-k=5`
        // returned 4 (always missing the first/lowest-distance row), confirmed live and
        // cross-checked against the same request via ailake_scan_json directly (correct)
        // and via the sibling `search`-mode InputFormat (correct, iterator-based) — not
        // root-caused further than "index-based reachedEnd/nextRecord loses the first
        // element in Flink's InputFormatSourceFunction driver loop, iterator-based
        // doesn't." Matching the already-correct sibling implementation exactly, rather
        // than continuing to debug an index-based scheme against undocumented Flink
        // runtime iteration order, is the safer fix.
        rows = (0 until response.num_rows).map { position ->
            val row = GenericRowData(columns.size)
            columns.forEachIndexed { i, col ->
                val raw = response.columns[col.name]?.getOrNull(position)
                row.setField(i, convert(raw, col.typeRoot))
            }
            row as RowData
        }.iterator()
    }

    companion object {
        private val log = LoggerFactory.getLogger(AilakeScanInputFormat::class.java)
    }

    override fun reachedEnd(): Boolean = !rows.hasNext()

    override fun nextRecord(reuse: RowData?): RowData = rows.next()

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
