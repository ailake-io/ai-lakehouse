// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import io.ailake.flink.internal.AilakeNativeLoader.PartitionFieldDef
import org.apache.flink.streaming.api.datastream.DataStream
import org.apache.flink.streaming.api.datastream.DataStreamSink
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction
import org.apache.flink.streaming.api.functions.sink.SinkFunction
import org.apache.flink.table.catalog.ResolvedSchema
import org.apache.flink.table.connector.ChangelogMode
import org.apache.flink.table.connector.ProviderContext
import org.apache.flink.table.connector.sink.DataStreamSinkProvider
import org.apache.flink.table.connector.sink.DynamicTableSink
import org.apache.flink.table.connector.sink.abilities.SupportsDeletePushDown
import org.apache.flink.table.data.RowData
import org.apache.flink.table.expressions.CallExpression
import org.apache.flink.table.expressions.FieldReferenceExpression
import org.apache.flink.table.expressions.ResolvedExpression
import org.apache.flink.table.expressions.ValueLiteralExpression
import org.apache.flink.table.functions.BuiltInFunctionDefinitions
import org.apache.flink.table.types.logical.LogicalTypeRoot
import org.slf4j.LoggerFactory
import java.util.Optional

/**
 * AI-Lake Flink table sink.  Buffers rows in memory and flushes each batch to the
 * native library via [AilakeNativeLoader.writeBatch].
 *
 * Expected row schema:
 *   - One BIGINT column named "id" (configurable via id_col index, default 0)
 *   - One ARRAY<FLOAT> or BYTES column for the embedding vector
 *   - Any number of additional STRING columns — persisted as AI-Lake extra
 *     metadata (`columns=` in the native writeBatch call) regardless of
 *     whether they're also listed in `fts.columns`; `fts.columns` only
 *     controls which of them additionally get a Tantivy FTS index.
 *
 * Flush is triggered when [BUFFER_SIZE] rows have accumulated or when the job finishes.
 */
class AilakeVectorTableSink(
    private val warehouse: String,
    private val namespace: String,
    private val tableName: String,
    private val vecCol: String,
    private val dim: Int,
    private val metric: String,
    private val precision: String,
    private val schema: ResolvedSchema,
    private val embeddingModel: String? = null,
    private val partitionFields: List<PartitionFieldDef> = emptyList(),
    private val formatVersion: Int = 2,
    private val ftsColumns: List<String> = emptyList(),
    private val ftsTokenizer: String = "default",
    private val hnswM: Int? = null,
    private val hnswEfConstruction: Int? = null,
    private val preNormalize: Boolean = false,
    private val deferred: Boolean = false,
    // Multi-column (Phase 8 multimodal) ingest — e.g. text + image embeddings on the same
    // row, each with its own HNSW index. When non-empty, one ARRAY<FLOAT> column per entry
    // (by name) is expected instead of the single vecCol, and writes go through
    // ailake_write_batch_multi_json. Configured via the `vector.columns` DDL option.
    private val vectorColumns: List<AilakeNativeLoader.VectorColSpec> = emptyList(),
) : DynamicTableSink, SupportsDeletePushDown {

    private val log = LoggerFactory.getLogger(AilakeVectorTableSink::class.java)

    // Captured by applyDeleteFilters, consumed by executeDeletion — see both below.
    private var deleteColumn: String? = null
    private var deleteValues: List<String>? = null

    companion object {
        const val BUFFER_SIZE = 10_000

        /**
         * Every declared STRING column except id/vector is persisted as
         * AI-Lake extra metadata — not just the fts.columns subset (columns=
         * is a general persisted-metadata mechanism on the native side, not
         * FTS-only). Non-string extras (e.g. a "_distance FLOAT" column
         * shared with the source side of the same table) are skipped — the
         * native side's columns= is Map<String, List<String>>, there's
         * nowhere to put a non-string value.
         *
         * `internal` (not `private`) so it's independently unit-testable
         * without spinning up the full DataStreamSinkProvider/DataStream
         * machinery.
         */
        /**
         * Regression: nothing validated that the id/vector columns had the right
         * Flink logical types before `AilakeSinkFunction.invoke()` unconditionally
         * called `row.getLong(idIdx)` / `row.getArray(vecIdx)` — a mismatch (e.g. the
         * `id STRING` shown in an earlier, incorrect doc example) surfaced only as an
         * opaque `ClassCastException` deep in `RowData` extraction on the first row,
         * not a clear DDL-time error naming the actual problem.
         */
        internal fun validateColumnType(colName: String, actual: LogicalTypeRoot, expected: Set<LogicalTypeRoot>, expectedDesc: String) {
            require(actual in expected) {
                "Column '$colName' must be $expectedDesc, got $actual"
            }
        }

        internal fun computeExtraColumnIndices(schema: ResolvedSchema, idIdx: Int, vecIdx: Int): Map<String, Int> =
            computeExtraColumnIndices(schema, idIdx, setOf(vecIdx))

        /** Multi-column (Phase 8 multimodal) overload — excludes every configured vector column index, not just one. */
        internal fun computeExtraColumnIndices(schema: ResolvedSchema, idIdx: Int, vecIndices: Set<Int>): Map<String, Int> {
            val colNames = schema.columnNames
            val dataTypes = schema.columnDataTypes
            return colNames
                .withIndex()
                .filter { (i, _) -> i != idIdx && i !in vecIndices }
                .filter { (i, _) ->
                    val root = dataTypes[i].logicalType.typeRoot
                    root == LogicalTypeRoot.VARCHAR || root == LogicalTypeRoot.CHAR
                }
                .associate { (i, name) -> name to i }
        }
    }

    override fun getChangelogMode(requestedMode: ChangelogMode): ChangelogMode =
        ChangelogMode.insertOnly()

    override fun getSinkRuntimeProvider(context: DynamicTableSink.Context): DynamicTableSink.SinkRuntimeProvider {
        val colNames = schema.columnNames
        val idIdx = colNames.indexOfFirst { it == "id" }.takeIf { it >= 0 } ?: 0
        validateColumnType(colNames[idIdx], schema.columnDataTypes[idIdx].logicalType.typeRoot, setOf(LogicalTypeRoot.BIGINT), "BIGINT")

        // Multi-column (Phase 8 multimodal) mode: one ARRAY<FLOAT> column per `vectorColumns`
        // entry, resolved by name against the declared schema instead of the single `vecCol`.
        val vecIdx: Int
        val vecIndices: List<Int>
        if (vectorColumns.isNotEmpty()) {
            vecIndices = vectorColumns.map { spec ->
                val idx = colNames.indexOfFirst { it == spec.column }
                require(idx >= 0) {
                    "Vector column '${spec.column}' (from vector.columns) not found in declared schema: ${colNames.toList()}"
                }
                validateColumnType(spec.column, schema.columnDataTypes[idx].logicalType.typeRoot, setOf(LogicalTypeRoot.ARRAY), "ARRAY<FLOAT>")
                idx
            }
            vecIdx = vecIndices[0]
        } else {
            vecIdx = colNames.indexOfFirst { it == vecCol }.takeIf { it >= 0 } ?: 1
            validateColumnType(colNames[vecIdx], schema.columnDataTypes[vecIdx].logicalType.typeRoot, setOf(LogicalTypeRoot.ARRAY), "ARRAY<FLOAT>")
            vecIndices = listOf(vecIdx)
        }
        val extraColumnIndices = computeExtraColumnIndices(schema, idIdx, vecIndices.toSet())
        return object : DataStreamSinkProvider {
            override fun consumeDataStream(
                context: ProviderContext,
                dataStream: DataStream<RowData>,
            ): DataStreamSink<*> {
                return dataStream.addSink(
                    AilakeSinkFunction(
                        warehouse          = warehouse,
                        namespace          = namespace,
                        tableName          = tableName,
                        vecCol             = vecCol,
                        dim                = dim,
                        metric             = metric,
                        precision          = precision,
                        idIdx              = idIdx,
                        vecIdx             = vecIdx,
                        embeddingModel     = embeddingModel,
                        partitionFields    = partitionFields,
                        formatVersion      = formatVersion,
                        ftsColumns         = ftsColumns,
                        ftsTokenizer       = ftsTokenizer,
                        extraColumnIndices = extraColumnIndices,
                        hnswM              = hnswM,
                        hnswEfConstruction = hnswEfConstruction,
                        preNormalize       = preNormalize,
                        deferred           = deferred,
                        vectorColumns      = vectorColumns,
                        vecIndices         = vecIndices,
                    )
                )
            }
        }
    }

    override fun copy(): DynamicTableSink = AilakeVectorTableSink(
        warehouse, namespace, tableName, vecCol, dim, metric, precision, schema,
        embeddingModel, partitionFields, formatVersion, ftsColumns, ftsTokenizer,
        hnswM, hnswEfConstruction, preNormalize, deferred, vectorColumns,
    )

    override fun asSummaryString(): String = "AI-Lake-Sink[$namespace.$tableName]"

    // ── DELETE FROM ... WHERE ... (equality/IN pushdown only) ─────────────────
    //
    // AilakeNativeLoader.deleteWhere was already fully implemented and tested
    // but had no SQL surface reachable from Flink — same "dead capability" gap
    // as compact/schema evolution. The native operation is an equality delete
    // file (column = one of N values), so only a WHERE clause that reduces to
    // a single-column equality/IN predicate can be supported. Anything else
    // is rejected (applyDeleteFilters returns false), which Flink surfaces as
    // "DELETE statement is not supported" rather than a silent partial delete.

    override fun applyDeleteFilters(filters: List<ResolvedExpression>): Boolean {
        if (filters.size != 1) return false
        val call = filters[0] as? CallExpression ?: return false
        val fn = call.functionDefinition
        if (fn != BuiltInFunctionDefinitions.EQUALS && fn != BuiltInFunctionDefinitions.IN) return false
        val children = call.resolvedChildren
        if (children.isEmpty()) return false
        val field = children[0] as? FieldReferenceExpression ?: return false
        val literals = children.drop(1)
        if (literals.isEmpty() || literals.any { it !is ValueLiteralExpression || it.isNull }) return false
        val values = literals.map { (it as ValueLiteralExpression).getValueAs(Any::class.java).get().toString() }
        deleteColumn = field.name
        deleteValues = values
        return true
    }

    override fun executeDeletion(): Optional<Long> {
        val col = deleteColumn ?: return Optional.empty()
        AilakeNativeLoader.deleteWhere(warehouse, namespace, tableName, col, deleteValues.orEmpty())
        log.info("[ailake] DELETE WHERE {} IN (...) executed for {}.{}", col, namespace, tableName)
        return Optional.empty() // native side doesn't report an exact row count
    }
}

class AilakeSinkFunction(
    private val warehouse: String,
    private val namespace: String,
    private val tableName: String,
    private val vecCol: String,
    private val dim: Int,
    private val metric: String,
    private val precision: String,
    private val idIdx: Int,
    private val vecIdx: Int,
    private val embeddingModel: String? = null,
    private val partitionFields: List<PartitionFieldDef> = emptyList(),
    private val formatVersion: Int = 2,
    private val ftsColumns: List<String> = emptyList(),
    private val ftsTokenizer: String = "default",
    private val extraColumnIndices: Map<String, Int> = emptyMap(),
    private val hnswM: Int? = null,
    private val hnswEfConstruction: Int? = null,
    private val preNormalize: Boolean = false,
    private val deferred: Boolean = false,
    // Multi-column (Phase 8 multimodal) mode — see AilakeVectorTableSink's `vectorColumns`
    // doc. Empty (default) = single-column path via vecIdx/writeBatch, unchanged from
    // before this existed.
    private val vectorColumns: List<AilakeNativeLoader.VectorColSpec> = emptyList(),
    private val vecIndices: List<Int> = emptyList(),
) : RichSinkFunction<RowData>() {

    private val idsBuffer = mutableListOf<Long>()
    private val embeddingsBuffer = mutableListOf<FloatArray>()
    private val multiEmbeddingsBuffers: List<MutableList<FloatArray>> =
        vectorColumns.map { mutableListOf() }
    private val textBuffers: Map<String, MutableList<String>> =
        extraColumnIndices.keys.associateWith { mutableListOf() }

    override fun invoke(row: RowData, context: SinkFunction.Context) {
        check(!row.isNullAt(idIdx)) { "id column cannot be NULL — every row must carry a real id for AI-Lake to index it" }
        val id = row.getLong(idIdx)
        idsBuffer.add(id)
        if (vectorColumns.isNotEmpty()) {
            vecIndices.forEachIndexed { i, idx ->
                check(!row.isNullAt(idx)) {
                    "Vector column '${vectorColumns[i].column}' cannot be NULL — every row must carry a real " +
                    "embedding for AI-Lake to index it"
                }
                multiEmbeddingsBuffers[i].add(row.getArray(idx).toFloatArray())
            }
        } else {
            check(!row.isNullAt(vecIdx)) { "Vector column '$vecCol' cannot be NULL — every row must carry a real embedding for AI-Lake to index it" }
            embeddingsBuffer.add(row.getArray(vecIdx).toFloatArray())
        }
        for ((col, idx) in extraColumnIndices) {
            val text = if (row.isNullAt(idx)) "" else row.getString(idx).toString()
            textBuffers[col]?.add(text)
        }
        if (idsBuffer.size >= AilakeVectorTableSink.BUFFER_SIZE) {
            flush()
        }
    }

    override fun close() {
        if (idsBuffer.isNotEmpty()) flush()
    }

    private fun flush() {
        val columnsSnapshot: Map<String, List<String>> =
            textBuffers.mapValues { (_, buf) -> buf.toList() }
        try {
            if (vectorColumns.isNotEmpty()) {
                AilakeNativeLoader.writeBatchMulti(
                    warehouse      = warehouse,
                    namespace      = namespace,
                    table          = tableName,
                    ids            = idsBuffer.toLongArray(),
                    vectorColumns  = vectorColumns.zip(multiEmbeddingsBuffers.map { it.toTypedArray() }),
                    embeddingModel = embeddingModel,
                    formatVersion  = formatVersion,
                    ftsColumns     = ftsColumns,
                    ftsTokenizer   = ftsTokenizer,
                    deferred       = deferred,
                    columns        = columnsSnapshot,
                )
            } else {
                AilakeNativeLoader.writeBatch(
                    warehouse       = warehouse,
                    namespace       = namespace,
                    table           = tableName,
                    vecCol          = vecCol,
                    dim             = dim,
                    metric          = metric,
                    precision       = precision,
                    ids             = idsBuffer.toLongArray(),
                    embeddings      = embeddingsBuffer.toTypedArray(),
                    embeddingModel  = embeddingModel,
                    partitionFields = partitionFields,
                    formatVersion   = formatVersion,
                    ftsColumns      = ftsColumns,
                    ftsTokenizer    = ftsTokenizer,
                    hnswM              = hnswM,
                    hnswEfConstruction = hnswEfConstruction,
                    preNormalize       = preNormalize,
                    deferred           = deferred,
                    columns         = columnsSnapshot,
                )
            }
        } finally {
            idsBuffer.clear()
            embeddingsBuffer.clear()
            multiEmbeddingsBuffers.forEach { it.clear() }
            textBuffers.values.forEach { it.clear() }
        }
    }

    // org.apache.flink.table.data.ArrayData does not have a toFloatArray() extension —
    // implement it inline
    private fun org.apache.flink.table.data.ArrayData.toFloatArray(): FloatArray =
        FloatArray(size()) { i -> getFloat(i) }
}
