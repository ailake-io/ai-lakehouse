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
import org.apache.flink.table.data.RowData

/**
 * AI-Lake Flink table sink.  Buffers rows in memory and flushes each batch to the
 * native library via [AilakeNativeLoader.writeBatch].
 *
 * Expected row schema:
 *   - One BIGINT column named "id" (configurable via id_col index, default 0)
 *   - One ARRAY<FLOAT> or BYTES column for the embedding vector
 *   - Any number of additional columns (currently ignored — future: stored in Parquet)
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
) : DynamicTableSink {

    companion object {
        const val BUFFER_SIZE = 10_000
    }

    override fun getChangelogMode(requestedMode: ChangelogMode): ChangelogMode =
        ChangelogMode.insertOnly()

    override fun getSinkRuntimeProvider(context: DynamicTableSink.Context): DynamicTableSink.SinkRuntimeProvider {
        val colNames = schema.columnNames
        val idIdx = colNames.indexOfFirst { it == "id" }.takeIf { it >= 0 } ?: 0
        val vecIdx = colNames.indexOfFirst { it == vecCol }.takeIf { it >= 0 } ?: 1
        val ftsColumnIndices: Map<String, Int> = ftsColumns
            .associateWith { col -> colNames.indexOf(col) }
            .filterValues { it >= 0 }
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
                        ftsColumnIndices   = ftsColumnIndices,
                    )
                )
            }
        }
    }

    override fun copy(): DynamicTableSink = AilakeVectorTableSink(
        warehouse, namespace, tableName, vecCol, dim, metric, precision, schema,
        embeddingModel, partitionFields, formatVersion, ftsColumns, ftsTokenizer
    )

    override fun asSummaryString(): String = "AI-Lake-Sink[$namespace.$tableName]"
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
    private val ftsColumnIndices: Map<String, Int> = emptyMap(),
) : RichSinkFunction<RowData>() {

    private val idsBuffer = mutableListOf<Long>()
    private val embeddingsBuffer = mutableListOf<FloatArray>()
    private val textBuffers: Map<String, MutableList<String>> =
        ftsColumnIndices.keys.associateWith { mutableListOf() }

    override fun invoke(row: RowData, context: SinkFunction.Context) {
        val id = row.getLong(idIdx)
        val embedding = row.getArray(vecIdx).toFloatArray()
        idsBuffer.add(id)
        embeddingsBuffer.add(embedding)
        for ((col, idx) in ftsColumnIndices) {
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
                columns         = columnsSnapshot,
            )
        } finally {
            idsBuffer.clear()
            embeddingsBuffer.clear()
            textBuffers.values.forEach { it.clear() }
        }
    }

    // org.apache.flink.table.data.ArrayData does not have a toFloatArray() extension —
    // implement it inline
    private fun org.apache.flink.table.data.ArrayData.toFloatArray(): FloatArray =
        FloatArray(size()) { i -> getFloat(i) }
}
