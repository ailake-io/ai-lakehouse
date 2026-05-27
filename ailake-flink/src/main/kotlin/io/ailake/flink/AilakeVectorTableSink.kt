package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
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
) : DynamicTableSink {

    companion object {
        const val BUFFER_SIZE = 10_000
    }

    override fun getChangelogMode(requestedMode: ChangelogMode): ChangelogMode =
        ChangelogMode.insertOnly()

    override fun getSinkRuntimeProvider(context: DynamicTableSink.Context): DynamicTableSink.SinkRuntimeProvider {
        val idIdx = schema.columnNames.indexOfFirst { it == "id" }.takeIf { it >= 0 } ?: 0
        val vecIdx = schema.columnNames.indexOfFirst { it == vecCol }.takeIf { it >= 0 } ?: 1
        return object : DataStreamSinkProvider {
            override fun consumeDataStream(
                context: ProviderContext,
                dataStream: DataStream<RowData>,
            ): DataStreamSink<*> {
                return dataStream.addSink(
                    AilakeSinkFunction(
                        warehouse  = warehouse,
                        namespace  = namespace,
                        tableName  = tableName,
                        vecCol     = vecCol,
                        dim        = dim,
                        metric     = metric,
                        precision  = precision,
                        idIdx      = idIdx,
                        vecIdx     = vecIdx,
                    )
                )
            }
        }
    }

    override fun copy(): DynamicTableSink = AilakeVectorTableSink(
        warehouse, namespace, tableName, vecCol, dim, metric, precision, schema
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
) : RichSinkFunction<RowData>() {

    private val idsBuffer = mutableListOf<Long>()
    private val embeddingsBuffer = mutableListOf<FloatArray>()

    override fun invoke(row: RowData, context: SinkFunction.Context) {
        val id = row.getLong(idIdx)
        val embedding = row.getArray(vecIdx).toFloatArray()
        idsBuffer.add(id)
        embeddingsBuffer.add(embedding)
        if (idsBuffer.size >= AilakeVectorTableSink.BUFFER_SIZE) {
            flush()
        }
    }

    override fun close() {
        if (idsBuffer.isNotEmpty()) flush()
    }

    private fun flush() {
        AilakeNativeLoader.writeBatch(
            warehouse  = warehouse,
            namespace  = namespace,
            table      = tableName,
            vecCol     = vecCol,
            dim        = dim,
            metric     = metric,
            precision  = precision,
            ids        = idsBuffer.toLongArray(),
            embeddings = embeddingsBuffer.toTypedArray(),
        )
        idsBuffer.clear()
        embeddingsBuffer.clear()
    }

    // org.apache.flink.table.data.ArrayData does not have a toFloatArray() extension —
    // implement it inline
    private fun org.apache.flink.table.data.ArrayData.toFloatArray(): FloatArray =
        FloatArray(size()) { i -> getFloat(i) }
}
