package io.ailake.trino

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import io.trino.spi.connector.ColumnHandle
import io.trino.spi.connector.ConnectorSplit
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTransactionHandle

object VectorScanTransactionHandle : ConnectorTransactionHandle

data class VectorScanTableHandle @JsonCreator constructor(
    @JsonProperty("tableUri") val tableUri: String,
    @JsonProperty("vectorColumn") val vectorColumn: String,
    @JsonProperty("dim") val dim: Int,
) : ConnectorTableHandle

data class VectorScanColumnHandle @JsonCreator constructor(
    @JsonProperty("name") val name: String,
    @JsonProperty("ordinal") val ordinal: Int,
) : ColumnHandle

/**
 * A single split carrying all search parameters. AI-Lake search is not
 * parallelised at the split level — the native library handles file-level
 * parallelism internally via Tokio.
 *
 * `queryVector` is a comma-separated list of f32 values, e.g. "0.1,-0.2,0.3"
 */
data class VectorScanSplit @JsonCreator constructor(
    @JsonProperty("tableUri") val tableUri: String,
    @JsonProperty("queryVector") val queryVector: String,
    @JsonProperty("topK") val topK: Int,
) : ConnectorSplit {
    override fun getSplitInfo(): Map<String, String> = mapOf(
        "tableUri" to tableUri,
        "topK" to topK.toString(),
    )
}
