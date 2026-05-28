package io.ailake.trino

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import io.trino.spi.HostAddress
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
 * `queryBytes` is Base64-encoded little-endian f32 array (4 bytes per dimension).
 * CSV→bytes conversion happens once in VectorScanSplitManager (planning phase),
 * not on every worker execution.
 */
data class VectorScanSplit @JsonCreator constructor(
    @JsonProperty("tableUri") val tableUri: String,
    @JsonProperty("queryBytes") val queryBytes: String,
    @JsonProperty("topK") val topK: Int,
) : ConnectorSplit {
    override fun isRemotelyAccessible(): Boolean = true
    override fun getAddresses(): List<HostAddress> = emptyList()
    override fun getInfo(): Any? = null
}
