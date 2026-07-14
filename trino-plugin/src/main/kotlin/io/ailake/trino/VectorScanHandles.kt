// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import io.trino.spi.HostAddress
import io.trino.spi.connector.ColumnHandle
import io.trino.spi.connector.ConnectorSplit
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTransactionHandle

// A Kotlin `object` compiles to a class with a private synthetic no-arg
// constructor — Trino's internal Jackson mapper (no kotlin-module registered,
// same root cause as the NB below) reflects on that constructor directly and
// fails with IllegalAccessException: "cannot access a member of class
// VectorScanTransactionHandle with modifiers 'private'" (confirmed live
// against a real Trino 430 server — this surfaced only after fixing the
// tableUri NPE below, since a real SELECT never got past that first). A
// @JsonCreator static factory sidesteps the constructor entirely.
object VectorScanTransactionHandle : ConnectorTransactionHandle {
    @JsonCreator
    @JvmStatic
    fun jsonCreator(): VectorScanTransactionHandle = VectorScanTransactionHandle
}

// NB: every property below carries BOTH @param: and @get: use-site targets.
// Trino's ObjectMapperProvider disables MapperFeature.AUTO_DETECT_GETTERS/FIELDS
// globally (it relies on io.airlift.json.RecordAutoDetectModule for genuine
// java.lang.Record types instead) — a bare @JsonProperty on a Kotlin primary-
// constructor `val` defaults to the PARAMETER site only, which is enough for
// deserialization (creator param resolution) but invisible to serialization
// (no getter/field Jackson is allowed to read), so every field silently
// serializes as absent. Root cause of the Trino SELECT NPE ("Parameter
// specified as non-null is null: ... parameter tableUri") — the coordinator's
// TaskUpdateRequest JSON never carried the handle's fields at all. See
// CHANGELOG.md and docs/specs/JVM_PLUGINS.md.
data class VectorScanTableHandle @JsonCreator constructor(
    @param:JsonProperty("tableUri") @get:JsonProperty("tableUri") val tableUri: String,
    @param:JsonProperty("vectorColumn") @get:JsonProperty("vectorColumn") val vectorColumn: String,
    @param:JsonProperty("dim") @get:JsonProperty("dim") val dim: Int,
    @param:JsonProperty("namespace") @get:JsonProperty("namespace") val namespace: String,
    @param:JsonProperty("tableName") @get:JsonProperty("tableName") val tableName: String,
) : ConnectorTableHandle

/** Table handle for `ailake.default.search_multimodal` — see [VectorScanMetadata]. */
data class MultimodalScanTableHandle @JsonCreator constructor(
    @param:JsonProperty("tableUri") @get:JsonProperty("tableUri") val tableUri: String,
    @param:JsonProperty("namespace") @get:JsonProperty("namespace") val namespace: String,
    @param:JsonProperty("tableName") @get:JsonProperty("tableName") val tableName: String,
) : ConnectorTableHandle

/**
 * Table handle for `ailake.default.search_full` (Fase 11 — search + full-row fetch, no JOIN
 * needed) — same shape as [VectorScanTableHandle], kept a distinct type purely so
 * [VectorScanSplitManager]/[VectorScanRecordSetProvider] can dispatch to `AilakeNative.scan`
 * instead of `AilakeNative.search` by table handle type. See [VectorScanMetadata].
 */
data class ScanTableHandle @JsonCreator constructor(
    @param:JsonProperty("tableUri") @get:JsonProperty("tableUri") val tableUri: String,
    @param:JsonProperty("vectorColumn") @get:JsonProperty("vectorColumn") val vectorColumn: String,
    @param:JsonProperty("dim") @get:JsonProperty("dim") val dim: Int,
    @param:JsonProperty("namespace") @get:JsonProperty("namespace") val namespace: String,
    @param:JsonProperty("tableName") @get:JsonProperty("tableName") val tableName: String,
) : ConnectorTableHandle

data class VectorScanColumnHandle @JsonCreator constructor(
    @param:JsonProperty("name") @get:JsonProperty("name") val name: String,
    @param:JsonProperty("ordinal") @get:JsonProperty("ordinal") val ordinal: Int,
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
    @param:JsonProperty("tableUri") @get:JsonProperty("tableUri") val tableUri: String,
    @param:JsonProperty("queryBytes") @get:JsonProperty("queryBytes") val queryBytes: String,
    @param:JsonProperty("topK") @get:JsonProperty("topK") val topK: Int,
    @param:JsonProperty("namespace") @get:JsonProperty("namespace") val namespace: String,
    @param:JsonProperty("tableName") @get:JsonProperty("tableName") val tableName: String,
    @param:JsonProperty("vectorColumn") @get:JsonProperty("vectorColumn") val vectorColumn: String,
    @param:JsonProperty("queryText") @get:JsonProperty("queryText") val queryText: String = "",
    @param:JsonProperty("hybridWeight") @get:JsonProperty("hybridWeight") val hybridWeight: Float = 0.5f,
) : ConnectorSplit {
    override fun isRemotelyAccessible(): Boolean = true
    override fun getAddresses(): List<HostAddress> = emptyList()
    override fun getInfo(): Any? = null
}

/**
 * Split for `ailake.default.search_multimodal`. Unlike [VectorScanSplit], the query
 * payload (N per-column vectors + RRF weights) isn't reduced to a single float array at
 * planning time — `queriesJson` carries the raw `SET SESSION ailake.multimodal_queries`
 * JSON straight through and is parsed once at execution in [VectorScanRecordSetProvider].
 */
data class MultimodalScanSplit @JsonCreator constructor(
    @param:JsonProperty("tableUri") @get:JsonProperty("tableUri") val tableUri: String,
    @param:JsonProperty("namespace") @get:JsonProperty("namespace") val namespace: String,
    @param:JsonProperty("tableName") @get:JsonProperty("tableName") val tableName: String,
    @param:JsonProperty("queriesJson") @get:JsonProperty("queriesJson") val queriesJson: String,
    @param:JsonProperty("topK") @get:JsonProperty("topK") val topK: Int,
) : ConnectorSplit {
    override fun isRemotelyAccessible(): Boolean = true
    override fun getAddresses(): List<HostAddress> = emptyList()
    override fun getInfo(): Any? = null
}
