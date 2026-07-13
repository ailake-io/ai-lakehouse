// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import io.trino.spi.connector.ConnectorInsertTableHandle
import io.trino.spi.connector.ConnectorTableHandle

/**
 * Represents the AI-Lake ingest table in both roles:
 *  - [ConnectorTableHandle]  — returned by getTableHandle for `ailake.default.ingest`
 *  - [ConnectorInsertTableHandle] — passed through beginInsert → AilakePageSink
 *
 * Schema exposed to Trino: (id BIGINT, embedding ARRAY<DOUBLE>, ...textColumns VARCHAR)
 */
// NB: every property below carries BOTH @param: and @get: use-site targets —
// see the matching note atop VectorScanHandles.kt (bare @JsonProperty on a
// Kotlin primary-constructor `val` only reaches the constructor parameter,
// invisible to Trino's ObjectMapperProvider on serialization since it disables
// MapperFeature.AUTO_DETECT_GETTERS/FIELDS globally).
data class AilakeIngestTableHandle @JsonCreator constructor(
    @param:JsonProperty("tableUri")        @get:JsonProperty("tableUri")        val tableUri:        String,
    @param:JsonProperty("namespace")       @get:JsonProperty("namespace")       val namespace:       String,
    @param:JsonProperty("tableName")       @get:JsonProperty("tableName")       val tableName:       String,
    @param:JsonProperty("vectorColumn")    @get:JsonProperty("vectorColumn")    val vectorColumn:    String,
    @param:JsonProperty("dim")             @get:JsonProperty("dim")             val dim:             Int,
    @param:JsonProperty("metric")          @get:JsonProperty("metric")          val metric:          String,
    @param:JsonProperty("precision")       @get:JsonProperty("precision")       val precision:       String,
    @param:JsonProperty("embeddingModel")  @get:JsonProperty("embeddingModel")  val embeddingModel:  String? = null,
    @param:JsonProperty("partitionFields") @get:JsonProperty("partitionFields") val partitionFields: List<AilakeNative.PartitionFieldDef> = emptyList(),
    @param:JsonProperty("formatVersion")   @get:JsonProperty("formatVersion")   val formatVersion:   Int = 2,
    // Extra VARCHAR column names, in Page-channel order starting at index 2
    // (0=id, 1=embedding) — see VectorScanMetadata.ingestColumns().
    @param:JsonProperty("textColumns")     @get:JsonProperty("textColumns")     val textColumns:     List<String> = emptyList(),
    // Write-tuning knobs — see VectorScanConnectorFactory's ailake.hnsw-m /
    // ailake.hnsw-ef-construction / ailake.pre-normalize / ailake.deferred /
    // ailake.fts-columns / ailake.fts-tokenizer catalog properties. All were
    // already supported by AilakeNative.writeBatch but never reachable from
    // Trino before — AilakePageSink.finish() always passed the defaults.
    @param:JsonProperty("hnswM")            @get:JsonProperty("hnswM")            val hnswM:            Int? = null,
    @param:JsonProperty("hnswEfConstruction") @get:JsonProperty("hnswEfConstruction") val hnswEfConstruction: Int? = null,
    @param:JsonProperty("preNormalize")     @get:JsonProperty("preNormalize")     val preNormalize:     Boolean = false,
    @param:JsonProperty("deferred")         @get:JsonProperty("deferred")         val deferred:         Boolean = false,
    @param:JsonProperty("ftsColumns")       @get:JsonProperty("ftsColumns")       val ftsColumns:       List<String> = emptyList(),
    @param:JsonProperty("ftsTokenizer")     @get:JsonProperty("ftsTokenizer")     val ftsTokenizer:     String = "default",
    // DELETE pushdown state — set by VectorScanMetadata.applyFilter when the
    // WHERE clause is a single-column equality/IN predicate (the only shape
    // AilakeNative.deleteWhere supports); read back by applyDelete/executeDelete.
    // null = no delete predicate captured yet (the normal INSERT-path state).
    @param:JsonProperty("deleteColumn")     @get:JsonProperty("deleteColumn")     val deleteColumn:     String? = null,
    @param:JsonProperty("deleteValues")     @get:JsonProperty("deleteValues")     val deleteValues:     List<String>? = null,
    // Multi-column (Phase 8 multimodal) ingest — see VectorScanMetadata.ingestColumns()'s
    // KDoc. Empty (default) = single-vector-column path via ailake_write_batch_json, same
    // as before this existed.
    @param:JsonProperty("vectorColumns")    @get:JsonProperty("vectorColumns")    val vectorColumns:    List<AilakeNative.VectorColSpec> = emptyList(),
) : ConnectorTableHandle, ConnectorInsertTableHandle
