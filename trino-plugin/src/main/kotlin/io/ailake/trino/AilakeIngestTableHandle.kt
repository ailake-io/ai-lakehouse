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
data class AilakeIngestTableHandle @JsonCreator constructor(
    @JsonProperty("tableUri")        val tableUri:        String,
    @JsonProperty("namespace")       val namespace:       String,
    @JsonProperty("tableName")       val tableName:       String,
    @JsonProperty("vectorColumn")    val vectorColumn:    String,
    @JsonProperty("dim")             val dim:             Int,
    @JsonProperty("metric")          val metric:          String,
    @JsonProperty("precision")       val precision:       String,
    @JsonProperty("embeddingModel")  val embeddingModel:  String? = null,
    @JsonProperty("partitionFields") val partitionFields: List<AilakeNative.PartitionFieldDef> = emptyList(),
    @JsonProperty("formatVersion")   val formatVersion:   Int = 2,
    // Extra VARCHAR column names, in Page-channel order starting at index 2
    // (0=id, 1=embedding) — see VectorScanMetadata.ingestColumns().
    @JsonProperty("textColumns")     val textColumns:     List<String> = emptyList(),
    // Write-tuning knobs — see VectorScanConnectorFactory's ailake.hnsw-m /
    // ailake.hnsw-ef-construction / ailake.pre-normalize / ailake.deferred /
    // ailake.fts-columns / ailake.fts-tokenizer catalog properties. All were
    // already supported by AilakeNative.writeBatch but never reachable from
    // Trino before — AilakePageSink.finish() always passed the defaults.
    @JsonProperty("hnswM")            val hnswM:            Int? = null,
    @JsonProperty("hnswEfConstruction") val hnswEfConstruction: Int? = null,
    @JsonProperty("preNormalize")     val preNormalize:     Boolean = false,
    @JsonProperty("deferred")         val deferred:         Boolean = false,
    @JsonProperty("ftsColumns")       val ftsColumns:       List<String> = emptyList(),
    @JsonProperty("ftsTokenizer")     val ftsTokenizer:     String = "default",
    // DELETE pushdown state — set by VectorScanMetadata.applyFilter when the
    // WHERE clause is a single-column equality/IN predicate (the only shape
    // AilakeNative.deleteWhere supports); read back by applyDelete/executeDelete.
    // null = no delete predicate captured yet (the normal INSERT-path state).
    @JsonProperty("deleteColumn")     val deleteColumn:     String? = null,
    @JsonProperty("deleteValues")     val deleteValues:     List<String>? = null,
    // Multi-column (Phase 8 multimodal) ingest — see VectorScanMetadata.ingestColumns()'s
    // KDoc. Empty (default) = single-vector-column path via ailake_write_batch_json, same
    // as before this existed.
    @JsonProperty("vectorColumns")    val vectorColumns:    List<AilakeNative.VectorColSpec> = emptyList(),
) : ConnectorTableHandle, ConnectorInsertTableHandle
