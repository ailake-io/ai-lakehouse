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
 * Schema exposed to Trino: (id BIGINT, embedding ARRAY<DOUBLE>)
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
) : ConnectorTableHandle, ConnectorInsertTableHandle
