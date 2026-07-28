// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.databind.ObjectMapper
import io.trino.spi.connector.Connector
import io.trino.spi.connector.ConnectorContext
import io.trino.spi.connector.ConnectorFactory

class VectorScanConnectorFactory : ConnectorFactory {
    override fun getName(): String = "ailake"

    override fun create(
        catalogName: String,
        config: Map<String, String>,
        context: ConnectorContext,
    ): Connector {
        val tableUri       = requireNotNull(config["ailake.table-uri"]) {
            "ailake.table-uri is required in catalog properties"
        }
        val vectorColumn   = config.getOrDefault("ailake.vector-column", "embedding")
        val dim            = config.getOrDefault("ailake.vector-dim", "1536").toInt()
        val metric         = config.getOrDefault("ailake.metric", "cosine")
        val precision      = config.getOrDefault("ailake.precision", "f16")
        val namespace      = config.getOrDefault("ailake.namespace", "default")
        val tableName      = config.getOrDefault("ailake.table-name",
            tableUri.trimEnd('/').substringAfterLast('/'))
        val embeddingModel = config["ailake.embedding-model"]?.takeIf { it.isNotEmpty() }
        val pfJson = config.getOrDefault("ailake.partition-fields", "[]")
        val partitionFields: List<AilakeNative.PartitionFieldDef> = if (pfJson == "[]" || pfJson.isBlank()) emptyList() else {
            val node = ObjectMapper().readTree(pfJson)
            (0 until node.size()).map { i ->
                val n = node.get(i)
                AilakeNative.PartitionFieldDef(n.get("column").asText(), n.get("transform").asText(), n.get("column_type").asText())
            }
        }
        val formatVersion = config.getOrDefault("ailake.format-version", "2").toInt()
        val textColumns = config.getOrDefault("ailake.text-columns", "")
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }
        val hnswM = config["ailake.hnsw-m"]?.toInt()
        val hnswEfConstruction = config["ailake.hnsw-ef-construction"]?.toInt()
        val preNormalize = config.getOrDefault("ailake.pre-normalize", "false").toBoolean()
        val deferred = config.getOrDefault("ailake.deferred", "false").toBoolean()
        val ftsColumns = config.getOrDefault("ailake.fts-columns", "")
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }
        val ftsTokenizer = config.getOrDefault("ailake.fts-tokenizer", "default")
        // Multi-column (Phase 8 multimodal) ingest — e.g. text + image embeddings on the
        // same row. When set, the `ingest` table's schema gets one ARRAY<DOUBLE> column per
        // entry (instead of the single `ailake.vector-column`) and INSERT calls
        // ailake_write_batch_multi_json instead of ailake_write_batch_json. Was already
        // exposed from Spark (`ailakeWriteMulti`) but had no wrapper here at all.
        val vcJson = config.getOrDefault("ailake.vector-columns", "[]")
        val vectorColumns: List<AilakeNative.VectorColSpec> = if (vcJson == "[]" || vcJson.isBlank()) emptyList() else {
            val node = ObjectMapper().readTree(vcJson)
            (0 until node.size()).map { i ->
                val n = node.get(i)
                AilakeNative.VectorColSpec(
                    column = n.get("column").asText(),
                    dim = n.get("dim").asInt(),
                    metric = n.path("metric").asText("cosine"),
                    precision = n.path("precision").asText("f16"),
                    modality = if (n.has("modality") && !n.get("modality").isNull) n.get("modality").asText() else null,
                )
            }
        }
        return VectorScanConnector(
            tableUri, vectorColumn, dim, metric, precision, namespace, tableName, embeddingModel,
            partitionFields, formatVersion, textColumns,
            hnswM, hnswEfConstruction, preNormalize, deferred, ftsColumns, ftsTokenizer,
            vectorColumns, catalogOptsFromConfig(config),
        )
    }

    /**
     * REST Catalog (Fase 17/19) config read from `ailake.catalog`/`ailake.rest-*`
     * catalog properties, translated to the snake_case field names
     * `ailake-jni`'s `CatalogOpts` expects. Empty (no `ailake.catalog` property
     * set) = default Hadoop catalog, unchanged behavior. Only reaches
     * `AilakeProcedures` today (see `VectorScanConnector`'s doc comment) — not
     * yet the search/scan/insert paths. See docs/guides/REST_CATALOG.md.
     */
    private fun catalogOptsFromConfig(config: Map<String, String>): Map<String, String> {
        val keys = listOf(
            "ailake.catalog"                   to "catalog",
            "ailake.rest-uri"                  to "rest_uri",
            "ailake.rest-prefix"                to "rest_prefix",
            "ailake.rest-warehouse"             to "rest_warehouse",
            "ailake.rest-auth"                  to "rest_auth",
            "ailake.rest-token"                 to "rest_token",
            "ailake.rest-oauth-token-endpoint"   to "rest_oauth_token_endpoint",
            "ailake.rest-oauth-client-id"        to "rest_oauth_client_id",
            "ailake.rest-oauth-client-secret"    to "rest_oauth_client_secret",
            "ailake.rest-oauth-scope"            to "rest_oauth_scope",
        )
        return keys.mapNotNull { (configKey, jniKey) ->
            config[configKey]?.takeIf { it.isNotEmpty() }?.let { jniKey to it }
        }.toMap()
    }
}
