// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink.internal

import com.sun.jna.Native
import com.sun.jna.Pointer
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Files

/**
 * Singleton that loads the ailake-jni native library and exposes safe Kotlin wrappers.
 *
 * The native library is located via (in order):
 *   1. System property `ailake.native.lib` — explicit path to the .so/.dll
 *   2. Environment variable `AILAKE_NATIVE_LIB`
 *   3. Library name "ailake_jni" via the standard JNA search path
 *      (jna.library.path, java.library.path, classpath resources)
 */
object AilakeNativeLoader {

    private val log = LoggerFactory.getLogger(AilakeNativeLoader::class.java)
    private val mapper = jacksonObjectMapper()

    val lib: AilakeNativeLib by lazy {
        val explicitPath =
            System.getProperty("ailake.native.lib")
                ?: System.getenv("AILAKE_NATIVE_LIB")

        val loaded = runCatching {
            if (explicitPath != null) {
                Native.load(explicitPath, AilakeNativeLib::class.java)
            } else {
                Native.load("ailake_jni", AilakeNativeLib::class.java)
            }
        }.onSuccess {
            log.info("[ailake] Native library libailake_jni loaded (path={})",
                explicitPath ?: "JNA default search path")
        }.onFailure {
            log.error(
                "[ailake] Failed to load native library libailake_jni (path={}). " +
                "Set ailake.native.lib system property or AILAKE_NATIVE_LIB env var. Error: {}",
                explicitPath ?: "JNA default search path", it.message
            )
        }.getOrThrow()

        loaded
    }

    val version: String by lazy { lib.ailake_version() }

    // ── Search ────────────────────────────────────────────────────────────────

    data class SearchResultItem(
        val row_id: Long,
        val distance: Float,
        val file_path: String,
    )

    data class SearchResponse(
        val ok: Boolean,
        val results: List<SearchResultItem> = emptyList(),
        val error: String? = null,
    )

    fun search(
        warehouse: String,
        namespace: String,
        table: String,
        vecCol: String,
        dim: Int,
        query: FloatArray,
        topK: Int = 10,
        efSearch: Int = 50,
        partitionFilter: String? = null,
        hybridText: String? = null,
        textColumn: String = "chunk_text",
        bm25Weight: Float = 0.5f,
    ): List<SearchResultItem> {
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse,
            "namespace" to namespace,
            "table" to table,
            "vec_col" to vecCol,
            "dim" to dim,
            "query" to query.toList(),
            "top_k" to topK,
            "ef_search" to efSearch,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        if (hybridText != null) {
            payload["hybrid_text"]  = hybridText
            payload["text_column"]  = textColumn
            payload["bm25_weight"]  = bm25Weight
        }
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_search_json(req)
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<SearchResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] ailake_search_json returned error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_search_json error: ${resp.error}")
            }
            log.debug("[ailake] search OK table={}.{} top_k={} results={}", namespace, table, topK, resp.results.size)
            resp.results
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    // ── BM25 text search ──────────────────────────────────────────────────────

    fun searchText(
        warehouse: String,
        namespace: String,
        table: String,
        queryText: String,
        topK: Int = 10,
        textColumn: String = "chunk_text",
        partitionFilter: String? = null,
    ): List<SearchResultItem> {
        val payload = mutableMapOf<String, Any>(
            "warehouse"   to warehouse,
            "namespace"   to namespace,
            "table"       to table,
            "query_text"  to queryText,
            "top_k"       to topK,
            "text_column" to textColumn,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_search_text_json(req)
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<SearchResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] ailake_search_text_json returned error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_search_text_json error: ${resp.error}")
            }
            log.debug("[ailake] searchText OK table={}.{} top_k={} results={}", namespace, table, topK, resp.results.size)
            resp.results
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    // ── Multimodal Search ─────────────────────────────────────────────────────

    data class MultimodalSearchResultItem(
        val row_id: Long,
        val rrf_score: Float,
        val file_path: String,
    )

    data class MultimodalSearchResponse(
        val ok: Boolean,
        val results: List<MultimodalSearchResultItem> = emptyList(),
        val error: String? = null,
    )

    /**
     * Cross-modal RRF search via the native library.
     *
     * @param queries  list of (column, query vector, weight) triples;
     *                 dim=0 means auto-detect from Iceberg metadata
     */
    fun searchMultimodal(
        warehouse: String,
        namespace: String,
        table: String,
        queries: List<Triple<String, FloatArray, Float>>,
        topK: Int = 10,
        partitionFilter: String? = null,
    ): List<MultimodalSearchResultItem> {
        require(queries.isNotEmpty()) { "queries must not be empty" }
        val queriesArr = queries.map { (col, q, w) ->
            mapOf("col" to col, "query" to q.toList(), "weight" to w, "dim" to 0)
        }
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse,
            "namespace" to namespace,
            "table"     to table,
            "queries"   to queriesArr,
            "top_k"     to topK,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_search_multimodal_json(req)
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<MultimodalSearchResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] searchMultimodal error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_search_multimodal_json error: ${resp.error}")
            }
            resp.results
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    data class WriteResponse(
        val ok: Boolean,
        val snapshot_id: Long = -1,
        val error: String? = null,
    )

    data class DeleteWhereResponse(
        val ok: Boolean,
        val error: String? = null,
    )

    data class EvolveSchemaResponse(
        val ok: Boolean,
        val new_schema_id: Int = -1,
        val error: String? = null,
    )

    /** Partition field definition for multi-column partition specs (Phase K). */
    data class PartitionFieldDef(val column: String, val transform: String, val columnType: String)

    /** Column addition request for schema evolution. */
    data class AddColReq(val name: String, val colType: String, val initialDefault: String? = null)

    /** Column rename request for schema evolution. */
    data class RenameColReq(val from: String, val to: String)

    /**
     * Write a batch of records to an AI-Lake table.
     *
     * @param partitionFields  multi-column partition spec (Phase K); empty = single-value partition_by/partition_value
     * @param formatVersion    Iceberg format version; 2 (default) or 3
     */
    fun writeBatch(
        warehouse: String,
        namespace: String,
        table: String,
        vecCol: String,
        dim: Int,
        metric: String = "euclidean",
        precision: String = "f16",
        ids: LongArray,
        embeddings: Array<FloatArray>,
        embeddingModel: String? = null,
        partitionBy: String? = null,
        partitionValue: String? = null,
        partitionFields: List<PartitionFieldDef> = emptyList(),
        formatVersion: Int = 2,
    ): Long {
        require(ids.size == embeddings.size) { "ids.size != embeddings.size" }
        val payload = mutableMapOf<String, Any>(
            "warehouse"      to warehouse,
            "namespace"      to namespace,
            "table"          to table,
            "vec_col"        to vecCol,
            "dim"            to dim,
            "metric"         to metric,
            "precision"      to precision,
            "ids"            to ids.toList(),
            "embeddings"     to embeddings.map { it.toList() },
            "format_version" to formatVersion,
        )
        if (embeddingModel != null) payload["embedding_model"] = embeddingModel
        if (partitionBy    != null) payload["partition_by"]    = partitionBy
        if (partitionValue != null) payload["partition_value"] = partitionValue
        if (partitionFields.isNotEmpty()) {
            payload["partition_fields"] = partitionFields.map { pf ->
                mapOf("column" to pf.column, "transform" to pf.transform, "column_type" to pf.columnType)
            }
        }
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_write_batch_json(req)
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<WriteResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] ailake_write_batch_json returned error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_write_batch_json error: ${resp.error}")
            }
            log.info("[ailake] write OK table={}.{} rows={} snapshot_id={}", namespace, table, ids.size, resp.snapshot_id)
            resp.snapshot_id
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    /**
     * Logically delete all rows where [column] equals any value in [values].
     * Throws [RuntimeException] on native error.
     */
    fun deleteWhere(
        warehouse: String,
        namespace: String,
        table: String,
        column: String,
        values: List<String>,
    ) {
        require(values.isNotEmpty()) { "values must not be empty" }
        val payload = mapOf(
            "warehouse" to warehouse,
            "namespace" to namespace,
            "table"     to table,
            "column"    to column,
            "values"    to values,
        )
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_delete_where_json(req)
        try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<DeleteWhereResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] ailake_delete_where_json returned error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_delete_where_json error: ${resp.error}")
            }
            log.info("[ailake] deleteWhere OK table={}.{} column={} values={}", namespace, table, column, values.size)
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    /**
     * Apply a metadata-only schema evolution to the table.
     * Returns the new schema_id.
     * Throws [RuntimeException] on native error.
     *
     * @param addCols     columns to add; [AddColReq.initialDefault] is a JSON literal (null, 0, "unknown", ...)
     * @param renameCols  columns to rename
     */
    fun evolveSchema(
        warehouse: String,
        namespace: String,
        table: String,
        addCols: List<AddColReq>,
        renameCols: List<RenameColReq>,
    ): Int {
        require(addCols.isNotEmpty() || renameCols.isNotEmpty()) {
            "at least one of addCols or renameCols must be non-empty"
        }
        // Build JSON manually for add_columns so initial_default is embedded as raw JSON literal.
        val addJson = addCols.joinToString(",", "[", "]") { ac ->
            val defPart = if (ac.initialDefault != null) ""","initial_default":${ac.initialDefault}""" else ""
            """{"name":${mapper.writeValueAsString(ac.name)},"type":${mapper.writeValueAsString(ac.colType)}$defPart}"""
        }
        val renJson = mapper.writeValueAsString(renameCols.map { rc -> mapOf("from" to rc.from, "to" to rc.to) })
        val baseJson = mapper.writeValueAsString(
            mapOf("warehouse" to warehouse, "namespace" to namespace, "table" to table)
        ).dropLast(1)
        val req = "$baseJson,\"add_columns\":$addJson,\"rename_columns\":$renJson}"

        val ptr = lib.ailake_evolve_schema_json(req)
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<EvolveSchemaResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] ailake_evolve_schema_json returned error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_evolve_schema_json error: ${resp.error}")
            }
            log.info("[ailake] evolveSchema OK table={}.{} new_schema_id={}", namespace, table, resp.new_schema_id)
            resp.new_schema_id
        } finally {
            lib.ailake_free_string(ptr)
        }
    }
}
