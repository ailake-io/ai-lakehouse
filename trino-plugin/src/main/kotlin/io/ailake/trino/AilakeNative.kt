// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import org.slf4j.LoggerFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

/**
 * JNA bridge to libailake_jni.so.
 *
 * The library must be on java.library.path or LD_LIBRARY_PATH.
 * If not found, search returns empty results (graceful degradation).
 */
object AilakeNative {

    private val log = LoggerFactory.getLogger(AilakeNative::class.java)

    data class SearchRow(val rowId: Long, val distance: Float, val filePath: String)

    /** Partition field definition for multi-column partition specs (Phase K). */
    data class PartitionFieldDef(val column: String, val transform: String, val columnType: String)

    /** Column addition request for schema evolution. */
    data class AddColReq(val name: String, val colType: String, val initialDefault: String? = null)

    /** Column rename request for schema evolution. */
    data class RenameColReq(val from: String, val to: String)

    private interface Lib : Library {
        /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
        fun ailake_search_json(requestJson: String): Pointer?

        /** Cross-modal RRF. Returns `{"ok":true,"results":[{"row_id":N,"rrf_score":F,"file_path":"..."}]}`. Caller must free. */
        fun ailake_search_multimodal_json(requestJson: String): Pointer?

        /** JSON-envelope write. Returns `{"ok":true,"snapshot_id":N}`. Caller must free. */
        fun ailake_write_batch_json(requestJson: String): Pointer?

        /** Logical delete via equality delete file. Returns `{"ok":true}`. Caller must free. */
        fun ailake_delete_where_json(requestJson: String): Pointer?

        /** Schema evolution. Returns `{"ok":true,"new_schema_id":N}`. Caller must free. */
        fun ailake_evolve_schema_json(requestJson: String): Pointer?

        fun ailake_free_string(ptr: Pointer)
    }

    private val lib: Lib? by lazy {
        runCatching { Native.load("ailake_jni", Lib::class.java) as Lib }
            .onSuccess { log.info("[ailake] Native library libailake_jni loaded successfully") }
            .onFailure {
                log.warn(
                    "[ailake] Native library libailake_jni not found — vector search disabled. " +
                    "Set java.library.path or LD_LIBRARY_PATH to the directory containing libailake_jni.so. " +
                    "Error: ${it.message}"
                )
            }
            .getOrNull()
    }

    private val mapper = jacksonObjectMapper()

    /**
     * Write a batch of rows to an AI-Lake table via the native library.
     * Returns the snapshot_id on success, null on failure.
     *
     * @param partitionFields  multi-column partition spec (Phase K); empty = single-value partition_by/partition_value
     * @param formatVersion    Iceberg format version; 2 (default) or 3
     */
    fun writeBatch(
        tableUri: String,
        namespace: String,
        tableName: String,
        vectorColumn: String,
        dim: Int,
        metric: String,
        precision: String,
        ids: List<Long>,
        embeddings: List<List<Float>>,
        embeddingModel: String? = null,
        partitionBy: String? = null,
        partitionValue: String? = null,
        partitionFields: List<PartitionFieldDef> = emptyList(),
        formatVersion: Int = 2,
    ): Long? {
        val native = lib ?: return null
        if (ids.isEmpty()) return null

        val payload = mutableMapOf<String, Any>(
            "warehouse"      to tableUri,
            "namespace"      to namespace,
            "table"          to tableName,
            "vec_col"        to vectorColumn,
            "dim"            to dim,
            "metric"         to metric,
            "precision"      to precision,
            "ids"            to ids,
            "embeddings"     to embeddings,
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
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_write_batch_json(requestJson) ?: run {
            log.warn("[ailake] ailake_write_batch_json returned null pointer for table={}", tableName)
            return null
        }

        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] writeBatch ok=false for table={}: {}", tableName, resp["error"])
                return null
            }
            (resp["snapshot_id"] as? Number)?.toLong()
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse writeBatch response for table={}: {}", tableName, e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }

    /**
     * Logically delete all rows where [column] equals any value in [values].
     * Writes an Iceberg equality delete file via the native library.
     * Returns true on success, false if the library is absent or the call fails.
     */
    fun deleteWhere(
        tableUri: String,
        namespace: String,
        tableName: String,
        column: String,
        values: List<String>,
    ): Boolean {
        if (values.isEmpty()) return false
        val native = lib ?: return false

        val payload = mapOf(
            "warehouse" to tableUri,
            "namespace" to namespace,
            "table"     to tableName,
            "column"    to column,
            "values"    to values,
        )
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_delete_where_json(requestJson) ?: run {
            log.warn("[ailake] ailake_delete_where_json returned null pointer for table={}", tableName)
            return false
        }
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] deleteWhere ok=false for table={}: {}", tableName, resp["error"])
                false
            } else true
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse deleteWhere response for table={}: {}", tableName, e.message, e)
            false
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }

    /**
     * Apply a metadata-only schema evolution to the table.
     * Returns the new schema_id on success, -1 on error, 0 when no-op (both lists empty).
     *
     * @param addCols     columns to add; [AddColReq.initialDefault] is a JSON literal (null, 0, "unknown", ...)
     * @param renameCols  columns to rename
     */
    fun evolveSchema(
        tableUri: String,
        namespace: String,
        tableName: String,
        addCols: List<AddColReq>,
        renameCols: List<RenameColReq>,
    ): Int {
        if (addCols.isEmpty() && renameCols.isEmpty()) return 0
        val native = lib ?: return -1

        val addArr = addCols.map { ac ->
            buildMap<String, Any> {
                put("name", ac.name)
                put("type", ac.colType)
                if (ac.initialDefault != null) put("initial_default_raw", ac.initialDefault)
            }
        }
        val renArr = renameCols.map { rc -> mapOf("from" to rc.from, "to" to rc.to) }

        // Build JSON manually for add_columns so initial_default is embedded as a raw JSON
        // literal (not re-quoted by Jackson).
        val addJson = addCols.joinToString(",", "[", "]") { ac ->
            val defPart = if (ac.initialDefault != null) ""","initial_default":${ac.initialDefault}""" else ""
            """{"name":${mapper.writeValueAsString(ac.name)},"type":${mapper.writeValueAsString(ac.colType)}$defPart}"""
        }
        val renJson = mapper.writeValueAsString(renArr)
        val baseJson = mapper.writeValueAsString(
            mapOf("warehouse" to tableUri, "namespace" to namespace, "table" to tableName)
        ).dropLast(1)
        val requestJson = "$baseJson,\"add_columns\":$addJson,\"rename_columns\":$renJson}"

        val ptr = native.ailake_evolve_schema_json(requestJson) ?: run {
            log.warn("[ailake] ailake_evolve_schema_json returned null pointer for table={}", tableName)
            return -1
        }
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] evolveSchema ok=false for table={}: {}", tableName, resp["error"])
                return -1
            }
            (resp["new_schema_id"] as? Number)?.toInt() ?: -1
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse evolveSchema response for table={}: {}", tableName, e.message, e)
            -1
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }

    data class MultimodalSearchRow(val rowId: Long, val rrfScore: Float, val filePath: String)

    /**
     * Cross-modal vector search via Reciprocal Rank Fusion.
     *
     * @param tableUri  path/URI of the AI-Lake table root
     * @param queries   list of (column, query vector, weight) triples
     * @param topK      number of results to return
     */
    fun searchMultimodal(
        tableUri: String,
        queries: List<Triple<String, List<Float>, Float>>,
        topK: Int,
        partitionFilter: String? = null,
    ): List<MultimodalSearchRow> {
        val native = lib ?: return emptyList()
        if (queries.isEmpty()) return emptyList()

        val queriesJson = queries.joinToString(",", "[", "]") { (col, q, w) ->
            val qArr = q.joinToString(",", "[", "]")
            """{"col":${mapper.writeValueAsString(col)},"query":$qArr,"weight":$w,"dim":0}"""
        }
        val partJson = if (partitionFilter != null) ""","partition_filter":${mapper.writeValueAsString(partitionFilter)}""" else ""
        val requestJson = mapper.writeValueAsString(
            mapOf("warehouse" to tableUri, "namespace" to "default", "table" to "table",
                  "top_k" to topK)
        ).dropLast(1) + ""","queries":$queriesJson$partJson}"""

        val ptr = native.ailake_search_multimodal_json(requestJson) ?: run {
            log.warn("[ailake] ailake_search_multimodal_json returned null for tableUri={}", tableUri)
            return emptyList()
        }
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] searchMultimodal ok=false for tableUri={}: {}", tableUri, resp["error"])
                return emptyList()
            }
            @Suppress("UNCHECKED_CAST")
            (resp["results"] as? List<Map<String, Any>> ?: emptyList()).map { m ->
                MultimodalSearchRow(
                    rowId    = (m["row_id"] as Number).toLong(),
                    rrfScore = (m["rrf_score"] as Number).toFloat(),
                    filePath = m["file_path"] as String,
                )
            }
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse multimodal response for tableUri={}: {}", tableUri, e.message, e)
            emptyList()
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }

    /**
     * Run a vector search via the native library.
     *
     * @param tableUri    path/URI of the AI-Lake table root
     * @param queryBytes  Base64-encoded little-endian f32 array
     * @param topK        number of nearest neighbors
     */
    fun search(tableUri: String, queryBytes: String, topK: Int, partitionFilter: String? = null): List<SearchRow> {
        val native = lib ?: return emptyList()
        if (queryBytes.isBlank()) return emptyList()

        val floats = runCatching {
            val bytes = Base64.getDecoder().decode(queryBytes)
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            (0 until bytes.size / 4).map { buf.getFloat() }
        }.getOrElse {
            log.error("[ailake] Failed to decode Base64 query vector for tableUri={}: {}", tableUri, it.message)
            return emptyList()
        }
        if (floats.isEmpty()) return emptyList()

        val payload = mutableMapOf<String, Any>(
            "warehouse" to tableUri,
            "namespace" to "default",
            "table" to "table",
            "query" to floats,
            "dim" to floats.size,
            "top_k" to topK,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_search_json(requestJson) ?: run {
            log.warn("[ailake] ailake_search_json returned null pointer for tableUri={}", tableUri)
            return emptyList()
        }

        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] Native search returned ok=false for tableUri={}: {}", tableUri, resp["error"])
                return emptyList()
            }
            @Suppress("UNCHECKED_CAST")
            (resp["results"] as? List<Map<String, Any>> ?: emptyList()).map { m ->
                SearchRow(
                    rowId = (m["row_id"] as Number).toLong(),
                    distance = (m["distance"] as Number).toFloat(),
                    filePath = m["file_path"] as String,
                )
            }
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse native search response for tableUri={}: {}", tableUri, e.message, e)
            emptyList()
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }
}
