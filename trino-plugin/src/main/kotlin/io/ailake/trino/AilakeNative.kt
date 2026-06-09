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

    private interface Lib : Library {
        /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
        fun ailake_search_json(requestJson: String): Pointer?

        /** JSON-envelope write. Returns `{"ok":true,"snapshot_id":N}`. Caller must free. */
        fun ailake_write_batch_json(requestJson: String): Pointer?

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
    ): Long? {
        val native = lib ?: return null
        if (ids.isEmpty()) return null

        val requestJson = mapper.writeValueAsString(
            mapOf(
                "warehouse"  to tableUri,
                "namespace"  to namespace,
                "table"      to tableName,
                "vec_col"    to vectorColumn,
                "dim"        to dim,
                "metric"     to metric,
                "precision"  to precision,
                "ids"        to ids,
                "embeddings" to embeddings,
            )
        )

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
     * Run a vector search via the native library.
     *
     * @param tableUri    path/URI of the AI-Lake table root
     * @param queryBytes  Base64-encoded little-endian f32 array
     * @param topK        number of nearest neighbors
     */
    fun search(tableUri: String, queryBytes: String, topK: Int): List<SearchRow> {
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

        val requestJson = mapper.writeValueAsString(
            mapOf(
                "warehouse" to tableUri,
                "namespace" to "default",
                "table" to "table",
                "query" to floats,
                "dim" to floats.size,
                "top_k" to topK,
            )
        )

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
