// SPDX-License-Identifier: MIT OR Apache-2.0
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
    ): List<SearchResultItem> {
        val req = mapper.writeValueAsString(
            mapOf(
                "warehouse" to warehouse,
                "namespace" to namespace,
                "table" to table,
                "vec_col" to vecCol,
                "dim" to dim,
                "query" to query.toList(),
                "top_k" to topK,
                "ef_search" to efSearch,
            )
        )
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

    // ── Write ─────────────────────────────────────────────────────────────────

    data class WriteResponse(
        val ok: Boolean,
        val snapshot_id: Long = -1,
        val error: String? = null,
    )

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
    ): Long {
        require(ids.size == embeddings.size) { "ids.size != embeddings.size" }
        val req = mapper.writeValueAsString(
            mapOf(
                "warehouse" to warehouse,
                "namespace" to namespace,
                "table" to table,
                "vec_col" to vecCol,
                "dim" to dim,
                "metric" to metric,
                "precision" to precision,
                "ids" to ids.toList(),
                "embeddings" to embeddings.map { it.toList() },
            )
        )
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
}
