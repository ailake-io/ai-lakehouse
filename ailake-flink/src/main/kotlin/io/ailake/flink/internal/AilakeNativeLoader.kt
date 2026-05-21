package io.ailake.flink.internal

import com.sun.jna.Native
import com.sun.jna.Pointer
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
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

    private val mapper = jacksonObjectMapper()

    val lib: AilakeNativeLib by lazy {
        val explicitPath =
            System.getProperty("ailake.native.lib")
                ?: System.getenv("AILAKE_NATIVE_LIB")

        if (explicitPath != null) {
            Native.load(explicitPath, AilakeNativeLib::class.java)
        } else {
            Native.load("ailake_jni", AilakeNativeLib::class.java)
        }
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
            if (!resp.ok) throw RuntimeException("ailake_search_json error: ${resp.error}")
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
            if (!resp.ok) throw RuntimeException("ailake_write_batch_json error: ${resp.error}")
            resp.snapshot_id
        } finally {
            lib.ailake_free_string(ptr)
        }
    }
}
