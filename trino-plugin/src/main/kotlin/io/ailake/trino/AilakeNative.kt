package io.ailake.trino

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer

/**
 * JNA bridge to libailake_jni.so.
 *
 * The library must be on java.library.path or LD_LIBRARY_PATH.
 * If not found, search returns empty results (graceful degradation).
 */
object AilakeNative {

    data class SearchRow(val rowId: Long, val distance: Float, val filePath: String)

    private interface Lib : Library {
        /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
        fun ailake_search_json(requestJson: String): Pointer?

        fun ailake_free_string(ptr: Pointer)
    }

    private val lib: Lib? by lazy {
        runCatching { Native.load("ailake_jni", Lib::class.java) as Lib }
            .onFailure { System.err.println("[ailake] Native library not found: ${it.message}") }
            .getOrNull()
    }

    private val mapper = jacksonObjectMapper()

    /**
     * Run a vector search via the native library.
     *
     * @param tableUri        path/URI of the AI-Lake table root
     * @param queryVectorCsv  comma-separated f32 values, e.g. "0.1,-0.2,0.3"
     * @param topK            number of nearest neighbors
     */
    fun search(tableUri: String, queryVectorCsv: String, topK: Int): List<SearchRow> {
        val native = lib ?: return emptyList()
        if (queryVectorCsv.isBlank()) return emptyList()

        val floats = runCatching {
            queryVectorCsv.split(',').map { it.trim().toFloat() }
        }.getOrElse { return emptyList() }
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

        val ptr = native.ailake_search_json(requestJson) ?: return emptyList()

        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) return emptyList()
            @Suppress("UNCHECKED_CAST")
            (resp["results"] as? List<Map<String, Any>> ?: emptyList()).map { m ->
                SearchRow(
                    rowId = (m["row_id"] as Number).toLong(),
                    distance = (m["distance"] as Number).toFloat(),
                    filePath = m["file_path"] as String,
                )
            }
        } catch (e: Exception) {
            emptyList()
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }
}
