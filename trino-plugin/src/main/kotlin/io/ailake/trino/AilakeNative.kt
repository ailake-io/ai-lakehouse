// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.databind.JsonNode
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

    /** One column of a [scan] response — `type` is one of the tags `ailake_scan_json` emits: `int64`, `float32`, `float64`, `utf8`, `bool`, `list_float32`. */
    data class ScanColumn(val name: String, val type: String)

    /** Result of [scan] — search + full-row fetch in one native call. Columnar, `_distance` always last. */
    data class ScanResult(
        val schema: List<ScanColumn> = emptyList(),
        val numRows: Int = 0,
        val columns: Map<String, List<Any?>> = emptyMap(),
    )

    /** Partition field definition for multi-column partition specs (Phase K). */
    data class PartitionFieldDef(val column: String, val transform: String, val columnType: String)

    /** Column addition request for schema evolution. */
    data class AddColReq(val name: String, val colType: String, val initialDefault: String? = null)

    /** Column rename request for schema evolution. */
    data class RenameColReq(val from: String, val to: String)

    private interface Lib : Library {
        /** Returns ailake-jni version string. Static — do NOT free this pointer. */
        fun ailake_version(): String

        /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
        fun ailake_search_json(requestJson: String): Pointer?

        /** Cross-modal RRF. Returns `{"ok":true,"results":[{"row_id":N,"rrf_score":F,"file_path":"..."}]}`. Caller must free. */
        fun ailake_search_multimodal_json(requestJson: String): Pointer?

        /** Search + full-row fetch. Returns `{"ok":true,"schema":[...],"num_rows":N,"columns":{...}}`. Caller must free. */
        fun ailake_scan_json(requestJson: String): Pointer?

        /** JSON-envelope write. Returns `{"ok":true,"snapshot_id":N}`. Caller must free. */
        fun ailake_write_batch_json(requestJson: String): Pointer?

        /** Logical delete via equality delete file. Returns `{"ok":true}`. Caller must free. */
        fun ailake_delete_where_json(requestJson: String): Pointer?

        /** Schema evolution. Returns `{"ok":true,"new_schema_id":N}`. Caller must free. */
        fun ailake_evolve_schema_json(requestJson: String): Pointer?

        /** Full-text search (Tantivy or BM25 fallback). Returns `{"ok":true,"results":[...]}`. Caller must free. */
        fun ailake_search_text_json(requestJson: String): Pointer?

        /** Compact small files. Returns `{"ok":true,"files_compacted":N}`. Caller must free. */
        fun ailake_compact_json(requestJson: String): Pointer?

        fun ailake_free_string(ptr: Pointer?)
    }

    private const val AILAKE_EXPECTED_MAJOR = "0"

    private val lib: Lib? by lazy {
        val explicitPath =
            System.getProperty("ailake.native.lib")
                ?: System.getenv("AILAKE_NATIVE_LIB")
        runCatching {
            if (explicitPath != null) Native.load(explicitPath, Lib::class.java) as Lib
            else Native.load("ailake_jni", Lib::class.java) as Lib
        }
            .onSuccess { loaded ->
                val version = loaded.ailake_version()
                val major = version.substringBefore('.')
                if (major != AILAKE_EXPECTED_MAJOR)
                    log.warn(
                        "[ailake] Version mismatch: loaded ailake-jni {} but expected major {}. " +
                        "Search results may be incorrect.", version, AILAKE_EXPECTED_MAJOR
                    )
                else
                    log.info("[ailake] Native library libailake_jni {} loaded (path={})",
                        version, explicitPath ?: "JNA default search path")
            }
            .onFailure {
                log.warn(
                    "[ailake] Native library libailake_jni not found — vector search disabled. " +
                    "Set ailake.native.lib system property or AILAKE_NATIVE_LIB env var. Error: ${it.message}"
                )
            }
            .getOrNull()
    }

    private val mapper = jacksonObjectMapper()

    /**
     * Write a batch of rows to an AI-Lake table via the native library.
     * Returns the snapshot_id on success, null on failure.
     *
     * @param partitionFields      multi-column partition spec (Phase K); empty = single-value partition_by/partition_value
     * @param formatVersion        Iceberg format version; 2 (default) or 3
     * @param ftsColumns           text columns to embed as Tantivy FTS index; empty = no FTS (default)
     * @param ftsTokenizer         Tantivy tokenizer name; default "default"
     * @param hnswM                HNSW graph connectivity (M). null = use table default.
     * @param hnswEfConstruction   HNSW ef_construction. null = use table default.
     * @param preNormalize         Normalize vectors to unit L2 at write time (recommended for cosine).
     * @param deferred             Build index asynchronously. Parquet committed immediately.
     * @param columns              Extra string columns sent with the batch for FTS indexing.
     *                             Map from column name to per-row string values.
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
        ftsColumns: List<String> = emptyList(),
        ftsTokenizer: String = "default",
        hnswM: Int? = null,
        hnswEfConstruction: Int? = null,
        preNormalize: Boolean = false,
        deferred: Boolean = false,
        columns: Map<String, List<String>> = emptyMap(),
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
        if (ftsColumns.isNotEmpty()) {
            payload["fts_columns"]   = ftsColumns
            payload["fts_tokenizer"] = ftsTokenizer
        }
        if (hnswM != null)              payload["hnsw_m"]              = hnswM
        if (hnswEfConstruction != null) payload["hnsw_ef_construction"] = hnswEfConstruction
        if (preNormalize)               payload["pre_normalize"]        = true
        if (deferred)                   payload["deferred"]             = true
        if (columns.isNotEmpty())       payload["columns"]              = columns
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

        val rootNode = mapper.createObjectNode()
        rootNode.put("warehouse", tableUri)
        rootNode.put("namespace", namespace)
        rootNode.put("table", tableName)

        val addArray = mapper.createArrayNode()
        for (ac in addCols) {
            val colNode = mapper.createObjectNode()
            colNode.put("name", ac.name)
            colNode.put("type", ac.colType)
            if (ac.initialDefault != null) {
                // parse as raw JSON so null/0/0.0/"string" embed correctly without re-quoting
                colNode.set<JsonNode>("initial_default", mapper.readTree(ac.initialDefault))
            }
            addArray.add(colNode)
        }
        rootNode.set<JsonNode>("add_columns", addArray)

        val renArray = mapper.createArrayNode()
        for (rc in renameCols) {
            val renNode = mapper.createObjectNode()
            renNode.put("from", rc.from)
            renNode.put("to", rc.to)
            renArray.add(renNode)
        }
        rootNode.set<JsonNode>("rename_columns", renArray)
        val requestJson = mapper.writeValueAsString(rootNode)

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

    /**
     * Full-text search via Tantivy (fast path when AILK_FTS present) or BM25 brute-force.
     * Returns empty on library absence or error.
     *
     * @param textColumns  columns to search; defaults to ["chunk_text"]
     * @param topK         number of results to return
     */
    fun searchText(
        tableUri: String,
        namespace: String,
        tableName: String,
        queryText: String,
        textColumns: List<String> = listOf("chunk_text"),
        topK: Int = 10,
        partitionFilter: String? = null,
    ): List<SearchRow> {
        val native = lib ?: return emptyList()
        if (queryText.isEmpty()) return emptyList()

        val payload = mutableMapOf<String, Any>(
            "warehouse"    to tableUri,
            "namespace"    to namespace,
            "table"        to tableName,
            "query_text"   to queryText,
            "text_columns" to textColumns,
            "top_k"        to topK,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_search_text_json(requestJson) ?: run {
            log.warn("[ailake] ailake_search_text_json returned null for tableUri={}", tableUri)
            return emptyList()
        }
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] searchText ok=false for tableUri={}: {}", tableUri, resp["error"])
                return emptyList()
            }
            @Suppress("UNCHECKED_CAST")
            (resp["results"] as? List<Map<String, Any>> ?: emptyList()).map { m ->
                SearchRow(
                    rowId    = (m["row_id"] as Number).toLong(),
                    distance = (m["distance"] as Number).toFloat(),
                    filePath = m["file_path"] as String,
                )
            }
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse searchText response for tableUri={}: {}", tableUri, e.message, e)
            emptyList()
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
        namespace: String = "default",
        tableName: String = "",
    ): List<MultimodalSearchRow> {
        val native = lib ?: return emptyList()
        if (queries.isEmpty()) return emptyList()

        val effectiveTable = tableName.ifBlank { tableUri.trimEnd('/').substringAfterLast('/') }
        val queriesArr = queries.map { (col, q, w) ->
            mapOf("col" to col, "query" to q, "weight" to w, "dim" to 0)
        }
        val payload = mutableMapOf<String, Any>(
            "warehouse" to tableUri,
            "namespace" to namespace,
            "table"     to effectiveTable,
            "queries"   to queriesArr,
            "top_k"     to topK,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        val requestJson = mapper.writeValueAsString(payload)

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
     * @param tableUri       path/URI of the AI-Lake table root
     * @param queryBytes     Base64-encoded little-endian f32 array
     * @param topK           number of nearest neighbors
     * @param hybridText     when non-null, enables hybrid BM25+vector RRF fusion
     * @param textColumn     Parquet column for BM25 scoring (default "chunk_text")
     * @param bm25Weight     BM25 weight in RRF (0.0 = pure vector, 1.0 = pure BM25)
     * @param vectorColumn   vector column name to search — must match the column the
     *                       table was written with (defaults to "embedding", the native
     *                       side's own default, but should be passed explicitly whenever
     *                       the caller knows the catalog's configured vector-column)
     */
    fun search(
        tableUri: String,
        queryBytes: String,
        topK: Int,
        partitionFilter: String? = null,
        hybridText: String? = null,
        textColumn: String = "chunk_text",
        bm25Weight: Float = 0.5f,
        namespace: String = "default",
        tableName: String = "",
        vectorColumn: String = "embedding",
    ): List<SearchRow> {
        val native = lib ?: return emptyList()
        if (queryBytes.isBlank()) return emptyList()

        val effectiveTable = tableName.ifBlank { tableUri.trimEnd('/').substringAfterLast('/') }
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
            "namespace" to namespace,
            "table" to effectiveTable,
            "vec_col" to vectorColumn,
            "query" to floats,
            "dim" to floats.size,
            "top_k" to topK,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        if (hybridText != null) {
            payload["hybrid_text"]  = hybridText
            payload["text_column"]  = textColumn
            payload["bm25_weight"]  = bm25Weight
        }
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

    /**
     * Vector search + full-row fetch in one native call (`ailake_scan_json`) — closes the
     * "SQL search only returns row_id/distance/file_path" gap: previously the only way to get
     * real columns (chunk_text, document_title, ...) back from a search was a manual `JOIN`
     * against a separately-registered Iceberg table pointing at the same physical location.
     * Result is columnar; every stored column comes back (vector column decoded to
     * `list_float32`), plus a trailing `_distance` column — there's no column-subset filter on
     * the native side, it always returns the full row width.
     */
    fun scan(
        tableUri: String,
        queryBytes: String,
        topK: Int,
        vectorColumn: String = "embedding",
        partitionFilter: String? = null,
        namespace: String = "default",
        tableName: String = "",
    ): ScanResult {
        val native = lib ?: return ScanResult()
        if (queryBytes.isBlank()) return ScanResult()

        val effectiveTable = tableName.ifBlank { tableUri.trimEnd('/').substringAfterLast('/') }
        val floats = runCatching {
            val bytes = Base64.getDecoder().decode(queryBytes)
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            (0 until bytes.size / 4).map { buf.getFloat() }
        }.getOrElse {
            log.error("[ailake] Failed to decode Base64 query vector for tableUri={}: {}", tableUri, it.message)
            return ScanResult()
        }
        if (floats.isEmpty()) return ScanResult()

        val payload = mutableMapOf<String, Any>(
            "warehouse" to tableUri,
            "namespace" to namespace,
            "table" to effectiveTable,
            "vec_col" to vectorColumn,
            "query" to floats,
            "dim" to floats.size,
            "top_k" to topK,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_scan_json(requestJson) ?: run {
            log.warn("[ailake] ailake_scan_json returned null pointer for tableUri={}", tableUri)
            return ScanResult()
        }
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] Native scan returned ok=false for tableUri={}: {}", tableUri, resp["error"])
                return ScanResult()
            }
            @Suppress("UNCHECKED_CAST")
            val schemaList = (resp["schema"] as? List<Map<String, Any>> ?: emptyList()).map { m ->
                ScanColumn(name = m["name"] as String, type = m["type"] as String)
            }
            val numRows = (resp["num_rows"] as? Number)?.toInt() ?: 0
            @Suppress("UNCHECKED_CAST")
            val columnsMap = resp["columns"] as? Map<String, List<Any?>> ?: emptyMap()
            ScanResult(schemaList, numRows, columnsMap)
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse native scan response for tableUri={}: {}", tableUri, e.message, e)
            ScanResult()
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }

    /**
     * Compact small files in an AI-Lake table.
     *
     * @return number of files compacted (0 = nothing to compact), or null when the library is absent.
     */
    fun compact(
        tableUri: String,
        namespace: String,
        tableName: String,
        minFiles: Int = 4,
        targetSizeBytes: Long = 128L * 1024 * 1024,
        maxFilesPerPass: Int = 20,
        deferred: Boolean = false,
    ): Int? {
        val native = lib ?: return null
        val payload = mutableMapOf<String, Any>(
            "warehouse"          to tableUri,
            "namespace"          to namespace,
            "table"              to tableName,
            "min_files"          to minFiles,
            "target_size_bytes"  to targetSizeBytes,
            "max_files_per_pass" to maxFilesPerPass,
        )
        if (deferred) payload["deferred"] = true
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_compact_json(requestJson) ?: run {
            log.warn("[ailake] ailake_compact_json returned null for table={}.{}", namespace, tableName)
            return null
        }
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.warn("[ailake] compact ok=false for table={}.{}: {}", namespace, tableName, resp["error"])
                return null
            }
            val n = (resp["files_compacted"] as? Number)?.toInt() ?: 0
            log.info("[ailake] compact OK table={}.{} files_compacted={}", namespace, tableName, n)
            n
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse compact response for table={}.{}: {}", namespace, tableName, e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }
}
