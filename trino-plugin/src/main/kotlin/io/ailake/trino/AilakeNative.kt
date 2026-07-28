// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
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

    /**
     * Partition field definition for multi-column partition specs (Phase K).
     *
     * Explicit @JsonCreator/@param:/@get: annotations (not just plain field
     * names) — this is embedded as a List<PartitionFieldDef> field inside
     * [AilakeIngestTableHandle], which round-trips through Trino's internal
     * cross-process JSON codec (ObjectMapperProvider disables
     * MapperFeature.AUTO_DETECT_GETTERS/FIELDS globally). See the note atop
     * VectorScanHandles.kt.
     */
    data class PartitionFieldDef @JsonCreator constructor(
        @param:JsonProperty("column") @get:JsonProperty("column") val column: String,
        @param:JsonProperty("transform") @get:JsonProperty("transform") val transform: String,
        @param:JsonProperty("columnType") @get:JsonProperty("columnType") val columnType: String,
    )

    /** Column addition request for schema evolution. */
    data class AddColReq(val name: String, val colType: String, val initialDefault: String? = null)

    /** Column rename request for schema evolution. */
    data class RenameColReq(val from: String, val to: String)

    /**
     * One vector column in a multi-column (Phase 8 multimodal) write batch — e.g. text +
     * image embeddings on the same row, each with its own HNSW index. See [writeBatchMulti].
     */
    data class VectorColSpec @JsonCreator constructor(
        @param:JsonProperty("column") @get:JsonProperty("column") val column: String,
        @param:JsonProperty("dim") @get:JsonProperty("dim") val dim: Int,
        @param:JsonProperty("metric") @get:JsonProperty("metric") val metric: String = "cosine",
        @param:JsonProperty("precision") @get:JsonProperty("precision") val precision: String = "f16",
        @param:JsonProperty("modality") @get:JsonProperty("modality") val modality: String? = null,
    )

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

        /** Multi-column (Phase 8 multimodal) write. Returns `{"ok":true,"snapshot_id":N}`. Caller must free. */
        fun ailake_write_batch_multi_json(requestJson: String): Pointer?

        /** Logical delete via equality delete file. Returns `{"ok":true}`. Caller must free. */
        fun ailake_delete_where_json(requestJson: String): Pointer?

        /** Schema evolution. Returns `{"ok":true,"new_schema_id":N}`. Caller must free. */
        fun ailake_evolve_schema_json(requestJson: String): Pointer?

        /** Full-text search (Tantivy or BM25 fallback). Returns `{"ok":true,"results":[...]}`. Caller must free. */
        fun ailake_search_text_json(requestJson: String): Pointer?

        /** Compact small files. Returns `{"ok":true,"files_compacted":N}`. Caller must free. */
        fun ailake_compact_json(requestJson: String): Pointer?

        /** Create an empty AI-Lake table. Returns `{"ok":true}`. Caller must free. */
        fun ailake_create_table_json(requestJson: String): Pointer?

        /** Recompute recency_weight (Phase 9 agent memory). Returns `{"ok":true,"files_updated":N}`. Caller must free. */
        fun ailake_decay_memories_json(requestJson: String): Pointer?

        /** Re-embed a column via an external embed command. Returns `{"ok":true}`. Caller must free. */
        fun ailake_migrate_json(requestJson: String): Pointer?

        /** Position-level delete via Iceberg V3 Deletion Vectors. Returns `{"ok":true}`. Caller must free. */
        fun ailake_delete_rows_json(requestJson: String): Pointer?

        /** Add a vector column to an existing table's schema (metadata-only). Returns `{"ok":true,"new_schema_id":N}`. Caller must free. */
        fun ailake_add_vector_column_json(requestJson: String): Pointer?

        /** Backfill embeddings for a column added via add_vector_column. Returns `{"ok":true}`. Caller must free. */
        fun ailake_backfill_vector_column_json(requestJson: String): Pointer?

        /** Storage/index size estimates — pure math, no I/O. Returns `{"ok":true,"estimates":[...]}`. Caller must free. */
        fun ailake_estimate_json(requestJson: String): Pointer?

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
        catalogOpts: Map<String, String> = emptyMap(),
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
        if (catalogOpts.isNotEmpty())   payload.putAll(catalogOpts)
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_write_batch_json(requestJson) ?: run {
            log.warn("[ailake] ailake_write_batch_json returned null pointer for table={}", tableName)
            return null
        }

        val resp: Map<String, Any>? = try {
            val json = ptr.getString(0)
            mapper.readValue<Map<String, Any>>(json)
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse writeBatch response for table={}: {}", tableName, e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
        if (resp == null) return null
        // A real backend rejection (e.g. NaN/Infinity embeddings, top_k over the cap on
        // other calls) must fail the write visibly — silently returning null here is
        // indistinguishable from "malformed response" and gets treated as a successful
        // (if snapshot-less) write by the caller, silently dropping the batch.
        if (resp["ok"] != true) {
            throw RuntimeException("ailake writeBatch failed for table=$tableName: ${resp["error"]}")
        }
        return (resp["snapshot_id"] as? Number)?.toLong()
    }

    /**
     * Write a batch of rows with N independent vector columns (Phase 8 multimodal — e.g.
     * text + image embeddings on the same row, searchable via [searchMultimodal]). Each
     * column gets its own HNSW section in the same AI-Lake file. Was already exposed from
     * Spark (`ailakeWriteMulti`) but had no wrapper here — same "dead capability" gap as
     * DELETE/ALTER TABLE/compact before them, closed the same way.
     *
     * @param vectorColumns  one spec per vector column, paired with its per-row embeddings
     *                       (`embeddings[i]` has one entry per row, in the same order as `ids`).
     *                       First entry is primary (used for geometric pruning in the manifest).
     */
    fun writeBatchMulti(
        tableUri: String,
        namespace: String,
        tableName: String,
        ids: List<Long>,
        vectorColumns: List<Pair<VectorColSpec, List<List<Float>>>>,
        embeddingModel: String? = null,
        formatVersion: Int = 2,
        ftsColumns: List<String> = emptyList(),
        ftsTokenizer: String = "default",
        deferred: Boolean = false,
        columns: Map<String, List<String>> = emptyMap(),
        catalogOpts: Map<String, String> = emptyMap(),
    ): Long? {
        val native = lib ?: return null
        if (ids.isEmpty() || vectorColumns.isEmpty()) return null

        val vecColsPayload = vectorColumns.map { (spec, embeddings) ->
            val m = mutableMapOf<String, Any>(
                "col" to spec.column,
                "dim" to spec.dim,
                "metric" to spec.metric,
                "precision" to spec.precision,
                "embeddings" to embeddings,
            )
            if (spec.modality != null) m["modality"] = spec.modality
            m
        }
        val payload = mutableMapOf<String, Any>(
            "warehouse" to tableUri,
            "namespace" to namespace,
            "table" to tableName,
            "ids" to ids,
            "vector_columns" to vecColsPayload,
            "format_version" to formatVersion,
        )
        if (embeddingModel != null) payload["embedding_model"] = embeddingModel
        if (ftsColumns.isNotEmpty()) {
            payload["fts_columns"] = ftsColumns
            payload["fts_tokenizer"] = ftsTokenizer
        }
        if (deferred) payload["deferred"] = true
        if (columns.isNotEmpty()) payload["columns"] = columns
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_write_batch_multi_json(requestJson) ?: run {
            log.warn("[ailake] ailake_write_batch_multi_json returned null pointer for table={}", tableName)
            return null
        }
        val resp: Map<String, Any>? = try {
            val json = ptr.getString(0)
            mapper.readValue<Map<String, Any>>(json)
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse writeBatchMulti response for table={}: {}", tableName, e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
        if (resp == null) return null
        // See writeBatch's identical comment: a real backend rejection must fail visibly.
        if (resp["ok"] != true) {
            throw RuntimeException("ailake writeBatchMulti failed for table=$tableName: ${resp["error"]}")
        }
        return (resp["snapshot_id"] as? Number)?.toLong()
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
        catalogOpts: Map<String, String> = emptyMap(),
    ): Boolean {
        if (values.isEmpty()) return false
        val native = lib ?: return false

        val payload = mutableMapOf<String, Any>(
            "warehouse" to tableUri,
            "namespace" to namespace,
            "table"     to tableName,
            "column"    to column,
            "values"    to values,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
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
        catalogOpts: Map<String, String> = emptyMap(),
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
        for ((k, v) in catalogOpts) rootNode.put(k, v)
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
        catalogOpts: Map<String, String> = emptyMap(),
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
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
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
        catalogOpts: Map<String, String> = emptyMap(),
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
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
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
        catalogOpts: Map<String, String> = emptyMap(),
        // ailake_search_json's Req struct (ailake-jni/src/lib.rs) accepts both —
        // server defaults (ef_search=50, pruning_threshold=infinity/no pruning)
        // apply when null. Regression: this was never forwarded from Trino
        // (or Spark — same gap, same fix) despite the JNI contract supporting
        // it since before either plugin existed; Flink already had it
        // (`search.ef` DDL option).
        efSearch: Int? = null,
        pruningThreshold: Float? = null,
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
        if (efSearch != null) payload["ef_search"] = efSearch
        if (pruningThreshold != null) payload["pruning_threshold"] = pruningThreshold
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
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
        catalogOpts: Map<String, String> = emptyMap(),
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
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
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
        catalogOpts: Map<String, String> = emptyMap(),
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
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
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

    /**
     * Create an empty AI-Lake table via the native library.
     * Returns true on success, false if the library is absent or the call fails.
     */
    fun createTable(
        warehouse: String,
        namespace: String,
        table: String,
        vectorColumn: String = "embedding",
        dim: Int = 1536,
        metric: String = "cosine",
        precision: String = "f16",
        formatVersion: Int = 2,
        catalogOpts: Map<String, String> = emptyMap(),
    ): Boolean {
        val native = lib ?: return false

        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse,
            "namespace" to namespace,
            "table" to table,
            "vector_column" to vectorColumn,
            "dim" to dim,
            "metric" to metric,
            "precision" to precision,
            "format_version" to formatVersion,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val requestJson = mapper.writeValueAsString(payload)

        val ptr = native.ailake_create_table_json(requestJson) ?: run {
            log.warn("[ailake] ailake_create_table_json returned null for table={}.{}", namespace, table)
            return false
        }
        val resp: Map<String, Any>? = try {
            val json = ptr.getString(0)
            mapper.readValue<Map<String, Any>>(json)
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse createTable response for table={}.{}: {}", namespace, table, e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
        if (resp == null) return false
        // Same as writeBatch/writeBatchMulti: a real backend rejection (e.g. the
        // table already exists) must fail visibly.
        if (resp["ok"] != true) {
            throw RuntimeException("ailake create_table failed for table=$namespace.$table: ${resp["error"]}")
        }
        log.info("[ailake] createTable OK table={}.{}", namespace, table)
        return true
    }

    // ── decay_memories / migrate / delete_rows / add_vector_column /
    // backfill_vector_column / estimate — closes a gap found auditing this
    // plugin: none of these 6 had a C-ABI path at all (ailake-jni itself never
    // exported them, unlike create_table which existed but was unwired). All
    // three JVM plugins (Spark/Trino/Flink) were equally affected since they
    // bind exclusively to ailake-jni via JNA — ailake-go/ailake-cpp reach
    // these only because they shell out to the ailake CLI binary instead.

    /** Recomputes recency_weight for every row (Phase 9 agent memory). Returns files_updated, or null on failure. */
    fun decayMemories(
        warehouse: String,
        namespace: String,
        table: String,
        lambda: Float,
        catalogOpts: Map<String, String> = emptyMap(),
    ): Int? {
        val native = lib ?: return null
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table, "lambda" to lambda,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = native.ailake_decay_memories_json(mapper.writeValueAsString(payload)) ?: run {
            log.warn("[ailake] ailake_decay_memories_json returned null for table={}.{}", namespace, table)
            return null
        }
        return try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                log.warn("[ailake] decayMemories failed for table={}.{}: {}", namespace, table, resp["error"])
                return null
            }
            (resp["files_updated"] as Number).toInt()
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse decayMemories response for table={}.{}: {}", namespace, table, e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }

    /** Re-embeds oldColumn → newColumn via an external embed command (spawned `sh -c embedCmd`, JSON stdin/stdout). Throws on failure. */
    fun migrate(
        warehouse: String,
        namespace: String,
        table: String,
        oldColumn: String,
        newColumn: String,
        textColumn: String,
        embedCmd: String,
        strategy: String = "atomic-replace",
        batchSize: Int = 10_000,
        modelName: String? = null,
        modelVersion: String? = null,
        catalogOpts: Map<String, String> = emptyMap(),
    ) {
        val native = lib ?: throw RuntimeException("ailake native library not loaded")
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "old_column" to oldColumn, "new_column" to newColumn, "text_column" to textColumn,
            "embed_cmd" to embedCmd, "strategy" to strategy, "batch_size" to batchSize,
        )
        if (modelName != null) payload["model_name"] = modelName
        if (modelVersion != null) payload["model_version"] = modelVersion
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = native.ailake_migrate_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_migrate_json returned null for table=$namespace.$table")
        val resp = try {
            mapper.readValue<Map<String, Any>>(ptr.getString(0))
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
        if (resp["ok"] != true) {
            throw RuntimeException("ailake migrate failed for table=$namespace.$table: ${resp["error"]}")
        }
        log.info("[ailake] migrate OK table={}.{} {}→{}", namespace, table, oldColumn, newColumn)
    }

    /** Deletes row positions from `file` via Iceberg V3 Deletion Vectors — different from [deleteWhere]'s equality predicate. Throws on failure. */
    fun deleteRows(
        warehouse: String,
        namespace: String,
        table: String,
        file: String,
        rowIds: List<Int>,
        catalogOpts: Map<String, String> = emptyMap(),
    ) {
        val native = lib ?: throw RuntimeException("ailake native library not loaded")
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "file" to file, "row_ids" to rowIds,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = native.ailake_delete_rows_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_delete_rows_json returned null for table=$namespace.$table")
        val resp = try {
            mapper.readValue<Map<String, Any>>(ptr.getString(0))
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
        if (resp["ok"] != true) {
            throw RuntimeException("ailake delete_rows failed for table=$namespace.$table: ${resp["error"]}")
        }
        log.info("[ailake] deleteRows OK table={}.{} file={} rows={}", namespace, table, file, rowIds.size)
    }

    /** Adds a new vector column to the schema (metadata-only, no backfill). Returns the new schema_id, or null on failure. */
    fun addVectorColumn(
        warehouse: String,
        namespace: String,
        table: String,
        column: String,
        dim: Int,
        metric: String = "cosine",
        precision: String = "f16",
        preNormalize: Boolean = false,
        hnswM: Int? = null,
        hnswEfConstruction: Int? = null,
        catalogOpts: Map<String, String> = emptyMap(),
    ): Int? {
        val native = lib ?: return null
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "column" to column, "dim" to dim, "metric" to metric, "precision" to precision,
            "pre_normalize" to preNormalize,
        )
        if (hnswM != null) payload["hnsw_m"] = hnswM
        if (hnswEfConstruction != null) payload["hnsw_ef_construction"] = hnswEfConstruction
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = native.ailake_add_vector_column_json(mapper.writeValueAsString(payload)) ?: run {
            log.warn("[ailake] ailake_add_vector_column_json returned null for table={}.{}", namespace, table)
            return null
        }
        return try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                log.warn("[ailake] addVectorColumn failed for table={}.{}: {}", namespace, table, resp["error"])
                return null
            }
            (resp["new_schema_id"] as Number).toInt()
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse addVectorColumn response for table={}.{}: {}", namespace, table, e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }

    /** Backfills embeddings for a column added via [addVectorColumn] — reads `textColumn`, embeds via `embedCmd`. Throws on failure. */
    fun backfillVectorColumn(
        warehouse: String,
        namespace: String,
        table: String,
        column: String,
        textColumn: String,
        embedCmd: String,
        batchSize: Int = 512,
        catalogOpts: Map<String, String> = emptyMap(),
    ) {
        val native = lib ?: throw RuntimeException("ailake native library not loaded")
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "column" to column, "text_column" to textColumn, "embed_cmd" to embedCmd,
            "batch_size" to batchSize,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = native.ailake_backfill_vector_column_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_backfill_vector_column_json returned null for table=$namespace.$table")
        val resp = try {
            mapper.readValue<Map<String, Any>>(ptr.getString(0))
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
        if (resp["ok"] != true) {
            throw RuntimeException("ailake backfill_vector_column failed for table=$namespace.$table: ${resp["error"]}")
        }
        log.info("[ailake] backfillVectorColumn OK table={}.{} column={}", namespace, table, column)
    }

    /** Storage/index size estimates for a hypothetical table — pure math, no warehouse/catalog needed. Returns the raw JSON response, or null on failure. */
    fun estimate(
        rows: Long,
        dim: Int,
        hnswM: Int = 16,
        pqM: Int? = null,
    ): Map<String, Any>? {
        val native = lib ?: return null
        val payload = mutableMapOf<String, Any>("rows" to rows, "dim" to dim, "hnsw_m" to hnswM)
        if (pqM != null) payload["pq_m"] = pqM
        val ptr = native.ailake_estimate_json(mapper.writeValueAsString(payload)) ?: return null
        return try {
            mapper.readValue<Map<String, Any>>(ptr.getString(0))
        } catch (e: Exception) {
            log.error("[ailake] Failed to parse estimate response: {}", e.message, e)
            null
        } finally {
            runCatching { native.ailake_free_string(ptr) }
        }
    }
}
