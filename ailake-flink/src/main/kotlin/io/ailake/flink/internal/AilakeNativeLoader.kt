// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink.internal

import com.fasterxml.jackson.databind.JsonNode
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

    private const val AILAKE_EXPECTED_MAJOR = "0"

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
        }.onSuccess { native ->
            val v = native.ailake_version()
            val major = v.substringBefore('.')
            if (major != AILAKE_EXPECTED_MAJOR)
                log.warn(
                    "[ailake] Version mismatch: loaded ailake-jni {} but expected major {}. " +
                    "Search results may be incorrect.",
                    v, AILAKE_EXPECTED_MAJOR
                )
            else
                log.info("[ailake] Native library libailake_jni {} loaded (path={})",
                    v, explicitPath ?: "JNA default search path")
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
        // ailake_search_json's Req struct (ailake-jni/src/lib.rs) has always
        // accepted pruning_threshold too — found auditing this plugin against
        // trino-plugin/spark-plugin, which both got this same fix moments
        // earlier this session (Fase 21). null = server default (no pruning).
        pruningThreshold: Float? = null,
        partitionFilter: String? = null,
        hybridText: String? = null,
        textColumn: String = "chunk_text",
        bm25Weight: Float = 0.5f,
        catalogOpts: Map<String, String> = emptyMap(),
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
        if (pruningThreshold != null) payload["pruning_threshold"] = pruningThreshold
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        if (hybridText != null) {
            payload["hybrid_text"]  = hybridText
            payload["text_column"]  = textColumn
            payload["bm25_weight"]  = bm25Weight
        }
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_search_json(req)
            ?: throw RuntimeException("ailake_search_json returned null for table=$namespace.$table")
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
        textColumns: List<String> = listOf("chunk_text"),
        partitionFilter: String? = null,
        catalogOpts: Map<String, String> = emptyMap(),
    ): List<SearchResultItem> {
        val payload = mutableMapOf<String, Any>(
            "warehouse"    to warehouse,
            "namespace"    to namespace,
            "table"        to table,
            "query_text"   to queryText,
            "top_k"        to topK,
            "text_columns" to textColumns,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_search_text_json(req)
            ?: throw RuntimeException("ailake_search_text_json returned null for table=$namespace.$table")
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
        catalogOpts: Map<String, String> = emptyMap(),
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
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_search_multimodal_json(req)
            ?: throw RuntimeException("ailake_search_multimodal_json returned null for table=$namespace.$table")
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

    // ── Scan (Fase 11 — search + full-row fetch, no JOIN needed) ──────────────

    /** One column of a [scan] response — `type` is one of the tags `ailake_scan_json` emits: `int64`, `float32`, `float64`, `utf8`, `bool`, `list_float32`. */
    data class ScanColumn(val name: String, val type: String)

    data class ScanResponse(
        val ok: Boolean,
        val schema: List<ScanColumn> = emptyList(),
        val num_rows: Int = 0,
        val columns: Map<String, List<Any?>> = emptyMap(),
        val error: String? = null,
    )

    /**
     * Vector search + full-row fetch in one native call — closes the "SQL search only returns
     * row_id/distance/file_path" gap: previously the only way to get real columns
     * (chunk_text, document_title, ...) back from a search was a manual JOIN against a
     * separately-registered Iceberg table pointing at the same physical location. Result is
     * columnar; every stored column comes back (vector column decoded to nested float lists),
     * plus a trailing `_distance` column — no column-subset filter on the native side.
     */
    fun scan(
        warehouse: String,
        namespace: String,
        table: String,
        vecCol: String,
        dim: Int,
        query: FloatArray,
        topK: Int = 10,
        partitionFilter: String? = null,
        catalogOpts: Map<String, String> = emptyMap(),
    ): ScanResponse {
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse,
            "namespace" to namespace,
            "table" to table,
            "vec_col" to vecCol,
            "dim" to dim,
            "query" to query.toList(),
            "top_k" to topK,
        )
        if (partitionFilter != null) payload["partition_filter"] = partitionFilter
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_scan_json(req)
            ?: throw RuntimeException("ailake_scan_json returned null for table=$namespace.$table")
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<ScanResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] ailake_scan_json returned error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_scan_json error: ${resp.error}")
            }
            log.debug("[ailake] scan OK table={}.{} top_k={} num_rows={}", namespace, table, topK, resp.num_rows)
            resp
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
        ftsColumns: List<String> = emptyList(),
        ftsTokenizer: String = "default",
        hnswM: Int? = null,
        hnswEfConstruction: Int? = null,
        preNormalize: Boolean = false,
        deferred: Boolean = false,
        columns: Map<String, List<String>> = emptyMap(),
        catalogOpts: Map<String, String> = emptyMap(),
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
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_write_batch_json(req)
            ?: throw RuntimeException("ailake_write_batch_json returned null for table=$namespace.$table")
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
     * One vector column in a multi-column (Phase 8 multimodal) write batch — e.g. text +
     * image embeddings on the same row, each with its own HNSW index. See [writeBatchMulti].
     */
    data class VectorColSpec(
        val column: String,
        val dim: Int,
        val metric: String = "cosine",
        val precision: String = "f16",
        val modality: String? = null,
    )

    /**
     * Write a batch of rows with N independent vector columns (Phase 8 multimodal). Each
     * column gets its own HNSW section in the same AI-Lake file. Was already exposed from
     * Spark (`ailakeWriteMulti`) but had no wrapper here — same "dead capability" gap as
     * DELETE/schema evolution before them, closed the same way. Throws on native error,
     * matching [writeBatch]'s existing contract.
     *
     * @param vectorColumns  one spec per vector column, paired with its per-row embeddings
     *                       (`embeddings[i]` has one entry per row, in the same order as `ids`).
     *                       First entry is primary (used for geometric pruning in the manifest).
     */
    fun writeBatchMulti(
        warehouse: String,
        namespace: String,
        table: String,
        ids: LongArray,
        vectorColumns: List<Pair<VectorColSpec, Array<FloatArray>>>,
        embeddingModel: String? = null,
        formatVersion: Int = 2,
        ftsColumns: List<String> = emptyList(),
        ftsTokenizer: String = "default",
        deferred: Boolean = false,
        columns: Map<String, List<String>> = emptyMap(),
        catalogOpts: Map<String, String> = emptyMap(),
    ): Long {
        require(vectorColumns.isNotEmpty()) { "vectorColumns must not be empty" }
        vectorColumns.forEach { (spec, embeddings) ->
            require(ids.size == embeddings.size) { "ids.size != embeddings.size for column '${spec.column}'" }
        }
        val vecColsPayload = vectorColumns.map { (spec, embeddings) ->
            val m = mutableMapOf<String, Any>(
                "col" to spec.column,
                "dim" to spec.dim,
                "metric" to spec.metric,
                "precision" to spec.precision,
                "embeddings" to embeddings.map { it.toList() },
            )
            if (spec.modality != null) m["modality"] = spec.modality
            m
        }
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse,
            "namespace" to namespace,
            "table" to table,
            "ids" to ids.toList(),
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
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_write_batch_multi_json(req)
            ?: throw RuntimeException("ailake_write_batch_multi_json returned null for table=$namespace.$table")
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<WriteResponse>(json)
            if (!resp.ok) {
                log.error("[ailake] ailake_write_batch_multi_json returned error for table={}.{}: {}", namespace, table, resp.error)
                throw RuntimeException("ailake_write_batch_multi_json error: ${resp.error}")
            }
            log.info("[ailake] writeBatchMulti OK table={}.{} rows={} snapshot_id={}", namespace, table, ids.size, resp.snapshot_id)
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
        catalogOpts: Map<String, String> = emptyMap(),
    ) {
        require(values.isNotEmpty()) { "values must not be empty" }
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse,
            "namespace" to namespace,
            "table"     to table,
            "column"    to column,
            "values"    to values,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_delete_where_json(req)
            ?: throw RuntimeException("ailake_delete_where_json returned null for table=$namespace.$table")
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
        catalogOpts: Map<String, String> = emptyMap(),
    ): Int {
        require(addCols.isNotEmpty() || renameCols.isNotEmpty()) {
            "at least one of addCols or renameCols must be non-empty"
        }
        val rootNode = mapper.createObjectNode()
        rootNode.put("warehouse", warehouse)
        rootNode.put("namespace", namespace)
        rootNode.put("table", table)

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
        val req = mapper.writeValueAsString(rootNode)

        val ptr = lib.ailake_evolve_schema_json(req)
            ?: throw RuntimeException("ailake_evolve_schema_json returned null for table=$namespace.$table")
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

    /**
     * Create an empty AI-Lake table.
     * Returns true on success.
     * Throws [RuntimeException] on native error.
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
        val req = mapper.writeValueAsString(payload)
        val ptr = lib.ailake_create_table_json(req)
            ?: throw RuntimeException("ailake_create_table_json returned null for table=$namespace.$table")
        return try {
            val json = ptr.getString(0)
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.error("[ailake] ailake_create_table_json returned error for table={}.{}: {}", namespace, table, resp["error"])
                throw RuntimeException("ailake_create_table_json error: ${resp["error"]}")
            }
            log.info("[ailake] createTable OK table={}.{}", namespace, table)
            true
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    // ── decayMemories / migrate / deleteRows / addVectorColumn /
    // backfillVectorColumn / estimate — closes a gap found auditing
    // trino-plugin (same for all 3 JVM plugins): none of these 6 had a
    // C-ABI path at all in ailake-jni until now, unlike createTable (which
    // existed but was simply unwired). ailake-go/ailake-cpp reach these only
    // by shelling out to the ailake CLI binary.

    /** Recomputes recency_weight for every row (Phase 9 agent memory). Returns files_updated. Throws on native error. */
    fun decayMemories(
        warehouse: String,
        namespace: String,
        table: String,
        lambda: Float,
        catalogOpts: Map<String, String> = emptyMap(),
    ): Int {
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table, "lambda" to lambda,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = lib.ailake_decay_memories_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_decay_memories_json returned null for table=$namespace.$table")
        return try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                throw RuntimeException("ailake_decay_memories_json error: ${resp["error"]}")
            }
            val filesUpdated = (resp["files_updated"] as Number).toInt()
            log.info("[ailake] decayMemories OK table={}.{} files_updated={}", namespace, table, filesUpdated)
            filesUpdated
        } finally {
            lib.ailake_free_string(ptr)
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
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "old_column" to oldColumn, "new_column" to newColumn, "text_column" to textColumn,
            "embed_cmd" to embedCmd, "strategy" to strategy, "batch_size" to batchSize,
        )
        if (modelName != null) payload["model_name"] = modelName
        if (modelVersion != null) payload["model_version"] = modelVersion
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = lib.ailake_migrate_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_migrate_json returned null for table=$namespace.$table")
        try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                throw RuntimeException("ailake_migrate_json error: ${resp["error"]}")
            }
            log.info("[ailake] migrate OK table={}.{} {}→{}", namespace, table, oldColumn, newColumn)
        } finally {
            lib.ailake_free_string(ptr)
        }
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
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "file" to file, "row_ids" to rowIds,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = lib.ailake_delete_rows_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_delete_rows_json returned null for table=$namespace.$table")
        try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                throw RuntimeException("ailake_delete_rows_json error: ${resp["error"]}")
            }
            log.info("[ailake] deleteRows OK table={}.{} file={} rows={}", namespace, table, file, rowIds.size)
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    /** Adds a new vector column to the schema (metadata-only, no backfill). Returns the new schema_id. Throws on failure. */
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
    ): Int {
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "column" to column, "dim" to dim, "metric" to metric, "precision" to precision,
            "pre_normalize" to preNormalize,
        )
        if (hnswM != null) payload["hnsw_m"] = hnswM
        if (hnswEfConstruction != null) payload["hnsw_ef_construction"] = hnswEfConstruction
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = lib.ailake_add_vector_column_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_add_vector_column_json returned null for table=$namespace.$table")
        return try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                throw RuntimeException("ailake_add_vector_column_json error: ${resp["error"]}")
            }
            val schemaId = (resp["new_schema_id"] as Number).toInt()
            log.info("[ailake] addVectorColumn OK table={}.{} column={} new_schema_id={}", namespace, table, column, schemaId)
            schemaId
        } finally {
            lib.ailake_free_string(ptr)
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
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
            "column" to column, "text_column" to textColumn, "embed_cmd" to embedCmd,
            "batch_size" to batchSize,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = lib.ailake_backfill_vector_column_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_backfill_vector_column_json returned null for table=$namespace.$table")
        try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                throw RuntimeException("ailake_backfill_vector_column_json error: ${resp["error"]}")
            }
            log.info("[ailake] backfillVectorColumn OK table={}.{} column={}", namespace, table, column)
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    /** Storage/index size estimates for a hypothetical table — pure math, no warehouse/catalog needed. Returns the raw JSON response as a Map. Throws on failure. */
    fun estimate(
        rows: Long,
        dim: Int,
        hnswM: Int = 16,
        pqM: Int? = null,
    ): Map<String, Any> {
        val payload = mutableMapOf<String, Any>("rows" to rows, "dim" to dim, "hnsw_m" to hnswM)
        if (pqM != null) payload["pq_m"] = pqM
        val ptr = lib.ailake_estimate_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_estimate_json returned null")
        return try {
            mapper.readValue<Map<String, Any>>(ptr.getString(0))
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    /**
     * Table metadata: current snapshot, file/row/size counts, index status
     * breakdown, and "foreign" files (written by a generic Iceberg engine —
     * no AI-Lake centroid/HNSW). Mirrors `ailake info --format json`.
     * Found in a later audit pass: had zero binding coverage anywhere
     * outside the CLI and the Airflow provider. Returns the raw JSON
     * response as a Map. Throws on failure.
     */
    fun info(
        warehouse: String,
        namespace: String,
        table: String,
        catalogOpts: Map<String, String> = emptyMap(),
    ): Map<String, Any> {
        val payload = mutableMapOf<String, Any>(
            "warehouse" to warehouse, "namespace" to namespace, "table" to table,
        )
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val ptr = lib.ailake_info_json(mapper.writeValueAsString(payload))
            ?: throw RuntimeException("ailake_info_json returned null for table=$namespace.$table")
        return try {
            val resp = mapper.readValue<Map<String, Any>>(ptr.getString(0))
            if (resp["ok"] != true) {
                throw RuntimeException("ailake_info_json error: ${resp["error"]}")
            }
            log.info("[ailake] info OK table={}.{}", namespace, table)
            resp
        } finally {
            lib.ailake_free_string(ptr)
        }
    }

    /**
     * Compact small files in an AI-Lake table.
     *
     * @param minFiles          minimum eligible files to trigger compaction (default 4)
     * @param targetSizeBytes   files smaller than this are candidates (default 128 MiB)
     * @param maxFilesPerPass   max files merged per run (default 20)
     * @param deferred          build index in background when true (default false)
     * @return number of files compacted (0 when nothing to compact)
     */
    fun compact(
        warehouse: String,
        namespace: String,
        table: String,
        minFiles: Int = 4,
        targetSizeBytes: Long = 128L * 1024 * 1024,
        maxFilesPerPass: Int = 20,
        deferred: Boolean = false,
        catalogOpts: Map<String, String> = emptyMap(),
    ): Int {
        val payload = mutableMapOf<String, Any>(
            "warehouse"    to warehouse,
            "namespace"    to namespace,
            "table"        to table,
            "min_files"    to minFiles,
            "target_size_bytes"  to targetSizeBytes,
            "max_files_per_pass" to maxFilesPerPass,
        )
        if (deferred) payload["deferred"] = true
        if (catalogOpts.isNotEmpty()) payload.putAll(catalogOpts)
        val req = mapper.writeValueAsString(payload)

        val ptr = lib.ailake_compact_json(req)
            ?: throw RuntimeException("ailake_compact_json returned null for table=$namespace.$table")
        return try {
            val json = ptr.getString(0)
            @Suppress("UNCHECKED_CAST")
            val resp = mapper.readValue<Map<String, Any>>(json)
            if (resp["ok"] != true) {
                log.error("[ailake] compact failed for table={}.{}: {}", namespace, table, resp["error"])
                throw RuntimeException("ailake_compact_json error: ${resp["error"]}")
            }
            val n = (resp["files_compacted"] as? Number)?.toInt() ?: 0
            log.info("[ailake] compact OK table={}.{} files_compacted={}", namespace, table, n)
            n
        } finally {
            lib.ailake_free_string(ptr)
        }
    }
}
