// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink.internal

import com.sun.jna.Library
import com.sun.jna.Pointer

/**
 * JNA mapping to the ailake-jni native library (libailake_jni.so / ailake_jni.dll).
 *
 * All `*const c_char` Rust parameters map to [String] on the JVM side (JNA handles
 * UTF-8 marshaling automatically).  Return values are [Pointer] — the caller must call
 * [ailake_free_string] after consuming each result.
 */
interface AilakeNativeLib : Library {

    /** Returns ailake-jni version string. Static — do NOT free this pointer. */
    fun ailake_version(): String

    /**
     * Perform ANN vector search via a JSON request envelope.
     *
     * Request JSON fields:
     *   warehouse         (String)  warehouse root path
     *   namespace         (String)  Iceberg namespace, default "default"
     *   table             (String)  table name
     *   vec_col           (String)  vector column name, default "embedding"
     *   dim               (Int)     vector dimensionality
     *   query             (Float[]) query vector as JSON float array
     *   top_k             (Int)     default 10
     *   ef_search         (Int)     default 50
     *   partition_filter  (String?) optional — restrict search to files where partition value matches
     *   hybrid_text       (String?) optional — enables BM25+vector hybrid when non-empty
     *   text_column       (String?) optional — Parquet column for BM25, default "chunk_text"
     *   bm25_weight       (Float?)  optional — BM25 weight in RRF fusion, default 0.5
     *
     * Response JSON: `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}`
     */
    fun ailake_search_json(requestJson: String): Pointer?

    /**
     * Write a batch of records to an AI-Lake table.
     *
     * Request JSON fields:
     *   warehouse         (String)    warehouse root path
     *   namespace         (String)    Iceberg namespace
     *   table             (String)    table name
     *   vec_col           (String)    vector column name
     *   dim               (Int)       vector dimensionality
     *   metric            (String?)   "euclidean" | "cosine" | "dot_product"
     *   precision         (String?)   "f32" | "f16" | "i8"
     *   ids               (Long[])    row IDs
     *   embeddings        (Float[][]) one embedding per row
     *   partition_by      (String?)   optional — Iceberg identity partition column (e.g. "agent_id")
     *   partition_value   (String?)   optional — value for partition_by in key_metadata of written files
     *   partition_fields  (Array?)    optional — multi-column partition spec (Phase K);
     *                                 each entry: {column, transform, column_type}
     *   format_version    (Int?)      optional — Iceberg format version, default 2
     *   fts_columns       (String[]?) optional — text columns to embed as Tantivy FTS index;
     *                                 empty/absent = no FTS (zero overhead)
     *   fts_tokenizer     (String?)   optional — Tantivy tokenizer, default "default"
     *
     * Response JSON: `{"ok":true,"snapshot_id":N}` or `{"ok":false,"error":"..."}`
     */
    fun ailake_write_batch_json(requestJson: String): Pointer?

    /**
     * Cross-modal RRF search across multiple vector columns.
     *
     * Request JSON fields:
     *   warehouse         (String)  warehouse root path
     *   namespace         (String)  Iceberg namespace, default "default"
     *   table             (String)  table name
     *   queries           (Array)   [{col, query: Float[], weight: Float, dim: Int (0=auto)}]
     *   top_k             (Int)     default 10
     *   partition_filter  (String?) optional — restrict to files matching partition value
     *
     * Response JSON: `{"ok":true,"results":[{"row_id":N,"rrf_score":F,"file_path":"..."}]}`
     */
    fun ailake_search_multimodal_json(requestJson: String): Pointer?

    /**
     * Low-level search: f32 pointer + length variant.  Prefer [ailake_search_json] for
     * JVM callers.  Returns JSON array `[{"row_id":N,"distance":F,"file_path":"..."}]`.
     */
    fun ailake_vector_search_json(
        tableUri: String,
        queryPtr: Pointer,
        queryLen: Int,
        topK: Int,
    ): Pointer?

    /**
     * Pure BM25 full-text search (no embedding required).
     *
     * Request JSON fields:
     *   warehouse         (String)  warehouse root path
     *   namespace         (String)  Iceberg namespace, default "default"
     *   table             (String)  table name
     *   query_text        (String)   text query to score against
     *   top_k             (Int)      default 10
     *   text_columns      (String[]) preferred — Parquet columns to search (multi-column)
     *   text_column       (String?)  legacy single-column fallback, default "chunk_text"
     *   partition_filter  (String?)  optional — restrict to files matching partition value
     *
     * Response JSON: `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}`
     * where distance = negated BM25 score (lower = more relevant).
     */
    fun ailake_search_text_json(requestJson: String): Pointer?

    /**
     * Logically delete rows matching a column equality predicate.
     *
     * Request JSON fields:
     *   warehouse  (String)   warehouse root path
     *   namespace  (String)   Iceberg namespace
     *   table      (String)   table name
     *   column     (String)   column name to match
     *   values     (String[]) values to delete (equality match)
     *
     * Response JSON: `{"ok":true}` or `{"ok":false,"error":"..."}`
     */
    fun ailake_delete_where_json(requestJson: String): Pointer?

    /**
     * Apply a metadata-only schema evolution to the table.
     *
     * Request JSON fields:
     *   warehouse        (String)  warehouse root path
     *   namespace        (String)  Iceberg namespace
     *   table            (String)  table name
     *   add_columns      (Array)   [{name, type, initial_default?}] — initial_default is a JSON literal
     *   rename_columns   (Array)   [{from, to}]
     *
     * Response JSON: `{"ok":true,"new_schema_id":N}` or `{"ok":false,"error":"..."}`
     */
    fun ailake_evolve_schema_json(requestJson: String): Pointer?

    /**
     * Compact small files in an AI-Lake table into a larger merged file.
     *
     * Request JSON fields:
     *   warehouse          (String)  warehouse root path
     *   namespace          (String)  Iceberg namespace, default "default"
     *   table              (String)  table name
     *   min_files          (Int?)    minimum eligible files to trigger compaction, default 4
     *   target_size_bytes  (Long?)   files smaller than this are candidates, default 128 MiB
     *   max_files_per_pass (Int?)    max files merged per run, default 20
     *   deferred           (Bool?)   build index in background when true, default false
     *
     * Response JSON: `{"ok":true,"files_compacted":N}` or `{"ok":false,"error":"..."}`
     */
    fun ailake_compact_json(requestJson: String): Pointer?

    /** Free a string pointer returned by any ailake_* function. */
    fun ailake_free_string(ptr: Pointer?)
}
