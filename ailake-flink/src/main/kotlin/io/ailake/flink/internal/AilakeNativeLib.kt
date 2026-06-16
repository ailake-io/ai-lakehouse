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
     *   warehouse    (String)  warehouse root path
     *   namespace    (String)  Iceberg namespace, default "default"
     *   table        (String)  table name
     *   vec_col      (String)  vector column name, default "embedding"
     *   dim          (Int)     vector dimensionality
     *   query        (Float[]) query vector as JSON float array
     *   top_k        (Int)     default 10
     *   ef_search    (Int)     default 50
     *
     * Response JSON: `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}`
     */
    fun ailake_search_json(requestJson: String): Pointer

    /**
     * Write a batch of records to an AI-Lake table.
     *
     * Request JSON fields:
     *   warehouse    (String)    warehouse root path
     *   namespace    (String)    Iceberg namespace
     *   table        (String)    table name
     *   vec_col      (String)    vector column name
     *   dim          (Int)       vector dimensionality
     *   metric       (String?)   "euclidean" | "cosine" | "dot_product"
     *   precision    (String?)   "f32" | "f16" | "i8"
     *   ids          (Long[])    row IDs
     *   embeddings   (Float[][]) one embedding per row
     *
     * Response JSON: `{"ok":true,"snapshot_id":N}` or `{"ok":false,"error":"..."}`
     */
    fun ailake_write_batch_json(requestJson: String): Pointer

    /**
     * Cross-modal RRF search across multiple vector columns.
     *
     * Request JSON fields:
     *   warehouse    (String)  warehouse root path
     *   namespace    (String)  Iceberg namespace, default "default"
     *   table        (String)  table name
     *   queries      (Array)   [{col, query: Float[], weight: Float, dim: Int (0=auto)}]
     *   top_k        (Int)     default 10
     *
     * Response JSON: `{"ok":true,"results":[{"row_id":N,"rrf_score":F,"file_path":"..."}]}`
     */
    fun ailake_search_multimodal_json(requestJson: String): Pointer

    /**
     * Low-level search: f32 pointer + length variant.  Prefer [ailake_search_json] for
     * JVM callers.  Returns JSON array `[{"row_id":N,"distance":F,"file_path":"..."}]`.
     */
    fun ailake_vector_search_json(
        tableUri: String,
        queryPtr: Pointer,
        queryLen: Int,
        topK: Int,
    ): Pointer

    /** Free a string pointer returned by any ailake_* function. */
    fun ailake_free_string(ptr: Pointer)
}
