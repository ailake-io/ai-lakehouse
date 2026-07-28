// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_add_vector_column(warehouse, namespace, table, column, dim)` —
 * adds a new vector column to the schema (metadata-only, no data files
 * rewritten, no embeddings backfilled — use [AilakeBackfillVectorColumnFunction]
 * for that). Returns the new schema_id. Defaults: metric="cosine", precision="f16".
 *
 * Closes a gap found auditing trino-plugin — see [AilakeDecayMemoriesFunction].
 *
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_add_vector_column AS 'io.ailake.flink.AilakeAddVectorColumnFunction';
 * SELECT ailake_add_vector_column('s3://my-lake/', 'default', 'docs', 'image_embedding', 512);
 * ```
 */
class AilakeAddVectorColumnFunction : ScalarFunction() {
    fun eval(warehouse: String, namespace: String, table: String, column: String, dim: Int): Int =
        AilakeNativeLoader.addVectorColumn(warehouse, namespace, table, column, dim)
}
