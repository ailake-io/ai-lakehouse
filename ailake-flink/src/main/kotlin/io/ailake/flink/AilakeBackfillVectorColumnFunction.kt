// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_backfill_vector_column(warehouse, namespace, table, column, textColumn, embedCmd)`
 * — backfills embeddings for a column added via [AilakeAddVectorColumnFunction].
 * Reads `textColumn` from every existing file missing `column`, embeds via
 * `embedCmd` (same `sh -c`/JSON stdin-stdout protocol as
 * [AilakeMigrateFunction]), writes new files with the column populated.
 * Idempotent — files that already have the column are skipped. Returns
 * `"ok"` on success, throws on failure.
 *
 * Closes a gap found auditing trino-plugin — see [AilakeDecayMemoriesFunction].
 *
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_backfill_vector_column AS 'io.ailake.flink.AilakeBackfillVectorColumnFunction';
 * SELECT ailake_backfill_vector_column('s3://my-lake/', 'default', 'docs',
 *   'image_embedding', 'image_uri', 'python3 embed_images.py');
 * ```
 */
class AilakeBackfillVectorColumnFunction : ScalarFunction() {
    fun eval(
        warehouse: String,
        namespace: String,
        table: String,
        column: String,
        textColumn: String,
        embedCmd: String,
    ): String {
        AilakeNativeLoader.backfillVectorColumn(warehouse, namespace, table, column, textColumn, embedCmd)
        return "ok"
    }
}
