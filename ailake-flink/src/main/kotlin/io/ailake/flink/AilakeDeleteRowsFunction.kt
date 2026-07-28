// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_delete_rows(warehouse, namespace, table, file, rowIdsCsv)` —
 * deletes row positions via Iceberg V3 Deletion Vectors. Different from
 * `DELETE FROM` (`AilakeSinkFunction`'s equality-predicate pushdown) — this
 * masks exact `(file_path, row_position)` pairs. `rowIdsCsv` is a
 * comma-separated list of 0-based row positions (e.g. `"0,5,42"`) — plain
 * scalar functions take positional arguments, not arrays, without
 * `ARRAY[...]` literal support this connector doesn't otherwise need.
 * Returns `"ok"` on success, throws on failure.
 *
 * Closes a gap found auditing trino-plugin — see [AilakeDecayMemoriesFunction].
 *
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_delete_rows AS 'io.ailake.flink.AilakeDeleteRowsFunction';
 * SELECT ailake_delete_rows('s3://my-lake/', 'default', 'docs', 'data/part-00000.parquet', '0,5,42');
 * ```
 */
class AilakeDeleteRowsFunction : ScalarFunction() {
    fun eval(warehouse: String, namespace: String, table: String, file: String, rowIdsCsv: String): String {
        val rowIds = rowIdsCsv.split(',').mapNotNull { it.trim().toIntOrNull() }
        AilakeNativeLoader.deleteRows(warehouse, namespace, table, file, rowIds)
        return "ok"
    }
}
