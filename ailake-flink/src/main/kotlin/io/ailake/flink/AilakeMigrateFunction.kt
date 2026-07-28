// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_migrate(warehouse, namespace, table, oldColumn, newColumn, textColumn, embedCmd)`
 * — re-embeds a column via an external embed command (spawned `sh -c
 * embedCmd`, JSON array-of-strings on stdin, JSON array-of-float-arrays on
 * stdout — same protocol as `ailake-cli`'s `--embed-cmd`). Returns `"ok"` on
 * success, throws on failure.
 *
 * Closes a gap found auditing trino-plugin — see [AilakeDecayMemoriesFunction].
 *
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_migrate AS 'io.ailake.flink.AilakeMigrateFunction';
 * SELECT ailake_migrate('s3://my-lake/', 'default', 'docs',
 *   'embedding', 'embedding_v2', 'chunk_text', 'python3 embed.py');
 * ```
 */
class AilakeMigrateFunction : ScalarFunction() {
    fun eval(
        warehouse: String,
        namespace: String,
        table: String,
        oldColumn: String,
        newColumn: String,
        textColumn: String,
        embedCmd: String,
    ): String {
        AilakeNativeLoader.migrate(warehouse, namespace, table, oldColumn, newColumn, textColumn, embedCmd)
        return "ok"
    }
}
