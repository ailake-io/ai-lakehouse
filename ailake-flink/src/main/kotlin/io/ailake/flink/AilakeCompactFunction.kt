// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_compact(warehouse, namespace, table)` — compacts small files in an
 * AI-Lake table, returning the number of files compacted.
 *
 * `AilakeNativeLoader.compact` was already fully implemented and tested but
 * had no SQL surface reachable from Flink at all — same "dead capability"
 * gap as DELETE/schema evolution, closed the same way for the other JVM
 * plugins in this codebase (Trino got a `CALL ailake.system.compact()`
 * procedure). Flink SQL has no equivalent stored-procedure mechanism for
 * connectors, so this is exposed as a plain scalar function instead — the
 * standard Flink idiom for a one-shot maintenance operation invoked from SQL.
 *
 * Register once per session (or job) via:
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_compact AS 'io.ailake.flink.AilakeCompactFunction';
 * SELECT ailake_compact('s3://my-lake/', 'default', 'docs');
 * ```
 */
class AilakeCompactFunction : ScalarFunction() {
    fun eval(warehouse: String, namespace: String, table: String): Int =
        AilakeNativeLoader.compact(warehouse, namespace, table)
}
