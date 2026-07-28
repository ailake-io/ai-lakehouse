// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_decay_memories(warehouse, namespace, table, lambda)` — recomputes
 * `recency_weight` for every row (Phase 9 agent memory), returning the number
 * of files rewritten.
 *
 * Closes a gap found auditing trino-plugin: `AilakeNativeLoader.decayMemories`
 * had no C-ABI export in `ailake-jni` at all until now (unlike `compact`,
 * which existed but was simply unwired) — same fix applied to all three JVM
 * plugins, since they bind exclusively to `ailake-jni` via JNA.
 *
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_decay_memories AS 'io.ailake.flink.AilakeDecayMemoriesFunction';
 * SELECT ailake_decay_memories('s3://my-lake/', 'default', 'agent_memory', CAST(0.1 AS FLOAT));
 * ```
 */
class AilakeDecayMemoriesFunction : ScalarFunction() {
    fun eval(warehouse: String, namespace: String, table: String, lambda: Float): Int =
        AilakeNativeLoader.decayMemories(warehouse, namespace, table, lambda)
}
