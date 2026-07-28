// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import com.fasterxml.jackson.databind.ObjectMapper
import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_estimate(rows, dim)` — storage/index size estimates for a
 * hypothetical table (pure math, no I/O, no warehouse/catalog needed).
 * Returns the result as a JSON string (Flink scalar functions need an
 * explicit type hint for complex return types; a JSON `VARCHAR` avoids that
 * and matches [AilakeMigrateFunction]/[AilakeBackfillVectorColumnFunction]'s
 * simple-`String`-return convention).
 *
 * Closes a gap found auditing trino-plugin — see [AilakeDecayMemoriesFunction].
 *
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_estimate AS 'io.ailake.flink.AilakeEstimateFunction';
 * SELECT ailake_estimate(1000000, 1536);
 * ```
 */
class AilakeEstimateFunction : ScalarFunction() {
    private val mapper = ObjectMapper()

    fun eval(rows: Long, dim: Int): String =
        mapper.writeValueAsString(AilakeNativeLoader.estimate(rows, dim))
}
