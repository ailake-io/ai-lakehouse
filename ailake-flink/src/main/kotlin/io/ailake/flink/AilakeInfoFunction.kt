// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import com.fasterxml.jackson.databind.ObjectMapper
import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.functions.ScalarFunction

/**
 * `ailake_info(warehouse, namespace, table)` — reports current snapshot,
 * file/row/size counts, index status breakdown, and "foreign" files (written
 * by a generic Iceberg engine — no AI-Lake centroid/HNSW). Returns the
 * result as a JSON string (same convention as [AilakeEstimateFunction]).
 *
 * Found in a later audit pass: `info` (`ailake info`) had zero binding
 * coverage anywhere outside the CLI and the Airflow provider — not even a
 * C-ABI export in `ailake-jni` until now.
 *
 * ```sql
 * CREATE TEMPORARY FUNCTION ailake_info AS 'io.ailake.flink.AilakeInfoFunction';
 * SELECT ailake_info('s3://my-lake/', 'default', 'docs');
 * ```
 */
class AilakeInfoFunction : ScalarFunction() {
    private val mapper = ObjectMapper()

    fun eval(warehouse: String, namespace: String, table: String): String =
        mapper.writeValueAsString(AilakeNativeLoader.info(warehouse, namespace, table))
}
