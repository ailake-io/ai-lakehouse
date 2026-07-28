// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test

/**
 * Regression: AilakeNativeLoader.compact was fully implemented and tested but
 * had no SQL surface reachable from Flink at all — same "dead capability" gap
 * as DELETE/schema evolution. Exposed here as a plain scalar function
 * (`CALL`-equivalent doesn't exist in Flink SQL for connectors).
 */
class AilakeCompactFunctionTest {

    @Test
    fun evalFailsClearlyWhenNativeLibraryAbsent() {
        // See AilakeVectorTableSinkTest.executeDeletionFailsClearlyWhenNativeLibraryAbsent
        // for why this checks AILAKE_NATIVE_LIB (the var Flink's own gradle-test step
        // sets — AILAKE_LIB_PATH is Spark/Trino's, never set here) and a fresh warehouse
        // path per run instead of the previously shared "file:///tmp/x/default/docs"
        // (now genuinely creatable by AilakeCatalog.createTable, so a stale table left
        // behind by another test class in the same job would make eval() succeed here
        // instead of throwing).
        val nativeLib = System.getenv("AILAKE_NATIVE_LIB") ?: System.getProperty("ailake.native.lib")
        assumeTrue(nativeLib == null || !java.io.File(nativeLib).exists(), "skipped: native library present")
        val fn = AilakeCompactFunction()
        val freshWarehouse = "file:///tmp/ailake-flink-compactfn-absent-${System.nanoTime()}"
        // AilakeNativeLoader.lib throws (via getOrThrow()) when the native lib isn't on
        // the library path — UnsatisfiedLinkError (a JVM Error), not a RuntimeException.
        assertThrows(Throwable::class.java) { fn.eval(freshWarehouse, "default", "docs") }
    }
}
