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
        assumeTrue(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        val fn = AilakeCompactFunction()
        // AilakeNativeLoader.lib throws (via getOrThrow()) when the native lib isn't on
        // the library path — UnsatisfiedLinkError (a JVM Error), not a RuntimeException.
        assertThrows(Throwable::class.java) { fn.eval("file:///tmp/x", "default", "docs") }
    }
}
