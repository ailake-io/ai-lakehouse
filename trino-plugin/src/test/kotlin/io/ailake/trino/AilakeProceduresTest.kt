// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.TrinoException
import io.trino.spi.connector.ConnectorSession
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.mockito.kotlin.mock

class AilakeProceduresTest {

    private val procedures = AilakeProcedures(
        tableUri = "file:///tmp/test-table",
        namespace = "default",
        tableName = "docs",
    )
    private val session = mock<ConnectorSession>()

    @Test
    fun getProceduresReturnsExactlyOneCompactProcedure() {
        val procs = procedures.getProcedures()
        assertEquals(1, procs.size)
        val p = procs.first()
        assertEquals("system", p.schema)
        assertEquals("compact", p.name)
        assertTrue(p.arguments.isEmpty())
    }

    @Test
    fun compactThrowsTrinoExceptionWhenNativeLibraryAbsent() {
        // AilakeNative.compact returns null when the native lib is absent (test env) —
        // CALL ailake.system.compact() must surface this as a clear SQL error, not silently no-op.
        assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        assertThrows(TrinoException::class.java) { procedures.compact(session) }
    }

    private fun assume(condition: Boolean, message: String) {
        org.junit.jupiter.api.Assumptions.assumeTrue(condition, message)
    }
}
