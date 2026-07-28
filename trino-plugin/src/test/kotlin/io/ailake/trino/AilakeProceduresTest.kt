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
    fun getProceduresReturnsCompactAndCreateTableProcedures() {
        val procs = procedures.getProcedures()
        assertEquals(2, procs.size)
        val byName = procs.associateBy { it.name }
        val compact = byName.getValue("compact")
        assertEquals("system", compact.schema)
        assertTrue(compact.arguments.isEmpty())
        val createTable = byName.getValue("create_table")
        assertEquals("system", createTable.schema)
        assertTrue(createTable.arguments.isEmpty())
    }

    @Test
    fun compactThrowsTrinoExceptionWhenNativeLibraryAbsent() {
        // AilakeNative.compact returns null when the native lib is absent (test env) —
        // CALL ailake.system.compact() must surface this as a clear SQL error, not silently no-op.
        assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        assertThrows(TrinoException::class.java) { procedures.compact(session) }
    }

    @Test
    fun createTableThrowsTrinoExceptionWhenNativeLibraryAbsent() {
        // Same reasoning as compactThrowsTrinoExceptionWhenNativeLibraryAbsent above —
        // AilakeNative.createTable returns false (not an exception) when the native lib
        // is absent; CALL ailake.system.create_table() must still surface a clear SQL error.
        assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        assertThrows(TrinoException::class.java) { procedures.createTable(session) }
    }

    private fun assume(condition: Boolean, message: String) {
        org.junit.jupiter.api.Assumptions.assumeTrue(condition, message)
    }
}
