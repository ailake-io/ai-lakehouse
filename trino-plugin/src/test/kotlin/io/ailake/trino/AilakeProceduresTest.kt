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
        // compact, create_table, decay_memories, migrate, delete_rows,
        // add_vector_column, backfill_vector_column, estimate (Fase 21) — all
        // no-arg, same reasoning as compact/create_table (parameters come
        // from SET SESSION properties, not typed CALL arguments).
        assertEquals(8, procs.size)
        val byName = procs.associateBy { it.name }
        for (name in listOf(
            "compact", "create_table", "decay_memories", "migrate",
            "delete_rows", "add_vector_column", "backfill_vector_column", "estimate",
        )) {
            val proc = byName.getValue(name)
            assertEquals("system", proc.schema)
            assertTrue(proc.arguments.isEmpty(), "expected $name to have no arguments")
        }
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
