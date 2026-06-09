// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class AilakePageSinkTest {

    private fun handle() = AilakeIngestTableHandle(
        tableUri     = "file:///tmp/test-table",
        namespace    = "default",
        tableName    = "docs",
        vectorColumn = "embedding",
        dim          = 4,
        metric       = "cosine",
        precision    = "f16",
    )

    @Test
    fun finishWithNoRowsReturnsEmptyCollection() {
        val sink = AilakePageSink(handle())
        val future = sink.finish()
        val fragments = future.get()
        // Native lib absent → writeBatch returns null → empty fragment list
        assertTrue(fragments.isEmpty())
    }

    @Test
    fun abortClearsBuffers() {
        val sink = AilakePageSink(handle())
        // No rows added; abort must not throw
        assertDoesNotThrow { sink.abort() }
    }
}
