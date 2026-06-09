// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.SchemaTableName
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.mockito.kotlin.mock

class AilakeIngestMetadataTest {

    private val metadata = VectorScanMetadata(
        tableUri     = "file:///tmp/test-table",
        vectorColumn = "embedding",
        dim          = 4,
        metric       = "cosine",
        precision    = "f16",
        namespace    = "default",
        tableName    = "docs",
    )
    private val session = mock<io.trino.spi.connector.ConnectorSession>()

    // ── getTableHandle ────────────────────────────────────────────────────────

    @Test
    fun getTableHandleReturnsIngestHandleForIngestTable() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))
        assertNotNull(handle)
        assertTrue(handle is AilakeIngestTableHandle)
        val h = handle as AilakeIngestTableHandle
        assertEquals("file:///tmp/test-table", h.tableUri)
        assertEquals("default", h.namespace)
        assertEquals("docs", h.tableName)
        assertEquals("embedding", h.vectorColumn)
        assertEquals(4, h.dim)
        assertEquals("cosine", h.metric)
        assertEquals("f16", h.precision)
    }

    @Test
    fun getTableHandleReturnsSearchHandleForSearchTable() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search"))
        assertNotNull(handle)
        assertTrue(handle is VectorScanTableHandle)
    }

    @Test
    fun getTableHandleReturnsNullForUnknownTable() {
        assertNull(metadata.getTableHandle(session, SchemaTableName("default", "unknown")))
    }

    // ── getTableMetadata ──────────────────────────────────────────────────────

    @Test
    fun ingestTableHasTwoColumns() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        val meta = metadata.getTableMetadata(session, handle)
        assertEquals(2, meta.columns.size)
        assertEquals("id", meta.columns[0].name)
        assertEquals("embedding", meta.columns[1].name)
    }

    // ── getColumnHandles ──────────────────────────────────────────────────────

    @Test
    fun ingestColumnHandlesHasIdAndEmbedding() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        val cols = metadata.getColumnHandles(session, handle)
        assertEquals(2, cols.size)
        assertTrue(cols.containsKey("id"))
        assertTrue(cols.containsKey("embedding"))
    }

    // ── beginInsert ───────────────────────────────────────────────────────────

    @Test
    fun beginInsertReturnsIngestHandle() {
        val tableHandle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        val insertHandle = metadata.beginInsert(session, tableHandle, emptyList(), mock())
        assertTrue(insertHandle is AilakeIngestTableHandle)
        assertSame(tableHandle, insertHandle)
    }

    // ── finishInsert ──────────────────────────────────────────────────────────

    @Test
    fun finishInsertReturnsEmpty() {
        val tableHandle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        val insertHandle = metadata.beginInsert(session, tableHandle, emptyList(), mock())
        val result = metadata.finishInsert(session, insertHandle, emptyList(), emptyList())
        assertTrue(result.isEmpty)
    }

    // ── pageSinkProvider ──────────────────────────────────────────────────────

    @Test
    fun pageSinkProviderReturnsAilakePageSinkProvider() {
        val connector = VectorScanConnector(
            tableUri     = "file:///tmp/test-table",
            vectorColumn = "embedding",
            dim          = 4,
            metric       = "cosine",
            precision    = "f16",
            namespace    = "default",
            tableName    = "docs",
        )
        assertTrue(connector.getPageSinkProvider() is AilakePageSinkProvider)
    }
}
