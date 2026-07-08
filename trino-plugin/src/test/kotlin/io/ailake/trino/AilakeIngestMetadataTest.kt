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

    // ── embeddingModel propagation ────────────────────────────────────────────

    @Test
    fun embeddingModelPropagatedToIngestHandle() {
        val meta = VectorScanMetadata(
            tableUri       = "file:///tmp/test-table",
            vectorColumn   = "embedding",
            dim            = 4,
            metric         = "cosine",
            precision      = "f16",
            namespace      = "default",
            tableName      = "docs",
            embeddingModel = "text-embedding-3-small@v1",
        )
        val handle = meta.getTableHandle(session, SchemaTableName("default", "ingest"))
        assertNotNull(handle)
        val h = handle as AilakeIngestTableHandle
        assertEquals("text-embedding-3-small@v1", h.embeddingModel)
    }

    @Test
    fun embeddingModelNullByDefault() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))
        assertNotNull(handle)
        assertNull((handle as AilakeIngestTableHandle).embeddingModel)
    }

    // ── textColumns (extra metadata columns) ──────────────────────────────────

    private val metadataWithTextColumns = VectorScanMetadata(
        tableUri     = "file:///tmp/test-table",
        vectorColumn = "embedding",
        dim          = 4,
        metric       = "cosine",
        precision    = "f16",
        namespace    = "default",
        tableName    = "docs",
        textColumns  = listOf("text", "source", "page"),
    )

    @Test
    fun ingestTableHasExtraTextColumnsWhenConfigured() {
        val handle = metadataWithTextColumns.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        val meta = metadataWithTextColumns.getTableMetadata(session, handle)
        assertEquals(5, meta.columns.size)
        assertEquals(listOf("id", "embedding", "text", "source", "page"), meta.columns.map { it.name })
    }

    @Test
    fun ingestColumnHandlesIncludeTextColumnsAtCorrectOrdinals() {
        val handle = metadataWithTextColumns.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        val cols = metadataWithTextColumns.getColumnHandles(session, handle) as Map<String, VectorScanColumnHandle>
        assertEquals(0, cols.getValue("id").ordinal)
        assertEquals(1, cols.getValue("embedding").ordinal)
        assertEquals(2, cols.getValue("text").ordinal)
        assertEquals(3, cols.getValue("source").ordinal)
        assertEquals(4, cols.getValue("page").ordinal)
    }

    @Test
    fun textColumnsPropagatedToIngestHandle() {
        val handle = metadataWithTextColumns.getTableHandle(session, SchemaTableName("default", "ingest"))
        assertEquals(listOf("text", "source", "page"), (handle as AilakeIngestTableHandle).textColumns)
    }

    @Test
    fun textColumnsEmptyByDefault() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))
        assertTrue((handle as AilakeIngestTableHandle).textColumns.isEmpty())
    }
}
