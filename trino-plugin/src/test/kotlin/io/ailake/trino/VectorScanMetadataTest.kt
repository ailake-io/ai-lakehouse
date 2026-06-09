// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.SchemaTableName
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.mockito.kotlin.mock
import java.util.Optional

class VectorScanMetadataTest {

    private val metadata = VectorScanMetadata(
        tableUri = "s3://bucket/table/",
        vectorColumn = "embedding",
        dim = 1536,
        metric = "cosine",
        precision = "f16",
        namespace = "default",
        tableName = "table",
    )
    private val session = mock<io.trino.spi.connector.ConnectorSession>()

    @Test
    fun listSchemaNameReturnDefault() {
        assertEquals(listOf("default"), metadata.listSchemaNames(session))
    }

    @Test
    fun getTableHandleFoundForKnownTable() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search"))
        assertNotNull(handle)
        val h = handle as VectorScanTableHandle
        assertEquals("s3://bucket/table/", h.tableUri)
        assertEquals("embedding", h.vectorColumn)
        assertEquals(1536, h.dim)
    }

    @Test
    fun getTableHandleNullForUnknownSchema() {
        assertNull(metadata.getTableHandle(session, SchemaTableName("other", "search")))
    }

    @Test
    fun getTableHandleNullForUnknownTable() {
        assertNull(metadata.getTableHandle(session, SchemaTableName("default", "other")))
    }

    @Test
    fun getTableMetadataHasThreeColumns() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search"))!!
        val tableMeta = metadata.getTableMetadata(session, handle)
        assertEquals(3, tableMeta.columns.size)
        assertEquals("row_id", tableMeta.columns[0].name)
        assertEquals("distance", tableMeta.columns[1].name)
        assertEquals("file_path", tableMeta.columns[2].name)
    }

    @Test
    fun listTablesReturnsSearchAndIngestTables() {
        val tables = metadata.listTables(session, Optional.empty())
        assertEquals(2, tables.size)
        assertTrue(SchemaTableName("default", "search") in tables)
        assertTrue(SchemaTableName("default", "ingest") in tables)
    }

    @Test
    fun getColumnHandlesReturnsThreeHandles() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search"))!!
        val cols = metadata.getColumnHandles(session, handle)
        assertEquals(3, cols.size)
        assertTrue(cols.containsKey("row_id"))
        assertTrue(cols.containsKey("distance"))
        assertTrue(cols.containsKey("file_path"))
    }

    @Test
    fun getColumnMetadataOrdinalConsistent() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search"))!!
        val colHandle = VectorScanColumnHandle("distance", 1)
        val colMeta = metadata.getColumnMetadata(session, handle, colHandle)
        assertEquals("distance", colMeta.name)
    }
}
