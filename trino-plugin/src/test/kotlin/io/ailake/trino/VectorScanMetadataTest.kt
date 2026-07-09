// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.airlift.slice.Slices
import io.trino.spi.TrinoException
import io.trino.spi.connector.ColumnMetadata
import io.trino.spi.connector.Constraint
import io.trino.spi.connector.SchemaTableName
import io.trino.spi.predicate.Domain
import io.trino.spi.predicate.TupleDomain
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.TimestampType
import io.trino.spi.type.VarcharType.VARCHAR
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Assumptions.assumeTrue
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

    // Regression: VectorScanTableHandle used to carry only tableUri/vectorColumn/dim,
    // silently dropping namespace/tableName — search always hit AilakeNative.search's
    // hardcoded "default" namespace and URI-derived table name, unfindable if the
    // catalog was configured with a custom namespace/table-name.
    @Test
    fun getTableHandleCarriesNamespaceAndTableName() {
        val m = VectorScanMetadata(
            tableUri = "s3://bucket/warehouse/", vectorColumn = "doc_vec", dim = 8,
            metric = "cosine", precision = "f16", namespace = "tenant_a", tableName = "docs",
        )
        val handle = m.getTableHandle(session, SchemaTableName("default", "search")) as VectorScanTableHandle
        assertEquals("tenant_a", handle.namespace)
        assertEquals("docs", handle.tableName)
        assertEquals("doc_vec", handle.vectorColumn)
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
    fun listTablesReturnsSearchMultimodalAndIngestTables() {
        val tables = metadata.listTables(session, Optional.empty())
        assertEquals(3, tables.size)
        assertTrue(SchemaTableName("default", "search") in tables)
        assertTrue(SchemaTableName("default", "search_multimodal") in tables)
        assertTrue(SchemaTableName("default", "ingest") in tables)
    }

    // ── search_multimodal (cross-modal RRF search) ────────────────────────────
    //
    // Regression: AilakeNative.searchMultimodal was fully implemented but had
    // no SQL surface in any of the three plugins — same "dead capability" gap
    // as DELETE/ALTER TABLE before it, closed the same way.

    @Test
    fun getTableHandleFoundForSearchMultimodal() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search_multimodal"))
        assertNotNull(handle)
        val h = handle as MultimodalScanTableHandle
        assertEquals("s3://bucket/table/", h.tableUri)
        assertEquals("default", h.namespace)
        assertEquals("table", h.tableName)
    }

    @Test
    fun getTableMetadataForSearchMultimodalHasThreeColumns() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search_multimodal"))!!
        val tableMeta = metadata.getTableMetadata(session, handle)
        assertEquals(3, tableMeta.columns.size)
        assertEquals("row_id", tableMeta.columns[0].name)
        assertEquals("rrf_score", tableMeta.columns[1].name)
        assertEquals("file_path", tableMeta.columns[2].name)
    }

    @Test
    fun getColumnHandlesForSearchMultimodalReturnsThreeHandles() {
        val handle = metadata.getTableHandle(session, SchemaTableName("default", "search_multimodal"))!!
        val cols = metadata.getColumnHandles(session, handle)
        assertEquals(3, cols.size)
        assertTrue(cols.containsKey("row_id"))
        assertTrue(cols.containsKey("rrf_score"))
        assertTrue(cols.containsKey("file_path"))
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

    // ── ALTER TABLE ADD/RENAME COLUMN ─────────────────────────────────────────
    //
    // Regression: AilakeNative.evolveSchema was fully implemented but had no
    // SQL surface — ALTER TABLE ADD/RENAME COLUMN did nothing (ConnectorMetadata's
    // default no-op implementations). Now wired to addColumn/renameColumn.

    @Test
    fun addColumnRejectsNonIngestTableHandle() {
        val searchHandle = metadata.getTableHandle(session, SchemaTableName("default", "search"))!!
        assertThrows(TrinoException::class.java) {
            metadata.addColumn(session, searchHandle, ColumnMetadata("source", VARCHAR))
        }
    }

    @Test
    fun addColumnRejectsUnsupportedType() {
        val ingestHandle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        val ex = assertThrows(TrinoException::class.java) {
            metadata.addColumn(session, ingestHandle, ColumnMetadata("ts", TimestampType.TIMESTAMP_MILLIS))
        }
        assertTrue(ex.message!!.contains("not supported"))
    }

    @Test
    fun addColumnFailsClearlyWhenNativeLibraryAbsent() {
        assumeTrue(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        val ingestHandle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        assertThrows(TrinoException::class.java) {
            metadata.addColumn(session, ingestHandle, ColumnMetadata("source", VARCHAR))
        }
    }

    @Test
    fun renameColumnRejectsNonIngestTableHandle() {
        val searchHandle = metadata.getTableHandle(session, SchemaTableName("default", "search"))!!
        assertThrows(TrinoException::class.java) {
            metadata.renameColumn(session, searchHandle, VectorScanColumnHandle("row_id", 0), "id2")
        }
    }

    @Test
    fun renameColumnFailsClearlyWhenNativeLibraryAbsent() {
        assumeTrue(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        val ingestHandle = metadata.getTableHandle(session, SchemaTableName("default", "ingest"))!!
        assertThrows(TrinoException::class.java) {
            metadata.renameColumn(session, ingestHandle, VectorScanColumnHandle("id", 0), "doc_id")
        }
    }

    // ── DELETE (equality/IN pushdown) ─────────────────────────────────────────
    //
    // Regression: AilakeNative.deleteWhere was fully implemented but had no
    // SQL surface — DELETE FROM ailake.default.ingest did nothing (Connector-
    // Metadata's default no-op applyDelete). Now wired via applyFilter/
    // applyDelete/executeDelete, equality/IN pushdown only (matches the
    // native operation's own capability — no row-level scan-and-delete).

    private val idCol = VectorScanColumnHandle("id", 0)
    private val textCol = VectorScanColumnHandle("source", 2)
    private val ingestHandle get() = metadata.getTableHandle(session, SchemaTableName("default", "ingest")) as AilakeIngestTableHandle

    @Test
    fun applyFilterCapturesSingleEqualityPredicate() {
        val constraint = Constraint(TupleDomain.withColumnDomains(mapOf(idCol to Domain.singleValue(BIGINT, 5L))))
        val result = metadata.applyFilter(session, ingestHandle, constraint)
        assertTrue(result.isPresent)
        val newHandle = result.get().handle as AilakeIngestTableHandle
        assertEquals("id", newHandle.deleteColumn)
        assertEquals(listOf("5"), newHandle.deleteValues)
        assertTrue(result.get().remainingFilter.isAll)
    }

    @Test
    fun applyFilterCapturesInPredicateOnVarcharColumn() {
        val domain = Domain.multipleValues(VARCHAR, listOf(Slices.utf8Slice("doc-a"), Slices.utf8Slice("doc-b")))
        val constraint = Constraint(TupleDomain.withColumnDomains(mapOf(textCol to domain)))
        val result = metadata.applyFilter(session, ingestHandle, constraint)
        assertTrue(result.isPresent)
        val newHandle = result.get().handle as AilakeIngestTableHandle
        assertEquals("source", newHandle.deleteColumn)
        assertEquals(setOf("doc-a", "doc-b"), newHandle.deleteValues!!.toSet())
    }

    @Test
    fun applyFilterRejectsMultiColumnPredicate() {
        val constraint = Constraint(TupleDomain.withColumnDomains(mapOf(
            idCol to Domain.singleValue(BIGINT, 5L),
            textCol to Domain.singleValue(VARCHAR, Slices.utf8Slice("doc-a")),
        )))
        assertTrue(metadata.applyFilter(session, ingestHandle, constraint).isEmpty)
    }

    @Test
    fun applyFilterRejectsRangePredicate() {
        val range = io.trino.spi.predicate.Range.greaterThan(BIGINT, 5L)
        val domain = Domain.create(io.trino.spi.predicate.ValueSet.ofRanges(range), false)
        val constraint = Constraint(TupleDomain.withColumnDomains(mapOf(idCol to domain)))
        assertTrue(metadata.applyFilter(session, ingestHandle, constraint).isEmpty)
    }

    @Test
    fun applyFilterRejectsNullableDomain() {
        val domain = Domain.create(io.trino.spi.predicate.ValueSet.of(BIGINT, 5L), true)
        val constraint = Constraint(TupleDomain.withColumnDomains(mapOf(idCol to domain)))
        assertTrue(metadata.applyFilter(session, ingestHandle, constraint).isEmpty)
    }

    @Test
    fun applyFilterRejectsAlwaysTrueConstraint() {
        assertTrue(metadata.applyFilter(session, ingestHandle, Constraint.alwaysTrue()).isEmpty)
    }

    @Test
    fun applyFilterReturnsEmptyForNonIngestHandle() {
        val searchHandle = metadata.getTableHandle(session, SchemaTableName("default", "search"))!!
        val constraint = Constraint(TupleDomain.withColumnDomains(mapOf(idCol to Domain.singleValue(BIGINT, 5L))))
        assertTrue(metadata.applyFilter(session, searchHandle, constraint).isEmpty)
    }

    @Test
    fun applyDeleteReturnsEmptyWithoutCapturedPredicate() {
        assertTrue(metadata.applyDelete(session, ingestHandle).isEmpty)
    }

    @Test
    fun applyDeleteReturnsHandleWithCapturedPredicate() {
        val h = ingestHandle.copy(deleteColumn = "id", deleteValues = listOf("5"))
        val result = metadata.applyDelete(session, h)
        assertTrue(result.isPresent)
        assertEquals(h, result.get())
    }

    @Test
    fun executeDeleteThrowsWhenNoPredicateCaptured() {
        val ex = assertThrows(TrinoException::class.java) { metadata.executeDelete(session, ingestHandle) }
        assertTrue(ex.message!!.contains("WHERE clause"))
    }

    @Test
    fun executeDeleteFailsClearlyWhenNativeLibraryAbsent() {
        assumeTrue(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
        val h = ingestHandle.copy(deleteColumn = "id", deleteValues = listOf("5"))
        assertThrows(TrinoException::class.java) { metadata.executeDelete(session, h) }
    }
}
