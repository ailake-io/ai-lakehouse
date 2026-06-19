// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class AilakeNativeTest {

    private fun base64Of(vararg floats: Float): String =
        VectorScanSplitManager.csvFloatsToBase64(floats.joinToString(","))

    @Test
    fun searchReturnsEmptyWhenNativeLibAbsent() {
        // Native lib is not available in test environment — graceful degradation.
        val results = AilakeNative.search("s3://bucket/table/", base64Of(0.1f, 0.2f, 0.3f), topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchReturnsEmptyForBlankQueryBytes() {
        val results = AilakeNative.search("s3://bucket/table/", "  ", topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchReturnsEmptyForEmptyQueryBytes() {
        val results = AilakeNative.search("s3://bucket/table/", "", topK = 10)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchRowDataClassEquality() {
        val r1 = AilakeNative.SearchRow(1L, 0.5f, "file.parquet")
        val r2 = AilakeNative.SearchRow(1L, 0.5f, "file.parquet")
        assertEquals(r1, r2)
    }

    @Test
    fun searchRowToString() {
        val r = AilakeNative.SearchRow(42L, 0.99f, "part-001.parquet")
        val s = r.toString()
        assertTrue(s.contains("42"))
        assertTrue(s.contains("part-001.parquet"))
    }

    // ── Phase P: writeBatch with partitionFields / formatVersion ─────────────

    @Test
    fun writeBatchReturnsNullWhenNativeLibAbsentWithPartitionFields() {
        val pf = AilakeNative.PartitionFieldDef("agent_id", "identity", "string")
        val result = AilakeNative.writeBatch(
            tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
            vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            ids = listOf(1L), embeddings = listOf(listOf(0.1f, 0.2f, 0.3f, 0.4f)),
            partitionFields = listOf(pf), formatVersion = 3,
        )
        assertNull(result)
    }

    @Test
    fun writeBatchReturnsNullForEmptyIdsRegardlessOfPartitionFields() {
        val pf = AilakeNative.PartitionFieldDef("col", "identity", "string")
        val result = AilakeNative.writeBatch(
            tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
            vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            ids = emptyList(), embeddings = emptyList(),
            partitionFields = listOf(pf),
        )
        assertNull(result)
    }

    @Test
    fun partitionFieldDefEquality() {
        val p1 = AilakeNative.PartitionFieldDef("col", "identity", "string")
        val p2 = AilakeNative.PartitionFieldDef("col", "identity", "string")
        assertEquals(p1, p2)
    }

    @Test
    fun partitionFieldDefToStringContainsColumn() {
        val p = AilakeNative.PartitionFieldDef("session_id", "truncate[4]", "string")
        assertTrue(p.toString().contains("session_id"))
    }

    // ── Phase P: deleteWhere ─────────────────────────────────────────────────

    @Test
    fun deleteWhereReturnsFalseWhenNativeLibAbsent() {
        val ok = AilakeNative.deleteWhere("s3://b/t/", "default", "tbl", "doc_id", listOf("x"))
        assertFalse(ok)
    }

    @Test
    fun deleteWhereReturnsFalseForEmptyValues() {
        val ok = AilakeNative.deleteWhere("s3://b/t/", "default", "tbl", "doc_id", emptyList())
        assertFalse(ok)
    }

    // ── Phase P: evolveSchema ────────────────────────────────────────────────

    @Test
    fun evolveSchemaReturnsMinusOneWhenNativeLibAbsent() {
        val id = AilakeNative.evolveSchema(
            tableUri = "s3://b/t/", namespace = "default", tableName = "tbl",
            addCols = listOf(AilakeNative.AddColReq("score", "float")),
            renameCols = emptyList(),
        )
        assertEquals(-1, id)
    }

    @Test
    fun evolveSchemaReturnsZeroForEmptyAddAndRename() {
        val id = AilakeNative.evolveSchema(
            tableUri = "s3://b/t/", namespace = "default", tableName = "tbl",
            addCols = emptyList(), renameCols = emptyList(),
        )
        assertEquals(0, id)
    }

    @Test
    fun addColReqDefaultInitialDefaultIsNull() {
        val r = AilakeNative.AddColReq("score", "float")
        assertNull(r.initialDefault)
    }

    @Test
    fun addColReqWithInitialDefault() {
        val r = AilakeNative.AddColReq("score", "float", "0.0")
        assertEquals("0.0", r.initialDefault)
    }

    @Test
    fun renameColReqEquality() {
        val r1 = AilakeNative.RenameColReq("old_col", "new_col")
        val r2 = AilakeNative.RenameColReq("old_col", "new_col")
        assertEquals(r1, r2)
    }
}
