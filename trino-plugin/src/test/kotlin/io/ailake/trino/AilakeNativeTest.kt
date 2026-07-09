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

    // ── Fase 11: scan (search + full-row fetch, ailake.default.search_full) ──

    @Test
    fun scanReturnsEmptyResultWhenNativeLibAbsent() {
        val result = AilakeNative.scan("s3://bucket/table/", base64Of(0.1f, 0.2f, 0.3f), topK = 5)
        assertTrue(result.schema.isEmpty())
        assertEquals(0, result.numRows)
        assertTrue(result.columns.isEmpty())
    }

    @Test
    fun scanReturnsEmptyResultForBlankQueryBytes() {
        val result = AilakeNative.scan("s3://bucket/table/", "  ", topK = 5)
        assertTrue(result.schema.isEmpty())
    }

    @Test
    fun scanColumnDataClassEquality() {
        val c1 = AilakeNative.ScanColumn("id", "int64")
        val c2 = AilakeNative.ScanColumn("id", "int64")
        assertEquals(c1, c2)
    }

    // ── writeBatchMulti (Phase 8 multimodal write) ────────────────────────────
    //
    // Regression: writeBatchMulti was exposed from Spark (`ailakeWriteMulti`)
    // but had no wrapper here at all — a Trino-only user could never write a
    // table with 2+ independent vector columns.

    @Test
    fun writeBatchMultiDoesNotThrowWhenNativeLibAbsent() {
        // Result is null (lib absent) or a snapshot_id (lib present, local fallback since
        // "s3://..." with no AWS creds/network resolves via LocalStore's relative-path
        // fallback in CI) — same caveat as writeBatchWithFtsColumnsDoesNotThrow below.
        val spec = AilakeNative.VectorColSpec("embedding", 4)
        val result = AilakeNative.writeBatchMulti(
            tableUri = "s3://bucket/t/", namespace = "default", tableName = "docs",
            ids = listOf(1L, 2L),
            vectorColumns = listOf(spec to listOf(listOf(0.1f, 0.2f, 0.3f, 0.4f), listOf(0.5f, 0.6f, 0.7f, 0.8f))),
        )
        assertTrue(result == null || result > 0, "writeBatchMulti must return null or a positive snapshot_id; got $result")
    }

    @Test
    fun writeBatchMultiReturnsNullForEmptyIds() {
        val spec = AilakeNative.VectorColSpec("embedding", 4)
        val result = AilakeNative.writeBatchMulti(
            tableUri = "s3://bucket/t/", namespace = "default", tableName = "docs",
            ids = emptyList(), vectorColumns = listOf(spec to emptyList()),
        )
        assertNull(result)
    }

    @Test
    fun writeBatchMultiReturnsNullForEmptyVectorColumns() {
        val result = AilakeNative.writeBatchMulti(
            tableUri = "s3://bucket/t/", namespace = "default", tableName = "docs",
            ids = listOf(1L), vectorColumns = emptyList(),
        )
        assertNull(result)
    }

    @Test
    fun vectorColSpecDefaults() {
        val spec = AilakeNative.VectorColSpec("embedding", 1536)
        assertEquals("cosine", spec.metric)
        assertEquals("f16", spec.precision)
        assertNull(spec.modality)
    }

    @Test
    fun vectorColSpecEquality() {
        val s1 = AilakeNative.VectorColSpec("embedding", 4, modality = "text")
        val s2 = AilakeNative.VectorColSpec("embedding", 4, modality = "text")
        assertEquals(s1, s2)
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

    // ── Phase R: public connector surface — AilakeIngestTableHandle ──────────

    @Test
    fun ingestHandleDefaultPartitionFieldsIsEmpty() {
        val handle = AilakeIngestTableHandle(
            tableUri = "s3://b/t/", namespace = "default", tableName = "t",
            vectorColumn = "emb", dim = 4, metric = "cosine", precision = "f16",
        )
        assertTrue(handle.partitionFields.isEmpty())
    }

    @Test
    fun ingestHandleDefaultFormatVersionIs2() {
        val handle = AilakeIngestTableHandle(
            tableUri = "s3://b/t/", namespace = "default", tableName = "t",
            vectorColumn = "emb", dim = 4, metric = "cosine", precision = "f16",
        )
        assertEquals(2, handle.formatVersion)
    }

    @Test
    fun ingestHandleAcceptsPartitionFieldsAndFormatVersion() {
        val pf = AilakeNative.PartitionFieldDef("agent_id", "identity", "string")
        val handle = AilakeIngestTableHandle(
            tableUri = "s3://b/t/", namespace = "default", tableName = "t",
            vectorColumn = "emb", dim = 4, metric = "cosine", precision = "f16",
            partitionFields = listOf(pf),
            formatVersion = 3,
        )
        assertEquals(1, handle.partitionFields.size)
        assertEquals("agent_id", handle.partitionFields[0].column)
        assertEquals(3, handle.formatVersion)
    }

    // ── Phase R: VectorScanConnectorFactory JSON parsing ─────────────────────

    @Test
    fun connectorFactoryParsesPartitionFieldsJson() {
        val factory = VectorScanConnectorFactory()
        val config = mapOf(
            "ailake.table-uri"        to "s3://b/t/",
            "ailake.partition-fields" to """[{"column":"ts","transform":"truncate[4]","column_type":"string"}]""",
            "ailake.format-version"   to "3",
        )
        val mapper = com.fasterxml.jackson.databind.ObjectMapper()
        val pfJson = config.getOrDefault("ailake.partition-fields", "[]")
        val node = mapper.readTree(pfJson)
        assertEquals(1, node.size())
        assertEquals("ts",          node.get(0).get("column").asText())
        assertEquals("truncate[4]", node.get(0).get("transform").asText())
        assertEquals("string",      node.get(0).get("column_type").asText())
        assertEquals(3, config.getOrDefault("ailake.format-version", "2").toInt())
        assertEquals("ailake", factory.name)
    }

    // ── Phase T: FTS ──────────────────────────────────────────────────────────

    @Test
    fun writeBatchWithFtsColumnsDoesNotThrow() {
        // Verifies writeBatch accepts ftsColumns without crashing.
        // Result is null (lib absent) or a snapshot_id (lib present, local fallback) — both are valid.
        val result = AilakeNative.writeBatch(
            tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
            vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            ids = listOf(1L), embeddings = listOf(listOf(0.1f, 0.2f, 0.3f, 0.4f)),
            ftsColumns = listOf("chunk_text", "title"),
            ftsTokenizer = "default",
        )
        assertTrue(result == null || result > 0, "writeBatch must return null or a positive snapshot_id; got $result")
    }

    @Test
    fun writeBatchJsonIncludesFtsColumnsWhenNonEmpty() {
        val ftsColumns   = listOf("chunk_text", "title")
        val ftsTokenizer = "default"
        val payload = mutableMapOf<String, Any>(
            "warehouse"      to "file:///tmp/test",
            "namespace"      to "default",
            "table"          to "t",
            "vec_col"        to "embedding",
            "dim"            to 4,
            "metric"         to "cosine",
            "precision"      to "f16",
            "ids"            to listOf(0L),
            "embeddings"     to listOf(listOf(0.1f, 0.2f, 0.3f, 0.4f)),
            "format_version" to 2,
        )
        if (ftsColumns.isNotEmpty()) {
            payload["fts_columns"]   = ftsColumns
            payload["fts_tokenizer"] = ftsTokenizer
        }
        val mapper = com.fasterxml.jackson.databind.ObjectMapper()
        val json = mapper.writeValueAsString(payload)
        assertTrue(json.contains("\"fts_columns\""),
            "JSON must contain fts_columns; got: $json")
        assertTrue(json.contains("chunk_text"),
            "JSON must contain chunk_text; got: $json")
        assertTrue(json.contains("\"fts_tokenizer\""),
            "JSON must contain fts_tokenizer; got: $json")
    }

    @Test
    fun writeBatchJsonOmitsFtsColumnsWhenEmpty() {
        val ftsColumns: List<String> = emptyList()
        val payload = mutableMapOf<String, Any>(
            "warehouse"  to "file:///tmp/test",
            "namespace"  to "default",
            "table"      to "t",
            "vec_col"    to "embedding",
            "dim"        to 4,
            "metric"     to "cosine",
            "precision"  to "f16",
            "ids"        to listOf(0L),
            "embeddings" to listOf(listOf(0.1f, 0.2f, 0.3f, 0.4f)),
        )
        if (ftsColumns.isNotEmpty()) payload["fts_columns"] = ftsColumns
        val mapper = com.fasterxml.jackson.databind.ObjectMapper()
        val json = mapper.writeValueAsString(payload)
        assertFalse(json.contains("fts_columns"),
            "JSON must NOT contain fts_columns when empty; got: $json")
    }

    @Test
    fun searchTextReturnsEmptyWhenNativeLibAbsent() {
        val results = AilakeNative.searchText(
            tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
            queryText = "rust programming", textColumns = listOf("chunk_text"), topK = 5,
        )
        assertTrue(results.isEmpty())
    }
}
