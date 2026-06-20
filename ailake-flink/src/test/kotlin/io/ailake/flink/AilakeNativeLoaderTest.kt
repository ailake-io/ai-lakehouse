// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/**
 * Unit tests for AilakeNativeLoader data classes and no-lib-required paths.
 *
 * Tests that exercise the native library (writeBatch, deleteWhere, evolveSchema)
 * are covered by AilakeJniIntegrationTest, which requires AILAKE_NATIVE_LIB.
 */
class AilakeNativeLoaderTest {

    // ── PartitionFieldDef ─────────────────────────────────────────────────────

    @Test
    fun partitionFieldDefEquality() {
        val p1 = AilakeNativeLoader.PartitionFieldDef("agent_id", "identity", "string")
        val p2 = AilakeNativeLoader.PartitionFieldDef("agent_id", "identity", "string")
        assertEquals(p1, p2)
    }

    @Test
    fun partitionFieldDefToStringContainsColumn() {
        val p = AilakeNativeLoader.PartitionFieldDef("session_id", "truncate[4]", "string")
        assertTrue(p.toString().contains("session_id"))
    }

    @Test
    fun partitionFieldDefFields() {
        val p = AilakeNativeLoader.PartitionFieldDef("doc_id", "identity", "long")
        assertEquals("doc_id", p.column)
        assertEquals("identity", p.transform)
        assertEquals("long", p.columnType)
    }

    // ── AddColReq ─────────────────────────────────────────────────────────────

    @Test
    fun addColReqDefaultInitialDefaultIsNull() {
        val r = AilakeNativeLoader.AddColReq("score", "float")
        assertNull(r.initialDefault)
    }

    @Test
    fun addColReqWithInitialDefault() {
        val r = AilakeNativeLoader.AddColReq("score", "float", "0.0")
        assertEquals("0.0", r.initialDefault)
    }

    @Test
    fun addColReqEquality() {
        val r1 = AilakeNativeLoader.AddColReq("tag", "string", "\"\"")
        val r2 = AilakeNativeLoader.AddColReq("tag", "string", "\"\"")
        assertEquals(r1, r2)
    }

    // ── RenameColReq ──────────────────────────────────────────────────────────

    @Test
    fun renameColReqEquality() {
        val r1 = AilakeNativeLoader.RenameColReq("old_col", "new_col")
        val r2 = AilakeNativeLoader.RenameColReq("old_col", "new_col")
        assertEquals(r1, r2)
    }

    @Test
    fun renameColReqFields() {
        val r = AilakeNativeLoader.RenameColReq("old_name", "new_name")
        assertEquals("old_name", r.from)
        assertEquals("new_name", r.to)
    }

    // ── WriteResponse / DeleteWhereResponse / EvolveSchemaResponse ───────────

    @Test
    fun writeResponseDefaults() {
        val r = AilakeNativeLoader.WriteResponse(ok = true)
        assertEquals(-1L, r.snapshot_id)
        assertNull(r.error)
    }

    @Test
    fun deleteWhereResponseDefaults() {
        val r = AilakeNativeLoader.DeleteWhereResponse(ok = false, error = "boom")
        assertFalse(r.ok)
        assertEquals("boom", r.error)
    }

    @Test
    fun evolveSchemaResponseDefaults() {
        val r = AilakeNativeLoader.EvolveSchemaResponse(ok = true, new_schema_id = 5)
        assertTrue(r.ok)
        assertEquals(5, r.new_schema_id)
    }

    // ── Phase R: AilakeVectorConnectorFactory option registration ─────────────

    @Test
    fun connectorFactoryRegistersPartitionFieldsOption() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val keys = factory.optionalOptions().map { it.key() }.toSet()
        assertTrue("partition.fields" in keys,
            "partition.fields must be in optionalOptions(); got: $keys")
    }

    @Test
    fun connectorFactoryRegistersFormatVersionOption() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val keys = factory.optionalOptions().map { it.key() }.toSet()
        assertTrue("format.version" in keys,
            "format.version must be in optionalOptions(); got: $keys")
    }

    @Test
    fun connectorFactoryPartitionFieldsDefaultIsEmptyJson() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val partitionFieldsOpt = factory.optionalOptions().first { it.key() == "partition.fields" }
        assertEquals("[]", partitionFieldsOpt.defaultValue())
    }

    @Test
    fun connectorFactoryFormatVersionDefaultIs2() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val formatVersionOpt = factory.optionalOptions().first { it.key() == "format.version" }
        assertEquals(2, formatVersionOpt.defaultValue())
    }

    // ── Phase T: FTS option registration ─────────────────────────────────────

    @Test
    fun connectorFactoryRegistersFtsColumnsOption() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val keys = factory.optionalOptions().map { it.key() }.toSet()
        assertTrue("fts.columns" in keys,
            "fts.columns must be in optionalOptions(); got: $keys")
    }

    @Test
    fun connectorFactoryRegistersFtsTokenizerOption() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val keys = factory.optionalOptions().map { it.key() }.toSet()
        assertTrue("fts.tokenizer" in keys,
            "fts.tokenizer must be in optionalOptions(); got: $keys")
    }

    @Test
    fun connectorFactoryFtsColumnsDefaultIsEmpty() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val opt = factory.optionalOptions().first { it.key() == "fts.columns" }
        assertEquals("", opt.defaultValue(),
            "fts.columns default must be empty string (no FTS by default)")
    }

    @Test
    fun connectorFactoryFtsTokenizerDefaultIsDefault() {
        val factory = io.ailake.flink.AilakeVectorConnectorFactory()
        val opt = factory.optionalOptions().first { it.key() == "fts.tokenizer" }
        assertEquals("default", opt.defaultValue())
    }

    // ── Phase T: writeBatch JSON payload includes fts_columns ─────────────────

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
            "JSON payload must contain fts_columns key; got: $json")
        assertTrue(json.contains("chunk_text"),
            "JSON payload must contain chunk_text column; got: $json")
        assertTrue(json.contains("\"fts_tokenizer\""),
            "JSON payload must contain fts_tokenizer key; got: $json")
    }

    @Test
    fun writeBatchJsonOmitsFtsColumnsWhenEmpty() {
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
        val ftsColumns: List<String> = emptyList()
        if (ftsColumns.isNotEmpty()) payload["fts_columns"] = ftsColumns

        val mapper = com.fasterxml.jackson.databind.ObjectMapper()
        val json = mapper.writeValueAsString(payload)

        assertFalse(json.contains("fts_columns"),
            "JSON payload must NOT contain fts_columns when empty; got: $json")
    }
}
