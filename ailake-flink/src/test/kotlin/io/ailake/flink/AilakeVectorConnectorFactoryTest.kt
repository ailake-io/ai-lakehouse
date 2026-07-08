// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.apache.flink.table.api.DataTypes
import org.apache.flink.table.api.ValidationException
import org.apache.flink.table.catalog.Column
import org.apache.flink.table.catalog.ResolvedSchema
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class AilakeVectorConnectorFactoryTest {

    @Test
    fun factoryIdentifier() {
        assertEquals("ailake", AilakeVectorConnectorFactory().factoryIdentifier())
    }

    @Test
    fun catalogFactoryIdentifier() {
        assertEquals("ailake", AilakeCatalogFactory().factoryIdentifier())
    }

    @Test
    fun requiredOptions() {
        val factory = AilakeVectorConnectorFactory()
        val keys = factory.requiredOptions().map { it.key() }
        assert("warehouse" in keys)
        assert("table-name" in keys)
        assert("vector.dim" in keys)
    }

    @Test
    fun optionalOptionsIncludesEmbeddingModel() {
        val keys = AilakeVectorConnectorFactory().optionalOptions().map { it.key() }
        assert("embedding.model" in keys) { "embedding.model missing from optionalOptions: $keys" }
    }

    @Test
    fun optionalOptionsIncludesFtsColumns() {
        val keys = AilakeVectorConnectorFactory().optionalOptions().map { it.key() }
        assert("fts.columns" in keys) { "fts.columns missing from optionalOptions: $keys" }
    }

    @Test
    fun optionalOptionsIncludesFtsTokenizer() {
        val keys = AilakeVectorConnectorFactory().optionalOptions().map { it.key() }
        assert("fts.tokenizer" in keys) { "fts.tokenizer missing from optionalOptions: $keys" }
    }

    @Test
    fun optionalOptionsIncludesWriteTuningKnobs() {
        val keys = AilakeVectorConnectorFactory().optionalOptions().map { it.key() }
        assert("hnsw.m" in keys) { "hnsw.m missing: $keys" }
        assert("hnsw.ef-construction" in keys) { "hnsw.ef-construction missing: $keys" }
        assert("pre-normalize" in keys) { "pre-normalize missing: $keys" }
        assert("deferred" in keys) { "deferred missing: $keys" }
    }

    // ── validateSearchResultSchema ──────────────────────────────────────────────
    //
    // Regression: AilakeInputFormat.nextRecord() always emitted a fixed
    // (row_id BIGINT, distance FLOAT, file_path STRING) row regardless of the
    // declared DDL schema — this connector's own doc example used to show a
    // 4-column ingest-shaped table for source tables too, which would
    // deserialize-crash on SELECT. Now validated at DDL-resolution time.

    @Test
    fun validateSearchResultSchemaAcceptsExactMatch() {
        val schema = ResolvedSchema.of(
            Column.physical("row_id", DataTypes.BIGINT()),
            Column.physical("distance", DataTypes.FLOAT()),
            Column.physical("file_path", DataTypes.STRING()),
        )
        assertDoesNotThrow { AilakeVectorConnectorFactory().validateSearchResultSchema(schema) }
    }

    @Test
    fun validateSearchResultSchemaRejectsIngestShapedSchema() {
        val schema = ResolvedSchema.of(
            Column.physical("id", DataTypes.BIGINT()),
            Column.physical("text", DataTypes.STRING()),
            Column.physical("embedding", DataTypes.BYTES()),
            Column.physical("_distance", DataTypes.FLOAT()),
        )
        val ex = assertThrows(ValidationException::class.java) {
            AilakeVectorConnectorFactory().validateSearchResultSchema(schema)
        }
        assert(ex.message!!.contains("row_id"))
    }

    @Test
    fun validateSearchResultSchemaRejectsWrongColumnOrder() {
        val schema = ResolvedSchema.of(
            Column.physical("distance", DataTypes.FLOAT()),
            Column.physical("row_id", DataTypes.BIGINT()),
            Column.physical("file_path", DataTypes.STRING()),
        )
        assertThrows(ValidationException::class.java) {
            AilakeVectorConnectorFactory().validateSearchResultSchema(schema)
        }
    }

    @Test
    fun validateSearchResultSchemaRejectsWrongColumnCount() {
        val schema = ResolvedSchema.of(
            Column.physical("row_id", DataTypes.BIGINT()),
            Column.physical("distance", DataTypes.FLOAT()),
        )
        assertThrows(ValidationException::class.java) {
            AilakeVectorConnectorFactory().validateSearchResultSchema(schema)
        }
    }
}
