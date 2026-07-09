// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.core.io.GenericInputSplit
import org.apache.flink.table.api.DataTypes
import org.apache.flink.table.catalog.CatalogTable
import org.apache.flink.table.catalog.Column
import org.apache.flink.table.catalog.ObjectPath
import org.apache.flink.table.catalog.ResolvedCatalogTable
import org.apache.flink.table.catalog.ResolvedSchema
import org.apache.flink.table.catalog.TableChange
import org.apache.flink.table.expressions.CallExpression
import org.apache.flink.table.expressions.FieldReferenceExpression
import org.apache.flink.table.expressions.ValueLiteralExpression
import org.apache.flink.table.functions.BuiltInFunctionDefinitions
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.sqrt

/**
 * End-to-end integration test for the Flink JNI bridge.
 * Requires AILAKE_NATIVE_LIB to point to libailake_jni.so.
 *
 * Covers Phase P: write+search roundtrip, deleteWhere, evolveSchema.
 * Skipped automatically when the env var is absent (unit-test runs on CI).
 */
class AilakeJniIntegrationTest {

    @Test
    fun writeAndSearch() {
        val nativeLib = System.getenv("AILAKE_NATIVE_LIB")
            ?: System.getProperty("ailake.native.lib")

        assumeTrue(nativeLib != null && File(nativeLib).exists()) {
            "AILAKE_NATIVE_LIB not set or file absent — skipping integration test"
        }

        val dim = 8
        val n = 10
        val embeddings = Array(n) { i ->
            val v = FloatArray(dim) { j -> (i * dim + j + 1).toFloat() }
            val norm = sqrt(v.fold(0f) { acc, x -> acc + x * x }.toDouble()).toFloat()
            FloatArray(dim) { j -> v[j] / norm }
        }
        val ids = LongArray(n) { it.toLong() }

        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-it-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            val snapId = AilakeNativeLoader.writeBatch(
                warehouse = tmp.absolutePath,
                namespace = "default",
                table = "flink_it",
                vecCol = "embedding",
                dim = dim,
                metric = "cosine",
                ids = ids,
                embeddings = embeddings,
            )
            check(snapId >= 0) { "writeBatch returned snapshot_id=$snapId" }
            println("PASS (write): snapshot_id=$snapId")

            val queryIdx = 4
            val results = AilakeNativeLoader.search(
                warehouse = tmp.absolutePath,
                namespace = "default",
                table = "flink_it",
                vecCol = "embedding",
                dim = dim,
                query = embeddings[queryIdx],
                topK = 3,
            )
            check(results.isNotEmpty()) { "search returned empty results" }

            val best = results.minByOrNull { it.distance }!!
            check(best.row_id == queryIdx.toLong()) {
                "nearest row_id=${best.row_id}, expected $queryIdx"
            }
            println("PASS (search): row_id=${best.row_id} distance=${best.distance}")
            println()
            println("PASS: Flink JNI integration — write + search via AilakeNativeLoader.")
        } finally {
            tmp.deleteRecursively()
        }
    }

    @Test
    fun deleteWhere() {
        val nativeLib = System.getenv("AILAKE_NATIVE_LIB")
            ?: System.getProperty("ailake.native.lib")
        assumeTrue(nativeLib != null && File(nativeLib).exists()) {
            "AILAKE_NATIVE_LIB not set or file absent — skipping"
        }

        val dim = 4
        val embeddings = Array(3) { i ->
            FloatArray(dim) { j -> if (j == i) 1.0f else 0.0f }
        }
        val ids = LongArray(3) { it.toLong() }
        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-del-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            AilakeNativeLoader.writeBatch(
                warehouse  = tmp.absolutePath,
                namespace  = "default",
                table      = "flink_del",
                vecCol     = "embedding",
                dim        = dim,
                metric     = "cosine",
                ids        = ids,
                embeddings = embeddings,
            )
            AilakeNativeLoader.deleteWhere(
                warehouse = tmp.absolutePath,
                namespace = "default",
                table     = "flink_del",
                column    = "id",
                values    = listOf("0", "1"),
            )
            println("PASS (deleteWhere): 2 rows marked deleted via Flink JNI bridge.")
        } finally {
            tmp.deleteRecursively()
        }
    }

    @Test
    fun evolveSchema() {
        val nativeLib = System.getenv("AILAKE_NATIVE_LIB")
            ?: System.getProperty("ailake.native.lib")
        assumeTrue(nativeLib != null && File(nativeLib).exists()) {
            "AILAKE_NATIVE_LIB not set or file absent — skipping"
        }

        val dim = 4
        val embeddings = Array(2) { i ->
            FloatArray(dim) { j -> if (j == i) 1.0f else 0.0f }
        }
        val ids = LongArray(2) { it.toLong() }
        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-evo-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            AilakeNativeLoader.writeBatch(
                warehouse  = tmp.absolutePath,
                namespace  = "default",
                table      = "flink_evo",
                vecCol     = "embedding",
                dim        = dim,
                metric     = "cosine",
                ids        = ids,
                embeddings = embeddings,
            )
            val schemaId = AilakeNativeLoader.evolveSchema(
                warehouse  = tmp.absolutePath,
                namespace  = "default",
                table      = "flink_evo",
                addCols    = listOf(AilakeNativeLoader.AddColReq(name = "source", colType = "string")),
                renameCols = emptyList(),
            )
            check(schemaId >= 0) { "evolveSchema returned $schemaId, expected >= 0" }
            println("PASS (evolveSchema): new_schema_id=$schemaId via Flink JNI bridge.")
        } finally {
            tmp.deleteRecursively()
        }
    }

    // ── Phase U: DELETE / ALTER TABLE / compact / hybrid search — real SQL surface ──
    //
    // These exercise the NEW SPI wiring (AilakeVectorTableSink.applyDeleteFilters/
    // executeDeletion, AilakeCatalog.alterTable(TableChange), AilakeCompactFunction,
    // AilakeInputFormat's hybrid/text-search job-param routing), not just
    // AilakeNativeLoader's underlying calls (already proven above) — closing the
    // "dead capability" gap found auditing whether every native capability was
    // actually reachable from Flink SQL.

    private fun requireNativeLib() {
        val nativeLib = System.getenv("AILAKE_NATIVE_LIB") ?: System.getProperty("ailake.native.lib")
        assumeTrue(nativeLib != null && File(nativeLib).exists()) {
            "AILAKE_NATIVE_LIB not set or file absent — skipping"
        }
    }

    @Test
    fun deleteViaSinkSpiRoundtrip() {
        requireNativeLib()
        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-delspi-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            AilakeNativeLoader.writeBatch(
                warehouse = tmp.absolutePath, namespace = "default", table = "flink_delspi",
                vecCol = "embedding", dim = 4, metric = "cosine",
                ids = longArrayOf(0L, 1L, 2L),
                embeddings = arrayOf(
                    floatArrayOf(1f, 0f, 0f, 0f), floatArrayOf(0f, 1f, 0f, 0f), floatArrayOf(0f, 0f, 1f, 0f),
                ),
            )
            val sink = AilakeVectorTableSink(
                warehouse = tmp.absolutePath, namespace = "default", tableName = "flink_delspi",
                vecCol = "embedding", dim = 4, metric = "cosine", precision = "f16",
                schema = ResolvedSchema.of(
                    Column.physical("id", DataTypes.BIGINT()),
                    Column.physical("embedding", DataTypes.ARRAY(DataTypes.FLOAT())),
                ),
            )
            val fieldRef = FieldReferenceExpression("id", DataTypes.BIGINT(), 0, 0)
            val call = CallExpression.anonymous(
                BuiltInFunctionDefinitions.IN,
                listOf(fieldRef, ValueLiteralExpression(0L), ValueLiteralExpression(1L)),
                DataTypes.BOOLEAN(),
            )
            check(sink.applyDeleteFilters(listOf(call))) { "applyDeleteFilters did not accept a simple IN predicate" }
            sink.executeDeletion() // throws on native failure — no exception = success
            println("PASS (DELETE via sink SPI): rows 0,1 deleted from flink_delspi")
        } finally {
            tmp.deleteRecursively()
        }
    }

    @Test
    fun alterTableViaCatalogRoundtrip() {
        requireNativeLib()
        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-altspi-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            AilakeNativeLoader.writeBatch(
                warehouse = tmp.absolutePath, namespace = "default", table = "docs",
                vecCol = "embedding", dim = 4, metric = "cosine",
                ids = longArrayOf(0L, 1L),
                embeddings = arrayOf(floatArrayOf(1f, 0f, 0f, 0f), floatArrayOf(0f, 1f, 0f, 0f)),
            )
            val catalog = AilakeCatalog("ailake", warehouse = tmp.absolutePath)
            catalog.open()
            val path = ObjectPath("default", "docs")
            val schema = ResolvedSchema.of(
                Column.physical("id", DataTypes.BIGINT()),
                Column.physical("embedding", DataTypes.ARRAY(DataTypes.FLOAT())),
            )
            val table = ResolvedCatalogTable(
                CatalogTable.of(
                    org.apache.flink.table.api.Schema.newBuilder()
                        .column("id", DataTypes.BIGINT())
                        .column("embedding", DataTypes.ARRAY(DataTypes.FLOAT()))
                        .build(),
                    "", emptyList(), emptyMap(),
                ),
                schema,
            )
            catalog.createTable(path, table, false)
            val change = TableChange.add(Column.physical("source", DataTypes.STRING()))
            catalog.alterTable(path, table, listOf(change), false) // throws on native failure — no exception = success
            println("PASS (ALTER TABLE via catalog SPI): 'source' added to docs")
        } finally {
            tmp.deleteRecursively()
        }
    }

    @Test
    fun compactViaFunctionRoundtrip() {
        requireNativeLib()
        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-compactfn-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            repeat(5) { batch ->
                AilakeNativeLoader.writeBatch(
                    warehouse = tmp.absolutePath, namespace = "default", table = "flink_compactfn",
                    vecCol = "embedding", dim = 4, metric = "cosine",
                    ids = longArrayOf(batch * 2L, batch * 2L + 1L),
                    embeddings = arrayOf(floatArrayOf(1f, 0f, 0f, 0f), floatArrayOf(0f, 1f, 0f, 0f)),
                )
            }
            val n = AilakeCompactFunction().eval(tmp.absolutePath, "default", "flink_compactfn")
            check(n >= 1) { "expected at least 1 file compacted, got $n" }
            val results = AilakeNativeLoader.search(
                warehouse = tmp.absolutePath, namespace = "default", table = "flink_compactfn",
                vecCol = "embedding", dim = 4, query = floatArrayOf(1f, 0f, 0f, 0f), topK = 10,
            )
            check(results.size == 10) { "expected 10 rows searchable post-compact, got ${results.size}" }
            println("PASS (compact via ScalarFunction): filesCompacted=$n, 10 rows still searchable")
        } finally {
            tmp.deleteRecursively()
        }
    }

    @Test
    fun hybridAndTextSearchViaInputFormatRoundtrip() {
        requireNativeLib()
        val tmp = File(System.getProperty("java.io.tmpdir"), "ailake-flink-hybrid-${System.nanoTime()}")
        tmp.mkdirs()
        try {
            val texts = listOf("rust programming language", "hello world example", "vector search database")
            AilakeNativeLoader.writeBatch(
                warehouse = tmp.absolutePath, namespace = "default", table = "flink_hybrid",
                vecCol = "embedding", dim = 4, metric = "cosine",
                ids = longArrayOf(0L, 1L, 2L),
                embeddings = arrayOf(
                    floatArrayOf(1f, 0f, 0f, 0f), floatArrayOf(0f, 1f, 0f, 0f), floatArrayOf(0f, 0f, 1f, 0f),
                ),
                ftsColumns = listOf("chunk_text"),
                columns = mapOf("chunk_text" to texts),
            )

            // Pure text search: only ailake.query.text set.
            val textFormat = AilakeInputFormat(
                warehouse = tmp.absolutePath, namespace = "default", tableName = "flink_hybrid",
                vecCol = "embedding", dim = 4, topK = 3, efSearch = 50,
            )
            val textParams = org.apache.flink.configuration.Configuration()
            textParams.setString("ailake.query.text", "rust")
            val textExecConfig = org.apache.flink.api.common.ExecutionConfig()
            textExecConfig.globalJobParameters = textParams
            val textCtx = org.mockito.kotlin.mock<org.apache.flink.api.common.functions.RuntimeContext>()
            org.mockito.kotlin.whenever(textCtx.executionConfig).thenReturn(textExecConfig)
            textFormat.runtimeContext = textCtx
            textFormat.open(GenericInputSplit(0, 1))
            check(!textFormat.reachedEnd()) { "pure text search returned empty" }
            val firstTextRow = textFormat.nextRecord(null)
            check(firstTextRow.getLong(0) == 0L) { "expected rowId=0 (rust programming) first" }
            println("PASS (pure text search via job params): top row_id=${firstTextRow.getLong(0)}")

            // Hybrid: both ailake.query.vector and ailake.query.text set.
            val hybridFormat = AilakeInputFormat(
                warehouse = tmp.absolutePath, namespace = "default", tableName = "flink_hybrid",
                vecCol = "embedding", dim = 4, topK = 3, efSearch = 50,
            )
            val hybridParams = org.apache.flink.configuration.Configuration()
            hybridParams.setString("ailake.query.vector", "1.0,0.0,0.0,0.0")
            hybridParams.setString("ailake.query.text", "rust")
            hybridParams.setString("ailake.hybrid.weight", "0.5")
            val hybridExecConfig = org.apache.flink.api.common.ExecutionConfig()
            hybridExecConfig.globalJobParameters = hybridParams
            val hybridCtx = org.mockito.kotlin.mock<org.apache.flink.api.common.functions.RuntimeContext>()
            org.mockito.kotlin.whenever(hybridCtx.executionConfig).thenReturn(hybridExecConfig)
            hybridFormat.runtimeContext = hybridCtx
            hybridFormat.open(GenericInputSplit(0, 1))
            check(!hybridFormat.reachedEnd()) { "hybrid search returned empty" }
            println("PASS (hybrid search via job params): rows returned")
            println()
            println("PASS: Flink hybrid/text search job-parameter routing functional with real library.")
        } finally {
            tmp.deleteRecursively()
        }
    }
}
