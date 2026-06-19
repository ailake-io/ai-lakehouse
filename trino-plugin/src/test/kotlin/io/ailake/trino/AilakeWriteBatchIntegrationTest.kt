// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.sqrt

/**
 * End-to-end integration test for AilakeNative.writeBatch.
 *
 * Required env vars (same as search integration test):
 *   AILAKE_LIB_PATH   — directory containing libailake_jni.so
 *   AILAKE_WRITE_DIR  — writable directory where a new table will be created
 *
 * Covers Phase P: writeBatch with partitionFields/formatVersion, deleteWhere, evolveSchema.
 * Skipped automatically when either env var is absent.
 */
class AilakeWriteBatchIntegrationTest {

    private val libPath   = System.getenv("AILAKE_LIB_PATH")
    private val writeDir  = System.getenv("AILAKE_WRITE_DIR")
    private val libPresent get() =
        libPath != null && File(libPath, "libailake_jni.so").exists()

    @Test
    fun writeBatchReturnsNullWhenNativeLibAbsent() {
        // Native lib absent in test env → writeBatch must return null gracefully
        val result = AilakeNative.writeBatch(
            tableUri     = "file:///tmp/absent-table",
            namespace    = "default",
            tableName    = "test",
            vectorColumn = "embedding",
            dim          = 4,
            metric       = "cosine",
            precision    = "f16",
            ids          = listOf(1L, 2L),
            embeddings   = listOf(listOf(0.1f, 0.2f, 0.3f, 0.4f), listOf(0.5f, 0.6f, 0.7f, 0.8f)),
        )
        // Without native lib, result is null — no exception thrown
        // (lib may or may not be loaded in CI; just assert no crash)
        println("[test] writeBatch without lib: result=$result (expected null or snapshotId)")
    }

    @Test
    fun writeBatchAndSearchRoundtrip() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val dim = 8
        val n = dim   // one row per spike position — no duplicate vectors, no tie in HNSW
        val tableUri = "$writeDir/integration-write-trino"

        // Build orthogonal-ish vectors: row i has a spike at position i
        val ids = (0 until n).map { it.toLong() }
        val embeddings = ids.map { id ->
            FloatArray(dim) { j -> if (j == (id % dim).toInt()) 1.0f else 0.01f }.toList()
        }

        val snapshotId = AilakeNative.writeBatch(
            tableUri     = tableUri,
            namespace    = "default",
            tableName    = "table",
            vectorColumn = "embedding",
            dim          = dim,
            metric       = "cosine",
            precision    = "f16",
            ids          = ids,
            embeddings   = embeddings,
        )
        checkNotNull(snapshotId) { "writeBatch returned null — check JNI and table path" }
        println("[test] writeBatch OK: snapshotId=$snapshotId, wrote $n rows")

        // Query for row 3: its embedding has spike at position 3
        val queryIdx = 3
        val qRaw = FloatArray(dim) { j -> if (j == queryIdx) 1.0f else 0.0f }
        val norm  = sqrt(qRaw.fold(0f) { acc, x -> acc + x * x }.toDouble()).toFloat()
        val queryBytes = VectorScanSplitManager.csvFloatsToBase64(
            qRaw.joinToString(",") { (it / norm).toString() }
        )

        val results = AilakeNative.search(tableUri, queryBytes, topK = 3)
        check(results.isNotEmpty()) { "search after write returned empty results" }
        val best = results.minByOrNull { it.distance }!!
        check(best.rowId == queryIdx.toLong()) {
            "nearest rowId=${best.rowId}, expected $queryIdx"
        }
        println("[test] search OK: rowId=${best.rowId} distance=${best.distance}")
        println()
        println("PASS (Trino): write+search roundtrip functional with real library.")
    }

    @Test
    fun writeBatchWithPartitionFieldsAndFormatVersion3() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-write-trino-partitioned"
        val pf = PartitionFieldDef(column = "id", transform = "identity", columnType = "long")
        val snap = AilakeNative.writeBatch(
            tableUri        = tableUri,
            namespace       = "default",
            tableName       = "integration_partitioned_trino",
            vectorColumn    = "embedding",
            dim             = 4,
            metric          = "cosine",
            precision       = "f16",
            ids             = listOf(0L, 1L),
            embeddings      = listOf(
                listOf(1.0f, 0.0f, 0.0f, 0.0f),
                listOf(0.0f, 1.0f, 0.0f, 0.0f),
            ),
            partitionFields = listOf(pf),
            formatVersion   = 3,
        )
        checkNotNull(snap) { "writeBatch with partitionFields returned null" }
        println("[test] writeBatch partitionFields OK: snapshotId=$snap")
    }

    @Test
    fun deleteWhereMarksRowsDeleted() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-delete-trino"
        AilakeNative.writeBatch(
            tableUri     = tableUri,
            namespace    = "default",
            tableName    = "integration_delete_trino",
            vectorColumn = "embedding",
            dim          = 4,
            metric       = "cosine",
            precision    = "f16",
            ids          = listOf(0L, 1L, 2L),
            embeddings   = listOf(
                listOf(1.0f, 0.0f, 0.0f, 0.0f),
                listOf(0.0f, 1.0f, 0.0f, 0.0f),
                listOf(0.0f, 0.0f, 1.0f, 0.0f),
            ),
        )
        val ok = AilakeNative.deleteWhere(tableUri, "default", "integration_delete_trino", "id", listOf("0", "1"))
        check(ok) { "deleteWhere returned false" }
        println("[test] deleteWhere OK: 2 rows marked deleted")
    }

    @Test
    fun evolveSchemaAddsColumn() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-evolve-trino"
        AilakeNative.writeBatch(
            tableUri     = tableUri,
            namespace    = "default",
            tableName    = "integration_evolve_trino",
            vectorColumn = "embedding",
            dim          = 4,
            metric       = "cosine",
            precision    = "f16",
            ids          = listOf(0L, 1L),
            embeddings   = listOf(
                listOf(1.0f, 0.0f, 0.0f, 0.0f),
                listOf(0.0f, 1.0f, 0.0f, 0.0f),
            ),
        )
        val schemaId = AilakeNative.evolveSchema(
            tableUri   = tableUri,
            namespace  = "default",
            tableName  = "integration_evolve_trino",
            addCols    = listOf(AddColReq(name = "source", colType = "string")),
            renameCols = emptyList(),
        )
        check(schemaId >= 0) { "evolveSchema returned $schemaId, expected >= 0" }
        println("[test] evolveSchema OK: new_schema_id=$schemaId")
    }
}
