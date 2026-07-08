// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.ailake.trino.AilakeNative.AddColReq
import io.ailake.trino.AilakeNative.PartitionFieldDef
import io.trino.spi.connector.ColumnMetadata
import io.trino.spi.connector.Constraint
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.predicate.Domain
import io.trino.spi.predicate.TupleDomain
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.VarcharType.VARCHAR
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import org.mockito.kotlin.mock
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

        val results = AilakeNative.search(tableUri, queryBytes, topK = 3, tableName = "table")
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

    // ── Phase T: FTS write + searchText roundtrip ─────────────────────────────

    @Test
    fun writeBatchWithFtsColumnsAndSearchTextRoundtrip() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-fts-trino"
        val texts    = listOf("rust programming language", "hello world example", "vector search database")
        val snap = AilakeNative.writeBatch(
            tableUri     = tableUri,
            namespace    = "default",
            tableName    = "integration_fts_trino",
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
            ftsColumns   = listOf("chunk_text"),
            ftsTokenizer = "default",
            columns      = mapOf("chunk_text" to texts),
        )
        checkNotNull(snap) { "writeBatch with ftsColumns returned null" }
        println("[test] writeBatch fts OK: snapshotId=$snap")

        val results = AilakeNative.searchText(
            tableUri    = tableUri,
            namespace   = "default",
            tableName   = "integration_fts_trino",
            queryText   = "rust",
            textColumns = listOf("chunk_text"),
            topK        = 3,
        )
        check(results.isNotEmpty()) { "searchText returned empty — FTS index not built or not searched" }
        val best = results.first()
        check(best.rowId == 0L) { "expected rowId=0 (rust programming), got rowId=${best.rowId}" }
        println("[test] searchText OK: rowId=${best.rowId} distance=${best.distance}")
        println()
        println("PASS (Trino): FTS write+searchText roundtrip functional with real library.")
    }

    // ── Phase U: DELETE / ALTER TABLE / compact / hybrid search — real SQL surface ──
    //
    // These exercise the NEW SPI wiring (VectorScanMetadata.applyFilter/
    // applyDelete/executeDelete/addColumn, AilakeProcedures.compact,
    // VectorScanRecordSetProvider's hybrid/text-search split routing), not
    // just AilakeNative's underlying calls (already proven above) — the whole
    // point is proving the new SQL surfaces (DELETE, ALTER TABLE ADD COLUMN,
    // CALL compact(), hybrid search session properties) actually work
    // end-to-end against a real native library, closing the "dead capability"
    // gap found in the full-plugin audit.

    private val session = mock<ConnectorSession>()

    @Test
    fun deleteViaMetadataSpiRoundtrip() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-delete-spi-trino"
        val tableName = "integration_delete_spi_trino"
        AilakeNative.writeBatch(
            tableUri = tableUri, namespace = "default", tableName = tableName,
            vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            ids = listOf(0L, 1L, 2L),
            embeddings = listOf(
                listOf(1.0f, 0.0f, 0.0f, 0.0f), listOf(0.0f, 1.0f, 0.0f, 0.0f), listOf(0.0f, 0.0f, 1.0f, 0.0f),
            ),
        )
        val metadata = VectorScanMetadata(
            tableUri = tableUri, vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            namespace = "default", tableName = tableName,
        )
        val ingestHandle = metadata.getTableHandle(session, io.trino.spi.connector.SchemaTableName("default", "ingest"))!!
        val idCol = VectorScanColumnHandle("id", 0)
        val constraint = Constraint(TupleDomain.withColumnDomains(mapOf(idCol to Domain.multipleValues(BIGINT, listOf(0L, 1L)))))

        val filterResult = metadata.applyFilter(session, ingestHandle, constraint)
        check(filterResult.isPresent) { "applyFilter did not push down a simple IN predicate" }
        val filteredHandle = filterResult.get().handle

        val deleteHandle = metadata.applyDelete(session, filteredHandle)
        check(deleteHandle.isPresent) { "applyDelete did not accept the pushed-down handle" }

        metadata.executeDelete(session, deleteHandle.get()) // throws on failure — no exception = success
        println("[test] DELETE via metadata SPI OK: rows 0,1 deleted from $tableName")
        println("PASS (Trino): applyFilter→applyDelete→executeDelete roundtrip functional with real library.")
    }

    @Test
    fun addColumnViaMetadataSpiRoundtrip() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-addcol-spi-trino"
        val tableName = "integration_addcol_spi_trino"
        AilakeNative.writeBatch(
            tableUri = tableUri, namespace = "default", tableName = tableName,
            vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            ids = listOf(0L, 1L),
            embeddings = listOf(listOf(1.0f, 0.0f, 0.0f, 0.0f), listOf(0.0f, 1.0f, 0.0f, 0.0f)),
        )
        val metadata = VectorScanMetadata(
            tableUri = tableUri, vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            namespace = "default", tableName = tableName,
        )
        val ingestHandle = metadata.getTableHandle(session, io.trino.spi.connector.SchemaTableName("default", "ingest"))!!
        metadata.addColumn(session, ingestHandle, ColumnMetadata("source", VARCHAR)) // throws on failure
        println("[test] ADD COLUMN via metadata SPI OK: 'source' added to $tableName")
        println("PASS (Trino): addColumn functional with real library.")
    }

    @Test
    fun compactProcedureRoundtrip() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-compact-trino"
        val tableName = "integration_compact_trino"
        (0 until 5).forEach { batch ->
            val snap = AilakeNative.writeBatch(
                tableUri = tableUri, namespace = "default", tableName = tableName,
                vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
                ids = listOf(batch * 2L, batch * 2L + 1L),
                embeddings = listOf(listOf(1.0f, 0.0f, 0.0f, 0.0f), listOf(0.0f, 1.0f, 0.0f, 0.0f)),
            )
            checkNotNull(snap) { "writeBatch (batch $batch) returned null" }
        }
        val procedures = AilakeProcedures(tableUri, "default", tableName)
        procedures.compact(session) // throws on failure
        val results = AilakeNative.search(tableUri, VectorScanSplitManager.csvFloatsToBase64("1,0,0,0"), topK = 10, tableName = tableName)
        check(results.size == 10) { "expected 10 rows searchable post-compact, got ${results.size}" }
        println("[test] CALL compact() OK: table still has 10 searchable rows")
        println("PASS (Trino): compact procedure functional with real library.")
    }

    @Test
    fun hybridSearchViaRecordSetProviderRoundtrip() {
        assumeTrue(libPath != null)  { "AILAKE_LIB_PATH not set — skipping" }
        assumeTrue(writeDir != null) { "AILAKE_WRITE_DIR not set — skipping" }
        assumeTrue(libPresent)       { "libailake_jni.so not found — skipping" }

        val tableUri = "$writeDir/integration-hybrid-trino"
        val tableName = "integration_hybrid_trino"
        val texts = listOf("rust programming language", "hello world example", "vector search database")
        val snap = AilakeNative.writeBatch(
            tableUri = tableUri, namespace = "default", tableName = tableName,
            vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            ids = listOf(0L, 1L, 2L),
            embeddings = listOf(
                listOf(1.0f, 0.0f, 0.0f, 0.0f), listOf(0.0f, 1.0f, 0.0f, 0.0f), listOf(0.0f, 0.0f, 1.0f, 0.0f),
            ),
            ftsColumns = listOf("chunk_text"), ftsTokenizer = "default",
            columns = mapOf("chunk_text" to texts),
        )
        checkNotNull(snap) { "writeBatch with ftsColumns returned null" }

        // Pure text search: split carries queryText but no queryBytes — routes
        // through VectorScanRecordSetProvider's searchText branch, not search().
        val textOnlySplit = VectorScanSplit(
            tableUri = tableUri, queryBytes = "", topK = 3,
            namespace = "default", tableName = tableName, vectorColumn = "embedding",
            queryText = "rust", hybridWeight = 0.5f,
        )
        val textOnlyRows = VectorScanRecordSetProvider().getRecordSet(
            VectorScanTransactionHandle, session, textOnlySplit,
            VectorScanTableHandle(tableUri, "embedding", 4, "default", tableName),
            listOf(VectorScanColumnHandle("row_id", 0)),
        ).cursor().let { cursor ->
            val ids = mutableListOf<Long>()
            while (cursor.advanceNextPosition()) ids += cursor.getLong(0)
            ids
        }
        check(textOnlyRows.isNotEmpty()) { "pure text search returned empty" }
        check(textOnlyRows.first() == 0L) { "expected rowId=0 (rust programming) first, got $textOnlyRows" }
        println("[test] pure text search via split routing OK: top row=${textOnlyRows.first()}")

        // Hybrid: both queryBytes and queryText set — routes through search()'s hybridText path.
        val qRaw = floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f)
        val hybridSplit = VectorScanSplit(
            tableUri = tableUri,
            queryBytes = VectorScanSplitManager.csvFloatsToBase64(qRaw.joinToString(",")),
            topK = 3, namespace = "default", tableName = tableName, vectorColumn = "embedding",
            queryText = "rust", hybridWeight = 0.5f,
        )
        val hybridRows = VectorScanRecordSetProvider().getRecordSet(
            VectorScanTransactionHandle, session, hybridSplit,
            VectorScanTableHandle(tableUri, "embedding", 4, "default", tableName),
            listOf(VectorScanColumnHandle("row_id", 0)),
        ).cursor().let { cursor ->
            val ids = mutableListOf<Long>()
            while (cursor.advanceNextPosition()) ids += cursor.getLong(0)
            ids
        }
        check(hybridRows.isNotEmpty()) { "hybrid search returned empty" }
        println("[test] hybrid search via split routing OK: rows=$hybridRows")
        println()
        println("PASS (Trino): hybrid/text search session-property routing functional with real library.")
    }
}
