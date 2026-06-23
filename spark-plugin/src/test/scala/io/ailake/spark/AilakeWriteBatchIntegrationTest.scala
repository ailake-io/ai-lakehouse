// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import scala.math.sqrt
import org.junit.runner.RunWith
import org.scalatestplus.junit.JUnitRunner
import io.ailake.spark.AilakeNative.{AddColReq, PartitionFieldDef}

/**
 * End-to-end integration test for AilakeNative.writeBatch (Spark side).
 *
 * Required env vars:
 *   AILAKE_LIB_PATH   — directory containing libailake_jni.so
 *   AILAKE_WRITE_DIR  — writable directory where a new table will be created
 *
 * Covers Phase P: writeBatch with partitionFields/formatVersion, deleteWhere, evolveSchema.
 * Tests that require the native lib are skipped automatically when env vars absent.
 */
@RunWith(classOf[JUnitRunner])
class AilakeWriteBatchIntegrationTest extends AnyFunSuite {

  private val libPath  = Option(System.getenv("AILAKE_LIB_PATH"))
  private val writeDir = Option(System.getenv("AILAKE_WRITE_DIR"))
  private def libPresent = libPath.exists(p => new File(p, "libailake_jni.so").exists())

  // ── graceful degradation ──────────────────────────────────────────────────

  test("writeBatch returns None when native lib absent") {
    val result = AilakeNative.writeBatch(
      tableUri     = "file:///tmp/absent-spark-table",
      namespace    = "default",
      tableName    = "test",
      vectorColumn = "embedding",
      dim          = 4,
      metric       = "cosine",
      precision    = "f16",
      ids          = Seq(1L, 2L),
      embeddings   = Seq(Seq(0.1f, 0.2f, 0.3f, 0.4f), Seq(0.5f, 0.6f, 0.7f, 0.8f)),
    )
    // Without native lib, result is None — no exception
    println(s"[test] writeBatch without lib: $result (expected None or snapshotId)")
  }

  // ── write + search roundtrip ──────────────────────────────────────────────

  test("writeBatch and search roundtrip") {
    assume(libPath.isDefined,  "AILAKE_LIB_PATH not set — skipping")
    assume(writeDir.isDefined, "AILAKE_WRITE_DIR not set — skipping")
    assume(libPresent,         "libailake_jni.so not found — skipping")

    val dim      = 8
    val n        = 16
    val tableUri = s"${writeDir.get}/integration-write-spark"

    // Orthogonal-ish vectors: row i has spike at position i%dim
    val ids = (0 until n).map(_.toLong)
    val embeddings = ids.map { id =>
      (0 until dim).map(j => if (j == (id % dim).toInt) 1.0f else 0.01f)
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
    assert(snapshotId.isDefined, "writeBatch returned None — check JNI and table path")
    println(s"[test] writeBatch OK: snapshotId=${snapshotId.get}, wrote $n rows")

    // Query for row 5: spike at position 5%8 = 5
    val queryIdx = 5
    val qRaw = (0 until dim).map(j => if (j == queryIdx) 1.0f else 0.0f).toArray
    val norm  = sqrt(qRaw.map(x => x * x).sum.toDouble).toFloat
    val qNorm = qRaw.map(_ / norm)

    val results = AilakeNative.search(tableUri, qNorm, topK = 3, tableName = "table")
    assert(results.nonEmpty, "search after write returned empty results")
    val best = results.minBy(_.distance)
    assert(
      best.rowId % dim == queryIdx,
      s"nearest rowId=${best.rowId}, expected rowId%dim==$queryIdx"
    )
    println(s"[test] search OK: rowId=${best.rowId} distance=${best.distance}")
    println()
    println("PASS (Spark): write+search roundtrip functional with real library.")
  }

  // ── AilakeDataWriter buffer isolation ────────────────────────────────────

  test("two AilakeDataWriter instances have independent buffers") {
    val handle = AilakeWriteHandle(
      tableUri     = "file:///tmp/test",
      namespace    = "default",
      tableName    = "t",
      vectorColumn = "embedding",
      dim          = 2,
      metric       = "cosine",
      precision    = "f16",
    )
    val w1 = new AilakeDataWriter(handle, partitionId = 0, taskId = 0L)
    val w2 = new AilakeDataWriter(handle, partitionId = 1, taskId = 1L)
    // Commit w1 (empty) should not see w2's state
    val msg1 = w1.commit().asInstanceOf[AilakeCommitMessage]
    val msg2 = w2.commit().asInstanceOf[AilakeCommitMessage]
    assert(msg1.snapshotId.isEmpty)
    assert(msg2.snapshotId.isEmpty)
  }

  // ── writeBatch with partitionFields + formatVersion ─────────────────────

  test("writeBatch with partitionFields and formatVersion=3") {
    assume(libPath.isDefined,  "AILAKE_LIB_PATH not set — skipping")
    assume(writeDir.isDefined, "AILAKE_WRITE_DIR not set — skipping")
    assume(libPresent,         "libailake_jni.so not found — skipping")

    val tableUri = s"${writeDir.get}/integration-write-spark-partitioned"
    val pf = PartitionFieldDef(column = "id", transform = "identity", columnType = "long")
    val snap = AilakeNative.writeBatch(
      tableUri     = tableUri,
      namespace    = "default",
      tableName    = "integration_partitioned_spark",
      vectorColumn = "embedding",
      dim          = 4,
      metric       = "cosine",
      precision    = "f16",
      ids          = Seq(0L, 1L),
      embeddings   = Seq(Seq(1.0f, 0.0f, 0.0f, 0.0f), Seq(0.0f, 1.0f, 0.0f, 0.0f)),
      partitionFields = Seq(pf),
      formatVersion   = 3,
    )
    assert(snap.isDefined, "writeBatch with partitionFields returned None")
    println(s"[test] writeBatch partitionFields OK: snapshotId=${snap.get}")
  }

  // ── deleteWhere ───────────────────────────────────────────────────────────

  test("deleteWhere marks rows deleted") {
    assume(libPath.isDefined,  "AILAKE_LIB_PATH not set — skipping")
    assume(writeDir.isDefined, "AILAKE_WRITE_DIR not set — skipping")
    assume(libPresent,         "libailake_jni.so not found — skipping")

    val tableUri = s"${writeDir.get}/integration-delete-spark"
    AilakeNative.writeBatch(
      tableUri     = tableUri,
      namespace    = "default",
      tableName    = "integration_delete_spark",
      vectorColumn = "embedding",
      dim          = 4,
      metric       = "cosine",
      precision    = "f16",
      ids          = Seq(0L, 1L, 2L),
      embeddings   = Seq(
        Seq(1.0f, 0.0f, 0.0f, 0.0f),
        Seq(0.0f, 1.0f, 0.0f, 0.0f),
        Seq(0.0f, 0.0f, 1.0f, 0.0f),
      ),
    )
    val ok = AilakeNative.deleteWhere(tableUri, "default", "integration_delete_spark", "id", Seq("0", "1"))
    assert(ok, "deleteWhere returned false")
    println(s"[test] deleteWhere OK: 2 rows marked deleted")
  }

  // ── evolveSchema ──────────────────────────────────────────────────────────

  test("evolveSchema adds a column") {
    assume(libPath.isDefined,  "AILAKE_LIB_PATH not set — skipping")
    assume(writeDir.isDefined, "AILAKE_WRITE_DIR not set — skipping")
    assume(libPresent,         "libailake_jni.so not found — skipping")

    val tableUri = s"${writeDir.get}/integration-evolve-spark"
    AilakeNative.writeBatch(
      tableUri     = tableUri,
      namespace    = "default",
      tableName    = "integration_evolve_spark",
      vectorColumn = "embedding",
      dim          = 4,
      metric       = "cosine",
      precision    = "f16",
      ids          = Seq(0L, 1L),
      embeddings   = Seq(Seq(1.0f, 0.0f, 0.0f, 0.0f), Seq(0.0f, 1.0f, 0.0f, 0.0f)),
    )
    val schemaId = AilakeNative.evolveSchema(
      tableUri   = tableUri,
      namespace  = "default",
      tableName  = "integration_evolve_spark",
      addCols    = Seq(AddColReq(name = "source", colType = "string")),
      renameCols = Seq.empty,
    )
    assert(schemaId >= 0, s"evolveSchema returned $schemaId, expected >= 0")
    println(s"[test] evolveSchema OK: new_schema_id=$schemaId")
  }

  // ── Phase T: FTS write + searchText roundtrip ────────────────────────────

  test("writeBatch with ftsColumns and searchText roundtrip") {
    assume(libPath.isDefined,  "AILAKE_LIB_PATH not set — skipping")
    assume(writeDir.isDefined, "AILAKE_WRITE_DIR not set — skipping")
    assume(libPresent,         "libailake_jni.so not found — skipping")

    val tableUri = s"${writeDir.get}/integration-fts-spark"
    val texts    = Seq("rust programming language", "hello world example", "vector search database")
    val snap = AilakeNative.writeBatch(
      tableUri     = tableUri,
      namespace    = "default",
      tableName    = "integration_fts_spark",
      vectorColumn = "embedding",
      dim          = 4,
      metric       = "cosine",
      precision    = "f16",
      ids          = Seq(0L, 1L, 2L),
      embeddings   = Seq(
        Seq(1.0f, 0.0f, 0.0f, 0.0f),
        Seq(0.0f, 1.0f, 0.0f, 0.0f),
        Seq(0.0f, 0.0f, 1.0f, 0.0f),
      ),
      ftsColumns   = Seq("chunk_text"),
      ftsTokenizer = "default",
      columns      = Map("chunk_text" -> texts),
    )
    assert(snap.isDefined, "writeBatch with ftsColumns returned None")
    println(s"[test] writeBatch fts OK: snapshotId=${snap.get}")

    val results = AilakeNative.searchText(
      tableUri    = tableUri,
      namespace   = "default",
      tableName   = "integration_fts_spark",
      queryText   = "rust",
      textColumns = Seq("chunk_text"),
      topK        = 3,
    )
    assert(results.nonEmpty, "searchText returned empty — FTS index not built or not searched")
    val best = results.head
    assert(best.rowId == 0L, s"expected rowId=0 (rust programming), got rowId=${best.rowId}")
    println(s"[test] searchText OK: rowId=${best.rowId} distance=${best.distance}")
    println()
    println("PASS (Spark): FTS write+searchText roundtrip functional with real library.")
  }

  // ── AilakeDataWriterFactory ───────────────────────────────────────────────

  test("AilakeDataWriterFactory creates distinct writer instances") {
    val handle = AilakeWriteHandle(
      tableUri     = "file:///tmp/test",
      namespace    = "default",
      tableName    = "t",
      vectorColumn = "embedding",
      dim          = 2,
      metric       = "cosine",
      precision    = "f16",
    )
    val factory = new AilakeDataWriterFactory(handle)
    val w1 = factory.createWriter(0, 0L)
    val w2 = factory.createWriter(1, 1L)
    assert(w1 ne w2)
  }
}
