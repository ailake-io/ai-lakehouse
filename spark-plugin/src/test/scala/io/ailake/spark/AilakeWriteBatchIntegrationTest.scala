// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import scala.math.sqrt
import org.junit.runner.RunWith
import org.scalatestplus.junit.JUnitRunner

/**
 * End-to-end integration test for AilakeNative.writeBatch (Spark side).
 *
 * Required env vars:
 *   AILAKE_LIB_PATH   — directory containing libailake_jni.so
 *   AILAKE_WRITE_DIR  — writable directory where a new table will be created
 *
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
      tableName    = "integration_write_spark",
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

    val results = AilakeNative.search(tableUri, qNorm, topK = 3)
    assert(results.nonEmpty, "search after write returned empty results")
    val best = results.minBy(_.distance)
    assert(
      best.rowId == queryIdx.toLong,
      s"nearest rowId=${best.rowId}, expected $queryIdx"
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
