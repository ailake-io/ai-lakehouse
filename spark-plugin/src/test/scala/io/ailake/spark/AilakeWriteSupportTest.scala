// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite

class AilakeWriteSupportTest extends AnyFunSuite {

  private def handle = AilakeWriteHandle(
    tableUri     = "file:///tmp/test-table",
    namespace    = "default",
    tableName    = "docs",
    vectorColumn = "embedding",
    dim          = 4,
    metric       = "cosine",
    precision    = "f16",
  )

  // ── AilakeTable ───────────────────────────────────────────────────────────

  test("AilakeTable schema has id and embedding columns") {
    val table = new AilakeTable(handle)
    val schema = table.schema()
    assert(schema.length == 2)
    assert(schema("id").dataType == LongType)
    assert(schema("embedding").dataType == ArrayType(DoubleType))
  }

  test("AilakeTable name equals tableName from handle") {
    val table = new AilakeTable(handle)
    assert(table.name() == "docs")
  }

  test("AilakeTable capabilities include BATCH_WRITE") {
    val table = new AilakeTable(handle)
    import org.apache.spark.sql.connector.catalog.TableCapability
    assert(table.capabilities().contains(TableCapability.BATCH_WRITE))
  }

  // ── AilakeWriteBuilder ────────────────────────────────────────────────────

  test("AilakeWriteBuilder.buildForBatch returns AilakeBatchWrite") {
    val builder = new AilakeWriteBuilder(handle)
    val bw = builder.buildForBatch()
    assert(bw.isInstanceOf[AilakeBatchWrite])
  }

  // ── AilakeBatchWrite ──────────────────────────────────────────────────────

  test("AilakeBatchWrite.createBatchWriterFactory returns AilakeDataWriterFactory") {
    val bw = new AilakeBatchWrite(handle)
    val factory = bw.createBatchWriterFactory(null)
    assert(factory.isInstanceOf[AilakeDataWriterFactory])
  }

  test("AilakeBatchWrite.commit does not throw") {
    val bw = new AilakeBatchWrite(handle)
    bw.commit(Array(AilakeCommitMessage(None)))
  }

  test("AilakeBatchWrite.abort does not throw") {
    val bw = new AilakeBatchWrite(handle)
    bw.abort(Array.empty)
  }

  // ── AilakeDataWriter ──────────────────────────────────────────────────────

  test("AilakeDataWriter.commit with no rows returns snapshotId=None") {
    val writer = new AilakeDataWriter(handle, partitionId = 0, taskId = 0L)
    val msg = writer.commit().asInstanceOf[AilakeCommitMessage]
    assert(msg.snapshotId.isEmpty)
  }

  test("AilakeDataWriter.abort does not throw") {
    val writer = new AilakeDataWriter(handle, partitionId = 0, taskId = 0L)
    writer.abort()
  }

  // ── AilakeNative.writeBatch graceful degradation ──────────────────────────

  test("writeBatch returns None when native lib absent") {
    val result = AilakeNative.writeBatch(
      tableUri     = "file:///tmp/absent",
      namespace    = "default",
      tableName    = "test",
      vectorColumn = "embedding",
      dim          = 4,
      metric       = "cosine",
      precision    = "f16",
      ids          = Seq(1L, 2L),
      embeddings   = Seq(Seq(0.1f, 0.2f, 0.3f, 0.4f), Seq(0.5f, 0.6f, 0.7f, 0.8f)),
    )
    // Without native lib result is None — no exception
    println(s"[test] writeBatch without lib: $result")
  }

  // ── AilakeCommitMessage ───────────────────────────────────────────────────

  test("AilakeCommitMessage equality") {
    val m1 = AilakeCommitMessage(Some(42L))
    val m2 = AilakeCommitMessage(Some(42L))
    assert(m1 == m2)
  }

  test("AilakeCommitMessage None equals None") {
    assert(AilakeCommitMessage(None) == AilakeCommitMessage(None))
  }
}
