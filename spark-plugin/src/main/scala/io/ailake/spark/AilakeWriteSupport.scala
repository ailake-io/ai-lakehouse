// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write._
import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer

// ── Config holder ─────────────────────────────────────────────────────────────

case class AilakeWriteHandle(
  tableUri:        String,
  namespace:       String,
  tableName:       String,
  vectorColumn:    String,
  dim:             Int,
  metric:          String,
  precision:       String,
  idColIndex:      Int = 0,
  vecColIndex:     Int = 1,
  embeddingModel:  Option[String] = None,
  partitionFields: Seq[PartitionFieldDef] = Seq.empty,
  formatVersion:   Int = 2,
)

// ── WriterCommitMessage ───────────────────────────────────────────────────────

case class AilakeCommitMessage(snapshotId: Option[Long]) extends WriterCommitMessage

// ── DataWriter ────────────────────────────────────────────────────────────────

class AilakeDataWriter(handle: AilakeWriteHandle, partitionId: Int, taskId: Long)
    extends DataWriter[InternalRow] {

  private val log = LoggerFactory.getLogger(classOf[AilakeDataWriter])
  private val ids        = ArrayBuffer[Long]()
  private val embeddings = ArrayBuffer[Seq[Float]]()
  private var autoId     = partitionId.toLong * Int.MaxValue

  def write(row: InternalRow): Unit = {
    val id  = if (!row.isNullAt(handle.idColIndex)) row.getLong(handle.idColIndex)
              else { val a = autoId; autoId += 1; a }
    val arr = row.getArray(handle.vecColIndex)
    val emb = arr.toDoubleArray().map(_.toFloat).toSeq
    ids        += id
    embeddings += emb
  }

  def commit(): WriterCommitMessage = {
    if (ids.isEmpty) return AilakeCommitMessage(None)
    val snapshotId = AilakeNative.writeBatch(
      tableUri        = handle.tableUri,
      namespace       = handle.namespace,
      tableName       = handle.tableName,
      vectorColumn    = handle.vectorColumn,
      dim             = handle.dim,
      metric          = handle.metric,
      precision       = handle.precision,
      ids             = ids.toSeq,
      embeddings      = embeddings.toSeq,
      embeddingModel  = handle.embeddingModel,
      partitionFields = handle.partitionFields,
      formatVersion   = handle.formatVersion,
    )
    log.info(s"[ailake] partition=$partitionId wrote ${ids.size} rows → snapshot=$snapshotId")
    AilakeCommitMessage(snapshotId)
  }

  def abort(): Unit = { ids.clear(); embeddings.clear() }

  def close(): Unit = {}
}

// ── DataWriterFactory ─────────────────────────────────────────────────────────

class AilakeDataWriterFactory(handle: AilakeWriteHandle) extends DataWriterFactory {
  def createWriter(partitionId: Int, taskId: Long): DataWriter[InternalRow] =
    new AilakeDataWriter(handle, partitionId, taskId)
}

// ── BatchWrite ────────────────────────────────────────────────────────────────

class AilakeBatchWrite(handle: AilakeWriteHandle) extends BatchWrite {
  def createBatchWriterFactory(info: PhysicalWriteInfo): DataWriterFactory =
    new AilakeDataWriterFactory(handle)

  def commit(messages: Array[WriterCommitMessage]): Unit = {}

  def abort(messages: Array[WriterCommitMessage]): Unit = {}
}

// ── WriteBuilder ──────────────────────────────────────────────────────────────

class AilakeWriteBuilder(handle: AilakeWriteHandle) extends WriteBuilder {
  override def buildForBatch(): BatchWrite = new AilakeBatchWrite(handle)
}
