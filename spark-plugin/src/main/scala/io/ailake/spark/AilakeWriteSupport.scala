// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write._
import org.apache.spark.sql.types.{ArrayType, DoubleType, LongType, StringType, StructType}
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
  // (column name, row-field index) for extra string columns written alongside
  // id + embedding — e.g. chunk text, source, page. Threaded through to
  // AilakeNative.writeBatch's `columns` map, same as the Flink connector does.
  textColIndices:  Seq[(String, Int)] = Seq.empty,
  embeddingModel:  Option[String] = None,
  partitionFields: Seq[AilakeNative.PartitionFieldDef] = Seq.empty,
  formatVersion:   Int = 2,
)

object AilakeWriteHandle {

  /**
   * Resolves (idColIndex, vecColIndex, textColIndices) from a caller-supplied
   * write schema. Any column that isn't `id` or the configured vector column
   * is treated as extra string metadata (chunk text, source, page, ...) and
   * threaded through to `AilakeNative.writeBatch`'s `columns` map — the same
   * capability the Flink connector (`AilakeVectorTableSink`) already uses;
   * previously the Spark plugin silently dropped every column but id/embedding.
   *
   * Extra columns must be StringType — `columns` on the native side is
   * `Map[String, Seq[String]]`. Cast non-string columns (e.g. `page: Int`) to
   * string before writing if you need them round-tripped. `id` must be
   * LongType and the vector column must be ArrayType(DoubleType) — Spark
   * would otherwise accept a write matching a looser CREATE TABLE schema
   * (e.g. `id INT` or `embedding ARRAY<FLOAT>`) that later throws an opaque
   * ClassCastException in AilakeDataWriter.write's unchecked getLong/getArray.
   *
   * An empty `schema` means no real DataFrame schema was resolved yet (e.g. a
   * caller only after option parsing) — falls back to the historical
   * `(id, embedding)`-only defaults instead of failing `fieldIndex` lookups.
   */
  def resolveColumns(schema: StructType, vectorColumn: String, idColumn: String = "id"): (Int, Int, Seq[(String, Int)]) = {
    if (schema.isEmpty) return (0, 1, Seq.empty)
    val idIdx  = schema.fieldIndex(idColumn)
    val vecIdx = schema.fieldIndex(vectorColumn)
    val idField  = schema.fields(idIdx)
    val vecField = schema.fields(vecIdx)
    if (idField.dataType != LongType) {
      throw new IllegalArgumentException(
        s"Column '$idColumn' must be LongType (bigint), got ${idField.dataType.simpleString}. " +
        s"Cast it first, e.g. col('$idColumn').cast('long').")
    }
    if (vecField.dataType != ArrayType(DoubleType, containsNull = false) &&
        vecField.dataType != ArrayType(DoubleType, containsNull = true)) {
      throw new IllegalArgumentException(
        s"Vector column '$vectorColumn' must be ArrayType(DoubleType), got ${vecField.dataType.simpleString}. " +
        s"Cast it first, e.g. col('$vectorColumn').cast('array<double>').")
    }
    val textCols = schema.fields.zipWithIndex.collect {
      case (f, i) if i != idIdx && i != vecIdx =>
        if (f.dataType != StringType) {
          throw new IllegalArgumentException(
            s"Column '${f.name}' must be StringType to be written as AI-Lake extra metadata " +
            s"(got ${f.dataType.simpleString}). Only 'id' (bigint) and the vector column " +
            s"('$vectorColumn', array<double>) may be non-string. Cast other columns to " +
            s"string first, e.g. col('page').cast('string').")
        }
        f.name -> i
    }
    (idIdx, vecIdx, textCols)
  }
}

// ── WriterCommitMessage ───────────────────────────────────────────────────────

case class AilakeCommitMessage(snapshotId: Option[Long]) extends WriterCommitMessage

// ── DataWriter ────────────────────────────────────────────────────────────────

class AilakeDataWriter(handle: AilakeWriteHandle, partitionId: Int, taskId: Long)
    extends DataWriter[InternalRow] {

  private val log = LoggerFactory.getLogger(classOf[AilakeDataWriter])
  private val ids        = ArrayBuffer[Long]()
  private val embeddings = ArrayBuffer[Seq[Float]]()
  private val textValues: Map[String, ArrayBuffer[String]] =
    handle.textColIndices.map { case (name, _) => name -> ArrayBuffer[String]() }.toMap
  private var autoId     = partitionId.toLong * Int.MaxValue

  def write(row: InternalRow): Unit = {
    val id  = if (!row.isNullAt(handle.idColIndex)) row.getLong(handle.idColIndex)
              else { val a = autoId; autoId += 1; a }
    val arr = row.getArray(handle.vecColIndex)
    val emb = arr.toDoubleArray().map(_.toFloat).toSeq
    ids        += id
    embeddings += emb
    handle.textColIndices.foreach { case (name, idx) =>
      val v = if (row.isNullAt(idx)) "" else row.getUTF8String(idx).toString
      textValues(name) += v
    }
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
      columns         = textValues.map { case (k, v) => k -> v.toSeq },
    )
    log.info(s"[ailake] partition=$partitionId wrote ${ids.size} rows → snapshot=$snapshotId")
    AilakeCommitMessage(snapshotId)
  }

  def abort(): Unit = { ids.clear(); embeddings.clear(); textValues.values.foreach(_.clear()) }

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
