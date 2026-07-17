// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import com.fasterxml.jackson.databind.ObjectMapper
import com.sun.jna.{Library, Native, Pointer}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.{BigIntVector, FieldVector, VarCharVector, VectorSchemaRoot}
import org.apache.arrow.vector.complex.ListVector
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType}
import org.slf4j.LoggerFactory
import java.io.ByteArrayOutputStream
import java.nio.channels.Channels
import java.nio.charset.StandardCharsets
import java.util.{ArrayList => JArrayList, Collections => JCollections}
import scala.util.Try

/**
 * JNA bridge to libailake_jni.so.
 *
 * The library must be on java.library.path or LD_LIBRARY_PATH.
 * If not found, all searches return empty sequences (graceful degradation).
 */
object AilakeNative {

  private val log = LoggerFactory.getLogger(getClass.getName)

  case class SearchRow(rowId: Long, distance: Float, filePath: String)

  /** One column of a [[scan]] response — `dataType` is one of the tags `ailake_scan_json` emits: `int64`, `float32`, `float64`, `utf8`, `bool`, `list_float32`. */
  case class ScanColumn(name: String, dataType: String)

  /** Result of [[scan]] — search + full-row fetch in one native call. Columnar, `_distance` always last. */
  case class ScanResult(schema: Seq[ScanColumn], numRows: Int, columns: Map[String, Seq[Any]])

  /** Partition field definition for multi-column partition specs (Phase K). */
  case class PartitionFieldDef(column: String, transform: String, columnType: String)

  /** Column addition request for schema evolution. */
  case class AddColReq(name: String, colType: String, initialDefault: Option[String] = None)

  /** Column rename request for schema evolution. */
  case class RenameColReq(from: String, to: String)

  /**
   * One vector column in a multi-column (Phase 8 multimodal) write batch —
   * e.g. text + image embeddings on the same row, each with its own HNSW index.
   * See [[writeBatchMulti]].
   */
  case class VectorColSpec(
    column:    String,
    dim:       Int,
    metric:    String         = "cosine",
    precision: String         = "f16",
    modality:  Option[String] = None,
  )

  private trait Lib extends Library {
    /** Returns ailake-jni version string. Static — do NOT free this pointer. */
    def ailake_version(): String

    /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
    def ailake_search_json(requestJson: String): Pointer

    /** Cross-modal RRF. Returns `{"ok":true,"results":[{"row_id":N,"rrf_score":F,"file_path":"..."}]}`. Caller must free. */
    def ailake_search_multimodal_json(requestJson: String): Pointer

    /** Search + full-row fetch. Returns `{"ok":true,"schema":[...],"num_rows":N,"columns":{...}}`. Caller must free. */
    def ailake_scan_json(requestJson: String): Pointer

    /** JSON-envelope write. Returns `{"ok":true,"snapshot_id":N}`. Caller must free. */
    def ailake_write_batch_json(requestJson: String): Pointer

    /**
     * Arrow-IPC write (Fase 10, ADR-017) — same result shape as
     * [[ailake_write_batch_json]], but `id`/embedding/extra-text-columns arrive
     * as a single Arrow IPC stream RecordBatch (`ipcBytes`) instead of JSON,
     * replacing `Float.toString`/`mkString` formatting (~150ms/1k×1536-dim
     * batch, measured) with a binary buffer write (~30ms). `ipcLen` is declared
     * `Long` — JNA marshals Scala/Java `Long` to native C `long`, which is
     * 8 bytes on the 64-bit Linux targets this plugin runs on, matching the
     * Rust side's `i64` exactly (see the safety doc on `ailake_write_batch_ipc`
     * in `ailake-jni/src/lib.rs` for why the width is pinned rather than left
     * to `usize`/native `long` inference).
     */
    def ailake_write_batch_ipc(ipcBytes: Array[Byte], ipcLen: Long, optsJson: String): Pointer

    /** Multi-column (Phase 8 multimodal) write. Returns `{"ok":true,"snapshot_id":N}`. Caller must free. */
    def ailake_write_batch_multi_json(requestJson: String): Pointer

    /** Logical delete via equality delete file. Returns `{"ok":true}`. Caller must free. */
    def ailake_delete_where_json(requestJson: String): Pointer

    /** Schema evolution. Returns `{"ok":true,"new_schema_id":N}`. Caller must free. */
    def ailake_evolve_schema_json(requestJson: String): Pointer

    /** Full-text search (Tantivy or BM25 fallback). Returns `{"ok":true,"results":[...]}`. Caller must free. */
    def ailake_search_text_json(requestJson: String): Pointer

    /** Compact small files. Returns `{"ok":true,"files_compacted":N}`. Caller must free. */
    def ailake_compact_json(requestJson: String): Pointer

    /** Create an empty AI-Lake table. Returns `{"ok":true}`. Caller must free. */
    def ailake_create_table_json(requestJson: String): Pointer

    def ailake_free_string(ptr: Pointer): Unit
  }

  private val AILAKE_EXPECTED_MAJOR = "0"

  private lazy val lib: Option[Lib] = {
    val explicitPath = Option(System.getProperty("ailake.native.lib"))
      .orElse(Option(System.getenv("AILAKE_NATIVE_LIB")))
    try {
      val loaded = explicitPath match {
        case Some(p) => Native.load(p, classOf[Lib]).asInstanceOf[Lib]
        case None    => Native.load("ailake_jni", classOf[Lib]).asInstanceOf[Lib]
      }
      val version = loaded.ailake_version()
      val major = version.takeWhile(_ != '.')
      if (major != AILAKE_EXPECTED_MAJOR)
        log.warn(s"[ailake] Version mismatch: loaded ailake-jni $version but expected major version $AILAKE_EXPECTED_MAJOR. Search results may be incorrect.")
      else
        log.info(s"[ailake] Native library libailake_jni $version loaded (path=${explicitPath.getOrElse("JNA default search path")})")
      Some(loaded)
    } catch {
      case e: Throwable =>
        log.warn(
          "[ailake] Native library libailake_jni not found — vector search disabled. " +
          "Set ailake.native.lib system property or AILAKE_NATIVE_LIB env var. Error: {}", e.getMessage)
        None
    }
  }

  // Single shared mapper; ObjectMapper is thread-safe after configuration.
  private val mapper = new ObjectMapper()

  /**
   * Builds the single-RecordBatch Arrow IPC **stream** payload `ailake_write_batch_ipc`
   * expects: `id` (Int64, non-null), `vectorColumn` (`List<Float32>`, non-null),
   * and any `columns` (Utf8, nullable) — the same logical shape
   * `ailake_write_batch_json` builds internally from `ids`/`embeddings`/`columns`,
   * just constructed here with primitive Arrow buffer writes instead of
   * `Float.toString`/`mkString` JSON formatting (Fase 10, ADR-017; see the doc
   * comment on `Lib.ailake_write_batch_ipc` above for the measured ~150ms → ~30ms
   * per 1k×1536-dim batch).
   *
   * Uses only public `org.apache.arrow.vector` API (`ListVector` + its
   * `UnionListWriter`, not any Spark-internal Arrow helper) — this dependency
   * is `compileOnly` against whatever Arrow version Spark itself bundles
   * (pinned to `12.0.1` in `build.gradle.kts` to match Spark 3.5.0 exactly),
   * so staying off Spark-private classes avoids a second coupling surface on
   * top of that version pin.
   */
  private def buildIpcBatch(
    ids:          Seq[Long],
    vectorColumn: String,
    embeddings:   Seq[Seq[Float]],
    columns:      Map[String, Seq[String]],
  ): Array[Byte] = {
    val allocator = new RootAllocator(Long.MaxValue)
    try {
      val idVector = new BigIntVector("id", allocator)
      idVector.allocateNew(ids.size)
      ids.zipWithIndex.foreach { case (v, i) => idVector.setSafe(i, v) }
      idVector.setValueCount(ids.size)

      val itemField = new Field(
        "item",
        FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
        null,
      )
      val embField = new Field(
        vectorColumn,
        FieldType.notNullable(new ArrowType.List()),
        JCollections.singletonList(itemField),
      )
      val embVector = embField.createVector(allocator).asInstanceOf[ListVector]
      embVector.allocateNew()
      val embWriter = embVector.getWriter
      for (i <- ids.indices) {
        embWriter.setPosition(i)
        embWriter.startList()
        embeddings(i).foreach(v => embWriter.float4().writeFloat4(v))
        embWriter.endList()
      }
      embVector.setValueCount(ids.size)

      val extraVectors: Seq[FieldVector] = columns.toSeq.sortBy(_._1).map { case (name, vals) =>
        val v = new VarCharVector(name, allocator)
        v.allocateNew()
        vals.zipWithIndex.foreach { case (s, i) => v.setSafe(i, s.getBytes(StandardCharsets.UTF_8)) }
        v.setValueCount(vals.size)
        v
      }

      val fieldVectors = new JArrayList[FieldVector]()
      fieldVectors.add(idVector)
      fieldVectors.add(embVector)
      extraVectors.foreach(fieldVectors.add(_))

      val root = new VectorSchemaRoot(fieldVectors)
      root.setRowCount(ids.size)
      try {
        val out = new ByteArrayOutputStream()
        val ipcWriter = new ArrowStreamWriter(root, null, Channels.newChannel(out))
        try {
          ipcWriter.start()
          ipcWriter.writeBatch()
          ipcWriter.end()
        } finally {
          ipcWriter.close()
        }
        out.toByteArray
      } finally {
        root.close() // also closes idVector/embVector/extraVectors, which it now owns
      }
    } finally {
      allocator.close()
    }
  }

  /**
   * Write a batch of rows to an AI-Lake table via the native library.
   * Returns the snapshot_id on success, None on failure.
   *
   * @param partitionFields      multi-column partition spec (Phase K); empty = single-value partition_by/partition_value
   * @param formatVersion        Iceberg format version; 2 (default) or 3
   * @param ftsColumns           text columns to embed as Tantivy FTS index; empty = no FTS (default)
   * @param ftsTokenizer         Tantivy tokenizer name; default "default"
   * @param hnswM                HNSW graph connectivity (M). None = use table default.
   * @param hnswEfConstruction   HNSW ef_construction. None = use table default.
   * @param preNormalize         Normalize vectors to unit L2 at write time (recommended for cosine).
   * @param deferred             Build index asynchronously (write_batch_auto_deferred). Parquet committed immediately.
   * @param columns              Extra string columns sent with the batch (required for FTS to index text content).
   *                             Map from column name to per-row string values, e.g. Map("chunk_text" -> Seq("row0", "row1", ...)).
   */
  def writeBatch(
    tableUri:           String,
    namespace:          String,
    tableName:          String,
    vectorColumn:       String,
    dim:                Int,
    metric:             String,
    precision:          String,
    ids:                Seq[Long],
    embeddings:         Seq[Seq[Float]],
    embeddingModel:     Option[String] = None,
    partitionBy:        Option[String] = None,
    partitionValue:     Option[String] = None,
    partitionFields:    Seq[PartitionFieldDef] = Seq.empty,
    formatVersion:      Int = 2,
    ftsColumns:         Seq[String] = Seq.empty,
    ftsTokenizer:       String = "default",
    hnswM:              Option[Int] = None,
    hnswEfConstruction: Option[Int] = None,
    preNormalize:       Boolean = false,
    deferred:           Boolean = false,
    columns:            Map[String, Seq[String]] = Map.empty,
  ): Option[Long] = {
    if (ids.isEmpty) return None
    lib match {
      case None => None
      case Some(native) =>
        val ipcBytes = buildIpcBatch(ids, vectorColumn, embeddings, columns)

        val modelJson   = embeddingModel.map(m => s""","embedding_model":${jsonStr(m)}""").getOrElse("")
        val partByJson  = partitionBy.map(v => s""","partition_by":${jsonStr(v)}""").getOrElse("")
        val partValJson = partitionValue.map(v => s""","partition_value":${jsonStr(v)}""").getOrElse("")
        val pfJson = if (partitionFields.nonEmpty) {
          val arr = partitionFields.map(pf =>
            s"""{"column":${jsonStr(pf.column)},"transform":${jsonStr(pf.transform)},"column_type":${jsonStr(pf.columnType)}}"""
          ).mkString("[", ",", "]")
          s""","partition_fields":$arr"""
        } else ""
        val fvJson  = s""","format_version":$formatVersion"""
        val ftsJson = if (ftsColumns.nonEmpty) {
          val arr = ftsColumns.map(c => jsonStr(c)).mkString("[", ",", "]")
          s""","fts_columns":$arr,"fts_tokenizer":${jsonStr(ftsTokenizer)}"""
        } else ""
        val hnswMJson              = hnswM.map(v => s""","hnsw_m":$v""").getOrElse("")
        val hnswEfJson             = hnswEfConstruction.map(v => s""","hnsw_ef_construction":$v""").getOrElse("")
        val preNormalizeJson       = if (preNormalize) ""","pre_normalize":true""" else ""
        val deferredJson           = if (deferred) ""","deferred":true""" else ""
        // No `ids`/`embeddings`/`columns` here — those three now live in
        // `ipcBytes` instead of being JSON-encoded (Fase 10, ADR-017).
        val optsJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},"vec_col":${jsonStr(vectorColumn)},""" +
          s""""dim":$dim,"metric":${jsonStr(metric)},"precision":${jsonStr(precision)}""" +
          s"""$modelJson$partByJson$partValJson$pfJson$fvJson$ftsJson$hnswMJson$hnswEfJson$preNormalizeJson$deferredJson}"""
        val ptr = native.ailake_write_batch_ipc(ipcBytes, ipcBytes.length.toLong, optsJson)
        if (ptr == null) {
          log.warn(s"[ailake] ailake_write_batch_ipc returned null for table=$tableName")
          return None
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read writeBatch result string for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return None
        }
        native.ailake_free_string(ptr)
        val root = mapper.readTree(json)
        // A real backend rejection (e.g. NaN/Infinity embeddings, top_k over the
        // cap on other calls) must fail the write visibly — silently returning None
        // here is indistinguishable from "no lib loaded"/"empty batch" (both
        // legitimate no-ops below) and gets treated as success by
        // AilakeDataWriter.commit(), silently dropping the batch.
        if (!root.path("ok").asBoolean(false)) {
          val errMsg = root.path("error").asText("unknown error")
          throw new RuntimeException(s"ailake writeBatch failed for table=$tableName: $errMsg")
        }
        val sid = root.path("snapshot_id")
        if (sid.isMissingNode) None else Some(sid.asLong())
    }
  }

  /**
   * Write a batch with N independent vector columns into a single AI-Lake file
   * (Phase 8 multimodal tables — e.g. text + image embeddings on the same row,
   * searchable via [[searchMultimodal]]). Each column gets its own HNSW section;
   * the first entry in `vectorColumns` is the primary column used for geometric
   * pruning in the manifest.
   *
   * @param vectorColumns  one or more (spec, embeddings) pairs; `embeddings(i)` must
   *                       have `ids.size` rows for every column. First entry is primary.
   * @param columns        extra string columns sent with the batch (same as [[writeBatch]]).
   */
  def writeBatchMulti(
    tableUri:       String,
    namespace:      String,
    tableName:      String,
    ids:            Seq[Long],
    vectorColumns:  Seq[(VectorColSpec, Seq[Seq[Float]])],
    embeddingModel: Option[String] = None,
    formatVersion:  Int = 2,
    ftsColumns:     Seq[String] = Seq.empty,
    ftsTokenizer:   String = "default",
    deferred:       Boolean = false,
    columns:        Map[String, Seq[String]] = Map.empty,
  ): Option[Long] = {
    if (ids.isEmpty || vectorColumns.isEmpty) return None
    lib match {
      case None => None
      case Some(native) =>
        val idsJson = ids.mkString("[", ",", "]")
        val vecColsJson = vectorColumns.map { case (spec, embeddings) =>
          val embJson = embeddings.map(_.mkString("[", ",", "]")).mkString("[", ",", "]")
          val modalityJson = spec.modality.map(m => s""","modality":${jsonStr(m)}""").getOrElse("")
          s"""{"col":${jsonStr(spec.column)},"dim":${spec.dim},"metric":${jsonStr(spec.metric)},""" +
          s""""precision":${jsonStr(spec.precision)}$modalityJson,"embeddings":$embJson}"""
        }.mkString("[", ",", "]")
        val modelJson = embeddingModel.map(m => s""","embedding_model":${jsonStr(m)}""").getOrElse("")
        val fvJson  = s""","format_version":$formatVersion"""
        val ftsJson = if (ftsColumns.nonEmpty) {
          val arr = ftsColumns.map(c => jsonStr(c)).mkString("[", ",", "]")
          s""","fts_columns":$arr,"fts_tokenizer":${jsonStr(ftsTokenizer)}"""
        } else ""
        val deferredJson = if (deferred) ""","deferred":true""" else ""
        val colsJson = if (columns.nonEmpty) {
          val inner = columns.map { case (col, vals) =>
            val arr = vals.map(v => jsonStr(v)).mkString("[", ",", "]")
            s"""${jsonStr(col)}:$arr"""
          }.mkString("{", ",", "}")
          s""","columns":$inner"""
        } else ""
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},"ids":$idsJson,"vector_columns":$vecColsJson""" +
          s"""$modelJson$fvJson$ftsJson$deferredJson$colsJson}"""
        val ptr = native.ailake_write_batch_multi_json(requestJson)
        if (ptr == null) {
          log.warn(s"[ailake] ailake_write_batch_multi_json returned null for table=$tableName")
          return None
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read writeBatchMulti result string for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return None
        }
        native.ailake_free_string(ptr)
        val root = mapper.readTree(json)
        // See writeBatch's identical comment: a real backend rejection must fail
        // visibly, not be swallowed into a None that AilakeDataWriter.commit()
        // treats as a successful (if snapshot-less) write.
        if (!root.path("ok").asBoolean(false)) {
          val errMsg = root.path("error").asText("unknown error")
          throw new RuntimeException(s"ailake writeBatchMulti failed for table=$tableName: $errMsg")
        }
        val sid = root.path("snapshot_id")
        if (sid.isMissingNode) None else Some(sid.asLong())
    }
  }

  /**
   * Logically delete all rows where `column` equals any value in `values`.
   * Writes an Iceberg equality delete file via the native library.
   * Returns true on success, false if the library is absent or the call fails.
   */
  def deleteWhere(
    tableUri:  String,
    namespace: String,
    tableName: String,
    column:    String,
    values:    Seq[String],
  ): Boolean = {
    if (values.isEmpty) return false
    lib match {
      case None => false
      case Some(native) =>
        val valsJson = values.map(v => jsonStr(v)).mkString("[", ",", "]")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},"column":${jsonStr(column)},"values":$valsJson}"""
        val ptr = native.ailake_delete_where_json(requestJson)
        if (ptr == null) {
          log.warn(s"[ailake] ailake_delete_where_json returned null for table=$tableName")
          return false
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read deleteWhere result string for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return false
        }
        native.ailake_free_string(ptr)
        try {
          val root = mapper.readTree(json)
          if (!root.path("ok").asBoolean(false)) {
            log.warn(s"[ailake] deleteWhere ok=false for table=$tableName: ${root.path("error").asText()}")
            false
          } else true
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception parsing deleteWhere response for table=$tableName: ${e.getMessage}", e)
            false
        }
    }
  }

  /**
   * Apply a metadata-only schema evolution to the table.
   * Returns the new schema_id on success, -1 on error, 0 when no-op (both lists empty).
   *
   * @param addCols     columns to add; `initialDefault` is a JSON literal (null, 0, "unknown", ...)
   * @param renameCols  columns to rename
   */
  def evolveSchema(
    tableUri:   String,
    namespace:  String,
    tableName:  String,
    addCols:    Seq[AddColReq],
    renameCols: Seq[RenameColReq],
  ): Int = {
    if (addCols.isEmpty && renameCols.isEmpty) return 0
    lib match {
      case None => -1
      case Some(native) =>
        val addJson = addCols.map { ac =>
          val defPart = ac.initialDefault.map(d => s""","initial_default":$d""").getOrElse("")
          s"""{"name":${jsonStr(ac.name)},"type":${jsonStr(ac.colType)}$defPart}"""
        }.mkString("[", ",", "]")
        val renJson = renameCols.map { rc =>
          s"""{"from":${jsonStr(rc.from)},"to":${jsonStr(rc.to)}}"""
        }.mkString("[", ",", "]")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},"add_columns":$addJson,"rename_columns":$renJson}"""
        val ptr = native.ailake_evolve_schema_json(requestJson)
        if (ptr == null) {
          log.warn(s"[ailake] ailake_evolve_schema_json returned null for table=$tableName")
          return -1
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read evolveSchema result string for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return -1
        }
        native.ailake_free_string(ptr)
        try {
          val root = mapper.readTree(json)
          if (!root.path("ok").asBoolean(false)) {
            log.warn(s"[ailake] evolveSchema ok=false for table=$tableName: ${root.path("error").asText()}")
            return -1
          }
          val sid = root.path("new_schema_id")
          if (sid.isMissingNode) -1 else sid.asInt(-1)
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception parsing evolveSchema response for table=$tableName: ${e.getMessage}", e)
            -1
        }
    }
  }

  /**
   * Run a vector search via the native library.
   *
   * @param tableUri    path/URI of the AI-Lake table root
   * @param query       f32 query vector
   * @param topK        number of nearest neighbors
   * @param hybridText  when non-empty, enables hybrid BM25+vector RRF fusion
   * @param textColumn  Parquet column for BM25 scoring (default "chunk_text")
   * @param bm25Weight  BM25 weight in RRF (0.0 = pure vector, 1.0 = pure BM25)
   */
  def search(
    tableUri:        String,
    query:           Array[Float],
    topK:            Int,
    partitionFilter: Option[String] = None,
    hybridText:      Option[String] = None,
    textColumn:      String = "chunk_text",
    bm25Weight:      Float = 0.5f,
    namespace:       String = "default",
    tableName:       String = "",
  ): Seq[SearchRow] = {
    if (query.isEmpty) return Seq.empty
    lib match {
      case None => Seq.empty
      case Some(native) =>
        val effectiveTable = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
        val queryJson  = query.mkString("[", ",", "]")
        val partJson   = partitionFilter.map(v => s""","partition_filter":${jsonStr(v)}""").getOrElse("")
        val hybridJson = hybridText.map(t =>
          s""","hybrid_text":${jsonStr(t)},"text_column":${jsonStr(textColumn)},"bm25_weight":$bm25Weight"""
        ).getOrElse("")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},"table":${jsonStr(effectiveTable)},""" +
          s""""query":$queryJson,"dim":${query.length},"top_k":$topK$partJson$hybridJson}"""
        val ptr = native.ailake_search_json(requestJson)
        if (ptr == null) {
          log.warn("[ailake] ailake_search_json returned null pointer for tableUri={}", tableUri)
          return Seq.empty
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read search result string for tableUri=$tableUri: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return Seq.empty
        }
        native.ailake_free_string(ptr)
        try { parseResponse(json, tableUri) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to parse search response for tableUri=$tableUri: ${e.getMessage}", e)
            Seq.empty
        }
    }
  }

  /**
   * Vector search + full-row fetch in one native call (`ailake_scan_json`) — closes the
   * "SQL search only returns row_id/distance/file_path" gap: previously the only way to get
   * real columns (chunk_text, document_title, ...) back from a search was a manual DataFrame
   * `JOIN` against a separately-registered Iceberg table pointing at the same physical
   * location. Result is columnar; every stored column comes back (vector column decoded to
   * `list_float32`), plus a trailing `_distance` column — there's no column-subset filter on
   * the native side, it always returns the full row width.
   *
   * @param tableUri    path/URI of the AI-Lake table root
   * @param query       f32 query vector
   * @param topK        number of nearest neighbors
   * @param vectorColumn vector column to search (default "embedding")
   */
  def scan(
    tableUri:        String,
    query:           Array[Float],
    topK:            Int,
    vectorColumn:    String = "embedding",
    partitionFilter: Option[String] = None,
    namespace:       String = "default",
    tableName:       String = "",
  ): ScanResult = {
    if (query.isEmpty) return ScanResult(Seq.empty, 0, Map.empty)
    lib match {
      case None => ScanResult(Seq.empty, 0, Map.empty)
      case Some(native) =>
        val effectiveTable = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
        val queryJson = query.mkString("[", ",", "]")
        val partJson = partitionFilter.map(v => s""","partition_filter":${jsonStr(v)}""").getOrElse("")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},"table":${jsonStr(effectiveTable)},""" +
          s""""vec_col":${jsonStr(vectorColumn)},"query":$queryJson,"dim":${query.length},"top_k":$topK$partJson}"""
        val ptr = native.ailake_scan_json(requestJson)
        if (ptr == null) {
          log.warn("[ailake] ailake_scan_json returned null pointer for tableUri={}", tableUri)
          return ScanResult(Seq.empty, 0, Map.empty)
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read scan result string for tableUri=$tableUri: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return ScanResult(Seq.empty, 0, Map.empty)
        }
        native.ailake_free_string(ptr)
        try { parseScanResponse(json, tableUri) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to parse scan response for tableUri=$tableUri: ${e.getMessage}", e)
            ScanResult(Seq.empty, 0, Map.empty)
        }
    }
  }

  /**
   * Full-text search via Tantivy (fast path when AILK_FTS present) or BM25 brute-force.
   * Returns empty on library absence or error.
   *
   * @param textColumns  columns to search; defaults to ["chunk_text"]
   */
  def searchText(
    tableUri:        String,
    namespace:       String,
    tableName:       String,
    queryText:       String,
    textColumns:     Seq[String] = Seq("chunk_text"),
    topK:            Int = 10,
    partitionFilter: Option[String] = None,
  ): Seq[SearchRow] = {
    if (queryText.isEmpty) return Seq.empty
    lib match {
      case None => Seq.empty
      case Some(native) =>
        val colsJson  = textColumns.map(c => jsonStr(c)).mkString("[", ",", "]")
        val partJson  = partitionFilter.map(v => s""","partition_filter":${jsonStr(v)}""").getOrElse("")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},"query_text":${jsonStr(queryText)},""" +
          s""""text_columns":$colsJson,"top_k":$topK$partJson}"""
        val ptr = native.ailake_search_text_json(requestJson)
        if (ptr == null) {
          log.warn("[ailake] ailake_search_text_json returned null for tableUri={}", tableUri)
          return Seq.empty
        }
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
          parseResponse(json, tableUri)
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception in searchText: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            Seq.empty
        }
    }
  }

  case class MultimodalSearchRow(rowId: Long, rrfScore: Float, filePath: String)

  /**
   * Cross-modal vector search via Reciprocal Rank Fusion.
   *
   * @param tableUri  path/URI of the AI-Lake table root
   * @param queries   list of (column, query vector, weight) triples
   * @param topK      number of results to return
   */
  def searchMultimodal(
    tableUri:        String,
    queries:         Seq[(String, Array[Float], Float)],
    topK:            Int,
    partitionFilter: Option[String] = None,
    namespace:       String = "default",
    tableName:       String = "",
  ): Seq[MultimodalSearchRow] = {
    if (queries.isEmpty) return Seq.empty
    lib match {
      case None => Seq.empty
      case Some(native) =>
        val effectiveTable = if (tableName.nonEmpty) tableName else tableUri.stripSuffix("/").split("/").last
        val queriesJson = queries.map { case (col, q, w) =>
          s"""{"col":${jsonStr(col)},"query":${q.mkString("[", ",", "]")},"weight":$w,"dim":0}"""
        }.mkString("[", ",", "]")
        val partJson = partitionFilter.map(v => s""","partition_filter":${jsonStr(v)}""").getOrElse("")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},"table":${jsonStr(effectiveTable)},""" +
          s""""queries":$queriesJson,"top_k":$topK$partJson}"""
        val ptr = native.ailake_search_multimodal_json(requestJson)
        if (ptr == null) {
          log.warn("[ailake] ailake_search_multimodal_json returned null for tableUri={}", tableUri)
          return Seq.empty
        }
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
          parseMultimodalResponse(json, tableUri)
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception in searchMultimodal: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            Seq.empty
        }
    }
  }

  private def parseMultimodalResponse(json: String, tableUri: String): Seq[MultimodalSearchRow] = {
    Try {
      val root = mapper.readTree(json)
      if (!root.path("ok").asBoolean(false)) {
        log.warn(s"[ailake] searchMultimodal ok=false for tableUri=$tableUri: ${root.path("error").asText()}")
        return Seq.empty
      }
      val nodes = root.path("results")
      (0 until nodes.size()).map { i =>
        val n = nodes.get(i)
        MultimodalSearchRow(
          rowId    = n.get("row_id").asLong(),
          rrfScore = n.get("rrf_score").floatValue(),
          filePath = n.get("file_path").asText(),
        )
      }.toSeq
    }.recover {
      case e: Exception =>
        log.error(s"[ailake] Failed to parse multimodal response: ${e.getMessage}", e)
        Seq.empty
    }.getOrElse(Seq.empty)
  }

  /**
   * Compact small files in an AI-Lake table.
   *
   * @param minFiles          minimum eligible files to trigger compaction (default 4)
   * @param targetSizeBytes   files smaller than this are candidates (default 128 MiB)
   * @param maxFilesPerPass   max files merged per run (default 20)
   * @param deferred          build index in background when true (default false)
   * @return Some(filesCompacted) on success, None when library absent or error
   */
  def compact(
    tableUri:        String,
    namespace:       String,
    tableName:       String,
    minFiles:        Int  = 4,
    targetSizeBytes: Long = 128L * 1024 * 1024,
    maxFilesPerPass: Int  = 20,
    deferred:        Boolean = false,
  ): Option[Int] = {
    lib match {
      case None => None
      case Some(native) =>
        val deferredJson = if (deferred) ""","deferred":true""" else ""
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},""" +
          s""""min_files":$minFiles,"target_size_bytes":$targetSizeBytes,""" +
          s""""max_files_per_pass":$maxFilesPerPass$deferredJson}"""
        val ptr = native.ailake_compact_json(requestJson)
        if (ptr == null) {
          log.warn(s"[ailake] ailake_compact_json returned null for table=$tableName")
          return None
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read compact result string for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return None
        }
        native.ailake_free_string(ptr)
        try {
          val root = mapper.readTree(json)
          if (!root.path("ok").asBoolean(false)) {
            log.warn(s"[ailake] compact ok=false for table=$tableName: ${root.path("error").asText()}")
            None
          } else {
            val n = root.path("files_compacted").asInt(0)
            log.info(s"[ailake] compact OK table=$namespace.$tableName files_compacted=$n")
            Some(n)
          }
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to parse compact response for table=$tableName: ${e.getMessage}", e)
            None
        }
    }
  }

  def createTable(
    warehouse:      String,
    namespace:      String,
    table:          String,
    vectorColumn:   String = "embedding",
    dim:            Int    = 1536,
    metric:         String = "cosine",
    precision:      String = "f16",
    formatVersion:  Int    = 2,
  ): Option[Unit] = {
    lib match {
      case None => None
      case Some(native) =>
        val requestJson =
          s"""{"warehouse":${jsonStr(warehouse)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(table)},"vector_column":${jsonStr(vectorColumn)},""" +
          s""""dim":$dim,"metric":${jsonStr(metric)},"precision":${jsonStr(precision)},""" +
          s""""format_version":$formatVersion}"""
        val ptr = native.ailake_create_table_json(requestJson)
        if (ptr == null) {
          log.warn(s"[ailake] ailake_create_table_json returned null for table=$namespace.$table")
          return None
        }
        val json = try { ptr.getString(0) } catch {
          case e: Exception =>
            log.error(s"[ailake] Failed to read create_table result string: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            return None
        }
        native.ailake_free_string(ptr)
        val root = mapper.readTree(json)
        // Same as writeBatch/writeBatchMulti: a real backend rejection (e.g. the
        // table already exists) must fail visibly, not be swallowed into a None
        // a caller could mistake for "nothing to do".
        if (!root.path("ok").asBoolean(false)) {
          val errMsg = root.path("error").asText("unknown error")
          throw new RuntimeException(s"ailake create_table failed for table=$namespace.$table: $errMsg")
        }
        log.info(s"[ailake] create_table OK table=$namespace.$table")
        Some(())
    }
  }

  private def jsonStr(s: String): String = mapper.writeValueAsString(s)

  private def parseScanResponse(json: String, tableUri: String): ScanResult = {
    val root = mapper.readTree(json)
    if (!root.path("ok").asBoolean(false)) {
      val err = root.path("error").asText("<no error field>")
      log.warn(s"[ailake] Native scan returned ok=false for tableUri=$tableUri: $err")
      return ScanResult(Seq.empty, 0, Map.empty)
    }
    val schemaNodes = root.path("schema")
    val schema = (0 until schemaNodes.size()).map { i =>
      val n = schemaNodes.get(i)
      ScanColumn(n.get("name").asText(), n.get("type").asText())
    }.toSeq
    val numRows = root.path("num_rows").asInt(0)
    val columnsNode = root.path("columns")
    val columns: Map[String, Seq[Any]] = schema.map { col =>
      val arr = columnsNode.path(col.name)
      val values: Seq[Any] = col.dataType match {
        case "int64"   => (0 until arr.size()).map(i => if (arr.get(i).isNull) null else arr.get(i).asLong())
        case "float32" => (0 until arr.size()).map(i => if (arr.get(i).isNull) null else arr.get(i).floatValue())
        case "float64" => (0 until arr.size()).map(i => if (arr.get(i).isNull) null else arr.get(i).doubleValue())
        case "bool"    => (0 until arr.size()).map(i => if (arr.get(i).isNull) null else arr.get(i).asBoolean())
        case "list_float32" =>
          (0 until arr.size()).map { i =>
            val inner = arr.get(i)
            (0 until inner.size()).map(j => inner.get(j).floatValue())
          }
        case _ /* utf8 and anything else */ =>
          (0 until arr.size()).map(i => if (arr.get(i).isNull) null else arr.get(i).asText())
      }
      col.name -> values
    }.toMap
    ScanResult(schema, numRows, columns)
  }

  private def parseResponse(json: String, tableUri: String): Seq[SearchRow] = {
    Try {
      val root = mapper.readTree(json)
      if (!root.path("ok").asBoolean(false)) {
        val err = root.path("error").asText("<no error field>")
        log.warn(s"[ailake] Native search returned ok=false for tableUri=$tableUri: $err")
        return Seq.empty
      }
      val nodes = root.path("results")
      (0 until nodes.size()).map { i =>
        val n = nodes.get(i)
        SearchRow(
          rowId = n.get("row_id").asLong(),
          distance = n.get("distance").floatValue(),
          filePath = n.get("file_path").asText(),
        )
      }.toSeq
    }.recover {
      case e: Exception =>
        log.error(s"[ailake] Failed to parse native search response: ${e.getMessage}", e)
        Seq.empty
    }.getOrElse(Seq.empty)
  }
}
