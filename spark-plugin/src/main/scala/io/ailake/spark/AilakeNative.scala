// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import com.fasterxml.jackson.databind.ObjectMapper
import com.sun.jna.{Library, Native, Pointer}
import org.slf4j.LoggerFactory
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

  /** Partition field definition for multi-column partition specs (Phase K). */
  case class PartitionFieldDef(column: String, transform: String, columnType: String)

  /** Column addition request for schema evolution. */
  case class AddColReq(name: String, colType: String, initialDefault: Option[String] = None)

  /** Column rename request for schema evolution. */
  case class RenameColReq(from: String, to: String)

  private trait Lib extends Library {
    /** Returns ailake-jni version string. Static — do NOT free this pointer. */
    def ailake_version(): String

    /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
    def ailake_search_json(requestJson: String): Pointer

    /** Cross-modal RRF. Returns `{"ok":true,"results":[{"row_id":N,"rrf_score":F,"file_path":"..."}]}`. Caller must free. */
    def ailake_search_multimodal_json(requestJson: String): Pointer

    /** JSON-envelope write. Returns `{"ok":true,"snapshot_id":N}`. Caller must free. */
    def ailake_write_batch_json(requestJson: String): Pointer

    /** Logical delete via equality delete file. Returns `{"ok":true}`. Caller must free. */
    def ailake_delete_where_json(requestJson: String): Pointer

    /** Schema evolution. Returns `{"ok":true,"new_schema_id":N}`. Caller must free. */
    def ailake_evolve_schema_json(requestJson: String): Pointer

    /** Full-text search (Tantivy or BM25 fallback). Returns `{"ok":true,"results":[...]}`. Caller must free. */
    def ailake_search_text_json(requestJson: String): Pointer

    /** Compact small files. Returns `{"ok":true,"files_compacted":N}`. Caller must free. */
    def ailake_compact_json(requestJson: String): Pointer

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
        log.info("[ailake] Native library libailake_jni {} loaded (path={})", version,
          explicitPath.getOrElse("JNA default search path"))
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
        val idsJson  = ids.mkString("[", ",", "]")
        val embJson  = embeddings.map(_.mkString("[", ",", "]")).mkString("[", ",", "]")
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
        val colsJson = if (columns.nonEmpty) {
          val inner = columns.map { case (col, vals) =>
            val arr = vals.map(v => jsonStr(v)).mkString("[", ",", "]")
            s"""${jsonStr(col)}:$arr"""
          }.mkString("{", ",", "}")
          s""","columns":$inner"""
        } else ""
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},"vec_col":${jsonStr(vectorColumn)},""" +
          s""""dim":$dim,"metric":${jsonStr(metric)},"precision":${jsonStr(precision)},""" +
          s""""ids":$idsJson,"embeddings":$embJson$modelJson$partByJson$partValJson$pfJson$fvJson$ftsJson$hnswMJson$hnswEfJson$preNormalizeJson$deferredJson$colsJson}"""
        val ptr = native.ailake_write_batch_json(requestJson)
        if (ptr == null) {
          log.warn(s"[ailake] ailake_write_batch_json returned null for table=$tableName")
          return None
        }
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
          val root = mapper.readTree(json)
          if (!root.path("ok").asBoolean(false)) {
            log.warn(s"[ailake] writeBatch ok=false for table=$tableName: ${root.path("error").asText()}")
            return None
          }
          val sid = root.path("snapshot_id")
          if (sid.isMissingNode) None else Some(sid.asLong())
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception in writeBatch for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
            None
        }
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
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
          val root = mapper.readTree(json)
          if (!root.path("ok").asBoolean(false)) {
            log.warn(s"[ailake] deleteWhere ok=false for table=$tableName: ${root.path("error").asText()}")
            false
          } else true
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception in deleteWhere for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
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
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
          val root = mapper.readTree(json)
          if (!root.path("ok").asBoolean(false)) {
            log.warn(s"[ailake] evolveSchema ok=false for table=$tableName: ${root.path("error").asText()}")
            return -1
          }
          val sid = root.path("new_schema_id")
          if (sid.isMissingNode) -1 else sid.asInt(-1)
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception in evolveSchema for table=$tableName: ${e.getMessage}", e)
            Try(native.ailake_free_string(ptr))
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
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
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
            Try(native.ailake_free_string(ptr))
            None
        }
    }
  }

  private def jsonStr(s: String): String = mapper.writeValueAsString(s)

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
