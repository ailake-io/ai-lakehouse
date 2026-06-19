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

    def ailake_free_string(ptr: Pointer): Unit
  }

  private lazy val lib: Option[Lib] =
    Try(Native.load("ailake_jni", classOf[Lib]).asInstanceOf[Lib])
      .fold(
        err => {
          log.warn(
            "[ailake] Native library libailake_jni not found — vector search disabled. " +
            "Set java.library.path or LD_LIBRARY_PATH to the directory containing libailake_jni.so. " +
            "Error: {}", err.getMessage)
          None
        },
        lib => {
          log.info("[ailake] Native library libailake_jni loaded successfully")
          Some(lib)
        }
      )

  // Single shared mapper; ObjectMapper is thread-safe after configuration.
  private val mapper = new ObjectMapper()

  /**
   * Write a batch of rows to an AI-Lake table via the native library.
   * Returns the snapshot_id on success, None on failure.
   *
   * @param partitionFields  multi-column partition spec (Phase K); empty = single-value partition_by/partition_value
   * @param formatVersion    Iceberg format version; 2 (default) or 3
   */
  def writeBatch(
    tableUri:        String,
    namespace:       String,
    tableName:       String,
    vectorColumn:    String,
    dim:             Int,
    metric:          String,
    precision:       String,
    ids:             Seq[Long],
    embeddings:      Seq[Seq[Float]],
    embeddingModel:  Option[String] = None,
    partitionBy:     Option[String] = None,
    partitionValue:  Option[String] = None,
    partitionFields: Seq[PartitionFieldDef] = Seq.empty,
    formatVersion:   Int = 2,
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
        val fvJson = s""","format_version":$formatVersion"""
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":${jsonStr(namespace)},""" +
          s""""table":${jsonStr(tableName)},"vec_col":${jsonStr(vectorColumn)},""" +
          s""""dim":$dim,"metric":${jsonStr(metric)},"precision":${jsonStr(precision)},""" +
          s""""ids":$idsJson,"embeddings":$embJson$modelJson$partByJson$partValJson$pfJson$fvJson}"""
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
   * @param tableUri  path/URI of the AI-Lake table root
   * @param query     f32 query vector
   * @param topK      number of nearest neighbors
   */
  def search(
    tableUri:        String,
    query:           Array[Float],
    topK:            Int,
    partitionFilter: Option[String] = None,
  ): Seq[SearchRow] = {
    if (query.isEmpty) return Seq.empty
    lib match {
      case None => Seq.empty
      case Some(native) =>
        val queryJson = query.mkString("[", ",", "]")
        val partJson  = partitionFilter.map(v => s""","partition_filter":${jsonStr(v)}""").getOrElse("")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":"default","table":"table",""" +
          s""""query":$queryJson,"dim":${query.length},"top_k":$topK$partJson}"""
        val ptr = native.ailake_search_json(requestJson)
        if (ptr == null) {
          log.warn("[ailake] ailake_search_json returned null pointer for tableUri={}", tableUri)
          return Seq.empty
        }
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
          parseResponse(json, tableUri)
        } catch {
          case e: Exception =>
            log.error(s"[ailake] Exception reading search result from native library: ${e.getMessage}", e)
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
  ): Seq[MultimodalSearchRow] = {
    if (queries.isEmpty) return Seq.empty
    lib match {
      case None => Seq.empty
      case Some(native) =>
        val queriesJson = queries.map { case (col, q, w) =>
          s"""{"col":${jsonStr(col)},"query":${q.mkString("[", ",", "]")},"weight":$w,"dim":0}"""
        }.mkString("[", ",", "]")
        val partJson = partitionFilter.map(v => s""","partition_filter":${jsonStr(v)}""").getOrElse("")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":"default","table":"table",""" +
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

  private def jsonStr(s: String): String =
    "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

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
