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

  private trait Lib extends Library {
    /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
    def ailake_search_json(requestJson: String): Pointer

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
   * Run a vector search via the native library.
   *
   * @param tableUri  path/URI of the AI-Lake table root
   * @param query     f32 query vector
   * @param topK      number of nearest neighbors
   */
  def search(tableUri: String, query: Array[Float], topK: Int): Seq[SearchRow] = {
    if (query.isEmpty) return Seq.empty
    lib match {
      case None => Seq.empty
      case Some(native) =>
        val queryJson = query.mkString("[", ",", "]")
        val requestJson =
          s"""{"warehouse":${jsonStr(tableUri)},"namespace":"default","table":"table",""" +
          s""""query":$queryJson,"dim":${query.length},"top_k":$topK}"""
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
        log.error("[ailake] Failed to parse native search response: {}", e.getMessage, e)
        Seq.empty
    }.getOrElse(Seq.empty)
  }
}
