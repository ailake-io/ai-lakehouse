package io.ailake.spark

import com.sun.jna.{Library, Native, Pointer}
import scala.util.Try

/**
 * JNA bridge to libailake_jni.so.
 *
 * The library must be on java.library.path or LD_LIBRARY_PATH.
 * If not found, all searches return empty sequences (graceful degradation).
 */
object AilakeNative {

  case class SearchRow(rowId: Long, distance: Float, filePath: String)

  private trait Lib extends Library {
    /** JSON-envelope search. Returns `{"ok":true,"results":[...]}`. Caller must free. */
    def ailake_search_json(requestJson: String): Pointer

    def ailake_free_string(ptr: Pointer): Unit
  }

  private lazy val lib: Option[Lib] =
    Try(Native.load("ailake_jni", classOf[Lib]).asInstanceOf[Lib])
      .toOption
      .orElse {
        System.err.println("[ailake] Native library not found — vector search disabled")
        None
      }

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
        if (ptr == null) return Seq.empty
        try {
          val json = ptr.getString(0)
          native.ailake_free_string(ptr)
          parseResponse(json)
        } catch {
          case _: Exception =>
            Try(native.ailake_free_string(ptr))
            Seq.empty
        }
    }
  }

  private def jsonStr(s: String): String =
    "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

  private def parseResponse(json: String): Seq[SearchRow] = {
    // Parse {"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}
    Try {
      val jackson = Class.forName("com.fasterxml.jackson.databind.ObjectMapper")
        .getDeclaredConstructor().newInstance()
      val mapper = jackson.asInstanceOf[com.fasterxml.jackson.databind.ObjectMapper]
      val root = mapper.readTree(json)
      if (!root.path("ok").asBoolean(false)) return Seq.empty
      val nodes = root.path("results")
      (0 until nodes.size()).map { i =>
        val n = nodes.get(i)
        SearchRow(
          rowId = n.get("row_id").asLong(),
          distance = n.get("distance").floatValue(),
          filePath = n.get("file_path").asText(),
        )
      }.toSeq
    }.getOrElse(Seq.empty)
  }
}
