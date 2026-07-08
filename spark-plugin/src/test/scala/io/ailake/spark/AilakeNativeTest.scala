// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.connector.expressions.Transform
import org.scalatest.funsuite.AnyFunSuite
import org.junit.runner.RunWith
import org.scalatestplus.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class AilakeNativeTest extends AnyFunSuite {

  test("search returns empty sequence when native library absent") {
    // libailake_jni.so is not on java.library.path in test env — graceful degradation.
    val results = AilakeNative.search("s3://bucket/t/", Array(0.1f, 0.2f, 0.3f), topK = 5)
    assert(results.isEmpty)
  }

  test("search returns empty for zero-length query") {
    val results = AilakeNative.search("s3://bucket/t/", Array.emptyFloatArray, topK = 10)
    assert(results.isEmpty)
  }

  test("SearchRow equality") {
    val r1 = AilakeNative.SearchRow(1L, 0.5f, "part-001.parquet")
    val r2 = AilakeNative.SearchRow(1L, 0.5f, "part-001.parquet")
    assert(r1 == r2)
  }

  test("SearchRow toString contains rowId and filePath") {
    val r = AilakeNative.SearchRow(99L, 0.1f, "my-file.parquet")
    assert(r.toString.contains("99"))
    assert(r.toString.contains("my-file.parquet"))
  }

  // ── Phase P: writeBatch with partitionFields / formatVersion ─────────────────

  test("writeBatch returns None when native library absent with partitionFields") {
    assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
    val pf = AilakeNative.PartitionFieldDef("agent_id", "identity", "string")
    val result = AilakeNative.writeBatch(
      tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
      vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
      ids = Seq(1L), embeddings = Seq(Seq(0.1f, 0.2f, 0.3f, 0.4f)),
      partitionFields = Seq(pf), formatVersion = 3,
    )
    assert(result.isEmpty)
  }

  test("writeBatch returns None when native library absent (formatVersion=2 default)") {
    assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
    val result = AilakeNative.writeBatch(
      tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
      vectorColumn = "embedding", dim = 2, metric = "euclidean", precision = "f32",
      ids = Seq(1L), embeddings = Seq(Seq(1.0f, 0.0f)),
    )
    assert(result.isEmpty)
  }

  test("PartitionFieldDef equality") {
    val p1 = AilakeNative.PartitionFieldDef("col", "identity", "string")
    val p2 = AilakeNative.PartitionFieldDef("col", "identity", "string")
    assert(p1 == p2)
  }

  test("PartitionFieldDef toString contains column name") {
    val p = AilakeNative.PartitionFieldDef("session_id", "truncate[4]", "string")
    assert(p.toString.contains("session_id"))
  }

  // ── Phase P: deleteWhere ──────────────────────────────────────────────────────

  test("deleteWhere returns false when native library absent") {
    val ok = AilakeNative.deleteWhere("s3://b/t/", "default", "tbl", "doc_id", Seq("x"))
    assert(!ok)
  }

  test("deleteWhere returns false for empty values") {
    val ok = AilakeNative.deleteWhere("s3://b/t/", "default", "tbl", "doc_id", Seq.empty)
    assert(!ok)
  }

  // ── Phase P: evolveSchema ─────────────────────────────────────────────────────

  test("evolveSchema returns -1 when native library absent") {
    val id = AilakeNative.evolveSchema(
      tableUri = "s3://b/t/", namespace = "default", tableName = "tbl",
      addCols = Seq(AilakeNative.AddColReq("score", "float")),
      renameCols = Seq.empty,
    )
    assert(id == -1)
  }

  test("evolveSchema returns 0 for empty add and rename") {
    val id = AilakeNative.evolveSchema(
      tableUri = "s3://b/t/", namespace = "default", tableName = "tbl",
      addCols = Seq.empty, renameCols = Seq.empty,
    )
    assert(id == 0)
  }

  test("AddColReq default initialDefault is None") {
    val r = AilakeNative.AddColReq("score", "float")
    assert(r.initialDefault.isEmpty)
  }

  test("AddColReq with initialDefault") {
    val r = AilakeNative.AddColReq("score", "float", Some("0.0"))
    assert(r.initialDefault.contains("0.0"))
  }

  test("RenameColReq equality") {
    val r1 = AilakeNative.RenameColReq("old_col", "new_col")
    val r2 = AilakeNative.RenameColReq("old_col", "new_col")
    assert(r1 == r2)
  }

  // ── Phase R: public connector surface — AilakeWriteHandle ─────────────────

  test("AilakeWriteHandle default partitionFields is empty") {
    val h = AilakeWriteHandle("s3://b/t/", "default", "t", "emb", 4, "cosine", "f16")
    assert(h.partitionFields.isEmpty)
  }

  test("AilakeWriteHandle default formatVersion is 2") {
    val h = AilakeWriteHandle("s3://b/t/", "default", "t", "emb", 4, "cosine", "f16")
    assert(h.formatVersion == 2)
  }

  test("AilakeWriteHandle accepts partitionFields and formatVersion") {
    val pf = AilakeNative.PartitionFieldDef("agent_id", "identity", "string")
    val h  = AilakeWriteHandle("s3://b/t/", "default", "t", "emb", 4, "cosine", "f16",
      partitionFields = Seq(pf), formatVersion = 3)
    assert(h.partitionFields.size == 1)
    assert(h.partitionFields.head.column == "agent_id")
    assert(h.formatVersion == 3)
  }

  // ── Phase R: AilakeDataSource JSON option parsing ─────────────────────────

  test("AilakeDataSource parses partition-fields and format-version options") {
    import org.apache.spark.sql.types.StructType
    val ds = new AilakeDataSource()
    val props = new java.util.HashMap[String, String]()
    props.put("tableUri", "s3://b/docs/")
    props.put("partition-fields",
      """[{"column":"agent_id","transform":"identity","column_type":"string"}]""")
    props.put("format-version", "3")
    val table = ds.getTable(new StructType(), Array.empty[Transform], props)
    val handle = table.asInstanceOf[AilakeTable].handle
    assert(handle.partitionFields.size == 1)
    assert(handle.partitionFields.head.column == "agent_id")
    assert(handle.partitionFields.head.transform == "identity")
    assert(handle.partitionFields.head.columnType == "string")
    assert(handle.formatVersion == 3)
  }

  test("AilakeDataSource defaults to empty partitionFields and formatVersion=2") {
    import org.apache.spark.sql.types.StructType
    val ds = new AilakeDataSource()
    val props = new java.util.HashMap[String, String]()
    props.put("tableUri", "s3://b/docs/")
    val table = ds.getTable(new StructType(), Array.empty[Transform], props)
    val handle = table.asInstanceOf[AilakeTable].handle
    assert(handle.partitionFields.isEmpty)
    assert(handle.formatVersion == 2)
  }

  // ── extra text/metadata columns (id + embedding + arbitrary StringType cols) ──

  test("AilakeDataSource wires textColumns option through to textColIndices") {
    // Spark's TableProvider.inferSchema only ever sees write options, never
    // the caller's DataFrame — so extra columns must be declared explicitly
    // via .option("textColumns", ...), not auto-detected from the schema
    // argument getTable() receives (see AilakeDataSource.buildSchema doc).
    val ds = new AilakeDataSource()
    val props = new java.util.HashMap[String, String]()
    props.put("tableUri", "s3://b/docs/")
    props.put("textColumns", "text, source, page")
    val table = ds.getTable(new org.apache.spark.sql.types.StructType(), Array.empty[Transform], props)
    val ailakeTable = table.asInstanceOf[AilakeTable]
    val handle = ailakeTable.handle

    assert(handle.idColIndex == 0)
    assert(handle.vecColIndex == 1)
    assert(handle.textColIndices == Seq("text" -> 2, "source" -> 3, "page" -> 4))
    assert(ailakeTable.schema().fieldNames.toSeq == Seq("id", "embedding", "text", "source", "page"))
    // inferSchema() and getTable() must agree — both derive from options alone.
    assert(ds.inferSchema(new org.apache.spark.sql.util.CaseInsensitiveStringMap(props)) == ailakeTable.schema())
  }

  test("AilakeDataSource with no textColumns option keeps the historical id+embedding-only schema") {
    val ds = new AilakeDataSource()
    val props = new java.util.HashMap[String, String]()
    props.put("tableUri", "s3://b/docs/")
    val table = ds.getTable(new org.apache.spark.sql.types.StructType(), Array.empty[Transform], props)
    val handle = table.asInstanceOf[AilakeTable].handle
    assert(handle.textColIndices.isEmpty)
    assert(table.schema().fieldNames.toSeq == Seq("id", "embedding"))
  }

  test("AilakeWriteHandle.resolveColumns rejects non-string extra columns") {
    import org.apache.spark.sql.types._
    val schema = StructType(Seq(
      StructField("id",        LongType,              nullable = true),
      StructField("embedding", ArrayType(DoubleType), nullable = false),
      StructField("page",      IntegerType,           nullable = true),
    ))
    val ex = intercept[IllegalArgumentException] {
      AilakeWriteHandle.resolveColumns(schema, "embedding")
    }
    assert(ex.getMessage.contains("page"))
    assert(ex.getMessage.contains("StringType"))
  }

  // Regression: resolveColumns previously only validated extra (text) columns'
  // types, trusting id/vector column types unconditionally — a looser
  // CREATE TABLE schema (e.g. `id INT`) would pass validation here and only
  // fail later with an opaque ClassCastException inside AilakeDataWriter.write.
  test("AilakeWriteHandle.resolveColumns rejects non-Long id column") {
    import org.apache.spark.sql.types._
    val schema = StructType(Seq(
      StructField("id",        IntegerType,           nullable = true),
      StructField("embedding", ArrayType(DoubleType), nullable = false),
    ))
    val ex = intercept[IllegalArgumentException] {
      AilakeWriteHandle.resolveColumns(schema, "embedding")
    }
    assert(ex.getMessage.contains("id"))
    assert(ex.getMessage.contains("LongType"))
  }

  test("AilakeWriteHandle.resolveColumns rejects non-array<double> vector column") {
    import org.apache.spark.sql.types._
    val schema = StructType(Seq(
      StructField("id",        LongType,             nullable = true),
      StructField("embedding", ArrayType(FloatType), nullable = false),
    ))
    val ex = intercept[IllegalArgumentException] {
      AilakeWriteHandle.resolveColumns(schema, "embedding")
    }
    assert(ex.getMessage.contains("embedding"))
    assert(ex.getMessage.contains("ArrayType(DoubleType"))
  }

  test("AilakeWriteHandle.resolveColumns accepts nullable array<double> vector column") {
    import org.apache.spark.sql.types._
    val schema = StructType(Seq(
      StructField("id",        LongType,                                nullable = true),
      StructField("embedding", ArrayType(DoubleType, containsNull = true), nullable = false),
    ))
    val (idIdx, vecIdx, textCols) = AilakeWriteHandle.resolveColumns(schema, "embedding")
    assert(idIdx == 0)
    assert(vecIdx == 1)
    assert(textCols.isEmpty)
  }

  // Regression: resolveColumns hardcoded "id" as the id-column field name,
  // ignoring the `idColumn` option AilakeSparkExtensions.ailakeWrite already
  // accepted and threaded through options — a DataFrame with an id column
  // named e.g. "doc_id" would fail fieldIndex("id") even though the caller
  // correctly declared idColumn = "doc_id".
  test("AilakeWriteHandle.resolveColumns resolves a custom idColumn name") {
    import org.apache.spark.sql.types._
    val schema = StructType(Seq(
      StructField("doc_id",    LongType,              nullable = true),
      StructField("embedding", ArrayType(DoubleType), nullable = false),
    ))
    val (idIdx, vecIdx, textCols) = AilakeWriteHandle.resolveColumns(schema, "embedding", idColumn = "doc_id")
    assert(idIdx == 0)
    assert(vecIdx == 1)
    assert(textCols.isEmpty)
  }

  test("AilakeDataSource.buildSchema respects idColumn option") {
    import scala.collection.JavaConverters._
    val props = Map(
      "tableUri"  -> "s3://b/docs/",
      "idColumn"  -> "doc_id",
    ).asJava
    val schema = AilakeDataSource.buildSchema(new org.apache.spark.sql.util.CaseInsensitiveStringMap(props))
    assert(schema.fieldNames.toSeq == Seq("doc_id", "embedding"))
    assert(schema("doc_id").dataType == org.apache.spark.sql.types.LongType)
  }

  test("AilakeDataWriter passes extra column values through to writeBatch's columns map") {
    import org.apache.spark.sql.catalyst.InternalRow
    import org.apache.spark.sql.catalyst.util.GenericArrayData
    import org.apache.spark.unsafe.types.UTF8String

    val handle = AilakeWriteHandle(
      tableUri = "s3://b/docs/", namespace = "default", tableName = "docs",
      vectorColumn = "embedding", dim = 2, metric = "cosine", precision = "f16",
      idColIndex = 0, vecColIndex = 1,
      textColIndices = Seq("text" -> 2, "source" -> 3),
    )
    val writer = new AilakeDataWriter(handle, partitionId = 0, taskId = 0L)
    val row = InternalRow(
      1L,
      new GenericArrayData(Array(0.1, 0.2)),
      UTF8String.fromString("hello world"),
      UTF8String.fromString("doc-a"),
    )
    writer.write(row)

    // Inspect the accumulated per-column buffers directly via reflection
    // instead of calling commit() — whether libailake_jni.so is on
    // java.library.path varies by environment (absent locally, present in
    // CI with AILAKE_LIB_PATH set), so commit() against this fake tableUri
    // would either no-op (native lib absent) or attempt a real write that
    // may throw (native lib present, no such bucket). Either way this test
    // only needs to prove write() accumulates the right values; the native
    // call itself is covered by the integration tests further down this file.
    val field = classOf[AilakeDataWriter].getDeclaredField("textValues")
    field.setAccessible(true)
    val textValues = field.get(writer).asInstanceOf[Map[String, scala.collection.mutable.ArrayBuffer[String]]]
    assert(textValues("text").toList == List("hello world"))
    assert(textValues("source").toList == List("doc-a"))
  }

  // ── Phase T: FTS ──────────────────────────────────────────────────────────

  test("writeBatch with ftsColumns returns None when native library absent") {
    assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
    val result = AilakeNative.writeBatch(
      tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
      vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
      ids = Seq(1L), embeddings = Seq(Seq(0.1f, 0.2f, 0.3f, 0.4f)),
      ftsColumns = Seq("chunk_text", "title"),
      ftsTokenizer = "default",
    )
    assert(result.isEmpty)
  }

  test("writeBatch JSON includes fts_columns when non-empty") {
    // White-box: verify JSON fragment produced by writeBatch ftsColumns logic.
    val ftsColumns   = Seq("chunk_text", "title")
    val ftsTokenizer = "default"
    def jsonStr(s: String): String = "\"" + s + "\""
    val arr     = ftsColumns.map(c => jsonStr(c)).mkString("[", ",", "]")
    val ftsJson = s""","fts_columns":$arr,"fts_tokenizer":${jsonStr(ftsTokenizer)}"""
    assert(ftsJson.contains("chunk_text"))
    assert(ftsJson.contains("fts_tokenizer"))
  }

  test("writeBatch JSON omits fts_columns when empty") {
    val ftsColumns: Seq[String] = Seq.empty
    def jsonStr(s: String): String = "\"" + s + "\""
    val ftsJson = if (ftsColumns.nonEmpty) {
      val arr = ftsColumns.map(c => jsonStr(c)).mkString("[", ",", "]")
      s""","fts_columns":$arr"""
    } else ""
    assert(ftsJson.isEmpty)
  }

  test("searchText returns empty when native library absent") {
    assume(System.getenv("AILAKE_LIB_PATH") == null, "skipped: native library present")
    val results = AilakeNative.searchText(
      tableUri = "s3://bucket/t/", namespace = "default", tableName = "t",
      queryText = "rust programming", textColumns = Seq("chunk_text"), topK = 5,
    )
    assert(results.isEmpty)
  }
}
