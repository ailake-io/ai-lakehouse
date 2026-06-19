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
}
