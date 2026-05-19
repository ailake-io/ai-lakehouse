package io.ailake.spark

import org.scalatest.funsuite.AnyFunSuite

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
}
