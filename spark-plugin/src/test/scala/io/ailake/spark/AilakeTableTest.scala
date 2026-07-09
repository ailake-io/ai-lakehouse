// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.sources.{EqualTo, Filter, GreaterThan, In}
import org.junit.runner.RunWith
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.junit.JUnitRunner

/**
 * Regression: AilakeNative.deleteWhere was fully implemented and tested but had no
 * SQL/DataFrame surface anywhere in this plugin — DELETE FROM was unreachable. Same
 * "dead capability" gap Trino/Flink already closed via equality/IN predicate pushdown,
 * closed the same way here via SupportsDelete.
 */
@RunWith(classOf[JUnitRunner])
class AilakeTableTest extends AnyFunSuite {

  private def handle(): AilakeWriteHandle =
    AilakeWriteHandle("file:///tmp/t", "default", "docs", "embedding", 4, "cosine", "f16")

  // SupportsDelete extends SupportsDeleteV2, which also has canDeleteWhere/deleteWhere
  // overloads taking Predicate[] — Array(EqualTo(...)) infers Array[EqualTo], too narrow
  // for Scala to pick a single overload by widening, so every call here is explicitly
  // typed Array[Filter] to disambiguate against the Predicate[] overload.

  test("canDeleteWhere accepts a single EqualTo filter") {
    val table = new AilakeTable(handle())
    assert(table.canDeleteWhere(Array[Filter](EqualTo("id", 5L))))
  }

  test("canDeleteWhere accepts a single In filter") {
    val table = new AilakeTable(handle())
    assert(table.canDeleteWhere(Array[Filter](In("id", Array(1L, 2L, 3L)))))
  }

  test("canDeleteWhere rejects multi-filter predicates") {
    val table = new AilakeTable(handle())
    assert(!table.canDeleteWhere(Array[Filter](EqualTo("id", 5L), EqualTo("source", "x"))))
  }

  test("canDeleteWhere rejects range predicates") {
    val table = new AilakeTable(handle())
    assert(!table.canDeleteWhere(Array[Filter](GreaterThan("id", 5L))))
  }

  test("canDeleteWhere rejects an empty filter array") {
    val table = new AilakeTable(handle())
    assert(!table.canDeleteWhere(Array.empty[Filter]))
  }

  test("deleteWhere with an unsupported filter throws UnsupportedOperationException") {
    val table = new AilakeTable(handle())
    val ex = intercept[UnsupportedOperationException] {
      table.deleteWhere(Array[Filter](GreaterThan("id", 5L)))
    }
    assert(ex.getMessage.contains("WHERE clause"))
  }

  test("deleteWhere with EqualTo throws RuntimeException when native lib absent") {
    val table = new AilakeTable(handle())
    val ex = intercept[RuntimeException] {
      table.deleteWhere(Array[Filter](EqualTo("id", 5L)))
    }
    assert(ex.getMessage.contains("DELETE WHERE"))
  }

  test("deleteWhere with In throws RuntimeException when native lib absent") {
    val table = new AilakeTable(handle())
    intercept[RuntimeException] {
      table.deleteWhere(Array[Filter](In("id", Array(1L, 2L))))
    }
  }

  test("capabilities still include BATCH_WRITE and TRUNCATE") {
    import org.apache.spark.sql.connector.catalog.TableCapability
    val table = new AilakeTable(handle())
    assert(table.capabilities().contains(TableCapability.BATCH_WRITE))
    assert(table.capabilities().contains(TableCapability.TRUNCATE))
  }
}
