// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.connector.catalog.Identifier
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.JavaConverters._
import org.junit.runner.RunWith
import org.scalatestplus.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class AilakeCatalogTest extends AnyFunSuite {

  private def makeCatalog(tableUri: String = "file:///tmp/test-table"): AilakeCatalog = {
    val catalog = new AilakeCatalog()
    val props = Map(
      "table-uri"     -> tableUri,
      "vector-column" -> "embedding",
      "vector-dim"    -> "4",
      "metric"        -> "cosine",
      "precision"     -> "f16",
    ).asJava
    catalog.initialize("ailake", new CaseInsensitiveStringMap(props))
    catalog
  }

  // ── initialize ────────────────────────────────────────────────────────────

  test("initialize sets catalog name") {
    val catalog = makeCatalog()
    assert(catalog.name() == "ailake")
  }

  // ── listTables ────────────────────────────────────────────────────────────

  test("listTables returns an array without throwing") {
    val catalog = makeCatalog()
    val tables = catalog.listTables(Array("default"))
    assert(tables != null)
  }

  // ── loadTable ─────────────────────────────────────────────────────────────

  test("loadTable returns AilakeTable") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "docs")
    val table = catalog.loadTable(ident)
    assert(table.isInstanceOf[AilakeTable])
  }

  test("loadTable creates handle with correct tableUri") {
    val catalog = makeCatalog("file:///my/lake")
    val ident = Identifier.of(Array("default"), "mytable")
    val table = catalog.loadTable(ident).asInstanceOf[AilakeTable]
    assert(table.handle.tableUri == "file:///my/lake")
  }

  test("loadTable creates handle with namespace from identifier") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("prod"), "docs")
    val table = catalog.loadTable(ident).asInstanceOf[AilakeTable]
    assert(table.handle.namespace == "prod")
  }

  test("loadTable creates handle with tableName from identifier") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "invoices")
    val table = catalog.loadTable(ident).asInstanceOf[AilakeTable]
    assert(table.handle.tableName == "invoices")
  }

  test("loadTable table has correct schema") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "docs")
    val schema = catalog.loadTable(ident).schema()
    assert(schema.length == 2)
    assert(schema.fieldNames.contains("id"))
    assert(schema.fieldNames.contains("embedding"))
    assert(schema("id").dataType == LongType)
    assert(schema("embedding").dataType == ArrayType(DoubleType))
  }

  test("loadTable uses default namespace when identifier namespace is empty") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array(), "docs")
    val table = catalog.loadTable(ident).asInstanceOf[AilakeTable]
    assert(table.handle.namespace == "default")
  }

  // ── tableExists ───────────────────────────────────────────────────────────

  test("tableExists returns true (catalog is open — loadTable never throws)") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "anything")
    assert(catalog.tableExists(ident))
  }

  // ── dropTable / renameTable no-op ─────────────────────────────────────────

  test("dropTable returns false") {
    val catalog = makeCatalog()
    assert(!catalog.dropTable(Identifier.of(Array("default"), "docs")))
  }

  test("renameTable throws UnsupportedOperationException") {
    val catalog = makeCatalog()
    intercept[UnsupportedOperationException] {
      catalog.renameTable(
        Identifier.of(Array("default"), "old"),
        Identifier.of(Array("default"), "new"),
      )
    }
  }
}
