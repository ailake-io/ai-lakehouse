// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.spark

import org.apache.spark.sql.connector.catalog.{Identifier, TableChange}
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

  // Regression: loadTable used to build against AilakeTable.WRITE_SCHEMA, which
  // hardcodes the field name "embedding" — a catalog configured with a
  // different `vector-column` name would fail resolveColumns' fieldIndex
  // lookup on bare `INSERT INTO`, even though tableExists claims the table is
  // always loadable. Fixed by deriving the default schema from the configured
  // vector-column name instead of a fixed literal.
  test("loadTable with custom vector-column name does not throw and uses that name in the schema") {
    val catalog = new AilakeCatalog()
    val props = Map(
      "table-uri"     -> "file:///tmp/test-table",
      "vector-column" -> "vec",
      "vector-dim"    -> "4",
      "metric"        -> "cosine",
      "precision"     -> "f16",
    ).asJava
    catalog.initialize("ailake", new CaseInsensitiveStringMap(props))
    val ident = Identifier.of(Array("default"), "docs")
    val table = catalog.loadTable(ident).asInstanceOf[AilakeTable]
    val schema = table.schema()
    assert(schema.length == 2)
    assert(schema.fieldNames.contains("vec"))
    assert(!schema.fieldNames.contains("embedding"))
    assert(table.handle.vecColIndex == 1)
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

  // ── ALTER TABLE ADD/RENAME COLUMN ─────────────────────────────────────────
  //
  // Regression: AilakeNative.evolveSchema was fully implemented and tested but
  // alterTable used to unconditionally throw UnsupportedOperationException —
  // same "dead capability" gap Trino/Flink already closed, closed the same way.

  test("alterTable with ADD COLUMN attempts a real evolveSchema call, not the old blanket throw") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "docs")
    val change = TableChange.addColumn(Array("source"), StringType)
    val ex = intercept[RuntimeException] {
      catalog.alterTable(ident, change)
    }
    // Native lib absent in test env → evolveSchema returns -1 → this message,
    // not the old "ALTER TABLE not supported by AI-Lake catalog".
    assert(ex.getMessage.contains("ALTER TABLE failed"))
  }

  test("alterTable rejects unsupported column type") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "docs")
    val change = TableChange.addColumn(Array("ts"), TimestampType)
    val ex = intercept[UnsupportedOperationException] {
      catalog.alterTable(ident, change)
    }
    assert(ex.getMessage.contains("not supported"))
  }

  test("alterTable rejects nested column path for ADD COLUMN") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "docs")
    val change = TableChange.addColumn(Array("parent", "child"), StringType)
    val ex = intercept[UnsupportedOperationException] {
      catalog.alterTable(ident, change)
    }
    assert(ex.getMessage.contains("nested"))
  }

  test("alterTable rejects nested column path for RENAME COLUMN") {
    val catalog = makeCatalog()
    val ident = Identifier.of(Array("default"), "docs")
    val change = TableChange.renameColumn(Array("parent", "child"), "newName")
    val ex = intercept[UnsupportedOperationException] {
      catalog.alterTable(ident, change)
    }
    assert(ex.getMessage.contains("nested"))
  }
}
