// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.apache.flink.table.api.DataTypes
import org.apache.flink.table.api.Schema
import org.apache.flink.table.catalog.CatalogTable
import org.apache.flink.table.catalog.Column
import org.apache.flink.table.catalog.ObjectPath
import org.apache.flink.table.catalog.ResolvedCatalogTable
import org.apache.flink.table.catalog.ResolvedSchema
import org.apache.flink.table.catalog.TableChange
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path

class AilakeCatalogTest {

    // JUnit5's default per-method test lifecycle injects a fresh @TempDir for every
    // @Test — required since AilakeCatalog.createTable (Fase 23) now genuinely creates
    // the table on disk instead of only touching in-memory state; the tests below used
    // to share a single hardcoded "file:///tmp/x" warehouse, which was harmless while
    // createTable was a no-op but now makes the second test to call createTable() on
    // the same ObjectPath("default", "docs") fail with "table already exists".
    @TempDir
    lateinit var tmpDir: Path

    private fun catalog(defaultNamespace: String = "default") =
        AilakeCatalog("ailake", warehouse = "file://$tmpDir", defaultNamespace = defaultNamespace).apply { open() }

    private fun ingestSchema() = Schema.newBuilder()
        .column("id", DataTypes.BIGINT())
        .column("embedding", DataTypes.ARRAY(DataTypes.FLOAT()))
        .build()

    private fun ingestResolvedSchema() = ResolvedSchema.of(
        Column.physical("id", DataTypes.BIGINT()),
        Column.physical("embedding", DataTypes.ARRAY(DataTypes.FLOAT())),
    )

    // AilakeCatalog.createTable reads the deprecated `.schema` (TableSchema) accessor
    // via `(table as CatalogTable).schema` — this only resolves on a ResolvedCatalogTable
    // (what Flink's CatalogManager actually passes to Catalog SPI methods after resolving
    // the DDL schema), not a bare CatalogTable.of(...). Match real usage here.
    private fun ingestTable(options: Map<String, String> = emptyMap()): CatalogTable =
        ResolvedCatalogTable(CatalogTable.of(ingestSchema(), "", emptyList(), options), ingestResolvedSchema())

    // ── createTable — table-name/namespace injection ──────────────────────────
    //
    // Regression: 'table-name' is a required connector option with no default —
    // createTable injected 'connector'/'warehouse' but never 'table-name'/'namespace',
    // so any CREATE TABLE through this catalog failed FactoryUtil validation unless
    // the user redundantly re-specified them inside WITH(...).

    @Test
    fun createTableInjectsTableNameFromObjectPath() {
        val cat = catalog()
        val path = ObjectPath("default", "docs")
        cat.createTable(path, ingestTable(), false)
        val stored = cat.getTable(path)
        assertEquals("docs", stored.options["table-name"])
    }

    @Test
    fun createTableInjectsNamespaceFromObjectPathDatabaseName() {
        val cat = catalog()
        val path = ObjectPath("tenant_a", "docs")
        cat.createTable(path, ingestTable(), false)
        val stored = cat.getTable(path)
        assertEquals("tenant_a", stored.options["namespace"])
    }

    @Test
    fun createTableDoesNotOverrideExplicitTableNameOrNamespace() {
        val cat = catalog()
        val path = ObjectPath("default", "docs")
        cat.createTable(path, ingestTable(mapOf("table-name" to "custom", "namespace" to "custom_ns")), false)
        val stored = cat.getTable(path)
        assertEquals("custom", stored.options["table-name"])
        assertEquals("custom_ns", stored.options["namespace"])
    }

    // ── alterTable(TableChange list) — schema evolution wiring ────────────────
    //
    // Regression: alterTable only ever swapped Flink's in-memory CatalogBaseTable —
    // ALTER TABLE appeared to succeed while never touching the real table on disk.

    @Test
    fun alterTableWithAddColumnChangeFailsClearlyWhenNativeLibraryAbsent() {
        // See AilakeVectorTableSinkTest.executeDeletionFailsClearlyWhenNativeLibraryAbsent
        // for why this checks AILAKE_NATIVE_LIB, not AILAKE_LIB_PATH (never set for
        // Flink's gradle-test step in any workflow — this assumeTrue was always true).
        val nativeLib = System.getenv("AILAKE_NATIVE_LIB") ?: System.getProperty("ailake.native.lib")
        assumeTrue(nativeLib == null || !java.io.File(nativeLib).exists(), "skipped: native library present")
        val cat = catalog()
        val path = ObjectPath("default", "docs")
        cat.createTable(path, ingestTable(), false)
        val change = TableChange.add(Column.physical("source", DataTypes.STRING()))
        // Native lib absent -> AilakeNativeLoader.evolveSchema throws (UnsatisfiedLinkError,
        // a JVM Error) rather than silently succeeding as a no-op.
        assertThrows(Throwable::class.java) {
            cat.alterTable(path, ingestTable(), listOf(change), false)
        }
    }

    @Test
    fun alterTableWithOnlyUnsupportedChangesUpdatesInMemoryStateWithoutNativeCall() {
        // e.g. a comment-only change — no AddColumn/ModifyColumnName in the list,
        // so no evolveSchema call should be attempted (and thus no native-lib
        // requirement), only the in-memory swap.
        val cat = catalog()
        val path = ObjectPath("default", "docs")
        cat.createTable(path, ingestTable(), false)
        val newTable = ingestTable(mapOf("some-marker" to "x"))
        val change = TableChange.modifyColumnComment(Column.physical("id", DataTypes.BIGINT()), "the id")
        cat.alterTable(path, newTable, listOf(change), false)
        assertEquals("x", cat.getTable(path).options["some-marker"])
    }

    @Test
    fun alterTableThreeArgOverloadStillJustSwapsInMemoryState() {
        val cat = catalog()
        val path = ObjectPath("default", "docs")
        cat.createTable(path, ingestTable(), false)
        val newTable = ingestTable(mapOf("some-marker" to "y"))
        cat.alterTable(path, newTable, false)
        assertEquals("y", cat.getTable(path).options["some-marker"])
    }
}
