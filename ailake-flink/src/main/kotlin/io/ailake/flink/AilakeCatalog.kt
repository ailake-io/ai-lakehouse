// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.table.catalog.*
import org.apache.flink.table.catalog.exceptions.*
import org.apache.flink.table.catalog.stats.CatalogColumnStatistics
import org.apache.flink.table.catalog.stats.CatalogTableStatistics
import org.apache.flink.table.expressions.Expression
import org.apache.flink.table.factories.Factory
import org.apache.flink.table.types.logical.LogicalTypeRoot
import java.util.Optional

/**
 * Flink catalog backed by an AI-Lake / Iceberg warehouse.
 *
 * Namespaces map to Iceberg namespaces; tables map to Iceberg tables.  Table DDL
 * properties prefixed with "ailake.*" are stored in the Iceberg table metadata and
 * interpreted by the AI-Lake SDK.
 *
 * This implementation uses the [AilakeVectorConnectorFactory] as the underlying
 * table factory so that all tables created via this catalog automatically use the
 * 'ailake' connector.
 */
class AilakeCatalog(
    name: String,
    private val warehouse: String,
    private val defaultNamespace: String = "default",
) : AbstractCatalog(name, defaultNamespace) {

    // In-memory store for catalog state during the session.
    // Production: delegate to Iceberg REST / Hadoop catalog via ailake-jni.
    private val databases = mutableMapOf<String, CatalogDatabase>()
    private val tables = mutableMapOf<ObjectPath, CatalogBaseTable>()

    override fun open() {
        if (!databases.containsKey(defaultDatabase)) {
            databases[defaultDatabase!!] = CatalogDatabaseImpl(emptyMap(), "default database")
        }
    }

    override fun close() {}

    override fun getFactory(): Optional<Factory> =
        Optional.of(AilakeVectorConnectorFactory())

    // ── Databases ──────────────────────────────────────────────────────────

    override fun listDatabases(): List<String> = databases.keys.toList()

    override fun getDatabase(databaseName: String): CatalogDatabase =
        databases[databaseName] ?: throw DatabaseNotExistException(name, databaseName)

    override fun databaseExists(databaseName: String): Boolean =
        databases.containsKey(databaseName)

    override fun createDatabase(
        name: String,
        database: CatalogDatabase,
        ignoreIfExists: Boolean,
    ) {
        if (databases.containsKey(name)) {
            if (!ignoreIfExists) throw DatabaseAlreadyExistException(this.name, name)
            return
        }
        databases[name] = database
    }

    override fun dropDatabase(name: String, ignoreIfNotExists: Boolean, cascade: Boolean) {
        if (!databases.containsKey(name)) {
            if (!ignoreIfNotExists) throw DatabaseNotExistException(this.name, name)
            return
        }
        databases.remove(name)
    }

    override fun alterDatabase(name: String, newDatabase: CatalogDatabase, ignoreIfNotExists: Boolean) {
        if (!databases.containsKey(name)) {
            if (!ignoreIfNotExists) throw DatabaseNotExistException(this.name, name)
            return
        }
        databases[name] = newDatabase
    }

    // ── Tables ─────────────────────────────────────────────────────────────

    override fun listTables(databaseName: String): List<String> =
        tables.keys.filter { it.databaseName == databaseName }.map { it.objectName }

    override fun listViews(databaseName: String): List<String> = emptyList()

    override fun getTable(tablePath: ObjectPath): CatalogBaseTable =
        tables[tablePath] ?: throw TableNotExistException(name, tablePath)

    override fun tableExists(tablePath: ObjectPath): Boolean = tables.containsKey(tablePath)

    override fun dropTable(tablePath: ObjectPath, ignoreIfNotExists: Boolean) {
        if (!tables.containsKey(tablePath)) {
            if (!ignoreIfNotExists) throw TableNotExistException(name, tablePath)
            return
        }
        tables.remove(tablePath)
    }

    override fun renameTable(tablePath: ObjectPath, newTableName: String, ignoreIfNotExists: Boolean) {
        val tbl = tables.remove(tablePath)
            ?: run {
                if (!ignoreIfNotExists) throw TableNotExistException(name, tablePath)
                return
            }
        tables[ObjectPath(tablePath.databaseName, newTableName)] = tbl
    }

    /**
     * `AilakeNativeLoader.createTable` was already fully implemented and tested but had
     * no SQL surface — this used to only insert into the catalog's in-memory `tables`
     * map, never actually creating the on-disk AI-Lake table (Iceberg metadata.json +
     * manifest). Same "dead capability" gap Trino/Spark had before them, closed the same
     * way: the native call now runs first, and `CREATE TABLE` only registers the
     * in-memory entry (so subsequent `INSERT`/`SELECT` resolve via [AilakeVectorConnectorFactory])
     * once the real table exists.
     */
    override fun createTable(tablePath: ObjectPath, table: CatalogBaseTable, ignoreIfExists: Boolean) {
        if (tables.containsKey(tablePath)) {
            if (!ignoreIfExists) throw TableAlreadyExistException(name, tablePath)
            return
        }
        // Inject 'connector'/'warehouse'/'table-name'/'namespace' if not already present.
        // Regression: 'table-name' is a *required* connector option with no default
        // (AilakeVectorConnectorFactory.TABLE_NAME) — without this, any `CREATE TABLE`
        // through this catalog failed FactoryUtil validation unless the user redundantly
        // re-specified 'table-name' (and 'namespace') inside WITH(...), defeating the
        // point of a catalog. 'namespace' also now respects this catalog's configured
        // defaultNamespace instead of always silently falling back to the connector's
        // own "default".
        val props = table.options.toMutableMap()
        props.putIfAbsent("connector", AilakeVectorConnectorFactory.IDENTIFIER)
        props.putIfAbsent("warehouse", warehouse)
        props.putIfAbsent("table-name", tablePath.objectName)
        props.putIfAbsent("namespace", tablePath.databaseName.takeIf { it.isNotBlank() } ?: defaultNamespace)

        val ok = AilakeNativeLoader.createTable(
            warehouse     = props.getValue("warehouse"),
            namespace     = props.getValue("namespace"),
            table         = props.getValue("table-name"),
            vectorColumn  = props["vector.column"] ?: "embedding",
            dim           = props["vector.dim"]?.toInt() ?: 1536,
            metric        = props["vector.metric"] ?: "cosine",
            precision     = props["vector.precision"] ?: "f16",
            formatVersion = props["format.version"]?.toInt() ?: 2,
            catalogOpts   = catalogOptsFromProps(props),
        )
        if (!ok) throw RuntimeException("ailake CREATE TABLE failed for ${tablePath} — see logs")

        tables[tablePath] = CatalogTableImpl(
            (table as CatalogTable).schema,
            props,
            table.comment,
        )
    }

    /**
     * Same translation as [AilakeVectorConnectorFactory]'s private
     * `catalogOptsFromConfig`, but reading raw `WITH (...)` string options directly —
     * this catalog builds/validates a [CatalogTableImpl] before Flink's
     * `FactoryUtil`/`ReadableConfig` machinery is available, so it can't reuse that
     * (private, `ReadableConfig`-typed) helper. Same keys, same "catalog left unset or
     * 'hadoop' → default Hadoop catalog" semantics. See docs/guides/REST_CATALOG.md.
     */
    private fun catalogOptsFromProps(props: Map<String, String>): Map<String, String> {
        val catalog = props["catalog"] ?: return emptyMap()
        if (catalog.isBlank() || catalog == "hadoop") return emptyMap()
        val entries = mutableMapOf("catalog" to catalog)
        props["rest-uri"]?.let { entries["rest_uri"] = it }
        props["rest-prefix"]?.let { entries["rest_prefix"] = it }
        props["rest-warehouse"]?.let { entries["rest_warehouse"] = it }
        entries["rest_auth"] = props["rest-auth"] ?: "none"
        props["rest-token"]?.let { entries["rest_token"] = it }
        props["rest-oauth-token-endpoint"]?.let { entries["rest_oauth_token_endpoint"] = it }
        props["rest-oauth-client-id"]?.let { entries["rest_oauth_client_id"] = it }
        props["rest-oauth-client-secret"]?.let { entries["rest_oauth_client_secret"] = it }
        props["rest-oauth-scope"]?.let { entries["rest_oauth_scope"] = it }
        return entries
    }

    override fun alterTable(tablePath: ObjectPath, newTable: CatalogBaseTable, ignoreIfNotExists: Boolean) {
        if (!tables.containsKey(tablePath)) {
            if (!ignoreIfNotExists) throw TableNotExistException(name, tablePath)
            return
        }
        tables[tablePath] = newTable
    }

    /**
     * `ALTER TABLE ... ADD COLUMN` / `RENAME COLUMN` — Flink calls this overload (not
     * the 3-arg one above) when it has structured [TableChange]s available, which is
     * exactly the case for `ADD COLUMN`/`RENAME COLUMN` DDL. Regression: this override
     * didn't exist at all — the 3-arg `alterTable` only ever swapped Flink's in-memory
     * `CatalogBaseTable`, so `ALTER TABLE` appeared to succeed while never touching the
     * real AI-Lake/Iceberg table on disk. `AilakeNativeLoader.evolveSchema` was already
     * fully implemented and tested; this just wires it in.
     *
     * Only [TableChange.AddColumn] and [TableChange.ModifyColumnName] map to
     * `evolveSchema`'s capability (add/rename columns, metadata-only). Other change
     * kinds (drop column, retype, reposition, comments, constraints, watermarks,
     * table options) have no equivalent in `evolveSchema` and are silently accepted
     * into Flink's in-memory catalog state only — same "static per-process schema"
     * caveat as trino-plugin's `addColumn`/`renameColumn`: the change is real on disk,
     * but this catalog's own DDL-derived table options don't auto-refresh, so a newly
     * added column isn't visible to `INSERT`/`SELECT` through this same catalog
     * instance until the table is re-created (session restart / catalog reload).
     */
    override fun alterTable(
        tablePath: ObjectPath,
        newTable: CatalogBaseTable,
        tableChanges: List<TableChange>,
        ignoreIfNotExists: Boolean,
    ) {
        val existing = tables[tablePath] ?: run {
            if (!ignoreIfNotExists) throw TableNotExistException(name, tablePath)
            return
        }
        val opts = existing.options
        val tableWarehouse = opts["warehouse"] ?: warehouse
        val ns = opts["namespace"] ?: tablePath.databaseName
        val tbl = opts["table-name"] ?: tablePath.objectName

        val addCols = mutableListOf<AilakeNativeLoader.AddColReq>()
        val renameCols = mutableListOf<AilakeNativeLoader.RenameColReq>()
        for (change in tableChanges) {
            when (change) {
                is TableChange.AddColumn ->
                    addCols += AilakeNativeLoader.AddColReq(
                        change.column.name,
                        flinkTypeToIcebergType(change.column.dataType.logicalType.typeRoot),
                    )
                is TableChange.ModifyColumnName ->
                    renameCols += AilakeNativeLoader.RenameColReq(change.oldColumnName, change.newColumnName)
                else -> {} // no evolveSchema equivalent — see KDoc above
            }
        }
        if (addCols.isNotEmpty() || renameCols.isNotEmpty()) {
            AilakeNativeLoader.evolveSchema(tableWarehouse, ns, tbl, addCols, renameCols)
        }
        tables[tablePath] = newTable
    }

    /**
     * Common Iceberg primitive types only — matches this connector's own minimal
     * column type surface (id BIGINT, embedding ARRAY<FLOAT>, text columns VARCHAR).
     * See ailake-catalog's `schema_evolution.rs` for the full set of Iceberg type
     * strings the native side accepts.
     */
    private fun flinkTypeToIcebergType(root: LogicalTypeRoot): String = when (root) {
        LogicalTypeRoot.BIGINT -> "long"
        LogicalTypeRoot.INTEGER -> "int"
        LogicalTypeRoot.DOUBLE -> "double"
        LogicalTypeRoot.FLOAT -> "float"
        LogicalTypeRoot.BOOLEAN -> "boolean"
        LogicalTypeRoot.DATE -> "date"
        LogicalTypeRoot.VARCHAR, LogicalTypeRoot.CHAR -> "string"
        else -> throw IllegalArgumentException(
            "Column type $root is not supported by ALTER TABLE ADD COLUMN for AI-Lake tables — " +
            "supported types: BIGINT, INTEGER, DOUBLE, FLOAT, BOOLEAN, DATE, VARCHAR/CHAR",
        )
    }

    // ── Partitions / Stats (not implemented) ──────────────────────────────

    override fun listPartitions(tablePath: ObjectPath): List<CatalogPartitionSpec> = emptyList()
    override fun listPartitions(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec): List<CatalogPartitionSpec> = emptyList()
    override fun listPartitionsByFilter(tablePath: ObjectPath, filters: List<Expression>): List<CatalogPartitionSpec> = emptyList()
    override fun getPartition(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec): CatalogPartition = throw PartitionNotExistException(name, tablePath, partitionSpec)
    override fun partitionExists(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec): Boolean = false
    override fun createPartition(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec, partition: CatalogPartition, ignoreIfExists: Boolean) {}
    override fun dropPartition(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec, ignoreIfNotExists: Boolean) {}
    override fun alterPartition(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec, newPartition: CatalogPartition, ignoreIfNotExists: Boolean) {}
    override fun getTableStatistics(tablePath: ObjectPath): CatalogTableStatistics = CatalogTableStatistics.UNKNOWN
    override fun getTableColumnStatistics(tablePath: ObjectPath): CatalogColumnStatistics = CatalogColumnStatistics.UNKNOWN
    override fun getPartitionStatistics(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec): CatalogTableStatistics = CatalogTableStatistics.UNKNOWN
    override fun getPartitionColumnStatistics(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec): CatalogColumnStatistics = CatalogColumnStatistics.UNKNOWN
    override fun alterTableStatistics(tablePath: ObjectPath, tableStatistics: CatalogTableStatistics, ignoreIfNotExists: Boolean) {}
    override fun alterTableColumnStatistics(tablePath: ObjectPath, columnStatistics: CatalogColumnStatistics, ignoreIfNotExists: Boolean) {}
    override fun alterPartitionStatistics(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec, partitionStatistics: CatalogTableStatistics, ignoreIfNotExists: Boolean) {}
    override fun alterPartitionColumnStatistics(tablePath: ObjectPath, partitionSpec: CatalogPartitionSpec, columnStatistics: CatalogColumnStatistics, ignoreIfNotExists: Boolean) {}

    // ── Functions (not implemented) ────────────────────────────────────────

    override fun listFunctions(dbName: String): List<String> = emptyList()
    override fun getFunction(functionPath: ObjectPath): CatalogFunction = throw FunctionNotExistException(name, functionPath)
    override fun functionExists(functionPath: ObjectPath): Boolean = false
    override fun createFunction(functionPath: ObjectPath, function: CatalogFunction, ignoreIfExists: Boolean) {}
    override fun alterFunction(functionPath: ObjectPath, newFunction: CatalogFunction, ignoreIfExists: Boolean) {}
    override fun dropFunction(functionPath: ObjectPath, ignoreIfNotExists: Boolean) {}
}
