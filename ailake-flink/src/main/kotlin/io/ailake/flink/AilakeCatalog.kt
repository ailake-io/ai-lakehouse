package io.ailake.flink

import org.apache.flink.table.catalog.*
import org.apache.flink.table.catalog.exceptions.*
import org.apache.flink.table.catalog.stats.CatalogColumnStatistics
import org.apache.flink.table.catalog.stats.CatalogTableStatistics
import org.apache.flink.table.expressions.Expression
import org.apache.flink.table.factories.Factory
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

    override fun createTable(tablePath: ObjectPath, table: CatalogBaseTable, ignoreIfExists: Boolean) {
        if (tables.containsKey(tablePath)) {
            if (!ignoreIfExists) throw TableAlreadyExistException(name, tablePath)
            return
        }
        // Inject 'connector' = 'ailake' and 'warehouse' if not already present
        val props = table.options.toMutableMap()
        props.putIfAbsent("connector", AilakeVectorConnectorFactory.IDENTIFIER)
        props.putIfAbsent("warehouse", warehouse)
        tables[tablePath] = CatalogTableImpl(
            (table as CatalogTable).schema,
            props,
            table.comment,
        )
    }

    override fun alterTable(tablePath: ObjectPath, newTable: CatalogBaseTable, ignoreIfNotExists: Boolean) {
        if (!tables.containsKey(tablePath)) {
            if (!ignoreIfNotExists) throw TableNotExistException(name, tablePath)
            return
        }
        tables[tablePath] = newTable
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
