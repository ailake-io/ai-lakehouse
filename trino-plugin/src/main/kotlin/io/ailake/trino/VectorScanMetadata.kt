// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.airlift.slice.Slice
import io.trino.spi.connector.ColumnHandle
import io.trino.spi.connector.ColumnMetadata
import io.trino.spi.connector.ConnectorInsertTableHandle
import io.trino.spi.connector.ConnectorMetadata
import io.trino.spi.connector.ConnectorOutputMetadata
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTableMetadata
import io.trino.spi.connector.RetryMode
import io.trino.spi.connector.SchemaTableName
import io.trino.spi.statistics.ComputedStatistics
import io.trino.spi.type.ArrayType
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.DoubleType.DOUBLE
import io.trino.spi.type.VarcharType.VARCHAR
import java.util.Optional

class VectorScanMetadata(
    private val tableUri: String,
    private val vectorColumn: String,
    private val dim: Int,
    private val metric: String,
    private val precision: String,
    private val namespace: String,
    private val tableName: String,
    private val embeddingModel: String? = null,
    private val partitionFields: List<AilakeNative.PartitionFieldDef> = emptyList(),
    private val formatVersion: Int = 2,
    // Extra VARCHAR columns (e.g. chunk text, source, page) written alongside
    // id + embedding via AilakeNative.writeBatch's `columns` map — see
    // ingestColumns() doc. Configured catalog-wide via ailake.text-columns
    // (VectorScanConnectorFactory); Trino's connector schema is fixed per
    // catalog, so there's no per-INSERT way to vary this.
    private val textColumns: List<String> = emptyList(),
) : ConnectorMetadata {

    companion object {
        const val SCHEMA = "default"
        const val TABLE_SEARCH = "search"
        const val TABLE_INGEST = "ingest"

        val SEARCH_COLUMNS = listOf(
            ColumnMetadata("row_id", BIGINT),
            ColumnMetadata("distance", DOUBLE),
            ColumnMetadata("file_path", VARCHAR),
        )
        val SEARCH_COLUMN_HANDLES: Map<String, ColumnHandle> = mapOf(
            "row_id"    to VectorScanColumnHandle("row_id", 0),
            "distance"  to VectorScanColumnHandle("distance", 1),
            "file_path" to VectorScanColumnHandle("file_path", 2),
        )
    }

    /**
     * `(id BIGINT, embedding ARRAY<DOUBLE>, ...textColumns VARCHAR)` — extra
     * columns are appended in the order configured via `ailake.text-columns`
     * on the catalog. `AilakePageSink` relies on this exact ordering (id=0,
     * vector=1, text columns starting at 2) to read the right Page channels.
     */
    private fun ingestColumns(): List<ColumnMetadata> =
        listOf(ColumnMetadata("id", BIGINT), ColumnMetadata("embedding", ArrayType(DOUBLE))) +
            textColumns.map { ColumnMetadata(it, VARCHAR) }

    private fun ingestColumnHandles(): Map<String, ColumnHandle> =
        ingestColumns().mapIndexed { i, c -> c.name to (VectorScanColumnHandle(c.name, i) as ColumnHandle) }.toMap()

    override fun listSchemaNames(session: ConnectorSession): List<String> = listOf(SCHEMA)

    override fun getTableHandle(
        session: ConnectorSession,
        schemaTableName: SchemaTableName,
    ): ConnectorTableHandle? {
        if (schemaTableName.schemaName != SCHEMA) return null
        return when (schemaTableName.tableName) {
            TABLE_SEARCH -> VectorScanTableHandle(tableUri, vectorColumn, dim)
            TABLE_INGEST -> AilakeIngestTableHandle(tableUri, namespace, tableName, vectorColumn, dim, metric, precision, embeddingModel, partitionFields, formatVersion, textColumns)
            else -> null
        }
    }

    override fun getTableMetadata(
        session: ConnectorSession,
        table: ConnectorTableHandle,
    ): ConnectorTableMetadata = when (table) {
        is AilakeIngestTableHandle -> ConnectorTableMetadata(SchemaTableName(SCHEMA, TABLE_INGEST), ingestColumns())
        else -> ConnectorTableMetadata(SchemaTableName(SCHEMA, TABLE_SEARCH), SEARCH_COLUMNS)
    }

    override fun listTables(
        session: ConnectorSession,
        schemaName: Optional<String>,
    ): List<SchemaTableName> = listOf(
        SchemaTableName(SCHEMA, TABLE_SEARCH),
        SchemaTableName(SCHEMA, TABLE_INGEST),
    )

    override fun getColumnHandles(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
    ): Map<String, ColumnHandle> = when (tableHandle) {
        is AilakeIngestTableHandle -> ingestColumnHandles()
        else -> SEARCH_COLUMN_HANDLES
    }

    override fun getColumnMetadata(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
        columnHandle: ColumnHandle,
    ): ColumnMetadata {
        val ordinal = (columnHandle as VectorScanColumnHandle).ordinal
        return when (tableHandle) {
            is AilakeIngestTableHandle -> ingestColumns()[ordinal]
            else -> SEARCH_COLUMNS[ordinal]
        }
    }

    // ── Write path ────────────────────────────────────────────────────────────

    override fun beginInsert(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
        columns: List<ColumnHandle>,
        retryMode: RetryMode,
    ): ConnectorInsertTableHandle = tableHandle as AilakeIngestTableHandle

    override fun finishInsert(
        session: ConnectorSession,
        insertHandle: ConnectorInsertTableHandle,
        fragments: Collection<Slice>,
        computedStatistics: Collection<ComputedStatistics>,
    ): Optional<ConnectorOutputMetadata> = Optional.empty()
}
