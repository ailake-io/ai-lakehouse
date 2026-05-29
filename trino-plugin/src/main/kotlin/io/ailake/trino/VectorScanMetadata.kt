// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.ColumnHandle
import io.trino.spi.connector.ColumnMetadata
import io.trino.spi.connector.ConnectorMetadata
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTableMetadata
import io.trino.spi.connector.SchemaTableName
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.DoubleType.DOUBLE
import io.trino.spi.type.VarcharType.VARCHAR
import java.util.Optional

class VectorScanMetadata(
    private val tableUri: String,
    private val vectorColumn: String,
    private val dim: Int,
) : ConnectorMetadata {

    companion object {
        const val SCHEMA = "default"
        const val TABLE = "search"

        val COLUMNS = listOf(
            ColumnMetadata("row_id", BIGINT),
            ColumnMetadata("distance", DOUBLE),
            ColumnMetadata("file_path", VARCHAR),
        )
        val COLUMN_HANDLES: Map<String, ColumnHandle> = mapOf(
            "row_id" to VectorScanColumnHandle("row_id", 0),
            "distance" to VectorScanColumnHandle("distance", 1),
            "file_path" to VectorScanColumnHandle("file_path", 2),
        )
    }

    override fun listSchemaNames(session: ConnectorSession): List<String> = listOf(SCHEMA)

    override fun getTableHandle(
        session: ConnectorSession,
        tableName: SchemaTableName,
    ): ConnectorTableHandle? {
        if (tableName.schemaName == SCHEMA && tableName.tableName == TABLE)
            return VectorScanTableHandle(tableUri, vectorColumn, dim)
        return null
    }

    override fun getTableMetadata(
        session: ConnectorSession,
        table: ConnectorTableHandle,
    ): ConnectorTableMetadata = ConnectorTableMetadata(
        SchemaTableName(SCHEMA, TABLE),
        COLUMNS,
    )

    override fun listTables(
        session: ConnectorSession,
        schemaName: Optional<String>,
    ): List<SchemaTableName> = listOf(SchemaTableName(SCHEMA, TABLE))

    override fun getColumnHandles(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
    ): Map<String, ColumnHandle> = COLUMN_HANDLES

    override fun getColumnMetadata(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
        columnHandle: ColumnHandle,
    ): ColumnMetadata = COLUMNS[(columnHandle as VectorScanColumnHandle).ordinal]
}
