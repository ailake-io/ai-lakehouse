// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.airlift.slice.Slice
import io.trino.spi.StandardErrorCode
import io.trino.spi.TrinoException
import io.trino.spi.connector.ColumnHandle
import io.trino.spi.connector.ColumnMetadata
import io.trino.spi.connector.ConnectorInsertTableHandle
import io.trino.spi.connector.ConnectorMetadata
import io.trino.spi.connector.ConnectorOutputMetadata
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorTableHandle
import io.trino.spi.connector.ConnectorTableMetadata
import io.trino.spi.connector.Constraint
import io.trino.spi.connector.ConstraintApplicationResult
import io.trino.spi.connector.RetryMode
import io.trino.spi.connector.SchemaTableName
import io.trino.spi.predicate.TupleDomain
import io.trino.spi.statistics.ComputedStatistics
import io.trino.spi.type.ArrayType
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.BooleanType.BOOLEAN
import io.trino.spi.type.DateType.DATE
import io.trino.spi.type.DoubleType.DOUBLE
import io.trino.spi.type.IntegerType.INTEGER
import io.trino.spi.type.RealType.REAL
import io.trino.spi.type.Type
import io.trino.spi.type.VarcharType
import io.trino.spi.type.VarcharType.VARCHAR
import org.slf4j.LoggerFactory
import java.util.Optional
import java.util.OptionalLong

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
    private val hnswM: Int? = null,
    private val hnswEfConstruction: Int? = null,
    private val preNormalize: Boolean = false,
    private val deferred: Boolean = false,
    private val ftsColumns: List<String> = emptyList(),
    private val ftsTokenizer: String = "default",
) : ConnectorMetadata {

    private val log = LoggerFactory.getLogger(VectorScanMetadata::class.java)

    companion object {
        const val SCHEMA = "default"
        const val TABLE_SEARCH = "search"
        const val TABLE_SEARCH_MULTIMODAL = "search_multimodal"
        const val TABLE_SEARCH_FULL = "search_full"
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

        /** `AilakeNative.searchMultimodal` was fully implemented but had no SQL surface — see [MultimodalScanTableHandle]. */
        val MULTIMODAL_SEARCH_COLUMNS = listOf(
            ColumnMetadata("row_id", BIGINT),
            ColumnMetadata("rrf_score", DOUBLE),
            ColumnMetadata("file_path", VARCHAR),
        )
        val MULTIMODAL_SEARCH_COLUMN_HANDLES: Map<String, ColumnHandle> = mapOf(
            "row_id"    to VectorScanColumnHandle("row_id", 0),
            "rrf_score" to VectorScanColumnHandle("rrf_score", 1),
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
        listOf(
            ColumnMetadata("id", BIGINT),
            ColumnMetadata.builder().setName("embedding").setType(ArrayType(DOUBLE)).setNullable(false).build(),
        ) + textColumns.map { ColumnMetadata(it, VARCHAR) }

    private fun ingestColumnHandles(): Map<String, ColumnHandle> =
        ingestColumns().mapIndexed { i, c -> c.name to (VectorScanColumnHandle(c.name, i) as ColumnHandle) }.toMap()

    /**
     * `(id BIGINT, <vectorColumn> VARCHAR, ...textColumns VARCHAR, _distance DOUBLE)` for
     * `ailake.default.search_full` (Fase 11) — same column set `ailake_scan_json` actually
     * returns for this table (every stored column, no subset filter on the native side), so
     * this reuses the same `ailake.text-columns` catalog config `ingestColumns()` already
     * uses rather than needing a new property. The vector column comes back from
     * `AilakeNative.scan` as `list_float32` (decoded F32 values) but is exposed here as
     * VARCHAR (JSON-encoded array, e.g. `"[0.1,-0.2]"`) rather than `ARRAY<DOUBLE>` — avoids
     * hand-rolling a Trino `Block`/`BlockBuilder` for `RecordCursor.getObject` (untested
     * without a live Trino SPI build to verify against); revisit once compat-heavy CI can
     * validate an ARRAY<DOUBLE> RecordCursor path.
     */
    private fun scanColumns(): List<ColumnMetadata> =
        listOf(ColumnMetadata("id", BIGINT), ColumnMetadata(vectorColumn, VARCHAR)) +
        textColumns.map { ColumnMetadata(it, VARCHAR) } +
        listOf(ColumnMetadata("_distance", DOUBLE))

    private fun scanColumnHandles(): Map<String, ColumnHandle> =
        scanColumns().mapIndexed { i, c -> c.name to (VectorScanColumnHandle(c.name, i) as ColumnHandle) }.toMap()

    override fun listSchemaNames(session: ConnectorSession): List<String> = listOf(SCHEMA)

    override fun getTableHandle(
        session: ConnectorSession,
        schemaTableName: SchemaTableName,
    ): ConnectorTableHandle? {
        if (schemaTableName.schemaName != SCHEMA) return null
        return when (schemaTableName.tableName) {
            TABLE_SEARCH -> VectorScanTableHandle(tableUri, vectorColumn, dim, namespace, tableName)
            TABLE_SEARCH_MULTIMODAL -> MultimodalScanTableHandle(tableUri, namespace, tableName)
            TABLE_SEARCH_FULL -> ScanTableHandle(tableUri, vectorColumn, dim, namespace, tableName)
            TABLE_INGEST -> AilakeIngestTableHandle(
                tableUri, namespace, tableName, vectorColumn, dim, metric, precision, embeddingModel,
                partitionFields, formatVersion, textColumns,
                hnswM, hnswEfConstruction, preNormalize, deferred, ftsColumns, ftsTokenizer,
            )
            else -> null
        }
    }

    override fun getTableMetadata(
        session: ConnectorSession,
        table: ConnectorTableHandle,
    ): ConnectorTableMetadata = when (table) {
        is AilakeIngestTableHandle -> ConnectorTableMetadata(SchemaTableName(SCHEMA, TABLE_INGEST), ingestColumns())
        is MultimodalScanTableHandle -> ConnectorTableMetadata(SchemaTableName(SCHEMA, TABLE_SEARCH_MULTIMODAL), MULTIMODAL_SEARCH_COLUMNS)
        is ScanTableHandle -> ConnectorTableMetadata(SchemaTableName(SCHEMA, TABLE_SEARCH_FULL), scanColumns())
        else -> ConnectorTableMetadata(SchemaTableName(SCHEMA, TABLE_SEARCH), SEARCH_COLUMNS)
    }

    override fun listTables(
        session: ConnectorSession,
        schemaName: Optional<String>,
    ): List<SchemaTableName> = listOf(
        SchemaTableName(SCHEMA, TABLE_SEARCH),
        SchemaTableName(SCHEMA, TABLE_SEARCH_MULTIMODAL),
        SchemaTableName(SCHEMA, TABLE_SEARCH_FULL),
        SchemaTableName(SCHEMA, TABLE_INGEST),
    )

    override fun getColumnHandles(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
    ): Map<String, ColumnHandle> = when (tableHandle) {
        is AilakeIngestTableHandle -> ingestColumnHandles()
        is MultimodalScanTableHandle -> MULTIMODAL_SEARCH_COLUMN_HANDLES
        is ScanTableHandle -> scanColumnHandles()
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
            is MultimodalScanTableHandle -> MULTIMODAL_SEARCH_COLUMNS[ordinal]
            is ScanTableHandle -> scanColumns()[ordinal]
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

    // ── DELETE (equality/IN pushdown only) ───────────────────────────────────
    //
    // AilakeNative.deleteWhere was already fully implemented and tested but
    // had no SQL surface — same "dead capability" gap as compact/ALTER TABLE.
    // The native operation is an equality delete file (column = one of N
    // values) — there is no row-level scan-and-delete capability, so only a
    // WHERE clause that reduces to a single-column equality/IN predicate can
    // be supported. Anything else (multi-column predicates, ranges, no WHERE
    // clause at all) is rejected: applyFilter returns Optional.empty() (no
    // pushdown captured), which leaves Trino unable to plan the DELETE via
    // this connector's read path (this table can't be scanned row-by-row —
    // see VectorScanSplitManager/VectorScanRecordSetProvider, which only
    // handle VectorScanTableHandle), so the engine reports a clear
    // "DELETE not supported" error rather than this connector silently doing
    // a partial or wrong delete.

    override fun applyFilter(
        session: ConnectorSession,
        table: ConnectorTableHandle,
        constraint: Constraint,
    ): Optional<ConstraintApplicationResult<ConnectorTableHandle>> {
        val handle = table as? AilakeIngestTableHandle ?: return Optional.empty()
        if (handle.deleteColumn != null) return Optional.empty() // already captured, avoid re-applying
        val domains = constraint.summary.domains.orElse(null) ?: return Optional.empty()
        if (domains.size != 1) return Optional.empty() // only single-column predicates supported
        val (colHandle, domain) = domains.entries.first()
        if (domain.isNullAllowed) return Optional.empty() // deleteWhere has no NULL-matching semantics
        val values = domain.values.takeIf { it.isDiscreteSet }?.discreteSet ?: return Optional.empty()
        val col = colHandle as VectorScanColumnHandle
        val newHandle = handle.copy(
            deleteColumn = col.name,
            deleteValues = values.map(::trinoValueToString),
        )
        // TupleDomain.all() as the remaining filter signals the predicate is
        // fully captured in newHandle — nothing left for the engine to enforce.
        return Optional.of(ConstraintApplicationResult(newHandle, TupleDomain.all(), false))
    }

    override fun applyDelete(session: ConnectorSession, handle: ConnectorTableHandle): Optional<ConnectorTableHandle> {
        val h = handle as? AilakeIngestTableHandle ?: return Optional.empty()
        return if (h.deleteColumn != null) Optional.of(h) else Optional.empty()
    }

    override fun executeDelete(session: ConnectorSession, handle: ConnectorTableHandle): OptionalLong {
        val h = handle as AilakeIngestTableHandle
        val col = h.deleteColumn
            ?: throw TrinoException(
                StandardErrorCode.NOT_SUPPORTED,
                "DELETE requires a WHERE clause that reduces to a single-column equality or IN predicate " +
                "(e.g. WHERE id = 5 or WHERE id IN (1,2,3)) — AI-Lake only supports equality deletes, no " +
                "row-level scan-and-delete is available for this table",
            )
        val ok = AilakeNative.deleteWhere(h.tableUri, h.namespace, h.tableName, col, h.deleteValues.orEmpty())
        if (!ok) {
            throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake DELETE WHERE $col IN (...) failed for ${h.namespace}.${h.tableName} — see logs",
            )
        }
        // Native side doesn't report an exact row count for equality deletes.
        return OptionalLong.empty()
    }

    private fun trinoValueToString(value: Any): String = when (value) {
        is Slice -> value.toStringUtf8()
        else -> value.toString()
    }

    // ── ALTER TABLE ADD/RENAME COLUMN ────────────────────────────────────────
    //
    // AilakeNative.evolveSchema was already fully implemented and tested but
    // had no SQL surface reachable from Trino — same "dead capability" gap as
    // DELETE and compact, closed the same way: wiring straight into the
    // corresponding ConnectorMetadata SPI method.
    //
    // IMPORTANT limitation: this connector's schema (ingestColumns()) is built
    // once at catalog startup from the ailake.text-columns catalog property —
    // it is NOT re-read per query. A column added here is genuinely persisted
    // to the AI-Lake table's Iceberg schema on disk (evolveSchema is a real
    // metadata-only operation, not a no-op), but THIS Trino worker's in-memory
    // schema won't include it for subsequent INSERT/SELECT until the catalog
    // is reconfigured with the new column added to ailake.text-columns and
    // Trino is restarted (or the catalog reloaded). Logged as a WARN so this
    // isn't a silent surprise.

    override fun addColumn(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
        column: ColumnMetadata,
    ) {
        val handle = tableHandle as? AilakeIngestTableHandle
            ?: throw TrinoException(StandardErrorCode.NOT_SUPPORTED, "ADD COLUMN is only supported on ailake.default.ingest")
        val icebergType = trinoTypeToIcebergType(column.type)
        val schemaId = AilakeNative.evolveSchema(
            tableUri = handle.tableUri, namespace = handle.namespace, tableName = handle.tableName,
            addCols = listOf(AilakeNative.AddColReq(column.name, icebergType)),
            renameCols = emptyList(),
        )
        if (schemaId < 0) {
            throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake ADD COLUMN '${column.name}' failed for ${handle.namespace}.${handle.tableName} — see logs",
            )
        }
        log.warn(
            "[ailake] ADD COLUMN '{}' ({}) applied to {}.{} on disk (new_schema_id={}) — this catalog's " +
            "in-memory schema is static per-process; add '{}' to ailake.text-columns and restart Trino (or " +
            "reload the catalog) for INSERT/SELECT to see it.",
            column.name, icebergType, handle.namespace, handle.tableName, schemaId, column.name,
        )
    }

    override fun renameColumn(
        session: ConnectorSession,
        tableHandle: ConnectorTableHandle,
        source: ColumnHandle,
        target: String,
    ) {
        val handle = tableHandle as? AilakeIngestTableHandle
            ?: throw TrinoException(StandardErrorCode.NOT_SUPPORTED, "RENAME COLUMN is only supported on ailake.default.ingest")
        val col = source as VectorScanColumnHandle
        val schemaId = AilakeNative.evolveSchema(
            tableUri = handle.tableUri, namespace = handle.namespace, tableName = handle.tableName,
            addCols = emptyList(),
            renameCols = listOf(AilakeNative.RenameColReq(col.name, target)),
        )
        if (schemaId < 0) {
            throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake RENAME COLUMN '${col.name}' to '$target' failed for ${handle.namespace}.${handle.tableName} — see logs",
            )
        }
        log.warn(
            "[ailake] RENAME COLUMN '{}' → '{}' applied to {}.{} on disk (new_schema_id={}) — update " +
            "ailake.text-columns and restart Trino (or reload the catalog) for INSERT/SELECT to see the new name.",
            col.name, target, handle.namespace, handle.tableName, schemaId,
        )
    }

    /**
     * Common Iceberg primitive types only — matches this connector's own
     * minimal column type surface (id BIGINT, embedding ARRAY<DOUBLE>, text
     * columns VARCHAR). Complex types (ARRAY/MAP/ROW) and other Trino
     * timestamp/decimal variants are rejected rather than silently
     * mis-mapped — see ailake-catalog's `schema_evolution.rs` for the full
     * set of Iceberg type strings the native side accepts.
     */
    private fun trinoTypeToIcebergType(type: Type): String = when {
        type == BIGINT -> "long"
        type == INTEGER -> "int"
        type == DOUBLE -> "double"
        type == REAL -> "float"
        type == BOOLEAN -> "boolean"
        type == DATE -> "date"
        type is VarcharType -> "string"
        else -> throw TrinoException(
            StandardErrorCode.NOT_SUPPORTED,
            "Column type $type is not supported by ALTER TABLE ADD COLUMN for AI-Lake tables — " +
            "supported types: bigint, integer, double, real, boolean, date, varchar",
        )
    }
}
