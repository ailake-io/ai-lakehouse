// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.StandardErrorCode
import io.trino.spi.TrinoException
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.procedure.Procedure
import org.slf4j.LoggerFactory
import java.lang.invoke.MethodHandles

/**
 * `CALL ailake.system.compact()` — compacts small files in the catalog's
 * configured ingest table.
 *
 * `AilakeNative.compact` was already fully implemented and tested but had no
 * SQL surface reachable from Trino at all — same "dead capability" gap as
 * DELETE/ALTER TABLE ADD COLUMN, fixed the same way Iceberg's own connector
 * exposes maintenance operations (`CALL iceberg.system.rollback_to_snapshot(...)`),
 * via `Connector.getProcedures()` rather than a heavier `ALTER TABLE ... EXECUTE`
 * table-procedure integration.
 *
 * No arguments: each catalog is configured for exactly one AI-Lake table
 * (table-uri/namespace/table-name are catalog-level properties, not
 * per-statement — see [VectorScanConnectorFactory]), so there's nothing to
 * parameterize.
 */
class AilakeProcedures(
    private val tableUri: String,
    private val namespace: String,
    private val tableName: String,
) {
    private val log = LoggerFactory.getLogger(AilakeProcedures::class.java)

    companion object {
        private val COMPACT = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("compact", ConnectorSession::class.java)
        )
    }

    fun getProcedures(): Set<Procedure> = setOf(
        Procedure(
            "system",
            "compact",
            emptyList(),
            COMPACT.bindTo(this),
        )
    )

    /** Invoked by the Trino engine as `CALL ailake.system.compact()`. */
    fun compact(session: ConnectorSession) {
        val filesCompacted = AilakeNative.compact(tableUri, namespace, tableName)
            ?: throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake compact failed for table=$namespace.$tableName — native library absent or the call " +
                "failed; check the coordinator/worker logs for [ailake] compact ok=false",
            )
        log.info("[ailake] CALL compact() table={}.{} files_compacted={}", namespace, tableName, filesCompacted)
    }
}
