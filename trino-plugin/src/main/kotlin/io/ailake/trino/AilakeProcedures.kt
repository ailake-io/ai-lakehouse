// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import com.fasterxml.jackson.databind.ObjectMapper
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
    // REST Catalog (Fase 17/19) config — see VectorScanConnectorFactory's
    // ailake.catalog/ailake.rest-* properties and docs/guides/REST_CATALOG.md.
    // Safe to thread here (unlike search/scan/insert): this class isn't a
    // JSON-serialized Trino handle, just a plain coordinator-side object.
    private val catalogOpts: Map<String, String> = emptyMap(),
    // create_table (Fase 23) config — same catalog-level properties
    // VectorScanConnectorFactory already parses for search/insert
    // (ailake.vector-column/vector-dim/metric/precision/format-version).
    private val vectorColumn: String = "embedding",
    private val dim: Int = 1536,
    private val metric: String = "cosine",
    private val precision: String = "f16",
    private val formatVersion: Int = 2,
) {
    private val log = LoggerFactory.getLogger(AilakeProcedures::class.java)

    companion object {
        private val COMPACT = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("compact", ConnectorSession::class.java)
        )
        private val CREATE_TABLE = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("createTable", ConnectorSession::class.java)
        )
        private val DECAY_MEMORIES = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("decayMemories", ConnectorSession::class.java)
        )
        private val MIGRATE = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("migrate", ConnectorSession::class.java)
        )
        private val DELETE_ROWS = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("deleteRows", ConnectorSession::class.java)
        )
        private val ADD_VECTOR_COLUMN = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("addVectorColumn", ConnectorSession::class.java)
        )
        private val BACKFILL_VECTOR_COLUMN = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("backfillVectorColumn", ConnectorSession::class.java)
        )
        private val ESTIMATE = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("estimate", ConnectorSession::class.java)
        )
        private val INFO = MethodHandles.lookup().unreflect(
            AilakeProcedures::class.java.getMethod("info", ConnectorSession::class.java)
        )
    }

    fun getProcedures(): Set<Procedure> = setOf(
        Procedure(
            "system",
            "compact",
            emptyList(),
            COMPACT.bindTo(this),
        ),
        Procedure(
            "system",
            "create_table",
            emptyList(),
            CREATE_TABLE.bindTo(this),
        ),
        // 6 procedures below close a gap found auditing this plugin:
        // AilakeNative.decayMemories/migrate/deleteRows/addVectorColumn/
        // backfillVectorColumn/estimate had zero SQL surface at all — same
        // "dead capability" pattern compact()/create_table() had before them,
        // but these 6 never even had a C-ABI export until now (see
        // ailake-jni's ailake_decay_memories_json and siblings). All no-arg,
        // same reasoning as compact()/create_table(): parameters come from
        // SET SESSION properties (see VectorScanConnector.getSessionProperties)
        // rather than typed CALL arguments — reuses the JSON-string-session-
        // property pattern `multimodal_queries` already established, instead
        // of introducing this codebase's first typed Procedure.Argument usage.
        Procedure("system", "decay_memories", emptyList(), DECAY_MEMORIES.bindTo(this)),
        Procedure("system", "migrate", emptyList(), MIGRATE.bindTo(this)),
        Procedure("system", "delete_rows", emptyList(), DELETE_ROWS.bindTo(this)),
        Procedure("system", "add_vector_column", emptyList(), ADD_VECTOR_COLUMN.bindTo(this)),
        Procedure("system", "backfill_vector_column", emptyList(), BACKFILL_VECTOR_COLUMN.bindTo(this)),
        Procedure("system", "estimate", emptyList(), ESTIMATE.bindTo(this)),
        // Found in a later audit pass: `info` (table metadata / foreign-file
        // report — `ailake info`) had zero binding coverage anywhere outside
        // the CLI and the Airflow provider, not even a C-ABI export. Same
        // no-arg/logged-result pattern as the 6 procedures above.
        Procedure("system", "info", emptyList(), INFO.bindTo(this)),
    )

    /** Invoked by the Trino engine as `CALL ailake.system.compact()`. */
    fun compact(session: ConnectorSession) {
        val filesCompacted = AilakeNative.compact(tableUri, namespace, tableName, catalogOpts = catalogOpts)
            ?: throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake compact failed for table=$namespace.$tableName — native library absent or the call " +
                "failed; check the coordinator/worker logs for [ailake] compact ok=false",
            )
        log.info("[ailake] CALL compact() table={}.{} files_compacted={}", namespace, tableName, filesCompacted)
    }

    /**
     * `CALL ailake.system.create_table()` — creates the catalog's configured
     * ingest table (empty, schema only) via `AilakeNative.createTable`, which
     * was already fully implemented and tested but had no SQL surface
     * reachable from Trino at all (same "dead capability" gap `compact` had
     * before it) — closes it the same way, via `Connector.getProcedures()`.
     * No arguments, same reasoning as `compact`: table-uri/namespace/
     * table-name/vector-column/dim/metric/precision/format-version are all
     * catalog-level properties (`ailake.*`), not per-statement.
     */
    fun createTable(session: ConnectorSession) {
        val ok = AilakeNative.createTable(
            tableUri, namespace, tableName, vectorColumn, dim, metric, precision, formatVersion,
            catalogOpts = catalogOpts,
        )
        if (!ok) {
            throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake create_table failed for table=$namespace.$tableName — native library absent or the " +
                "call failed; check the coordinator/worker logs for [ailake] create_table ok=false",
            )
        }
        log.info("[ailake] CALL create_table() table={}.{}", namespace, tableName)
    }

    private val mapper = ObjectMapper()

    /**
     * `CALL ailake.system.decay_memories()` — recomputes recency_weight for
     * every row (Phase 9 agent memory). Set `SET SESSION ailake.decay_lambda
     * = 0.1` first.
     */
    fun decayMemories(session: ConnectorSession) {
        val lambda = session.getProperty("decay_lambda", Double::class.javaObjectType) ?: 0.1
        val filesUpdated = AilakeNative.decayMemories(tableUri, namespace, tableName, lambda.toFloat(), catalogOpts)
            ?: throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake decay_memories failed for table=$namespace.$tableName — native library absent or " +
                "the call failed; check the coordinator/worker logs for [ailake] decayMemories failed",
            )
        log.info("[ailake] CALL decay_memories() table={}.{} files_updated={}", namespace, tableName, filesUpdated)
    }

    /**
     * `CALL ailake.system.migrate()` — re-embeds a column via an external
     * embed command. Set `SET SESSION ailake.migrate_config = '{"old_column":
     * "embedding","new_column":"embedding_v2","text_column":"chunk_text",
     * "embed_cmd":"python3 embed.py"}'` first.
     */
    fun migrate(session: ConnectorSession) {
        val configJson = session.getProperty("migrate_config", String::class.java) ?: ""
        if (configJson.isBlank()) {
            throw TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "ailake.migrate_config session property is required")
        }
        val cfg = mapper.readTree(configJson)
        AilakeNative.migrate(
            tableUri, namespace, tableName,
            oldColumn = cfg.get("old_column").asText(),
            newColumn = cfg.get("new_column").asText(),
            textColumn = cfg.get("text_column").asText(),
            embedCmd = cfg.get("embed_cmd").asText(),
            strategy = cfg.path("strategy").asText("atomic-replace"),
            batchSize = cfg.path("batch_size").asInt(10_000),
            modelName = if (cfg.has("model_name")) cfg.get("model_name").asText() else null,
            modelVersion = if (cfg.has("model_version")) cfg.get("model_version").asText() else null,
            catalogOpts = catalogOpts,
        )
        log.info("[ailake] CALL migrate() table={}.{}", namespace, tableName)
    }

    /**
     * `CALL ailake.system.delete_rows()` — deletes row positions via Iceberg
     * V3 Deletion Vectors. Set `SET SESSION ailake.delete_rows_config =
     * '{"file":"data/part-00000.parquet","row_ids":[0,5,42]}'` first.
     */
    fun deleteRows(session: ConnectorSession) {
        val configJson = session.getProperty("delete_rows_config", String::class.java) ?: ""
        if (configJson.isBlank()) {
            throw TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "ailake.delete_rows_config session property is required")
        }
        val cfg = mapper.readTree(configJson)
        val rowIds = cfg.get("row_ids").map { it.asInt() }
        AilakeNative.deleteRows(tableUri, namespace, tableName, cfg.get("file").asText(), rowIds, catalogOpts)
        log.info("[ailake] CALL delete_rows() table={}.{} rows={}", namespace, tableName, rowIds.size)
    }

    /**
     * `CALL ailake.system.add_vector_column()` — adds a vector column to the
     * schema (metadata-only). Set `SET SESSION ailake.add_vector_column_config
     * = '{"column":"image_embedding","dim":512}'` first.
     */
    fun addVectorColumn(session: ConnectorSession) {
        val configJson = session.getProperty("add_vector_column_config", String::class.java) ?: ""
        if (configJson.isBlank()) {
            throw TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "ailake.add_vector_column_config session property is required")
        }
        val cfg = mapper.readTree(configJson)
        val newSchemaId = AilakeNative.addVectorColumn(
            tableUri, namespace, tableName,
            column = cfg.get("column").asText(),
            dim = cfg.get("dim").asInt(),
            metric = cfg.path("metric").asText("cosine"),
            precision = cfg.path("precision").asText("f16"),
            preNormalize = cfg.path("pre_normalize").asBoolean(false),
            hnswM = if (cfg.has("hnsw_m")) cfg.get("hnsw_m").asInt() else null,
            hnswEfConstruction = if (cfg.has("hnsw_ef_construction")) cfg.get("hnsw_ef_construction").asInt() else null,
            catalogOpts = catalogOpts,
        ) ?: throw TrinoException(
            StandardErrorCode.GENERIC_USER_ERROR,
            "ailake add_vector_column failed for table=$namespace.$tableName — native library absent or " +
            "the call failed; check the coordinator/worker logs for [ailake] addVectorColumn failed",
        )
        log.info("[ailake] CALL add_vector_column() table={}.{} new_schema_id={}", namespace, tableName, newSchemaId)
    }

    /**
     * `CALL ailake.system.backfill_vector_column()` — backfills embeddings
     * for a column added via `add_vector_column`. Set `SET SESSION
     * ailake.backfill_vector_column_config = '{"column":"image_embedding",
     * "text_column":"image_uri","embed_cmd":"python3 embed_images.py"}'` first.
     */
    fun backfillVectorColumn(session: ConnectorSession) {
        val configJson = session.getProperty("backfill_vector_column_config", String::class.java) ?: ""
        if (configJson.isBlank()) {
            throw TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "ailake.backfill_vector_column_config session property is required")
        }
        val cfg = mapper.readTree(configJson)
        AilakeNative.backfillVectorColumn(
            tableUri, namespace, tableName,
            column = cfg.get("column").asText(),
            textColumn = cfg.get("text_column").asText(),
            embedCmd = cfg.get("embed_cmd").asText(),
            batchSize = cfg.path("batch_size").asInt(512),
            catalogOpts = catalogOpts,
        )
        log.info("[ailake] CALL backfill_vector_column() table={}.{}", namespace, tableName)
    }

    /**
     * `CALL ailake.system.estimate()` — storage/index size estimates for a
     * hypothetical table (pure math, no I/O). Set `SET SESSION
     * ailake.estimate_config = '{"rows":1000000,"dim":1536}'` first. `CALL`
     * procedures have no result set — the estimate is logged server-side
     * (INFO) instead of returned to the client.
     */
    fun estimate(session: ConnectorSession) {
        val configJson = session.getProperty("estimate_config", String::class.java) ?: ""
        if (configJson.isBlank()) {
            throw TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "ailake.estimate_config session property is required")
        }
        val cfg = mapper.readTree(configJson)
        val result = AilakeNative.estimate(
            rows = cfg.get("rows").asLong(),
            dim = cfg.get("dim").asInt(),
            hnswM = cfg.path("hnsw_m").asInt(16),
            pqM = if (cfg.has("pq_m")) cfg.get("pq_m").asInt() else null,
        ) ?: throw TrinoException(
            StandardErrorCode.GENERIC_USER_ERROR,
            "ailake estimate failed — native library absent or the call failed",
        )
        log.info("[ailake] CALL estimate() result={}", mapper.writeValueAsString(result))
    }

    /**
     * `CALL ailake.system.info()` — reports current snapshot, file/row/size
     * counts, index status breakdown, and "foreign" files (written by a
     * generic Iceberg engine — no AI-Lake centroid/HNSW) for the catalog's
     * configured table. No arguments, same reasoning as `compact()`. `CALL`
     * procedures have no result set — logged server-side (INFO) instead.
     */
    fun info(session: ConnectorSession) {
        val result = AilakeNative.info(tableUri, namespace, tableName, catalogOpts)
            ?: throw TrinoException(
                StandardErrorCode.GENERIC_USER_ERROR,
                "ailake info failed for table=$namespace.$tableName — native library absent or the call " +
                "failed; check the coordinator/worker logs for [ailake] info failed",
            )
        log.info("[ailake] CALL info() table={}.{} result={}", namespace, tableName, mapper.writeValueAsString(result))
    }
}
