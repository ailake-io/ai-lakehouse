// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import com.fasterxml.jackson.databind.ObjectMapper
import io.ailake.flink.internal.AilakeNativeLoader
import io.ailake.flink.internal.AilakeNativeLoader.PartitionFieldDef
import org.apache.flink.configuration.ConfigOption
import org.apache.flink.configuration.ConfigOptions
import org.apache.flink.table.api.ValidationException
import org.apache.flink.table.connector.sink.DynamicTableSink
import org.apache.flink.table.connector.source.DynamicTableSource
import org.apache.flink.table.factories.DynamicTableFactory
import org.apache.flink.table.factories.DynamicTableSinkFactory
import org.apache.flink.table.factories.DynamicTableSourceFactory
import org.apache.flink.table.factories.FactoryUtil
import org.apache.flink.table.types.logical.LogicalTypeRoot

/**
 * Flink SQL connector factory for AI-Lake vector tables.
 *
 * This one connector serves two DIFFERENT DDL shapes depending on direction —
 * INSERT (write) and SELECT (search) use unrelated column sets, exactly like
 * Spark/Trino's separate `ingest`/`search` tables, just modeled here as two
 * separate Flink `CREATE TABLE` statements sharing the same `warehouse`/
 * `namespace`/`table-name` (i.e. the same physical AI-Lake table).
 *
 * Write (sink) DDL example — id + vector + any number of extra STRING columns:
 * ```sql
 * CREATE TABLE docs_ingest (
 *   id        BIGINT,
 *   embedding ARRAY<FLOAT>,
 *   text      STRING       -- extra metadata, persisted via columns= (see fts.columns)
 * ) WITH (
 *   'connector'        = 'ailake',
 *   'warehouse'        = 's3://my-lake/',
 *   'namespace'        = 'default',
 *   'table-name'       = 'docs',
 *   'vector.column'    = 'embedding',
 *   'vector.dim'       = '128',
 *   'vector.metric'    = 'euclidean',
 *   'vector.precision' = 'f16',
 *   'hnsw.m'                = '32',     -- optional, table default if unset
 *   'hnsw.ef-construction'  = '200',    -- optional, table default if unset
 *   'pre-normalize'         = 'false',  -- optional
 *   'deferred'              = 'false'   -- optional
 * );
 * ```
 *
 * Read (source) DDL example — fixed 3-column search-result shape, validated
 * at DDL-resolution time (see [validateSearchResultSchema]):
 * ```sql
 * CREATE TABLE docs_search (
 *   row_id    BIGINT,
 *   distance  FLOAT,
 *   file_path STRING
 * ) WITH (
 *   'connector'     = 'ailake',
 *   'warehouse'     = 's3://my-lake/',
 *   'namespace'     = 'default',
 *   'table-name'    = 'docs',
 *   'vector.column' = 'embedding',
 *   'vector.dim'    = '128',
 *   'search.top-k'  = '10',
 *   'search.ef'     = '50'
 * );
 * ```
 */
class AilakeVectorConnectorFactory : DynamicTableSourceFactory, DynamicTableSinkFactory {

    companion object {
        const val IDENTIFIER = "ailake"

        val WAREHOUSE       = ConfigOptions.key("warehouse").stringType().noDefaultValue()
        val NAMESPACE       = ConfigOptions.key("namespace").stringType().defaultValue("default")
        val TABLE_NAME      = ConfigOptions.key("table-name").stringType().noDefaultValue()
        val VEC_COL         = ConfigOptions.key("vector.column").stringType().defaultValue("embedding")
        val VEC_DIM         = ConfigOptions.key("vector.dim").intType().noDefaultValue()
        val VEC_METRIC      = ConfigOptions.key("vector.metric").stringType().defaultValue("euclidean")
        val VEC_PREC        = ConfigOptions.key("vector.precision").stringType().defaultValue("f16")
        val SEARCH_TOPK       = ConfigOptions.key("search.top-k").intType().defaultValue(10)
        val SEARCH_EF         = ConfigOptions.key("search.ef").intType().defaultValue(50)
        /** `"search"` (default, fixed 3-column shape) or `"full"` (Fase 11 — search + full-row fetch, dynamic columns from the DDL, no JOIN needed). */
        val SEARCH_MODE       = ConfigOptions.key("search.mode").stringType().defaultValue("search")
        val EMBEDDING_MODEL   = ConfigOptions.key("embedding.model").stringType().noDefaultValue()
        val PARTITION_FIELDS  = ConfigOptions.key("partition.fields").stringType().defaultValue("[]")
        val FORMAT_VERSION    = ConfigOptions.key("format.version").intType().defaultValue(2)
        /** Comma-separated text columns to index with Tantivy FTS. Empty = no FTS. */
        val FTS_COLUMNS       = ConfigOptions.key("fts.columns").stringType().defaultValue("")
        val FTS_TOKENIZER     = ConfigOptions.key("fts.tokenizer").stringType().defaultValue("default")
        // Write-tuning knobs — AilakeNativeLoader.writeBatch already accepted all
        // four, but nothing exposed them as DDL options and AilakeSinkFunction.flush()
        // always passed the defaults (null/null/false/false).
        val HNSW_M                = ConfigOptions.key("hnsw.m").intType().noDefaultValue()
        val HNSW_EF_CONSTRUCTION  = ConfigOptions.key("hnsw.ef-construction").intType().noDefaultValue()
        val PRE_NORMALIZE         = ConfigOptions.key("pre-normalize").booleanType().defaultValue(false)
        val DEFERRED              = ConfigOptions.key("deferred").booleanType().defaultValue(false)
        /**
         * Multi-column (Phase 8 multimodal) ingest — e.g. text + image embeddings on the
         * same row, each with its own HNSW index. JSON array of
         * `{"column","dim","metric"?,"precision"?,"modality"?}`. When set, the sink expects
         * one `ARRAY<FLOAT>` column per entry (by name) instead of the single `vector.column`,
         * and writes via `ailake_write_batch_multi_json`. Was already exposed from Spark
         * (`ailakeWriteMulti`) but had no DDL option here at all.
         */
        val VEC_COLUMNS           = ConfigOptions.key("vector.columns").stringType().defaultValue("[]")
    }

    override fun factoryIdentifier(): String = IDENTIFIER

    override fun requiredOptions(): Set<ConfigOption<*>> = setOf(WAREHOUSE, TABLE_NAME, VEC_DIM)

    override fun optionalOptions(): Set<ConfigOption<*>> = setOf(
        NAMESPACE, VEC_COL, VEC_METRIC, VEC_PREC, SEARCH_TOPK, SEARCH_EF, SEARCH_MODE, EMBEDDING_MODEL,
        PARTITION_FIELDS, FORMAT_VERSION, FTS_COLUMNS, FTS_TOKENIZER,
        HNSW_M, HNSW_EF_CONSTRUCTION, PRE_NORMALIZE, DEFERRED, VEC_COLUMNS,
    )

    /**
     * Regression: `AilakeInputFormat.nextRecord()` always emits a fixed
     * `(row_id BIGINT, distance FLOAT, file_path STRING)` row regardless of what
     * `CREATE TABLE` declared — the connector's own DDL example (this class's KDoc)
     * used to show a 4-column ingest-shaped table for BOTH source and sink, which
     * would deserialize-crash on `SELECT`. A source table must be declared with
     * exactly this 3-column search-result shape (matching Spark/Trino's separate
     * `ailake.default.search` table) — validated here at DDL-resolution time instead
     * of failing opaquely on the first row.
     */
    internal fun validateSearchResultSchema(schema: org.apache.flink.table.catalog.ResolvedSchema) {
        val expected = listOf(
            "row_id" to LogicalTypeRoot.BIGINT,
            "distance" to LogicalTypeRoot.FLOAT,
            "file_path" to LogicalTypeRoot.VARCHAR,
        )
        val actual = schema.columns.map { it.name to it.dataType.logicalType.typeRoot }
        if (actual != expected) {
            throw ValidationException(
                "AI-Lake source tables must declare exactly (row_id BIGINT, distance FLOAT, " +
                "file_path STRING) in that order — got ${schema.columns.joinToString { "${it.name} ${it.dataType}" }}. " +
                "Use a separate CREATE TABLE for writes (id BIGINT, embedding ARRAY<FLOAT>, ...) — " +
                "see AilakeVectorConnectorFactory's KDoc for both DDL shapes.",
            )
        }
    }

    /**
     * `search.mode = 'full'` (Fase 11) tables aren't validated against the fixed 3-column
     * shape — the whole point is dynamic columns read straight from the DDL, one native
     * `ailake_scan_json` call fetching every stored column instead of the `(row_id, distance,
     * file_path)` triple a manual JOIN against a separately-registered Iceberg table would
     * otherwise be needed to enrich. Only requirement: the last declared column must be
     * `_distance` (FLOAT or DOUBLE) — matches `ailake_scan_json`'s "`_distance` always last"
     * contract, so [AilakeScanInputFormat] always has a stable slot to place it in.
     */
    internal fun validateScanResultSchema(schema: org.apache.flink.table.catalog.ResolvedSchema) {
        val last = schema.columns.lastOrNull()
            ?: throw ValidationException("AI-Lake 'search.mode=full' tables must declare at least the _distance column")
        val lastRoot = last.dataType.logicalType.typeRoot
        if (last.name != "_distance" || lastRoot !in setOf(LogicalTypeRoot.FLOAT, LogicalTypeRoot.DOUBLE)) {
            throw ValidationException(
                "AI-Lake 'search.mode=full' tables must declare '_distance' (FLOAT or DOUBLE) as the " +
                "last column — got ${schema.columns.joinToString { "${it.name} ${it.dataType}" }}",
            )
        }
    }

    override fun createDynamicTableSource(context: DynamicTableFactory.Context): DynamicTableSource {
        val helper = FactoryUtil.createTableFactoryHelper(this, context)
        helper.validate()
        val opts = helper.options
        return if (opts.get(SEARCH_MODE) == "full") {
            validateScanResultSchema(context.catalogTable.resolvedSchema)
            AilakeScanTableSource(
                warehouse = opts.get(WAREHOUSE),
                namespace = opts.get(NAMESPACE),
                tableName = opts.get(TABLE_NAME),
                vecCol    = opts.get(VEC_COL),
                dim       = opts.get(VEC_DIM),
                topK      = opts.get(SEARCH_TOPK),
                columns   = context.catalogTable.resolvedSchema.columns.map {
                    ScanColumnSpec(it.name, it.dataType.logicalType.typeRoot)
                },
            )
        } else {
            validateSearchResultSchema(context.catalogTable.resolvedSchema)
            AilakeVectorTableSource(
                warehouse  = opts.get(WAREHOUSE),
                namespace  = opts.get(NAMESPACE),
                tableName  = opts.get(TABLE_NAME),
                vecCol     = opts.get(VEC_COL),
                dim        = opts.get(VEC_DIM),
                topK       = opts.get(SEARCH_TOPK),
                efSearch   = opts.get(SEARCH_EF),
                schema     = context.catalogTable.resolvedSchema,
            )
        }
    }

    override fun createDynamicTableSink(context: DynamicTableFactory.Context): DynamicTableSink {
        val helper = FactoryUtil.createTableFactoryHelper(this, context)
        helper.validate()
        val opts = helper.options
        val embeddingModel = runCatching { opts.get(EMBEDDING_MODEL) }.getOrNull()
            ?.takeIf { it.isNotEmpty() }
        val pfJson = opts.get(PARTITION_FIELDS)
        val partitionFields: List<PartitionFieldDef> = if (pfJson == "[]" || pfJson.isBlank()) emptyList() else {
            val node = ObjectMapper().readTree(pfJson)
            (0 until node.size()).map { i ->
                val n = node.get(i)
                PartitionFieldDef(n.get("column").asText(), n.get("transform").asText(), n.get("column_type").asText())
            }
        }
        val ftsColsRaw = opts.get(FTS_COLUMNS)
        val ftsColumns = if (ftsColsRaw.isBlank()) emptyList()
                         else ftsColsRaw.split(",").map { it.trim() }.filter { it.isNotEmpty() }
        val vcJson = opts.get(VEC_COLUMNS)
        val vectorColumns: List<AilakeNativeLoader.VectorColSpec> = if (vcJson == "[]" || vcJson.isBlank()) emptyList() else {
            val node = ObjectMapper().readTree(vcJson)
            (0 until node.size()).map { i ->
                val n = node.get(i)
                AilakeNativeLoader.VectorColSpec(
                    column = n.get("column").asText(),
                    dim = n.get("dim").asInt(),
                    metric = n.path("metric").asText("cosine"),
                    precision = n.path("precision").asText("f16"),
                    modality = if (n.has("modality") && !n.get("modality").isNull) n.get("modality").asText() else null,
                )
            }
        }
        return AilakeVectorTableSink(
            warehouse       = opts.get(WAREHOUSE),
            namespace       = opts.get(NAMESPACE),
            tableName       = opts.get(TABLE_NAME),
            vecCol          = opts.get(VEC_COL),
            dim             = opts.get(VEC_DIM),
            metric          = opts.get(VEC_METRIC),
            precision       = opts.get(VEC_PREC),
            schema          = context.catalogTable.resolvedSchema,
            embeddingModel  = embeddingModel,
            partitionFields = partitionFields,
            formatVersion   = opts.get(FORMAT_VERSION),
            ftsColumns      = ftsColumns,
            ftsTokenizer    = opts.get(FTS_TOKENIZER),
            hnswM              = runCatching { opts.get(HNSW_M) }.getOrNull(),
            hnswEfConstruction = runCatching { opts.get(HNSW_EF_CONSTRUCTION) }.getOrNull(),
            preNormalize       = opts.get(PRE_NORMALIZE),
            deferred           = opts.get(DEFERRED),
            vectorColumns      = vectorColumns,
        )
    }
}
