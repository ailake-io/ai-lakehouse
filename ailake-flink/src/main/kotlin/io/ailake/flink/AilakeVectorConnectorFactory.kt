// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.flink

import io.ailake.flink.internal.AilakeNativeLoader
import org.apache.flink.configuration.ConfigOption
import org.apache.flink.configuration.ConfigOptions
import org.apache.flink.table.connector.sink.DynamicTableSink
import org.apache.flink.table.connector.source.DynamicTableSource
import org.apache.flink.table.factories.DynamicTableFactory
import org.apache.flink.table.factories.DynamicTableSinkFactory
import org.apache.flink.table.factories.DynamicTableSourceFactory
import org.apache.flink.table.factories.FactoryUtil

/**
 * Flink SQL connector factory for AI-Lake vector tables.
 *
 * DDL example:
 * ```sql
 * CREATE TABLE docs (
 *   id        BIGINT,
 *   text      STRING,
 *   embedding BYTES,
 *   _distance FLOAT     -- populated by vector search, ignored on writes
 * ) WITH (
 *   'connector'        = 'ailake',
 *   'warehouse'        = 's3://my-lake/',
 *   'namespace'        = 'default',
 *   'table-name'       = 'docs',
 *   'vector.column'    = 'embedding',
 *   'vector.dim'       = '128',
 *   'vector.metric'    = 'euclidean',
 *   'vector.precision' = 'f16',
 *   'search.top-k'     = '10',
 *   'search.ef'        = '50'
 * );
 * ```
 */
class AilakeVectorConnectorFactory : DynamicTableSourceFactory, DynamicTableSinkFactory {

    companion object {
        const val IDENTIFIER = "ailake"

        val WAREHOUSE    = ConfigOptions.key("warehouse").stringType().noDefaultValue()
        val NAMESPACE    = ConfigOptions.key("namespace").stringType().defaultValue("default")
        val TABLE_NAME   = ConfigOptions.key("table-name").stringType().noDefaultValue()
        val VEC_COL      = ConfigOptions.key("vector.column").stringType().defaultValue("embedding")
        val VEC_DIM      = ConfigOptions.key("vector.dim").intType().noDefaultValue()
        val VEC_METRIC   = ConfigOptions.key("vector.metric").stringType().defaultValue("euclidean")
        val VEC_PREC     = ConfigOptions.key("vector.precision").stringType().defaultValue("f16")
        val SEARCH_TOPK  = ConfigOptions.key("search.top-k").intType().defaultValue(10)
        val SEARCH_EF    = ConfigOptions.key("search.ef").intType().defaultValue(50)
    }

    override fun factoryIdentifier(): String = IDENTIFIER

    override fun requiredOptions(): Set<ConfigOption<*>> = setOf(WAREHOUSE, TABLE_NAME, VEC_DIM)

    override fun optionalOptions(): Set<ConfigOption<*>> =
        setOf(NAMESPACE, VEC_COL, VEC_METRIC, VEC_PREC, SEARCH_TOPK, SEARCH_EF)

    override fun createDynamicTableSource(context: DynamicTableFactory.Context): DynamicTableSource {
        val helper = FactoryUtil.createTableFactoryHelper(this, context)
        helper.validate()
        val opts = helper.options
        return AilakeVectorTableSource(
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

    override fun createDynamicTableSink(context: DynamicTableFactory.Context): DynamicTableSink {
        val helper = FactoryUtil.createTableFactoryHelper(this, context)
        helper.validate()
        val opts = helper.options
        return AilakeVectorTableSink(
            warehouse  = opts.get(WAREHOUSE),
            namespace  = opts.get(NAMESPACE),
            tableName  = opts.get(TABLE_NAME),
            vecCol     = opts.get(VEC_COL),
            dim        = opts.get(VEC_DIM),
            metric     = opts.get(VEC_METRIC),
            precision  = opts.get(VEC_PREC),
            schema     = context.catalogTable.resolvedSchema,
        )
    }
}
