// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.flink

import org.apache.flink.configuration.ConfigOption
import org.apache.flink.configuration.ConfigOptions
import org.apache.flink.table.catalog.Catalog
import org.apache.flink.table.factories.CatalogFactory

/**
 * Flink catalog factory for AI-Lake.
 *
 * Register via SQL:
 * ```sql
 * CREATE CATALOG ailake WITH (
 *   'type'      = 'ailake',
 *   'warehouse' = 's3://my-lake/'
 * );
 * USE CATALOG ailake;
 * ```
 */
class AilakeCatalogFactory : CatalogFactory {

    companion object {
        const val IDENTIFIER = "ailake"
        val WAREHOUSE = ConfigOptions.key("warehouse").stringType().noDefaultValue()
            .withDescription("Warehouse root path (local or s3://)")
        val DEFAULT_NAMESPACE = ConfigOptions.key("default-namespace").stringType().defaultValue("default")
    }

    override fun factoryIdentifier(): String = IDENTIFIER

    override fun requiredOptions(): Set<ConfigOption<*>> = setOf(WAREHOUSE)

    override fun optionalOptions(): Set<ConfigOption<*>> = setOf(DEFAULT_NAMESPACE)

    override fun createCatalog(context: CatalogFactory.Context): Catalog {
        val opts = context.options
        return AilakeCatalog(
            name             = context.name,
            warehouse        = opts[WAREHOUSE.key()]
                ?: throw IllegalArgumentException("'warehouse' is required for ailake catalog"),
            defaultNamespace = opts.getOrDefault(DEFAULT_NAMESPACE.key(), "default"),
        )
    }
}
