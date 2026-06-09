// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.Connector
import io.trino.spi.connector.ConnectorContext
import io.trino.spi.connector.ConnectorFactory

class VectorScanConnectorFactory : ConnectorFactory {
    override fun getName(): String = "ailake"

    override fun create(
        catalogName: String,
        config: Map<String, String>,
        context: ConnectorContext,
    ): Connector {
        val tableUri = requireNotNull(config["ailake.table-uri"]) {
            "ailake.table-uri is required in catalog properties"
        }
        val vectorColumn = config.getOrDefault("ailake.vector-column", "embedding")
        val dim          = config.getOrDefault("ailake.vector-dim", "1536").toInt()
        val metric       = config.getOrDefault("ailake.metric", "cosine")
        val precision    = config.getOrDefault("ailake.precision", "f16")
        val namespace    = config.getOrDefault("ailake.namespace", "default")
        val tableName    = config.getOrDefault("ailake.table-name",
            tableUri.trimEnd('/').substringAfterLast('/'))
        return VectorScanConnector(tableUri, vectorColumn, dim, metric, precision, namespace, tableName)
    }
}
