// SPDX-License-Identifier: MIT OR Apache-2.0
package io.ailake.trino

import io.trino.spi.Plugin
import io.trino.spi.connector.ConnectorFactory

class AilakePlugin : Plugin {
    override fun getConnectorFactories(): Iterable<ConnectorFactory> =
        listOf(VectorScanConnectorFactory())
}
