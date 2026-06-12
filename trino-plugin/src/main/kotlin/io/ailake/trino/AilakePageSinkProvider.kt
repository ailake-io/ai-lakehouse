// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.ConnectorInsertTableHandle
import io.trino.spi.connector.ConnectorOutputTableHandle
import io.trino.spi.connector.ConnectorPageSink
import io.trino.spi.connector.ConnectorPageSinkId
import io.trino.spi.connector.ConnectorPageSinkProvider
import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.ConnectorTransactionHandle

class AilakePageSinkProvider : ConnectorPageSinkProvider {

    override fun createPageSink(
        transactionHandle: ConnectorTransactionHandle,
        session: ConnectorSession,
        tableHandle: ConnectorOutputTableHandle,
        pageSinkId: ConnectorPageSinkId,
    ): ConnectorPageSink = throw UnsupportedOperationException("CREATE TABLE AS SELECT not supported by AI-Lake connector")

    override fun createPageSink(
        transactionHandle: ConnectorTransactionHandle,
        session: ConnectorSession,
        tableHandle: ConnectorInsertTableHandle,
        pageSinkId: ConnectorPageSinkId,
    ): ConnectorPageSink = AilakePageSink(tableHandle as AilakeIngestTableHandle)
}
