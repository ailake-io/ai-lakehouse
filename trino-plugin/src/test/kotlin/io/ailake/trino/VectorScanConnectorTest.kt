// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.ConnectorSession
import io.trino.spi.transaction.IsolationLevel
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.mockito.kotlin.mock

class VectorScanConnectorTest {

    private val connector = VectorScanConnector(
        tableUri = "s3://bucket/table/",
        vectorColumn = "embedding",
        dim = 1536,
        metric = "cosine",
        precision = "f16",
        namespace = "default",
        tableName = "table",
    )
    private val session = mock<ConnectorSession>()

    @Test
    fun sessionPropertiesRegistersQueryVectorAndTopK() {
        val props = connector.getSessionProperties()
        assertEquals(13, props.size)
        val names = props.map { it.name }.toSet()
        assertTrue("query_vector" in names)
        assertTrue("top_k" in names)
        assertTrue("query_text" in names)
        assertTrue("hybrid_weight" in names)
        assertTrue("multimodal_queries" in names)
        assertTrue("ef_search" in names)
        assertTrue("pruning_threshold" in names)
        assertTrue("decay_lambda" in names)
        assertTrue("migrate_config" in names)
        assertTrue("delete_rows_config" in names)
        assertTrue("add_vector_column_config" in names)
        assertTrue("backfill_vector_column_config" in names)
        assertTrue("estimate_config" in names)
    }

    @Test
    fun queryVectorDefaultIsEmptyString() {
        val prop = connector.getSessionProperties().first { it.name == "query_vector" }
        assertEquals("", prop.defaultValue)
    }

    @Test
    fun topKDefaultIsTen() {
        val prop = connector.getSessionProperties().first { it.name == "top_k" }
        assertEquals(10, prop.defaultValue)
    }

    @Test
    fun efSearchDefaultIsZeroSentinel() {
        val prop = connector.getSessionProperties().first { it.name == "ef_search" }
        assertEquals(0, prop.defaultValue)
    }

    @Test
    fun pruningThresholdDefaultIsNegativeOneSentinel() {
        val prop = connector.getSessionProperties().first { it.name == "pruning_threshold" }
        assertEquals(-1.0, prop.defaultValue)
    }

    @Test
    fun beginTransactionAlwaysReturnsSameHandle() {
        val h1 = connector.beginTransaction(IsolationLevel.READ_UNCOMMITTED, true, true)
        val h2 = connector.beginTransaction(IsolationLevel.SERIALIZABLE, false, false)
        assertSame(h1, h2)
        assertSame(VectorScanTransactionHandle, h1)
    }

    @Test
    fun getMetadataReturnsVectorScanMetadata() {
        val meta = connector.getMetadata(session, VectorScanTransactionHandle)
        assertTrue(meta is VectorScanMetadata)
    }

    @Test
    fun getSplitManagerReturnsVectorScanSplitManager() {
        assertTrue(connector.getSplitManager() is VectorScanSplitManager)
    }

    @Test
    fun getRecordSetProviderReturnsProvider() {
        assertTrue(connector.getRecordSetProvider() is VectorScanRecordSetProvider)
    }
}
