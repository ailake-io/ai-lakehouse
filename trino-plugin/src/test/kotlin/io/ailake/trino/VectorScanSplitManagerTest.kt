// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.Constraint
import io.trino.spi.connector.DynamicFilter
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.mockito.kotlin.doReturn
import org.mockito.kotlin.mock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

class VectorScanSplitManagerTest {

    private val splitManager = VectorScanSplitManager()
    private val tableHandle = VectorScanTableHandle("s3://bucket/table/", "embedding", 1536, "default", "docs")
    private val dynamicFilter = mock<DynamicFilter>()
    private val constraint = Constraint.alwaysTrue()

    private fun session(
        queryVector: String = "0.1,-0.2,0.3",
        topK: Int = 5,
        queryText: String? = null,
        hybridWeight: Double? = null,
    ): ConnectorSession =
        mock {
            on { getProperty("query_vector", String::class.java) } doReturn queryVector
            on { getProperty("top_k", Int::class.java) } doReturn topK
            on { getProperty("query_text", String::class.java) } doReturn queryText
            on { getProperty("hybrid_weight", Double::class.java) } doReturn hybridWeight
        }

    @Test
    fun getSplitsReturnsOneSplit() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(), tableHandle,
            dynamicFilter, constraint,
        )
        val splits = source.getNextBatch(1000).get().splits
        assertEquals(1, splits.size)
    }

    @Test
    fun splitCarriesTableUri() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(), tableHandle,
            dynamicFilter, constraint,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals("s3://bucket/table/", split.tableUri)
    }

    @Test
    fun splitCarriesQueryBytesDecodableToExpectedFloats() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(queryVector = "1.0,2.0,3.0"), tableHandle,
            dynamicFilter, constraint,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        val bytes = Base64.getDecoder().decode(split.queryBytes)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4) { buf.getFloat() }
        assertArrayEquals(floatArrayOf(1.0f, 2.0f, 3.0f), floats, 1e-6f)
    }

    @Test
    fun splitCarriesTopKFromSession() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(topK = 42), tableHandle,
            dynamicFilter, constraint,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals(42, split.topK)
    }

    // Regression: VectorScanTableHandle used to carry only tableUri/vectorColumn/dim,
    // silently dropping the catalog's configured namespace/tableName — search always
    // hit AilakeNative.search's hardcoded "default" namespace and URI-derived table
    // name, unfindable if the catalog was configured with a custom namespace/table-name
    // or if the table-uri's last path segment didn't match. Now threaded through the
    // split into AilakeNative.search's namespace/tableName/vectorColumn params.
    @Test
    fun splitCarriesNamespaceTableNameAndVectorColumnFromHandle() {
        val handle = VectorScanTableHandle("s3://bucket/table/", "doc_vec", 1536, "tenant_a", "docs")
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(), handle,
            dynamicFilter, constraint,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals("tenant_a", split.namespace)
        assertEquals("docs", split.tableName)
        assertEquals("doc_vec", split.vectorColumn)
    }

    @Test
    fun splitCarriesQueryTextAndHybridWeightFromSession() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(queryText = "rust programming", hybridWeight = 0.7),
            tableHandle, dynamicFilter, constraint,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals("rust programming", split.queryText)
        assertEquals(0.7f, split.hybridWeight, 1e-6f)
    }

    @Test
    fun splitDefaultsQueryTextEmptyAndHybridWeightHalfWhenSessionPropsUnset() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(), tableHandle, dynamicFilter, constraint,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals("", split.queryText)
        assertEquals(0.5f, split.hybridWeight, 1e-6f)
    }

    @Test
    fun splitIsRemotelyAccessible() {
        val split = VectorScanSplit("s3://t/", VectorScanSplitManager.csvFloatsToBase64("0.1,0.2"), 10, "default", "docs", "embedding")
        assertTrue(split.isRemotelyAccessible())
    }

    @Test
    fun csvFloatsToBase64RoundTrip() {
        val base64 = VectorScanSplitManager.csvFloatsToBase64("0.5,1.5,-0.5")
        val bytes = Base64.getDecoder().decode(base64)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4) { buf.getFloat() }
        assertArrayEquals(floatArrayOf(0.5f, 1.5f, -0.5f), floats, 1e-6f)
    }

    @Test
    fun csvFloatsToBase64ReturnsEmptyForBlankInput() {
        assertEquals("", VectorScanSplitManager.csvFloatsToBase64(""))
        assertEquals("", VectorScanSplitManager.csvFloatsToBase64("   "))
    }
}
