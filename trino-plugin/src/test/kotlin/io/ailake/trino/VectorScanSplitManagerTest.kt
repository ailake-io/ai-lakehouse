package io.ailake.trino

import io.trino.spi.connector.ConnectorSession
import io.trino.spi.connector.DynamicFilter
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.mockito.kotlin.doReturn
import org.mockito.kotlin.mock

class VectorScanSplitManagerTest {

    private val splitManager = VectorScanSplitManager()
    private val tableHandle = VectorScanTableHandle("s3://bucket/table/", "embedding", 1536)
    private val dynamicFilter = mock<DynamicFilter>()

    private fun session(queryVector: String = "0.1,-0.2,0.3", topK: Int = 5): ConnectorSession =
        mock {
            on { getProperty("query_vector", String::class.java) } doReturn queryVector
            on { getProperty("top_k", Int::class.java) } doReturn topK
        }

    @Test
    fun getSplitsReturnsOneSplit() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(), tableHandle,
            dynamicFilter,
        )
        val splits = source.getNextBatch(1000).get().splits
        assertEquals(1, splits.size)
    }

    @Test
    fun splitCarriesTableUri() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(), tableHandle,
            dynamicFilter,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals("s3://bucket/table/", split.tableUri)
    }

    @Test
    fun splitCarriesQueryVectorFromSession() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(queryVector = "1.0,2.0,3.0"), tableHandle,
            dynamicFilter,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals("1.0,2.0,3.0", split.queryVector)
    }

    @Test
    fun splitCarriesTopKFromSession() {
        val source = splitManager.getSplits(
            VectorScanTransactionHandle, session(topK = 42), tableHandle,
            dynamicFilter,
        )
        val split = source.getNextBatch(1).get().splits.first() as VectorScanSplit
        assertEquals(42, split.topK)
    }

    @Test
    fun splitIsRemotelyAccessible() {
        val split = VectorScanSplit("s3://t/", "0.1,0.2", 10)
        assertTrue(split.isRemotelyAccessible())
    }
}
