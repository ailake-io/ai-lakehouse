package io.ailake.trino

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class AilakeNativeTest {

    @Test
    fun searchReturnsEmptyWhenNativeLibAbsent() {
        // Native lib is not available in test environment — graceful degradation.
        val results = AilakeNative.search("s3://bucket/table/", "0.1,0.2,0.3", topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchReturnsEmptyForBlankQueryVector() {
        val results = AilakeNative.search("s3://bucket/table/", "  ", topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchReturnsEmptyForEmptyQueryVector() {
        val results = AilakeNative.search("s3://bucket/table/", "", topK = 10)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchRowDataClassEquality() {
        val r1 = AilakeNative.SearchRow(1L, 0.5f, "file.parquet")
        val r2 = AilakeNative.SearchRow(1L, 0.5f, "file.parquet")
        assertEquals(r1, r2)
    }

    @Test
    fun searchRowToString() {
        val r = AilakeNative.SearchRow(42L, 0.99f, "part-001.parquet")
        val s = r.toString()
        assertTrue(s.contains("42"))
        assertTrue(s.contains("part-001.parquet"))
    }
}
