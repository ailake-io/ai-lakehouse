// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class AilakeNativeTest {

    private fun base64Of(vararg floats: Float): String =
        VectorScanSplitManager.csvFloatsToBase64(floats.joinToString(","))

    @Test
    fun searchReturnsEmptyWhenNativeLibAbsent() {
        // Native lib is not available in test environment — graceful degradation.
        val results = AilakeNative.search("s3://bucket/table/", base64Of(0.1f, 0.2f, 0.3f), topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchReturnsEmptyForBlankQueryBytes() {
        val results = AilakeNative.search("s3://bucket/table/", "  ", topK = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun searchReturnsEmptyForEmptyQueryBytes() {
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
