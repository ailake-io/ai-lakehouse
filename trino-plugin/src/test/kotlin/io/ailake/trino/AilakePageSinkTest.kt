// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.trino.spi.Page
import io.trino.spi.block.ArrayBlockBuilder
import io.trino.spi.block.Block
import io.trino.spi.type.ArrayType
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.DoubleType.DOUBLE
import io.trino.spi.type.VarcharType.VARCHAR
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.lang.reflect.Field

class AilakePageSinkTest {

    /** Builds a real Trino Page: id BIGINT, embedding ARRAY<DOUBLE>, ...textCols VARCHAR. */
    private fun buildPage(ids: List<Long>, vectors: List<List<Double>>, textCols: List<Pair<String, List<String>>>): Page {
        val n = ids.size
        val idBuilder = BIGINT.createBlockBuilder(null, n)
        ids.forEach { BIGINT.writeLong(idBuilder, it) }

        val arrayType = ArrayType(DOUBLE)
        val vecBuilder = arrayType.createBlockBuilder(null, n) as ArrayBlockBuilder
        vectors.forEach { vec ->
            vecBuilder.buildEntry<RuntimeException> { elementBuilder -> vec.forEach { DOUBLE.writeDouble(elementBuilder, it) } }
        }

        val textBlocks = textCols.map { (_, values) ->
            val b = VARCHAR.createBlockBuilder(null, n)
            values.forEach { VARCHAR.writeSlice(b, io.airlift.slice.Slices.utf8Slice(it)) }
            b.build()
        }

        return Page(*(listOf(idBuilder.build(), vecBuilder.build()) + textBlocks).toTypedArray<Block>())
    }

    /** Reads a private field via reflection — these buffers have no public getter. */
    @Suppress("UNCHECKED_CAST")
    private fun <T> privateField(target: Any, name: String): T {
        val f: Field = target.javaClass.getDeclaredField(name)
        f.isAccessible = true
        return f.get(target) as T
    }

    private fun handle() = AilakeIngestTableHandle(
        tableUri     = "file:///tmp/test-table",
        namespace    = "default",
        tableName    = "docs",
        vectorColumn = "embedding",
        dim          = 4,
        metric       = "cosine",
        precision    = "f16",
    )

    @Test
    fun finishWithNoRowsReturnsEmptyCollection() {
        val sink = AilakePageSink(handle())
        val future = sink.finish()
        val fragments = future.get()
        // Native lib absent → writeBatch returns null → empty fragment list
        assertTrue(fragments.isEmpty())
    }

    @Test
    fun abortClearsBuffers() {
        val sink = AilakePageSink(handle())
        // No rows added; abort must not throw
        assertDoesNotThrow { sink.abort() }
    }

    @Test
    fun sinkWithEmbeddingModelFinishesGracefully() {
        val h = AilakeIngestTableHandle(
            tableUri       = "file:///tmp/test-table",
            namespace      = "default",
            tableName      = "docs",
            vectorColumn   = "embedding",
            dim            = 4,
            metric         = "cosine",
            precision      = "f16",
            embeddingModel = "text-embedding-3-small@v1",
        )
        val sink = AilakePageSink(h)
        // Native lib absent — should return empty, not throw
        val fragments = sink.finish().get()
        assertTrue(fragments.isEmpty())
    }

    // ── textColumns (extra metadata columns) ──────────────────────────────────

    @Test
    fun appendPageAccumulatesExtraTextColumnsPerRow() {
        val h = AilakeIngestTableHandle(
            tableUri     = "file:///tmp/test-table",
            namespace    = "default",
            tableName    = "docs",
            vectorColumn = "embedding",
            dim          = 2,
            metric       = "cosine",
            precision    = "f16",
            textColumns  = listOf("text", "source"),
        )
        val sink = AilakePageSink(h)
        val page = buildPage(
            ids = listOf(1L, 2L),
            vectors = listOf(listOf(0.1, 0.2), listOf(0.3, 0.4)),
            textCols = listOf(
                "text" to listOf("hello world", "second row"),
                "source" to listOf("doc-a", "doc-b"),
            ),
        )
        sink.appendPage(page).get()

        val textValues: Map<String, List<String>> = privateField(sink, "textValues")
        assertEquals(listOf("hello world", "second row"), textValues.getValue("text"))
        assertEquals(listOf("doc-a", "doc-b"), textValues.getValue("source"))
    }

    @Test
    fun appendPageWithNoTextColumnsConfiguredReadsOnlyIdAndEmbedding() {
        val sink = AilakePageSink(handle())  // no textColumns
        val page = buildPage(
            ids = listOf(1L),
            vectors = listOf(listOf(0.1, 0.2, 0.3, 0.4)),
            textCols = emptyList(),
        )
        assertDoesNotThrow { sink.appendPage(page).get() }
        val textValues: Map<String, List<String>> = privateField(sink, "textValues")
        assertTrue(textValues.isEmpty())
    }

    @Test
    fun finishPassesTextColumnsAsColumnsMapToNativeWriteBatch() {
        val h = AilakeIngestTableHandle(
            tableUri     = "file:///tmp/test-table",
            namespace    = "default",
            tableName    = "docs",
            vectorColumn = "embedding",
            dim          = 2,
            metric       = "cosine",
            precision    = "f16",
            textColumns  = listOf("text"),
        )
        val sink = AilakePageSink(h)
        val page = buildPage(
            ids = listOf(1L),
            vectors = listOf(listOf(0.1, 0.2)),
            textCols = listOf("text" to listOf("hello")),
        )
        sink.appendPage(page).get()
        // Inspect the accumulated per-column buffer directly via reflection
        // instead of calling finish() — whether libailake_jni.so is on the
        // classpath varies by environment (absent locally, present in CI via
        // AILAKE_LIB_PATH), so finish() against this fake tableUri would
        // either no-op (native lib absent) or attempt a real write (native
        // lib present). Either way this test only needs to prove appendPage
        // accumulates the right values into what finish() passes as
        // columns=; the actual native wiring is exercised by
        // AilakeWriteBatchIntegrationTest against a real native lib.
        val textValues: Map<String, List<String>> = privateField(sink, "textValues")
        assertEquals(listOf("hello"), textValues.getValue("text"))
    }
}
