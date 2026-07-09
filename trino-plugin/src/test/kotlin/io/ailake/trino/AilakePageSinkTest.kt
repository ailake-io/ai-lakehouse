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

    // ── null embedding guard ───────────────────────────────────────────────────
    //
    // Regression: appendPage null-checked id and text columns (auto-generated
    // id fallback, empty-string text fallback) but not the vector column —
    // a NULL embedding either threw an obscure exception from extractVector's
    // unchecked getObject/positionCount, or silently produced a wrong-length
    // vector that would corrupt the whole native writeBatch call for every row
    // in the same INSERT. Now fails fast with a clear message. The ingest
    // table's embedding column is also declared NOT NULL in
    // VectorScanMetadata.ingestColumns(), so Trino itself should reject this
    // before a page ever reaches the sink — this is defense in depth for
    // whatever internal engine path might still construct one.

    @Test
    fun appendPageThrowsClearErrorOnNullEmbedding() {
        val idBuilder = BIGINT.createBlockBuilder(null, 1)
        BIGINT.writeLong(idBuilder, 1L)
        val vecBuilder = ArrayType(DOUBLE).createBlockBuilder(null, 1) as ArrayBlockBuilder
        vecBuilder.appendNull()
        val page = Page(idBuilder.build(), vecBuilder.build())

        val sink = AilakePageSink(handle())
        val ex = assertThrows(IllegalStateException::class.java) { sink.appendPage(page).get() }
        assertTrue(ex.message!!.contains("embedding"), "expected message to mention the vector column, got: ${ex.message}")
    }

    // ── write-tuning knobs (hnsw_m / hnsw_ef_construction / pre_normalize / deferred / fts_columns) ──
    //
    // Regression: AilakeNative.writeBatch already supported these six params,
    // but AilakePageSink.finish() never passed any of them, and there was no
    // catalog property to configure them — see VectorScanConnectorFactory's
    // ailake.hnsw-m / ailake.hnsw-ef-construction / ailake.pre-normalize /
    // ailake.deferred / ailake.fts-columns / ailake.fts-tokenizer.

    @Test
    fun handleCarriesWriteTuningKnobsThroughToDefaults() {
        // No knobs configured — defaults must match writeBatch's own historical defaults.
        val h = handle()
        assertNull(h.hnswM)
        assertNull(h.hnswEfConstruction)
        assertFalse(h.preNormalize)
        assertFalse(h.deferred)
        assertTrue(h.ftsColumns.isEmpty())
        assertEquals("default", h.ftsTokenizer)
    }

    // ── multi-column (Phase 8 multimodal) ingest — ailake.vector-columns ──────
    //
    // Regression: AilakeNative.writeBatchMulti was exposed from Spark
    // (`ailakeWriteMulti`) but had no wrapper or SQL surface here at all — a
    // Trino-only user could never write a table with 2+ independent vector
    // columns, only single-vector-column ingest existed.

    private fun buildMultiPage(ids: List<Long>, vecCols: List<List<List<Double>>>): Page {
        val n = ids.size
        val idBuilder = BIGINT.createBlockBuilder(null, n)
        ids.forEach { BIGINT.writeLong(idBuilder, it) }
        val arrayType = ArrayType(DOUBLE)
        val vecBlocks = vecCols.map { rows ->
            val b = arrayType.createBlockBuilder(null, n) as ArrayBlockBuilder
            rows.forEach { vec -> b.buildEntry<RuntimeException> { eb -> vec.forEach { DOUBLE.writeDouble(eb, it) } } }
            b.build()
        }
        return Page(*(listOf(idBuilder.build()) + vecBlocks).toTypedArray<Block>())
    }

    private fun multiHandle() = AilakeIngestTableHandle(
        tableUri = "file:///tmp/test-table", namespace = "default", tableName = "docs",
        vectorColumn = "embedding", dim = 2, metric = "cosine", precision = "f16",
        vectorColumns = listOf(
            AilakeNative.VectorColSpec("embedding", 2),
            AilakeNative.VectorColSpec("image_embedding", 2),
        ),
    )

    @Test
    fun appendPageAccumulatesMultipleVectorColumns() {
        val sink = AilakePageSink(multiHandle())
        val page = buildMultiPage(
            ids = listOf(1L, 2L),
            vecCols = listOf(
                listOf(listOf(0.1, 0.2), listOf(0.3, 0.4)),
                listOf(listOf(0.5, 0.6), listOf(0.7, 0.8)),
            ),
        )
        sink.appendPage(page).get()

        val multiEmbeddings: List<MutableList<List<Float>>> = privateField(sink, "multiEmbeddings")
        assertEquals(2, multiEmbeddings.size)
        assertEquals(listOf(listOf(0.1f, 0.2f), listOf(0.3f, 0.4f)), multiEmbeddings[0])
        assertEquals(listOf(listOf(0.5f, 0.6f), listOf(0.7f, 0.8f)), multiEmbeddings[1])
    }

    @Test
    fun appendPageWithMultiColumnHandleDoesNotTouchSingleColumnBuffer() {
        val sink = AilakePageSink(multiHandle())
        val page = buildMultiPage(ids = listOf(1L), vecCols = listOf(listOf(listOf(0.1, 0.2)), listOf(listOf(0.3, 0.4))))
        sink.appendPage(page).get()

        val embeddings: MutableList<List<Float>> = privateField(sink, "embeddings")
        assertTrue(embeddings.isEmpty())
    }

    @Test
    fun finishWithVectorColumnsConfiguredReturnsEmptyWhenNativeLibAbsent() {
        val sink = AilakePageSink(multiHandle())
        val page = buildMultiPage(ids = listOf(1L), vecCols = listOf(listOf(listOf(0.1, 0.2)), listOf(listOf(0.3, 0.4))))
        sink.appendPage(page).get()
        val fragments = sink.finish().get()
        assertTrue(fragments.isEmpty())
    }

    @Test
    fun handleDefaultsVectorColumnsToEmptyList() {
        assertTrue(handle().vectorColumns.isEmpty())
    }

    @Test
    fun handleCarriesConfiguredVectorColumns() {
        val h = multiHandle()
        assertEquals(2, h.vectorColumns.size)
        assertEquals("embedding", h.vectorColumns[0].column)
        assertEquals("image_embedding", h.vectorColumns[1].column)
    }

    @Test
    fun handleCarriesConfiguredWriteTuningKnobs() {
        val h = AilakeIngestTableHandle(
            tableUri = "file:///tmp/test-table", namespace = "default", tableName = "docs",
            vectorColumn = "embedding", dim = 4, metric = "cosine", precision = "f16",
            hnswM = 32, hnswEfConstruction = 200, preNormalize = true, deferred = true,
            ftsColumns = listOf("chunk_text"), ftsTokenizer = "en_stem",
        )
        assertEquals(32, h.hnswM)
        assertEquals(200, h.hnswEfConstruction)
        assertTrue(h.preNormalize)
        assertTrue(h.deferred)
        assertEquals(listOf("chunk_text"), h.ftsColumns)
        assertEquals("en_stem", h.ftsTokenizer)
    }
}
