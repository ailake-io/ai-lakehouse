// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.trino

import io.airlift.slice.Slice
import io.airlift.slice.Slices
import io.trino.spi.Page
import io.trino.spi.block.Block
import io.trino.spi.connector.ConnectorPageSink
import io.trino.spi.connector.ConnectorPageSink.NOT_BLOCKED
import io.trino.spi.type.BigintType.BIGINT
import io.trino.spi.type.DoubleType.DOUBLE
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture

/**
 * Buffers rows from INSERT INTO and flushes to AI-Lake via JNI on [finish].
 *
 * Expected page layout (matches INGEST_COLUMNS in VectorScanMetadata):
 *   channel 0 — id BIGINT
 *   channel 1 — embedding ARRAY<DOUBLE>
 */
class AilakePageSink(private val handle: AilakeIngestTableHandle) : ConnectorPageSink {

    private val log = LoggerFactory.getLogger(AilakePageSink::class.java)
    private val ids = mutableListOf<Long>()
    private val embeddings = mutableListOf<List<Float>>()
    private var autoId = 0L

    override fun appendPage(page: Page): CompletableFuture<*> {
        val idBlock  = page.getBlock(0)
        val vecBlock = page.getBlock(1)

        for (pos in 0 until page.positionCount) {
            ids += if (!idBlock.isNull(pos)) BIGINT.getLong(idBlock, pos) else autoId
            embeddings += extractVector(vecBlock, pos)
            autoId++
        }
        return NOT_BLOCKED
    }

    private fun extractVector(block: Block, pos: Int): List<Float> {
        val inner = block.getObject(pos, Block::class.java)
        return (0 until inner.positionCount).map { i -> DOUBLE.getDouble(inner, i).toFloat() }
    }

    override fun finish(): CompletableFuture<Collection<Slice>> {
        if (ids.isEmpty()) return CompletableFuture.completedFuture(emptyList())

        val snapshotId = AilakeNative.writeBatch(
            tableUri     = handle.tableUri,
            namespace    = handle.namespace,
            tableName    = handle.tableName,
            vectorColumn = handle.vectorColumn,
            dim          = handle.dim,
            metric       = handle.metric,
            precision    = handle.precision,
            ids          = ids,
            embeddings   = embeddings,
        )

        if (snapshotId == null) {
            log.warn("[ailake] writeBatch returned null — INSERT may not have persisted")
            return CompletableFuture.completedFuture(emptyList())
        }

        log.info("[ailake] INSERT {} rows → snapshot {}", ids.size, snapshotId)
        val fragment = Slices.utf8Slice(snapshotId.toString())
        return CompletableFuture.completedFuture(listOf(fragment))
    }

    override fun abort() {
        ids.clear()
        embeddings.clear()
    }
}
