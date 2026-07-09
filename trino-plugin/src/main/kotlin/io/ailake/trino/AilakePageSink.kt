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
import io.trino.spi.type.VarcharType.VARCHAR
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture

/**
 * Buffers rows from INSERT INTO and flushes to AI-Lake via JNI on [finish].
 *
 * Expected page layout (matches VectorScanMetadata.ingestColumns()):
 *   channel 0        — id BIGINT
 *   channel 1        — embedding ARRAY<DOUBLE>  (or channels 1..N, one per
 *                       `handle.vectorColumns` entry, in multi-column mode)
 *   channel N+1..M   — handle.textColumns, VARCHAR, same order
 */
class AilakePageSink(private val handle: AilakeIngestTableHandle) : ConnectorPageSink {

    private val log = LoggerFactory.getLogger(AilakePageSink::class.java)
    private val ids = mutableListOf<Long>()
    // Single-column mode (handle.vectorColumns is empty)
    private val embeddings = mutableListOf<List<Float>>()
    // Multi-column mode — one embeddings list per handle.vectorColumns entry, same order
    private val multiEmbeddings: List<MutableList<List<Float>>> =
        handle.vectorColumns.map { mutableListOf() }
    private val textValues: Map<String, MutableList<String>> =
        handle.textColumns.associateWith { mutableListOf<String>() }
    private var autoId = 0L

    override fun appendPage(page: Page): CompletableFuture<*> {
        val idBlock = page.getBlock(0)
        val vecColCount = if (handle.vectorColumns.isNotEmpty()) handle.vectorColumns.size else 1
        val vecBlocks = (0 until vecColCount).map { i -> page.getBlock(1 + i) }
        val textStart = 1 + vecColCount
        val textBlocks = handle.textColumns.mapIndexed { i, name -> name to page.getBlock(textStart + i) }

        for (pos in 0 until page.positionCount) {
            ids += if (!idBlock.isNull(pos)) BIGINT.getLong(idBlock, pos) else autoId
            vecBlocks.forEachIndexed { i, block ->
                val colName = if (handle.vectorColumns.isNotEmpty()) handle.vectorColumns[i].column else handle.vectorColumn
                check(!block.isNull(pos)) {
                    "Vector column '$colName' cannot be NULL (row with id-position=$pos in this page) " +
                    "— every row must carry a real embedding for AI-Lake to index it."
                }
                if (handle.vectorColumns.isNotEmpty()) multiEmbeddings[i] += extractVector(block, pos)
                else embeddings += extractVector(block, pos)
            }
            textBlocks.forEach { (name, block) ->
                textValues.getValue(name) += if (block.isNull(pos)) "" else VARCHAR.getSlice(block, pos).toStringUtf8()
            }
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

        val snapshotId = if (handle.vectorColumns.isNotEmpty()) {
            AilakeNative.writeBatchMulti(
                tableUri       = handle.tableUri,
                namespace      = handle.namespace,
                tableName      = handle.tableName,
                ids            = ids,
                vectorColumns  = handle.vectorColumns.zip(multiEmbeddings),
                embeddingModel = handle.embeddingModel,
                formatVersion  = handle.formatVersion,
                ftsColumns     = handle.ftsColumns,
                ftsTokenizer   = handle.ftsTokenizer,
                deferred       = handle.deferred,
                columns        = textValues,
            )
        } else {
            AilakeNative.writeBatch(
                tableUri        = handle.tableUri,
                namespace       = handle.namespace,
                tableName       = handle.tableName,
                vectorColumn    = handle.vectorColumn,
                dim             = handle.dim,
                metric          = handle.metric,
                precision       = handle.precision,
                ids             = ids,
                embeddings      = embeddings,
                embeddingModel  = handle.embeddingModel,
                partitionFields = handle.partitionFields,
                formatVersion   = handle.formatVersion,
                hnswM              = handle.hnswM,
                hnswEfConstruction = handle.hnswEfConstruction,
                preNormalize       = handle.preNormalize,
                deferred           = handle.deferred,
                ftsColumns         = handle.ftsColumns,
                ftsTokenizer       = handle.ftsTokenizer,
                columns         = textValues,
            )
        }

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
        multiEmbeddings.forEach { it.clear() }
        textValues.values.forEach { it.clear() }
    }
}
