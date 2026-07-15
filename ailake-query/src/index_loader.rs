// SPDX-License-Identifier: MIT OR Apache-2.0
//! Range-GET fast path: load just the HNSW/IVF-PQ index of a file's
//! *primary* vector column, without ever downloading the tabular/vector data
//! section. Used by `scanner.rs::search_one_file` when the query needs
//! nothing else from the file (no rerank, hybrid, `score_fn`, equality
//! deletes, or `column_filter` — see Fase 16 in `CLAUDE.md`).
//!
//! Offset discovery reads the `ailake.footer_offset` Parquet KV entry (via
//! `ailake_parquet::ParquetVectorReader::kv_metadata`) from a speculative
//! tail `get_range` of just the footer thrift, with an exact one-shot
//! follow-up on the rare miss — never the whole file. This is the KV path,
//! not `AilakeFileReader`'s `AilakeTrailer` bootstrap fallback: multi-column
//! files (`AilakeFileWriter::write_multi`) write one self-pointing
//! `AilakeTrailer` per column section, so the trailer physically nearest
//! EOF belongs to whichever column was written *last* — not necessarily the
//! primary one. The KV entry has no such ambiguity: `write_multi` always
//! tags column `0` (primary) with the plain `ailake.footer_offset` key
//! regardless of how many columns follow it.
//!
//! The AILK header and HNSW blob are sliced straight out of the same
//! speculative tail buffer whenever they happen to fall inside it (common
//! for small-to-medium files, or any file where the AILK section is smaller
//! than the tail window) instead of issuing separate `get_range` calls —
//! without this, those calls would frequently *re-fetch* bytes the tail read
//! already has, since the tail window's whole purpose is landing on the
//! footer thrift, which sits immediately after the AILK section.
//!
//! Only the primary column is supported — a secondary/multimodal column's
//! offset lives behind `ailake.<col>.footer_offset` instead, and this module
//! doesn't thread a column name through the request, since callers already
//! gate on `vector_column == primary_col` before invoking it (see
//! `scanner.rs::search_one_file`).
//!
//! Every failure here (parse error, missing KV, out-of-range GET) is
//! recoverable by the caller falling back to the existing full-file GET —
//! this module never turns a query that would have succeeded into one that
//! fails; at worst it wastes one or two small extra `get_range` calls.

use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use ailake_file::{parquet_footer_start, AilakeHeader, Precision, FLAG_INDEX_IVF_PQ, HEADER_SIZE};
use ailake_index::{AnyIndex, IvfPqSerializer, MmapLoader};
use ailake_parquet::ParquetVectorReader;
use ailake_store::Store;
use bytes::Bytes;

/// Speculative tail-`get_range` size for footer-thrift discovery. Sized
/// generously for the KV metadata + schema + per-row-group column statistics
/// (Fase 5's `column_stats`) that make up the footer thrift; `resolve_ailk_offset`
/// falls back to one exact follow-up `get_range` on the rare miss (very wide
/// schema, many row groups).
const SPECULATIVE_TAIL_BYTES: u64 = 65_536;

/// Bytes already fetched during offset discovery, kept around so the header
/// and (if it fits) the HNSW blob reads can slice out of them directly
/// instead of re-fetching overlapping bytes from the store.
struct TailBuf {
    /// Absolute file offset of `bytes[0]`.
    base: u64,
    bytes: Bytes,
}

impl TailBuf {
    /// Returns the requested absolute `[start, end)` range, sliced from this
    /// buffer if it's fully contained, without touching the store.
    fn slice(&self, start: u64, end: u64) -> Option<Bytes> {
        if start < self.base || end > self.base + self.bytes.len() as u64 {
            return None;
        }
        let rel_start = (start - self.base) as usize;
        let rel_end = (end - self.base) as usize;
        Some(self.bytes.slice(rel_start..rel_end))
    }
}

/// Loads the primary column's HNSW/IVF-PQ index via small `get_range` calls
/// (footer-thrift discovery, 64-byte AILK header, index blob) instead of one
/// whole-file `get`. Returns `Err` for any file this fast path can't handle —
/// callers must treat that as "use the full-file path", not as a hard failure.
pub async fn load_primary_index(store: &Arc<dyn Store>, path: &str) -> AilakeResult<AnyIndex> {
    let file_size = store.file_size(path).await?;
    let (ailk_offset, tail) = resolve_ailk_offset(store, path, file_size).await?;

    let header_end = ailk_offset
        .checked_add(HEADER_SIZE as u64)
        .ok_or(AilakeError::NotAnAilakeFile)?;
    let header_bytes = match tail.slice(ailk_offset, header_end) {
        Some(b) => b,
        None => store.get_range(path, ailk_offset..header_end).await?,
    };
    let header_arr: [u8; HEADER_SIZE] = header_bytes
        .as_ref()
        .try_into()
        .map_err(|_| AilakeError::NotAnAilakeFile)?;
    let header = AilakeHeader::from_bytes(&header_arr)?;

    let index_start = ailk_offset
        .checked_add(header.hnsw_offset)
        .ok_or(AilakeError::NotAnAilakeFile)?;
    let index_end = index_start
        .checked_add(header.hnsw_len)
        .ok_or(AilakeError::NotAnAilakeFile)?;
    let index_bytes = match tail.slice(index_start, index_end) {
        Some(b) => b,
        None => store.get_range(path, index_start..index_end).await?,
    };

    if header.flags & FLAG_INDEX_IVF_PQ != 0 {
        let idx = IvfPqSerializer::from_bytes(&index_bytes)?;
        Ok(AnyIndex::IvfPq(idx))
    } else {
        let mut idx = MmapLoader::from_bytes(&index_bytes)?;
        if header.precision == Precision::F16 {
            idx.quantize_to_f16();
        }
        Ok(AnyIndex::Hnsw(idx))
    }
}

/// Returns the absolute byte offset of the primary AILK section (read from
/// the `ailake.footer_offset` Parquet KV entry) plus the tail buffer it was
/// found in, so the caller can reuse those same bytes for the header/HNSW
/// reads when they happen to fall inside it.
///
/// `parquet`'s own metadata parser (wrapped by `ParquetVectorReader::kv_metadata`)
/// only ever reads backward from the end of the buffer it's given; as long as
/// that buffer's own end aligns with the file's true EOF and is large enough
/// to hold the full footer thrift, feeding it a tail slice (not the whole
/// file) parses correctly — no need to fetch anything before the footer.
async fn resolve_ailk_offset(
    store: &Arc<dyn Store>,
    path: &str,
    file_size: u64,
) -> AilakeResult<(u64, TailBuf)> {
    let guess_len = SPECULATIVE_TAIL_BYTES.min(file_size);
    let mut base = file_size - guess_len;
    let mut tail = store.get_range(path, base..file_size).await?;

    if parquet_footer_start(&tail).is_err() {
        // Speculative window too small for the real footer thrift (or a
        // corrupt/non-AI-Lake file — `kv_metadata` below will error out
        // either way). We know the exact footer_thrift_len from the tail's
        // own trailing 8 bytes, so the follow-up fetch is sized precisely —
        // no further guessing, no retry loop.
        if tail.len() < 8 {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let footer_thrift_len =
            u32::from_le_bytes(tail[tail.len() - 8..tail.len() - 4].try_into().unwrap()) as u64;
        let exact_len = (8 + footer_thrift_len).min(file_size);
        base = file_size - exact_len;
        tail = store.get_range(path, base..file_size).await?;
    }

    let reader = ParquetVectorReader::new(tail.clone(), "");
    let ailk_offset = match reader.kv_metadata("ailake.footer_offset")? {
        Some(v) => v.parse::<u64>().map_err(|_| AilakeError::NotAnAilakeFile)?,
        None => return Err(AilakeError::NotAnAilakeFile),
    };
    Ok((ailk_offset, TailBuf { base, bytes: tail }))
}
