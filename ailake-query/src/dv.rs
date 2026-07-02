// SPDX-License-Identifier: MIT OR Apache-2.0
// Iceberg V3 Deletion Vector support — Phase B (read only).
//
// A Deletion Vector (DV) is a Roaring Bitmap stored inside a Puffin `.dvd` file.
// The manifest entry carries `(path, offset, length)` that address the bitmap blob
// bytes directly, so no full Puffin footer parse is needed for read support.
//
// Phase C will add write support (producing DVs on row delete operations).

use ailake_catalog::provider::DeletionVector;
use ailake_core::{AilakeError, AilakeResult};
use ailake_store::Store;
use arrow_array::RecordBatch;
use roaring::RoaringBitmap;
use std::sync::Arc;

/// Fetch and deserialize a Deletion Vector bitmap from a Puffin `.dvd` file.
///
/// Uses a range GET (`offset..offset+length`) so only the bitmap bytes are
/// transferred from S3 — no full file download required.
///
/// The returned bitmap contains the row positions (0-based within the data file)
/// that have been deleted. Callers must filter HNSW results against this bitmap:
/// ```ignore
/// results.retain(|(row_id, _)| !bitmap.contains(row_id.as_u64() as u32));
/// ```
pub async fn load_deletion_vector(
    store: &Arc<dyn Store>,
    dv: &DeletionVector,
) -> AilakeResult<RoaringBitmap> {
    let bytes = store
        .get_range(&dv.path, dv.offset..dv.offset + dv.length)
        .await?;

    RoaringBitmap::deserialize_from(bytes.as_ref()).map_err(|e| {
        AilakeError::Io(std::io::Error::other(format!(
            "ailake: failed to deserialize Deletion Vector bitmap from '{}' \
             (offset={}, length={}): {e}",
            dv.path, dv.offset, dv.length
        )))
    })
}

/// Removes DV-masked rows from a Parquet-read `(batch, parallel)` pair, where
/// `parallel[i]` corresponds positionally to `batch` row `i` (embeddings, texts, ...).
///
/// Used by rewrite jobs (compaction, migration, backfill) that read a file's raw rows
/// and write them into a brand-new physical file: since the new file has fresh row
/// positions, a deleted row can't just be re-masked with the old bitmap after the fact —
/// it must be dropped from the input before the merge, or it silently reappears (no DV
/// on the new `DataFileEntry`, or a DV bitmap that no longer lines up with the new
/// row order).
pub fn filter_deleted_rows<T>(
    batch: RecordBatch,
    parallel: Vec<T>,
    bitmap: &RoaringBitmap,
) -> AilakeResult<(RecordBatch, Vec<T>)> {
    if bitmap.is_empty() {
        return Ok((batch, parallel));
    }
    let n = batch.num_rows();
    let keep: Vec<bool> = (0..n).map(|i| !bitmap.contains(i as u32)).collect();
    let filtered_parallel: Vec<T> = parallel
        .into_iter()
        .zip(keep.iter())
        .filter_map(|(v, &k)| k.then_some(v))
        .collect();
    let mask = arrow_array::BooleanArray::from(keep);
    let filtered_batch = arrow_select::filter::filter_record_batch(&batch, &mask)
        .map_err(|e| AilakeError::Arrow(e.to_string()))?;
    Ok((filtered_batch, filtered_parallel))
}

/// Returns true when any row in `row_ids` is deleted according to `bitmap`.
/// Used for early-exit pruning: if zero deletions touch the candidate set, skip
/// the per-row check.
#[inline]
pub fn has_deletions(bitmap: &RoaringBitmap, row_ids: &[u64]) -> bool {
    row_ids.iter().any(|&id| bitmap.contains(id as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_store::LocalStore;
    use bytes::Bytes;
    use roaring::RoaringBitmap;

    fn make_bitmap_bytes(deleted: &[u32]) -> Vec<u8> {
        let mut bm = RoaringBitmap::new();
        for &r in deleted {
            bm.insert(r);
        }
        let mut buf = Vec::new();
        bm.serialize_into(&mut buf).unwrap();
        buf
    }

    #[tokio::test]
    async fn load_dv_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let bitmap_bytes = make_bitmap_bytes(&[0, 5, 42, 1000]);

        // Write a minimal Puffin-like file: just the bitmap bytes at a known offset.
        // In real Puffin files there is a header and footer; the DV manifest entry
        // gives us the exact offset+length so we only fetch those bytes.
        let offset: u64 = 16; // simulate a 16-byte Puffin header before the blob
        let mut file_bytes = vec![0u8; offset as usize]; // fake puffin header
        file_bytes.extend_from_slice(&bitmap_bytes);

        let dvd_path = "data/dv-0001.dvd";
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        store.put(dvd_path, Bytes::from(file_bytes)).await.unwrap();

        let dv = DeletionVector {
            path: dvd_path.to_string(),
            offset,
            length: bitmap_bytes.len() as u64,
            cardinality: 4,
        };

        let bm = load_deletion_vector(&store, &dv).await.unwrap();
        assert!(bm.contains(0));
        assert!(bm.contains(5));
        assert!(bm.contains(42));
        assert!(bm.contains(1000));
        assert!(!bm.contains(1)); // not deleted
        assert_eq!(bm.len(), 4);
    }

    #[test]
    fn has_deletions_detects_overlap() {
        let mut bm = RoaringBitmap::new();
        bm.insert(10);
        bm.insert(20);

        assert!(has_deletions(&bm, &[5, 10, 15])); // 10 is deleted
        assert!(!has_deletions(&bm, &[1, 2, 3])); // none deleted
    }
}
