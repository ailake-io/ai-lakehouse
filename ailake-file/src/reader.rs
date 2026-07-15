// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::footer::Precision;
use ailake_core::{AilakeError, AilakeResult, Centroid, VectorMetric};
use ailake_index::{AnyIndex, HnswIndex, IvfPqSerializer, MmapLoader};
use ailake_parquet::ParquetVectorReader;
use arrow_array::RecordBatch;
use bytes::Bytes;

use crate::footer::{
    parquet_footer_start, AilakeHeader, AilakeTrailer, DistanceMetric, AILK_FTS_HEADER_SIZE,
    AILK_FTS_MAGIC, FLAG_INDEX_IVF_PQ, HEADER_SIZE, KV_FTS_OFFSET, TRAILER_SIZE,
};

pub struct AilakeFileReader {
    bytes: Bytes,
    vector_column: String,
    #[allow(dead_code)]
    dim: u32,
}

impl AilakeFileReader {
    pub fn new(bytes: Bytes, vector_column: &str, dim: u32) -> Self {
        Self {
            bytes,
            vector_column: vector_column.to_string(),
            dim,
        }
    }

    /// Returns the absolute byte offset of the primary AILK section.
    ///
    /// Tries `ailake.footer_offset` from Parquet KV metadata first (files written
    /// by `write()` / `write_multi()`). Falls back to `AilakeTrailer` bootstrap for
    /// files produced by `write_single_pass()` / `write_multi_single_pass()`.
    pub fn ailk_offset(&self) -> AilakeResult<u64> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        if let Some(val) = reader.kv_metadata("ailake.footer_offset")? {
            return val.parse::<u64>().map_err(|_| AilakeError::NotAnAilakeFile);
        }
        self.ailk_offset_from_trailer()
    }

    /// Returns the absolute byte offset of the AILK section for a named vector column.
    ///
    /// Resolution order:
    ///   1. `ailake.{column}.footer_offset` KV (extra columns in multi-column files)
    ///   2. `ailake.footer_offset` KV (primary column or single-column files)
    ///   3. `AilakeTrailer` scan (streaming / single-pass files without KV injection)
    pub fn ailk_offset_for_column(&self, column: &str) -> AilakeResult<u64> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), column);
        let col_key = format!("ailake.{column}.footer_offset");
        if let Some(val) = reader.kv_metadata(&col_key)? {
            return val.parse::<u64>().map_err(|_| AilakeError::NotAnAilakeFile);
        }
        if let Some(val) = reader.kv_metadata("ailake.footer_offset")? {
            return val.parse::<u64>().map_err(|_| AilakeError::NotAnAilakeFile);
        }
        self.ailk_offset_from_trailer()
    }

    /// True iff this file has its own AILK section specifically for `column` — i.e. a
    /// `ailake.{column}.footer_offset` KV entry is present.
    ///
    /// Unlike `ailk_offset_for_column()`, this does **not** fall back to the primary
    /// column's footer or the trailer bootstrap: those fallbacks make
    /// `ailk_offset_for_column(...).is_ok()` return `true` for *any* AI-Lake file
    /// regardless of `column`, which is correct for "give me an offset to read" but
    /// wrong for "does this specific secondary column exist" — e.g. an idempotency
    /// check before adding a new vector column to files that don't have it yet.
    pub fn has_column_footer(&self, column: &str) -> bool {
        let reader = ParquetVectorReader::new(self.bytes.clone(), column);
        let col_key = format!("ailake.{column}.footer_offset");
        matches!(reader.kv_metadata(&col_key), Ok(Some(_)))
    }

    /// Bootstrap AILK offset from the `AilakeTrailer` embedded just before the Parquet footer.
    ///
    /// The `AilakeTrailer` (24 bytes) ends immediately before the Parquet footer thrift
    /// in every AI-Lake file. On S3, the trailer bytes are already present in the initial
    /// footer range-GET (same GET that fetches the Parquet footer), so this bootstrap path
    /// costs no additional I/O compared to the KV path.
    fn ailk_offset_from_trailer(&self) -> AilakeResult<u64> {
        let buf = self.bytes.as_ref();
        let footer_start = parquet_footer_start(buf)?;
        if footer_start < TRAILER_SIZE {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let trailer_start = footer_start - TRAILER_SIZE;
        let trailer_bytes: &[u8; TRAILER_SIZE] = buf[trailer_start..footer_start]
            .try_into()
            .map_err(|_| AilakeError::NotAnAilakeFile)?;
        let trailer = AilakeTrailer::from_bytes(trailer_bytes)?;
        Ok(trailer.footer_offset)
    }

    /// Returns true if the file contains an embedded AILK section.
    pub fn is_ailake_file(&self) -> bool {
        self.ailk_offset().is_ok()
    }

    /// Parse the 64-byte AI-Lake header from the embedded AILK section.
    pub fn read_header(&self) -> AilakeResult<AilakeHeader> {
        self.read_header_at_offset(self.ailk_offset()?)
    }

    /// Parse the 64-byte AI-Lake header for a named vector column.
    ///
    /// Uses `ailake.{column}.footer_offset` for extra columns and falls back to
    /// `ailake.footer_offset` for the primary column (single-column files).
    pub fn read_header_for_column(&self, column: &str) -> AilakeResult<AilakeHeader> {
        self.read_header_at_offset(self.ailk_offset_for_column(column)?)
    }

    fn read_header_at_offset(&self, offset: u64) -> AilakeResult<AilakeHeader> {
        let offset = offset as usize;
        let header_end = offset
            .checked_add(HEADER_SIZE)
            .ok_or(AilakeError::NotAnAilakeFile)?;
        if header_end > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let header_bytes: &[u8; HEADER_SIZE] = self.bytes[offset..header_end]
            .try_into()
            .map_err(|_| AilakeError::NotAnAilakeFile)?;
        AilakeHeader::from_bytes(header_bytes)
    }

    /// Read centroid + radius from the AILK section.
    pub fn get_centroid(&self) -> AilakeResult<Centroid> {
        let ailk_start = self.ailk_offset()? as usize;
        let header = self.read_header()?;
        let centroid_start = ailk_start
            .checked_add(header.centroid_offset as usize)
            .ok_or(AilakeError::NotAnAilakeFile)?;
        let centroid_end = centroid_start
            .checked_add(header.centroid_len as usize)
            .ok_or(AilakeError::NotAnAilakeFile)?;

        if centroid_end > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }

        let centroid_data = &self.bytes[centroid_start..centroid_end];
        let dim = header.dim as usize;
        let expected_len = dim.checked_mul(4).and_then(|v| v.checked_add(4)).ok_or(
            AilakeError::InvalidCentroidLength {
                expected_dim: header.dim,
                actual: centroid_data.len(),
            },
        )?;
        if centroid_data.len() != expected_len {
            return Err(AilakeError::InvalidCentroidLength {
                expected_dim: header.dim,
                actual: centroid_data.len(),
            });
        }

        let values: Vec<f32> = centroid_data[..dim * 4]
            .chunks_exact(4)
            .map(|b| {
                f32::from_le_bytes(
                    b.try_into()
                        .expect("chunks_exact(4) guarantees 4-byte slices"),
                )
            })
            .collect();
        let radius = f32::from_le_bytes(
            centroid_data[dim * 4..]
                .try_into()
                .expect("invariant: validated len == dim*4 + 4 above"),
        );
        let metric = distance_metric_to_vector_metric(header.distance_metric);

        Ok(Centroid {
            values,
            radius,
            metric,
        })
    }

    /// Load the HNSW index from the primary AILK section.
    pub fn load_index(&self) -> AilakeResult<HnswIndex> {
        self.load_index_for_column(&self.vector_column.clone())
    }

    /// Load the HNSW index for a specific vector column.
    ///
    /// Works for both single-column files (falls back to primary AILK) and
    /// multi-column files written with `AilakeFileWriter::write_multi`.
    pub fn load_index_for_column(&self, column: &str) -> AilakeResult<HnswIndex> {
        let ailk_start = self.ailk_offset_for_column(column)? as usize;
        let header = self.read_header_at_offset(ailk_start as u64)?;

        let hnsw_start = ailk_start
            .checked_add(header.hnsw_offset as usize)
            .ok_or(AilakeError::NotAnAilakeFile)?;
        let hnsw_end = hnsw_start
            .checked_add(header.hnsw_len as usize)
            .ok_or(AilakeError::NotAnAilakeFile)?;

        if hnsw_end > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let mut idx = MmapLoader::from_bytes(&self.bytes[hnsw_start..hnsw_end])?;
        if header.precision == Precision::F16 {
            idx.quantize_to_f16();
        }
        Ok(idx)
    }

    /// Load primary index as `AnyIndex`, dispatching on header flags.
    pub fn load_any_index(&self) -> AilakeResult<AnyIndex> {
        self.load_any_index_for_column(&self.vector_column.clone())
    }

    /// Load index for a specific vector column as `AnyIndex`.
    pub fn load_any_index_for_column(&self, column: &str) -> AilakeResult<AnyIndex> {
        let ailk_start = self.ailk_offset_for_column(column)? as usize;
        let header = self.read_header_at_offset(ailk_start as u64)?;

        let index_start = ailk_start
            .checked_add(header.hnsw_offset as usize)
            .ok_or(AilakeError::NotAnAilakeFile)?;
        let index_end = index_start
            .checked_add(header.hnsw_len as usize)
            .ok_or(AilakeError::NotAnAilakeFile)?;

        if index_end > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let index_bytes = &self.bytes[index_start..index_end];

        if header.flags & FLAG_INDEX_IVF_PQ != 0 {
            let idx = IvfPqSerializer::from_bytes(index_bytes)?;
            Ok(AnyIndex::IvfPq(idx))
        } else {
            let mut idx = MmapLoader::from_bytes(index_bytes)?;
            if header.precision == Precision::F16 {
                idx.quantize_to_f16();
            }
            Ok(AnyIndex::Hnsw(idx))
        }
    }

    /// Read the Parquet section (tabular data + decoded embeddings).
    /// The full file is valid Parquet; the AILK section is invisible to standard readers.
    pub fn read_parquet(&self) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        reader.read_all()
    }

    /// Same as `read_parquet`, but pushes `filter` down to the Parquet read —
    /// see `ParquetVectorReader::read_all_filtered` for the row-group skip +
    /// exact `RowFilter` mechanics.
    pub fn read_parquet_filtered(
        &self,
        filter: &ailake_core::ColumnFilter,
    ) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        reader.read_all_filtered(filter)
    }

    /// Returns the set of original (file-relative) row indices matching
    /// `filter`, without disturbing row identity — see
    /// `ParquetVectorReader::matching_row_ids`. Use this (not
    /// `read_parquet_filtered`) wherever the caller also indexes rows by an
    /// externally-computed position (HNSW `row_id`, deletion vector bitmap).
    pub fn matching_row_ids(
        &self,
        filter: &ailake_core::ColumnFilter,
    ) -> AilakeResult<std::collections::HashSet<u64>> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        reader.matching_row_ids(filter)
    }

    /// Load the raw FTS blob from the AILK_FTS section, if present.
    ///
    /// Returns `Ok(None)` when the file has no FTS section (opt-in feature).
    /// The returned bytes can be passed directly to `ailake_fts::FtsSearcher::from_blob`.
    pub fn load_fts_blob(&self) -> AilakeResult<Option<Bytes>> {
        let pq_reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        let fts_offset_str = match pq_reader.kv_metadata(KV_FTS_OFFSET)? {
            Some(s) => s,
            None => return Ok(None),
        };
        let fts_abs: usize = fts_offset_str
            .parse::<u64>()
            .map_err(|e| {
                AilakeError::Fts(format!(
                    "invalid FTS section offset '{fts_offset_str}': {e}"
                ))
            })?
            .try_into()
            .map_err(|_| AilakeError::Fts("FTS section offset exceeds address space".into()))?;

        // Read AILK_FTS header: magic(4) | version(2) | reserved(2) | blob_len(8)
        if fts_abs + AILK_FTS_HEADER_SIZE > self.bytes.len() {
            return Err(AilakeError::Fts("AILK_FTS header out of bounds".into()));
        }
        let hdr = &self.bytes[fts_abs..fts_abs + AILK_FTS_HEADER_SIZE];
        if hdr[0..4] != AILK_FTS_MAGIC {
            return Err(AilakeError::Fts(format!(
                "bad AILK_FTS magic: {:?}",
                &hdr[0..4]
            )));
        }
        let blob_len = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
        let blob_start = fts_abs + AILK_FTS_HEADER_SIZE;
        let blob_end = blob_start + blob_len;
        if blob_end > self.bytes.len() {
            return Err(AilakeError::Fts("AILK_FTS blob out of bounds".into()));
        }
        Ok(Some(self.bytes.slice(blob_start..blob_end)))
    }

    /// Verify the positional invariant: Parquet record_count == index node_count.
    ///
    /// Dispatches via `load_any_index()` (HNSW or IVF-PQ, per `header.flags`) — unlike
    /// `load_index()`, which always assumes HNSW and errors out on an IVF-PQ-indexed file.
    pub fn verify_integrity(&self) -> AilakeResult<()> {
        let header = self.read_header()?;
        let index = self.load_any_index()?;
        let reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        let parquet_count = reader.record_count()?;

        if parquet_count != index.node_count() {
            return Err(AilakeError::RowCountMismatch {
                parquet: parquet_count,
                hnsw: index.node_count(),
            });
        }
        if parquet_count != header.record_count {
            return Err(AilakeError::RowCountMismatch {
                parquet: parquet_count,
                hnsw: header.record_count,
            });
        }
        Ok(())
    }
}

fn distance_metric_to_vector_metric(dm: DistanceMetric) -> VectorMetric {
    match dm {
        DistanceMetric::Cosine => VectorMetric::Cosine,
        DistanceMetric::Euclidean => VectorMetric::Euclidean,
        DistanceMetric::DotProduct => VectorMetric::DotProduct,
        DistanceMetric::NormalizedCosine => VectorMetric::NormalizedCosine,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::AilakeFileWriter;
    use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_policy(dim: u32) -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
            partition_column_type: None,
            partition_fields: vec![],
        }
    }

    fn write_file(rows: usize, dim: u32) -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let ids: Vec<i32> = (0..rows as i32).collect();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();
        let embs: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut v = vec![0.0f32; dim as usize];
                v[i % dim as usize] = 1.0;
                v
            })
            .collect();
        AilakeFileWriter::new(make_policy(dim))
            .write(&batch, &embs)
            .unwrap()
    }

    #[test]
    fn is_ailake_file() {
        let file = write_file(3, 4);
        let reader = AilakeFileReader::new(file, "embedding", 4);
        assert!(reader.is_ailake_file());
    }

    #[test]
    fn integrity_check_passes() {
        let file = write_file(10, 8);
        let reader = AilakeFileReader::new(file, "embedding", 8);
        reader.verify_integrity().unwrap();
    }

    #[test]
    fn centroid_has_correct_dim() {
        let file = write_file(5, 4);
        let reader = AilakeFileReader::new(file, "embedding", 4);
        let centroid = reader.get_centroid().unwrap();
        assert_eq!(centroid.values.len(), 4);
    }

    #[test]
    fn search_finds_nearest() {
        let dim = 4u32;
        let file = write_file(4, dim);
        let reader = AilakeFileReader::new(file, "embedding", dim);
        let index = reader.load_index().unwrap();
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let results = index.search(&query, 1, 50);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, ailake_core::RowId::new(0));
    }

    #[test]
    fn parquet_read_returns_tabular_data() {
        let file = write_file(3, 4);
        let reader = AilakeFileReader::new(file, "embedding", 4);
        let (batch, embs) = reader.read_parquet().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(embs.len(), 3);
    }

    /// Regression: header offset/length fields are parsed straight from file bytes with
    /// no bound validation — a corrupted or hostile file with `hnsw_offset` near
    /// `u64::MAX` used to overflow the `ailk_start + header.hnsw_offset` addition. In a
    /// release build the wrapped result could slip past the `> self.bytes.len()` bounds
    /// check and panic later on a backwards slice range instead of returning
    /// `NotAnAilakeFile`. `checked_add` must reject this cleanly.
    #[test]
    fn corrupted_hnsw_offset_errors_instead_of_panicking() {
        let file = write_file(3, 4);
        let reader = AilakeFileReader::new(file.clone(), "embedding", 4);
        let ailk_start = reader.ailk_offset().unwrap() as usize;

        // hnsw_offset is a little-endian u64 at header bytes [40..48], per footer.rs.
        let mut corrupted = file.to_vec();
        let field_start = ailk_start + 40;
        corrupted[field_start..field_start + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let corrupted = Bytes::from(corrupted);

        let r1 = AilakeFileReader::new(corrupted.clone(), "embedding", 4);
        assert!(r1.load_index_for_column("embedding").is_err());
        let r2 = AilakeFileReader::new(corrupted, "embedding", 4);
        assert!(r2.load_any_index_for_column("embedding").is_err());
    }

    /// Regression: `ailk_offset_for_column("some_col_that_was_never_written")` used to
    /// be misused as an existence check (e.g. `BackfillJob`'s idempotency skip) — but it
    /// falls back to the *primary* column's footer when no per-column KV key exists, so
    /// `.is_ok()` is `true` for any AI-Lake file regardless of `column`, silently
    /// skipping every file on the very first backfill run. `has_column_footer` checks
    /// the per-column KV key directly, with no such fallback.
    #[test]
    fn has_column_footer_does_not_false_positive_on_primary_fallback() {
        let file = write_file(3, 4);
        let reader = AilakeFileReader::new(file, "embedding", 4);

        // A column that was never written must NOT be reported as present, even though
        // ailk_offset_for_column() incorrectly succeeds via the primary-column fallback
        // (the exact behavior has_column_footer exists to avoid).
        assert!(!reader.has_column_footer("embedding_v2"));
        assert!(
            reader.ailk_offset_for_column("embedding_v2").is_ok(),
            "sanity: ailk_offset_for_column's primary-fallback behavior is what \
             has_column_footer exists to avoid — if this assert ever fails, the \
             fallback was removed and has_column_footer may be redundant"
        );
    }

    /// True-positive counterpart: a genuine extra column written via `write_multi`
    /// (its own `ailake.{column}.footer_offset` KV entry, per the per-column convention)
    /// must be detected as present.
    #[test]
    fn has_column_footer_detects_genuine_extra_column() {
        use crate::writer::VectorColumnBatch;

        let dim = 4u32;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![0i32, 1, 2]))])
                .unwrap();
        let primary_embs: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0]; 3];
        let extra_embs: Vec<Vec<f32>> = vec![vec![0.0, 1.0, 0.0, 0.0]; 3];
        let extra_policy = make_policy(dim);
        let mut extra_policy = extra_policy.clone();
        extra_policy.column_name = "embedding_v2".to_string();

        let primary_policy = make_policy(dim);
        let file_bytes = AilakeFileWriter::new(primary_policy.clone())
            .write_multi(
                &batch,
                &[
                    VectorColumnBatch {
                        policy: &primary_policy,
                        embeddings: &primary_embs,
                    },
                    VectorColumnBatch {
                        policy: &extra_policy,
                        embeddings: &extra_embs,
                    },
                ],
            )
            .unwrap();

        let reader = AilakeFileReader::new(file_bytes, "embedding", dim);
        assert!(reader.has_column_footer("embedding_v2"));
        // Still correctly absent for a column that really was never written.
        assert!(!reader.has_column_footer("embedding_v3"));
    }
}
