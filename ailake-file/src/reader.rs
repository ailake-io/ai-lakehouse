// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::footer::Precision;
use ailake_core::{AilakeError, AilakeResult, Centroid, VectorMetric};
use ailake_index::{AnyIndex, HnswIndex, IvfPqSerializer, MmapLoader};
use ailake_parquet::ParquetVectorReader;
use arrow_array::RecordBatch;
use bytes::Bytes;

use crate::footer::{
    parquet_footer_start, AilakeHeader, AilakeTrailer, DistanceMetric, FLAG_INDEX_IVF_PQ,
    AILK_FTS_HEADER_SIZE, AILK_FTS_MAGIC, HEADER_SIZE, KV_FTS_OFFSET, TRAILER_SIZE,
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
        if offset + HEADER_SIZE > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let header_bytes: &[u8; HEADER_SIZE] = self.bytes[offset..offset + HEADER_SIZE]
            .try_into()
            .map_err(|_| AilakeError::NotAnAilakeFile)?;
        AilakeHeader::from_bytes(header_bytes)
    }

    /// Read centroid + radius from the AILK section.
    pub fn get_centroid(&self) -> AilakeResult<Centroid> {
        let ailk_start = self.ailk_offset()? as usize;
        let header = self.read_header()?;
        let centroid_start = ailk_start + header.centroid_offset as usize;
        let centroid_end = centroid_start + header.centroid_len as usize;

        if centroid_end > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }

        let centroid_data = &self.bytes[centroid_start..centroid_end];
        let dim = header.dim as usize;
        let expected_len = dim * 4 + 4;
        if centroid_data.len() != expected_len {
            return Err(AilakeError::InvalidCentroidLength {
                expected_dim: header.dim,
                actual: centroid_data.len(),
            });
        }

        let values: Vec<f32> = centroid_data[..dim * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let radius = f32::from_le_bytes(centroid_data[dim * 4..].try_into().unwrap());
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

        if ailk_start + HEADER_SIZE > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let header_bytes: &[u8; HEADER_SIZE] = self.bytes[ailk_start..ailk_start + HEADER_SIZE]
            .try_into()
            .map_err(|_| AilakeError::NotAnAilakeFile)?;
        let header = AilakeHeader::from_bytes(header_bytes)?;

        let hnsw_start = ailk_start + header.hnsw_offset as usize;
        let hnsw_end = hnsw_start + header.hnsw_len as usize;

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

        if ailk_start + HEADER_SIZE > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let header_bytes: &[u8; HEADER_SIZE] = self.bytes[ailk_start..ailk_start + HEADER_SIZE]
            .try_into()
            .map_err(|_| AilakeError::NotAnAilakeFile)?;
        let header = AilakeHeader::from_bytes(header_bytes)?;

        let index_start = ailk_start + header.hnsw_offset as usize;
        let index_end = index_start + header.hnsw_len as usize;

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
            .map_err(|_| AilakeError::NotAnAilakeFile)? as usize;

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

    /// Verify the positional invariant: Parquet record_count == HNSW node_count.
    pub fn verify_integrity(&self) -> AilakeResult<()> {
        let header = self.read_header()?;
        let index = self.load_index()?;
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
}
