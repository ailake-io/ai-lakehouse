// SPDX-License-Identifier: MIT OR Apache-2.0
use ailake_core::{AilakeError, AilakeResult, Centroid, VectorMetric};
use ailake_index::{AnyIndex, BinarySerializer, HnswIndex, IvfPqSerializer, MmapLoader, RaBitQSerializer};
use ailake_parquet::ParquetVectorReader;
use arrow_array::RecordBatch;
use bytes::Bytes;

use crate::footer::{
    AilakeHeader, DistanceMetric, FLAG_INDEX_BINARY, FLAG_INDEX_IVF_PQ, FLAG_INDEX_RABITQ,
    HEADER_SIZE,
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
    /// Reads `ailake.footer_offset` from the Parquet footer key-value metadata.
    pub fn ailk_offset(&self) -> AilakeResult<u64> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        let val = reader
            .kv_metadata("ailake.footer_offset")?
            .ok_or(AilakeError::NotAnAilakeFile)?;
        val.parse::<u64>().map_err(|_| AilakeError::NotAnAilakeFile)
    }

    /// Returns the absolute byte offset of the AILK section for a named vector column.
    ///
    /// For additional columns tries `ailake.{column}.footer_offset` first,
    /// then falls back to `ailake.footer_offset` (primary / single-column files).
    pub fn ailk_offset_for_column(&self, column: &str) -> AilakeResult<u64> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), column);
        let col_key = format!("ailake.{column}.footer_offset");
        if let Some(val) = reader.kv_metadata(&col_key)? {
            return val.parse::<u64>().map_err(|_| AilakeError::NotAnAilakeFile);
        }
        let val = reader
            .kv_metadata("ailake.footer_offset")?
            .ok_or(AilakeError::NotAnAilakeFile)?;
        val.parse::<u64>().map_err(|_| AilakeError::NotAnAilakeFile)
    }

    /// Returns true if the file contains an embedded AILK section.
    pub fn is_ailake_file(&self) -> bool {
        self.ailk_offset().is_ok()
    }

    /// Parse the 64-byte AI-Lake header from the embedded AILK section.
    pub fn read_header(&self) -> AilakeResult<AilakeHeader> {
        let offset = self.ailk_offset()? as usize;
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
        MmapLoader::from_bytes(&self.bytes[hnsw_start..hnsw_end])
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

        if header.flags & FLAG_INDEX_BINARY != 0 {
            let idx = BinarySerializer::from_bytes(index_bytes)?;
            Ok(AnyIndex::Binary(idx))
        } else if header.flags & FLAG_INDEX_RABITQ != 0 {
            let idx = RaBitQSerializer::from_bytes(index_bytes)?;
            Ok(AnyIndex::RaBitQ(idx))
        } else if header.flags & FLAG_INDEX_IVF_PQ != 0 {
            let idx = IvfPqSerializer::from_bytes(index_bytes)?;
            Ok(AnyIndex::IvfPq(idx))
        } else {
            let idx = MmapLoader::from_bytes(index_bytes)?;
            Ok(AnyIndex::Hnsw(idx))
        }
    }

    /// Read the Parquet section (tabular data + decoded embeddings).
    /// The full file is valid Parquet; the AILK section is invisible to standard readers.
    pub fn read_parquet(&self) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let reader = ParquetVectorReader::new(self.bytes.clone(), &self.vector_column);
        reader.read_all()
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
            keep_raw_for_reranking: false,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            rabitq: None,
            binary: None,
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
