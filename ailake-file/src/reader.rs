use ailake_core::{AilakeError, AilakeResult, Centroid, VectorMetric};
use ailake_index::{HnswIndex, MmapLoader};
use ailake_parquet::ParquetVectorReader;
use arrow_array::RecordBatch;
use bytes::Bytes;

use crate::footer::{
    AilakeHeader, AilakeTrailer, DistanceMetric, HEADER_SIZE, TRAILER_SIZE,
};

pub struct AilakeFileReader {
    bytes: Bytes,
    vector_column: String,
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

    /// Detect AILK magic in the last 4 bytes of the trailer.
    pub fn is_ailake_file(&self) -> bool {
        let len = self.bytes.len();
        if len < TRAILER_SIZE {
            return false;
        }
        &self.bytes[len - 4..] == b"AILK"
    }

    /// Parse the 24-byte trailer from the end of the file.
    pub fn read_trailer(&self) -> AilakeResult<AilakeTrailer> {
        let len = self.bytes.len();
        if len < TRAILER_SIZE {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let trailer_bytes: &[u8; TRAILER_SIZE] = self.bytes[len - TRAILER_SIZE..]
            .try_into()
            .map_err(|_| AilakeError::NotAnAilakeFile)?;
        AilakeTrailer::from_bytes(trailer_bytes)
    }

    /// Parse the 64-byte AI-Lake header.
    pub fn read_header(&self) -> AilakeResult<AilakeHeader> {
        let trailer = self.read_trailer()?;
        let offset = trailer.footer_offset as usize;
        if offset + HEADER_SIZE > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }
        let header_bytes: &[u8; HEADER_SIZE] = self.bytes[offset..offset + HEADER_SIZE]
            .try_into()
            .map_err(|_| AilakeError::NotAnAilakeFile)?;
        AilakeHeader::from_bytes(header_bytes)
    }

    /// Read centroid + radius from the centroid section. Does NOT load the HNSW graph.
    pub fn get_centroid(&self) -> AilakeResult<Centroid> {
        let trailer = self.read_trailer()?;
        let header = self.read_header()?;
        let footer_start = trailer.footer_offset as usize;
        let centroid_start = footer_start + header.centroid_offset as usize;
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

    /// Load the HNSW index from the AI-Lake footer.
    pub fn load_index(&self) -> AilakeResult<HnswIndex> {
        let trailer = self.read_trailer()?;
        let header = self.read_header()?;
        let footer_start = trailer.footer_offset as usize;
        let hnsw_start = footer_start + header.hnsw_offset as usize;
        let hnsw_end = hnsw_start + header.hnsw_len as usize;

        if hnsw_end > self.bytes.len() {
            return Err(AilakeError::NotAnAilakeFile);
        }

        let hnsw_data = &self.bytes[hnsw_start..hnsw_end];
        MmapLoader::from_bytes(hnsw_data)
    }

    /// Read the Parquet section (tabular data + decoded embeddings).
    pub fn read_parquet(&self) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let trailer = self.read_trailer()?;
        let parquet_bytes = self.bytes.slice(0..trailer.footer_offset as usize);
        let reader = ParquetVectorReader::new(parquet_bytes, &self.vector_column);
        reader.read_all()
    }

    /// Verify the positional invariant: Parquet record_count == HNSW node_count.
    pub fn verify_integrity(&self) -> AilakeResult<()> {
        let header = self.read_header()?;
        let index = self.load_index()?;
        let parquet_count = {
            let trailer = self.read_trailer()?;
            let parquet_bytes = self.bytes.slice(0..trailer.footer_offset as usize);
            let reader = ParquetVectorReader::new(parquet_bytes, &self.vector_column);
            reader.record_count()?
        };
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
        }
    }

    fn write_file(rows: usize, dim: u32) -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let ids: Vec<i32> = (0..rows as i32).collect();
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();
        let embs: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut v = vec![0.0f32; dim as usize];
                v[i % dim as usize] = 1.0;
                v
            })
            .collect();
        AilakeFileWriter::new(make_policy(dim)).write(&batch, &embs).unwrap()
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
