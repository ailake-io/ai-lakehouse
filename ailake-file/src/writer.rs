// SPDX-License-Identifier: MIT OR Apache-2.0
use ailake_core::{AilakeResult, Centroid, RowId, VectorStoragePolicy};
use ailake_index::{
    HnswBuilder, HnswConfig, HnswSerializer, IvfPqCodebook, IvfPqConfig, IvfPqIndex,
    IvfPqSerializer,
};
use ailake_parquet::ParquetVectorWriter;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::RecordBatch;
use bytes::{BufMut, Bytes, BytesMut};

use crate::footer::{
    AilakeHeader, AilakeTrailer, DistanceMetric, Precision, AILAKE_FORMAT_VERSION,
    FLAG_INDEX_IVF_PQ, HEADER_SIZE, TRAILER_SIZE,
};

/// Which index algorithm to embed in the AILK section.
#[derive(Debug, Clone)]
pub enum IndexType {
    /// HNSW (default). Best recall for in-memory workloads.
    Hnsw(HnswConfig),
    /// IVF-PQ. Best for S3: 10-100x smaller index, sequential inverted-list reads.
    IvfPq(IvfPqConfig),
    /// Detect hardware at write time and pick the best index automatically.
    ///
    /// Chooses IVF-PQ when a GPU or ≥8 CPU cores are available AND the dataset
    /// has ≥5 000 vectors. Falls back to HNSW otherwise (local/low-power hardware).
    Auto,
}

impl Default for IndexType {
    fn default() -> Self {
        IndexType::Hnsw(HnswConfig::default())
    }
}

/// One vector column to embed in a multi-column write.
pub struct VectorColumnBatch<'a> {
    pub policy: &'a VectorStoragePolicy,
    pub embeddings: &'a [Vec<f32>],
}

pub struct AilakeFileWriter {
    policy: VectorStoragePolicy,
    index_type: IndexType,
    /// Pre-trained shared codebook. When set, skips k-means for IVF-PQ builds.
    shared_codebook: Option<std::sync::Arc<IvfPqCodebook>>,
}

impl AilakeFileWriter {
    pub fn new(policy: VectorStoragePolicy) -> Self {
        Self {
            policy,
            index_type: IndexType::default(),
            shared_codebook: None,
        }
    }

    /// Use a pre-trained IVF-PQ codebook instead of running k-means.
    /// Shards built from the same codebook produce comparable ADC distances.
    pub fn with_shared_ivf_codebook(mut self, codebook: std::sync::Arc<IvfPqCodebook>) -> Self {
        self.shared_codebook = Some(codebook);
        self
    }

    pub fn with_hnsw_config(mut self, config: HnswConfig) -> Self {
        self.index_type = IndexType::Hnsw(config);
        self
    }

    pub fn with_ivf_pq(mut self, config: IvfPqConfig) -> Self {
        self.index_type = IndexType::IvfPq(config);
        self
    }

    pub fn with_index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Use `IndexType::Auto`: detect GPU / CPU cores at write time and pick the
    /// best index. Equivalent to `.with_index_type(IndexType::Auto)`.
    pub fn with_auto_index(mut self) -> Self {
        self.index_type = IndexType::Auto;
        self
    }

    /// Write RecordBatch + embeddings as plain Parquet, with no AILK section.
    ///
    /// Used by `TableWriter::write_batch_deferred()` to persist data immediately
    /// while the HNSW index is built asynchronously in the background.
    /// The resulting file is a valid Parquet readable by any standard reader,
    /// but `AilakeFileReader::is_ailake_file()` returns false until the HNSW
    /// section is appended by the background indexing task.
    pub fn write_parquet_only(
        &self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<Bytes> {
        let parquet_writer = ParquetVectorWriter::new(self.policy.clone());
        let (bytes, _) = parquet_writer.write_batch(batch, embeddings)?;
        Ok(bytes)
    }

    /// Write RecordBatch + embeddings into a single AI-Lake file.
    ///
    /// Layout:
    ///   [PAR1][row groups][AILK header+centroid+HNSW+trailer][Parquet footer][footer_len][PAR1]
    ///
    /// Standard Parquet readers find PAR1 at the end, read the footer, skip directly to row
    /// group offsets. The AILK section sits between row groups and footer and is never touched.
    /// AI-Lake readers find the AILK section via `ailake.footer_offset` in the Parquet footer KV.
    pub fn write(&self, batch: &RecordBatch, embeddings: &[Vec<f32>]) -> AilakeResult<Bytes> {
        let col = VectorColumnBatch {
            policy: &self.policy,
            embeddings,
        };
        self.write_multi(batch, &[col])
    }

    /// Write RecordBatch + multiple vector columns into a single AI-Lake file.
    ///
    /// Each column gets its own AILK section appended sequentially before the Parquet footer.
    /// Offsets are recorded in Parquet KV metadata:
    ///   - Primary (first) column: `ailake.footer_offset`
    ///   - Additional columns: `ailake.{column_name}.footer_offset`
    ///
    /// Readers use the column-specific KV key to locate the right AILK section.
    pub fn write_multi(
        &self,
        batch: &RecordBatch,
        columns: &[VectorColumnBatch<'_>],
    ) -> AilakeResult<Bytes> {
        use ailake_core::AilakeError;

        if columns.is_empty() {
            return Err(AilakeError::InvalidArgument(
                "write_multi requires at least one vector column".into(),
            ));
        }

        let primary = &columns[0];
        let parquet_writer = ParquetVectorWriter::new(primary.policy.clone());

        // Pass 1 — write Parquet without KV to measure the data section size.
        let (parquet_v1, record_count) = parquet_writer.write_batch(batch, primary.embeddings)?;
        let footer_start = parquet_footer_start(&parquet_v1)?;

        // Build all AILK sections sequentially; track running absolute offset.
        let mut ailk_sections: Vec<Bytes> = Vec::with_capacity(columns.len());
        let mut kv_owned: Vec<(String, String)> = Vec::with_capacity(columns.len());
        let mut current_offset = footer_start as u64;

        for (i, col) in columns.iter().enumerate() {
            let section = build_ailk_section(
                col.policy,
                col.embeddings,
                record_count,
                current_offset,
                &self.index_type,
                self.shared_codebook.as_deref(),
            )?;
            let kv_key = if i == 0 {
                "ailake.footer_offset".to_string()
            } else {
                format!("ailake.{}.footer_offset", col.policy.column_name)
            };
            kv_owned.push((kv_key, current_offset.to_string()));
            current_offset += section.len() as u64;
            ailk_sections.push(section);
        }

        // Pass 2 — write Parquet with all AILK offset KVs embedded.
        let kv_refs: Vec<(&str, &str)> = kv_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let (parquet_v2, _) =
            parquet_writer.write_batch_with_kv(batch, primary.embeddings, &kv_refs)?;
        let footer_start_v2 = parquet_footer_start(&parquet_v2)?;

        // Splice: [PAR1 + row groups] + [all AILK sections] + [Parquet footer + PAR1]
        let total_ailk: usize = ailk_sections.iter().map(|s| s.len()).sum();
        let total = footer_start_v2 + total_ailk + (parquet_v2.len() - footer_start_v2);
        let mut out = BytesMut::with_capacity(total);
        out.put_slice(&parquet_v2[..footer_start_v2]);
        for section in ailk_sections {
            out.put(section);
        }
        out.put_slice(&parquet_v2[footer_start_v2..]);

        Ok(out.freeze())
    }
}

/// Build a complete AILK section (header + centroid + index + trailer) for one vector column.
fn build_ailk_section(
    policy: &VectorStoragePolicy,
    embeddings: &[Vec<f32>],
    record_count: u64,
    ailk_abs_offset: u64,
    index_type: &IndexType,
    shared_codebook: Option<&IvfPqCodebook>,
) -> AilakeResult<Bytes> {
    // Normalize to unit L2 when pre_normalize is set.
    // Enables the NormalizedCosine fast path: 1-dot(a,b) instead of full cosine.
    let norm_storage: Vec<Vec<f32>>;
    let (embeddings, hnsw_metric) =
        if policy.pre_normalize && policy.metric == ailake_core::VectorMetric::Cosine {
            norm_storage = embeddings
                .iter()
                .map(|v| ailake_vec::normalize_l2(v))
                .collect();
            (
                norm_storage.as_slice(),
                ailake_core::VectorMetric::NormalizedCosine,
            )
        } else {
            (embeddings, policy.metric)
        };

    let centroid: Centroid = compute_centroid_and_radius(embeddings, hnsw_metric);
    let centroid_bytes = encode_centroid(&centroid);

    // Resolve Auto to a concrete variant before matching.
    let resolved: IndexType;
    let index_type = if matches!(index_type, IndexType::Auto) {
        let profile = ailake_index::HardwareProfile::detect();
        resolved = if profile.recommend_ivf_pq(embeddings.len()) {
            IndexType::IvfPq(ailake_index::IvfPqConfig::for_dataset(
                policy.dim as usize,
                embeddings.len(),
            ))
        } else {
            IndexType::Hnsw(ailake_index::HnswConfig::default())
        };
        &resolved
    } else {
        index_type
    };

    let (index_bytes, flags) = match index_type {
        IndexType::Hnsw(hnsw_config) => {
            // Policy-level M/ef_construction override the IndexType defaults when set.
            let config = HnswConfig {
                m: policy.hnsw_m.map(|v| v as usize).unwrap_or(hnsw_config.m),
                ef_construction: policy
                    .hnsw_ef_construction
                    .map(|v| v as usize)
                    .unwrap_or(hnsw_config.ef_construction),
                max_elements: hnsw_config.max_elements,
            };
            let mut builder = HnswBuilder::new(policy.dim, hnsw_metric, config);
            for (i, v) in embeddings.iter().enumerate() {
                builder.insert(RowId::new(i as u64), v.clone());
            }
            let index = builder.build();
            (HnswSerializer::to_bytes(&index)?, 0u16)
        }
        IndexType::IvfPq(ivf_config) => {
            let row_ids: Vec<RowId> = (0..embeddings.len() as u64).map(RowId::new).collect();
            let index = if let Some(cb) = shared_codebook {
                IvfPqIndex::build_with_codebook(&row_ids, embeddings, cb)?
            } else {
                ailake_index::IvfPqIndex::train(
                    &row_ids,
                    embeddings,
                    policy.metric,
                    ivf_config.clone(),
                )?
            };
            (IvfPqSerializer::to_bytes(&index)?, FLAG_INDEX_IVF_PQ)
        }
        IndexType::Auto => unreachable!("Auto resolved above"),
    };

    let centroid_offset = HEADER_SIZE as u64;
    let centroid_len = centroid_bytes.len() as u64;
    let index_offset_in_ailk = centroid_offset + centroid_len;
    let index_len = index_bytes.len() as u64;
    let ailk_total_len = HEADER_SIZE as u64 + centroid_len + index_len + TRAILER_SIZE as u64;

    let header = AilakeHeader {
        format_version: AILAKE_FORMAT_VERSION,
        flags,
        dim: policy.dim,
        precision: Precision::from(policy.precision),
        distance_metric: DistanceMetric::from(policy.metric),
        record_count,
        centroid_offset,
        centroid_len,
        hnsw_offset: index_offset_in_ailk,
        hnsw_len: index_len,
    };
    let trailer = AilakeTrailer {
        footer_offset: ailk_abs_offset,
        footer_len: ailk_total_len,
        format_version: AILAKE_FORMAT_VERSION,
        flags,
    };

    let mut buf = BytesMut::with_capacity(ailk_total_len as usize);
    buf.put_slice(&header.to_bytes());
    buf.put_slice(&centroid_bytes);
    buf.put_slice(&index_bytes);
    buf.put_slice(&trailer.to_bytes());
    Ok(buf.freeze())
}

/// Returns the byte offset in `buf` where the Parquet footer thrift starts.
/// Layout of buf tail: [...footer_thrift...][footer_len u32 LE][PAR1 4 bytes]
fn parquet_footer_start(buf: &[u8]) -> AilakeResult<usize> {
    use ailake_core::AilakeError;
    let len = buf.len();
    if len < 8 {
        return Err(AilakeError::Parquet("file too small".into()));
    }
    if &buf[len - 4..] != b"PAR1" {
        return Err(AilakeError::Parquet("missing PAR1 footer magic".into()));
    }
    let footer_thrift_len = u32::from_le_bytes(buf[len - 8..len - 4].try_into().unwrap()) as usize;
    let start = len
        .checked_sub(8 + footer_thrift_len)
        .ok_or_else(|| AilakeError::Parquet("footer length overflow".into()))?;
    Ok(start)
}

fn encode_centroid(c: &Centroid) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(c.values.len() * 4 + 4);
    for &v in &c.values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&c.radius.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_core::{VectorMetric, VectorPrecision};
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
        }
    }

    #[test]
    fn write_ends_with_par1() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
        let embs: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();

        let writer = AilakeFileWriter::new(make_policy(4));
        let file = writer.write(&batch, &embs).unwrap();

        assert_eq!(&file[file.len() - 4..], b"PAR1");
        assert_eq!(&file[..4], b"PAR1");
        assert!(file.windows(4).any(|w| w == b"AILK"));
    }

    #[test]
    fn write_multi_two_columns() {
        use ailake_core::{VectorMetric, VectorPrecision};

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

        let embs: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let ctx_embs: Vec<Vec<f32>> = (0..3).map(|i| vec![0.0, i as f32, 0.0, 0.0]).collect();

        let policy1 = make_policy(4);
        let policy2 = VectorStoragePolicy {
            column_name: "context_embedding".to_string(),
            dim: 4,
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
        };

        let writer = AilakeFileWriter::new(policy1.clone());
        let file = writer
            .write_multi(
                &batch,
                &[
                    VectorColumnBatch {
                        policy: &policy1,
                        embeddings: &embs,
                    },
                    VectorColumnBatch {
                        policy: &policy2,
                        embeddings: &ctx_embs,
                    },
                ],
            )
            .unwrap();

        // Valid Parquet envelope
        assert_eq!(&file[..4], b"PAR1");
        assert_eq!(&file[file.len() - 4..], b"PAR1");
        // Two AILK sections — magic appears at least twice
        let ailk_count = file.windows(4).filter(|w| *w == b"AILK").count();
        assert!(
            ailk_count >= 2,
            "expected >= 2 AILK markers, got {ailk_count}"
        );
    }
}
