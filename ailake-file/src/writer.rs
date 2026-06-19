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
    parquet_footer_start, AilakeHeader, AilakeTrailer, DistanceMetric, Precision,
    AILAKE_FORMAT_VERSION, FLAG_INDEX_IVF_PQ, HEADER_SIZE, TRAILER_SIZE,
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

    /// Write `batch` + `embeddings` using a **pre-built** `HnswIndex`.
    ///
    /// Skips the O(N log N) HNSW construction — the caller is responsible for
    /// building and populating the index. The index must contain exactly
    /// `embeddings.len()` nodes with RowIds `0..N` matching `embeddings[0..N]`
    /// in order.
    ///
    /// Used by incremental compaction to reuse the dominant file's existing
    /// index and only insert vectors from smaller files.
    pub fn write_with_prebuilt_hnsw(
        &self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
        hnsw: &ailake_index::HnswIndex,
    ) -> AilakeResult<Bytes> {
        use ailake_core::AilakeError;

        let parquet_writer = ParquetVectorWriter::new(self.policy.clone());

        // Pass 1: write Parquet without KV to measure the row-group section size.
        let (parquet_v1, record_count) = parquet_writer.write_batch(batch, embeddings)?;
        let footer_start = parquet_footer_start(&parquet_v1)?;

        let index_bytes = HnswSerializer::to_bytes(hnsw)?;
        let ailk_section = build_ailk_section_from_index_bytes(
            &self.policy,
            embeddings,
            record_count,
            footer_start as u64,
            &index_bytes,
            0u16, // flags=0: HNSW (not IVF-PQ)
        )?;

        let kv_val = footer_start.to_string();
        let kv_refs: &[(&str, &str)] = &[("ailake.footer_offset", kv_val.as_str())];

        // Pass 2: write Parquet with AILK offset KV embedded.
        let (parquet_v2, _) = parquet_writer.write_batch_with_kv(batch, embeddings, kv_refs)?;
        let footer_start_v2 = parquet_footer_start(&parquet_v2).map_err(|e| {
            AilakeError::Parquet(format!(
                "footer_start unstable in write_with_prebuilt_hnsw: {e}"
            ))
        })?;
        debug_assert_eq!(
            footer_start, footer_start_v2,
            "footer_start must be stable across KV injection"
        );

        let footer_len_v2 = parquet_v2.len() - footer_start_v2;
        let mut out = BytesMut::with_capacity(footer_start + ailk_section.len() + footer_len_v2);
        out.put_slice(&parquet_v1[..footer_start]);
        drop(parquet_v1);
        out.put(ailk_section);
        out.put_slice(&parquet_v2[footer_start_v2..]);
        Ok(out.freeze())
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

    /// Single-pass streaming write — no `ailake.footer_offset` KV injection.
    ///
    /// Produces a valid AI-Lake file in a **single Parquet write pass** without any
    /// seek or footer rewrite. Safe for append-only destinations (HDFS strict mode,
    /// piped stdout, write-once distributed filesystems).
    ///
    /// Readers bootstrap the AILK section from the `AilakeTrailer` (the 24 bytes
    /// immediately preceding the Parquet footer) instead of from the Parquet KV
    /// metadata. Both bootstrap paths are supported by `AilakeFileReader`.
    ///
    /// Trade-off vs `write()`:
    /// - One Parquet write instead of two — saves CPU + memory for large batches.
    /// - On S3, the AILK offset is derived from bytes already fetched in the
    ///   footer range-GET (trailer is the 24 bytes before the footer). No extra GET.
    pub fn write_single_pass(
        &self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<Bytes> {
        let col = VectorColumnBatch {
            policy: &self.policy,
            embeddings,
        };
        self.write_multi_single_pass(batch, &[col])
    }

    /// Multi-column variant of `write_single_pass`.
    pub fn write_multi_single_pass(
        &self,
        batch: &RecordBatch,
        columns: &[VectorColumnBatch<'_>],
    ) -> AilakeResult<Bytes> {
        use ailake_core::AilakeError;

        if columns.is_empty() {
            return Err(AilakeError::InvalidArgument(
                "write_multi_single_pass requires at least one vector column".into(),
            ));
        }

        let primary = &columns[0];
        let parquet_writer = ParquetVectorWriter::new(primary.policy.clone());

        // Single Parquet write — no KV injection for ailake.footer_offset.
        // Extra KV still present (dim, metric, record_count, etc.) for Iceberg compat.
        let (parquet_bytes, record_count) =
            parquet_writer.write_batch(batch, primary.embeddings)?;
        let footer_start = parquet_footer_start(&parquet_bytes)?;

        // Build all AILK sections. Offsets are relative to the final assembled file,
        // where AILK sections start at `footer_start` (right after row groups).
        let mut ailk_sections: Vec<Bytes> = Vec::with_capacity(columns.len());
        let mut current_offset = footer_start as u64;
        for col in columns.iter() {
            let section = build_ailk_section(
                col.policy,
                col.embeddings,
                record_count,
                current_offset,
                &self.index_type,
                self.shared_codebook.as_deref(),
            )?;
            current_offset += section.len() as u64;
            ailk_sections.push(section);
        }

        // Assemble: [row groups] + [AILK sections] + [original Parquet footer].
        // No footer rewrite needed — reader uses AilakeTrailer for bootstrap.
        let total_ailk: usize = ailk_sections.iter().map(|s| s.len()).sum();
        let mut out = BytesMut::with_capacity(parquet_bytes.len() + total_ailk);
        out.put_slice(&parquet_bytes[..footer_start]);
        for section in ailk_sections {
            out.put(section);
        }
        out.put_slice(&parquet_bytes[footer_start..]);

        Ok(out.freeze())
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
        // Only the Parquet footer changes between pass 1 and pass 2 (KV metadata
        // is stored in the footer thrift, not in row groups). Row group offsets and
        // payloads are byte-for-byte identical, so footer_start is stable across
        // both passes. We reuse pass-1's row groups and take only the footer from pass 2.
        let kv_refs: Vec<(&str, &str)> = kv_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let (parquet_v2, _) =
            parquet_writer.write_batch_with_kv(batch, primary.embeddings, &kv_refs)?;
        let footer_start_v2 = parquet_footer_start(&parquet_v2)?;
        debug_assert_eq!(
            footer_start, footer_start_v2,
            "footer_start must be stable across KV injection (row groups unchanged)"
        );

        // Splice: [PAR1 + row groups from v1] + [AILK sections] + [Parquet footer+PAR1 from v2]
        // Using v1's row groups here lets us release v2's large row group allocation sooner.
        let total_ailk: usize = ailk_sections.iter().map(|s| s.len()).sum();
        let footer_len_v2 = parquet_v2.len() - footer_start_v2;
        let total = footer_start + total_ailk + footer_len_v2;
        let mut out = BytesMut::with_capacity(total);
        out.put_slice(&parquet_v1[..footer_start]);
        drop(parquet_v1); // row groups copied; free the v1 allocation
        for section in ailk_sections {
            out.put(section);
        }
        out.put_slice(&parquet_v2[footer_start_v2..]);

        Ok(out.freeze())
    }
}

/// Build a complete AILK section using **pre-serialized** index bytes.
/// Same layout as `build_ailk_section` but skips the index build step.
fn build_ailk_section_from_index_bytes(
    policy: &VectorStoragePolicy,
    embeddings: &[Vec<f32>],
    record_count: u64,
    ailk_abs_offset: u64,
    index_bytes: &[u8],
    flags: u16,
) -> AilakeResult<Bytes> {
    let norm_storage: Vec<Vec<f32>>;
    let (emb_for_centroid, centroid_metric) =
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

    let centroid = compute_centroid_and_radius(emb_for_centroid, centroid_metric);
    let centroid_bytes = encode_centroid(&centroid);

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
    buf.put_slice(index_bytes);
    buf.put_slice(&trailer.to_bytes());
    Ok(buf.freeze())
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
            partition_by: None,
            partition_value: None,
            partition_column_type: None,
            partition_fields: vec![],
        }
    }

    #[test]
    fn write_single_pass_valid_parquet_and_ailk() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
        let embs: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();

        let writer = AilakeFileWriter::new(make_policy(4));
        let file = writer.write_single_pass(&batch, &embs).unwrap();

        // Must be valid Parquet envelope
        assert_eq!(&file[..4], b"PAR1");
        assert_eq!(&file[file.len() - 4..], b"PAR1");
        // AILK magic present
        assert!(file.windows(4).any(|w| w == b"AILK"));
    }

    #[test]
    fn write_single_pass_reader_bootstrap_from_trailer() {
        use crate::reader::AilakeFileReader;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![10, 20, 30]))])
                .unwrap();
        let embs: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();

        let writer = AilakeFileWriter::new(make_policy(4));
        let file_bytes = writer.write_single_pass(&batch, &embs).unwrap();

        // Single-pass files have NO ailake.footer_offset in Parquet KV.
        // AilakeFileReader must bootstrap from AilakeTrailer instead.
        let reader = AilakeFileReader::new(file_bytes, "embedding", 4);
        assert!(
            reader.is_ailake_file(),
            "single-pass file must be recognised as AI-Lake file via trailer bootstrap"
        );
        let header = reader.read_header().expect("must read AILK header");
        assert_eq!(header.dim, 4);
        assert_eq!(header.record_count, 3);
    }

    #[test]
    fn write_and_write_single_pass_same_index() {
        // Both write paths must produce an index that returns the same nearest neighbour
        // for a fixed query — verifying that single-pass doesn't corrupt the HNSW.
        use crate::reader::AilakeFileReader;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )
        .unwrap();
        let embs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];
        let policy = make_policy(4);
        let writer = AilakeFileWriter::new(policy);

        let bytes_two_pass = writer.write(&batch, &embs).unwrap();
        let bytes_single_pass = writer.write_single_pass(&batch, &embs).unwrap();

        let query = vec![1.0f32, 0.0, 0.0, 0.0];

        let reader_tp = AilakeFileReader::new(bytes_two_pass, "embedding", 4);
        let reader_sp = AilakeFileReader::new(bytes_single_pass, "embedding", 4);

        let idx_tp = reader_tp.load_index().unwrap();
        let idx_sp = reader_sp.load_index().unwrap();

        let res_tp = idx_tp.search(&query, 1, 50);
        let res_sp = idx_sp.search(&query, 1, 50);

        assert_eq!(res_tp[0].0, res_sp[0].0, "nearest neighbour must match");
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
            partition_by: None,
            partition_value: None,
            partition_column_type: None,
            partition_fields: vec![],
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
