// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-file column statistics (Iceberg manifest `value_counts`/`null_value_counts`/
//! `column_sizes`/`lower_bounds`/`upper_bounds`) extracted from a written file's own
//! Parquet footer.
//!
//! CLAUDE.md Fase 5 "Iceberg V3 — Column Statistics estendidas" originally framed
//! this as a V3-only requirement; verified against the spec directly (see ADR) that
//! these fields are optional at v1/v2/v3 alike — AI-Lake just never populated them at
//! any version, so Spark/Trino/DuckDB reading AI-Lake tables as plain Iceberg got zero
//! row-group pruning from them despite the underlying Parquet files already carrying
//! real statistics (parquet-rs's default `EnabledStatistics::Page`).
//!
//! For `Int32`/`Int64`/`Float`/`Double`/`Boolean`/`ByteArray`/`FixedLenByteArray` —
//! every physical type parquet-rs's `ArrowWriter` ever emits — Parquet's own min/max
//! byte encoding (little-endian for numerics, raw bytes for byte arrays) is identical
//! to Iceberg's single-value serialization (spec Appendix D), so bounds are reused
//! as-is with no re-encoding, only cross-row-group reduction and, for byte arrays,
//! truncation to keep manifest size bounded.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use bytes::Bytes;
use parquet::basic::Type as PhysicalType;
use parquet::file::statistics::Statistics;
use serde::{Deserialize, Serialize};

/// Max bytes kept for a truncated string/binary bound — matches the convention used
/// by Java/PyIceberg writers to keep manifest size bounded regardless of how long the
/// widest value in a text column is.
const BOUND_TRUNCATE_LEN: usize = 16;

/// One column's aggregated stats across all row groups in a file, keyed by Iceberg
/// field id (`column_index + 1` — see `writer.rs`'s `field_id` convention; Parquet's
/// row-group column order mirrors the Arrow schema order that convention assumes).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FieldStats {
    pub value_count: i64,
    pub null_count: i64,
    pub column_size: i64,
    /// base64-encoded Iceberg single-value-serialization bytes (spec Appendix D).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower_bound_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper_bound_b64: Option<String>,
}

#[derive(Default)]
struct Accum {
    value_count: i64,
    null_count: i64,
    column_size: i64,
    physical_type: Option<PhysicalType>,
    lower: Option<Vec<u8>>,
    upper: Option<Vec<u8>>,
}

/// Extract per-column stats from a just-written file's own Parquet footer.
///
/// `skip_column_names` excludes columns that shouldn't get bounds — the vector
/// column(s): large binary blobs with no pruning value and real risk of bloating the
/// manifest. Returns `None` (never fails the write) if the footer can't be parsed;
/// callers fall back to the pre-existing null encoding, which is always spec-valid.
pub fn extract_column_stats(
    file_bytes: &Bytes,
    skip_column_names: &[&str],
) -> Option<BTreeMap<i32, FieldStats>> {
    let meta = parquet::file::footer::parse_metadata(file_bytes).ok()?;
    let schema_descr = meta.file_metadata().schema_descr();
    let mut acc: BTreeMap<i32, Accum> = BTreeMap::new();

    for rg in meta.row_groups() {
        for (idx, col) in rg.columns().iter().enumerate() {
            let name = schema_descr.column(idx).name().to_string();
            if skip_column_names.contains(&name.as_str()) {
                continue;
            }
            let field_id = (idx as i32) + 1;
            let entry = acc.entry(field_id).or_default();
            entry.value_count += col.num_values();
            entry.column_size += col.compressed_size();
            if let Some(stats) = col.statistics() {
                entry.null_count += stats.null_count() as i64;
                if stats.has_min_max_set() {
                    merge_bounds(entry, stats);
                }
            }
        }
    }

    Some(acc.into_iter().map(|(id, a)| (id, finish(a))).collect())
}

fn merge_bounds(entry: &mut Accum, stats: &Statistics) {
    let ty = stats.physical_type();
    entry.physical_type = Some(ty);
    let (min, max) = (stats.min_bytes(), stats.max_bytes());

    entry.lower = Some(match &entry.lower {
        Some(cur) if cmp_stat_bytes(ty, cur, min) != Ordering::Greater => cur.clone(),
        _ => min.to_vec(),
    });
    entry.upper = Some(match &entry.upper {
        Some(cur) if cmp_stat_bytes(ty, cur, max) != Ordering::Less => cur.clone(),
        _ => max.to_vec(),
    });
}

/// Compares two Iceberg-encoded bound values of the same physical type. Numeric types
/// decode their little-endian bytes for a real numeric comparison (raw byte-lexical
/// order does NOT match numeric order for little-endian ints/floats); byte arrays
/// compare lexicographically, which matches Iceberg's own string/binary/decimal
/// ordering. Malformed (wrong-length) bytes compare `Equal` — a merge no-op — rather
/// than panicking; a single corrupt row group degrades to a slightly looser bound
/// instead of failing the write.
fn cmp_stat_bytes(ty: PhysicalType, a: &[u8], b: &[u8]) -> Ordering {
    match ty {
        PhysicalType::BOOLEAN => a.first().cmp(&b.first()),
        PhysicalType::INT32 => match (le_i32(a), le_i32(b)) {
            (Some(x), Some(y)) => x.cmp(&y),
            _ => Ordering::Equal,
        },
        PhysicalType::INT64 => match (le_i64(a), le_i64(b)) {
            (Some(x), Some(y)) => x.cmp(&y),
            _ => Ordering::Equal,
        },
        PhysicalType::FLOAT => match (le_f32(a), le_f32(b)) {
            (Some(x), Some(y)) => x.total_cmp(&y),
            _ => Ordering::Equal,
        },
        PhysicalType::DOUBLE => match (le_f64(a), le_f64(b)) {
            (Some(x), Some(y)) => x.total_cmp(&y),
            _ => Ordering::Equal,
        },
        PhysicalType::BYTE_ARRAY | PhysicalType::FIXED_LEN_BYTE_ARRAY => a.cmp(b),
        // INT96 is a deprecated legacy timestamp encoding arrow-rs's ArrowWriter never
        // emits; treated as incomparable rather than guessed at.
        PhysicalType::INT96 => Ordering::Equal,
    }
}

fn le_i32(b: &[u8]) -> Option<i32> {
    Some(i32::from_le_bytes(b.try_into().ok()?))
}
fn le_i64(b: &[u8]) -> Option<i64> {
    Some(i64::from_le_bytes(b.try_into().ok()?))
}
fn le_f32(b: &[u8]) -> Option<f32> {
    Some(f32::from_le_bytes(b.try_into().ok()?))
}
fn le_f64(b: &[u8]) -> Option<f64> {
    Some(f64::from_le_bytes(b.try_into().ok()?))
}

fn finish(a: Accum) -> FieldStats {
    use base64::Engine;
    let is_byte_array = matches!(
        a.physical_type,
        Some(PhysicalType::BYTE_ARRAY) | Some(PhysicalType::FIXED_LEN_BYTE_ARRAY)
    );
    let lower_bound_b64 = a.lower.as_deref().map(|raw| {
        let bytes = if is_byte_array {
            truncate_lower(raw)
        } else {
            raw.to_vec()
        };
        base64::engine::general_purpose::STANDARD.encode(bytes)
    });
    // Truncating the upper bound can fail (all-0xFF byte run) — in that rare case
    // omit it rather than emit a bound that could exclude real matches.
    let upper_bound_b64 = a.upper.as_deref().and_then(|raw| {
        let bytes = if is_byte_array {
            truncate_upper(raw)?
        } else {
            raw.to_vec()
        };
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    });
    FieldStats {
        value_count: a.value_count,
        null_count: a.null_count,
        column_size: a.column_size,
        lower_bound_b64,
        upper_bound_b64,
    }
}

/// A prefix of a byte string always compares `<=` the original under lexicographic
/// order, so truncating a lower bound never invalidates it.
fn truncate_lower(bytes: &[u8]) -> Vec<u8> {
    if bytes.len() <= BOUND_TRUNCATE_LEN {
        bytes.to_vec()
    } else {
        bytes[..BOUND_TRUNCATE_LEN].to_vec()
    }
}

/// A prefix alone can compare `<` the original, which would make it an invalid upper
/// bound — increments the last non-0xFF byte after truncating (dropping any trailing
/// 0xFF bytes) so the result is guaranteed `>=` every value the true bound covered.
/// Returns `None` only when every byte in the truncated prefix is 0xFF (no safe
/// truncated value exists) — callers omit the bound in that case.
fn truncate_upper(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() <= BOUND_TRUNCATE_LEN {
        return Some(bytes.to_vec());
    }
    let mut truncated = bytes[..BOUND_TRUNCATE_LEN].to_vec();
    while let Some(&last) = truncated.last() {
        if last < 0xFF {
            *truncated.last_mut().unwrap() += 1;
            return Some(truncated);
        }
        truncated.pop();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_lower_keeps_short_bytes_unchanged() {
        assert_eq!(truncate_lower(b"short"), b"short".to_vec());
    }

    #[test]
    fn truncate_lower_cuts_to_max_len() {
        let long = vec![b'a'; 40];
        assert_eq!(truncate_lower(&long).len(), BOUND_TRUNCATE_LEN);
        assert_eq!(truncate_lower(&long), vec![b'a'; BOUND_TRUNCATE_LEN]);
    }

    #[test]
    fn truncate_upper_keeps_short_bytes_unchanged() {
        assert_eq!(truncate_upper(b"short"), Some(b"short".to_vec()));
    }

    #[test]
    fn truncate_upper_increments_last_byte_after_cut() {
        let mut long = vec![b'a'; BOUND_TRUNCATE_LEN];
        long.extend_from_slice(b"zzzz"); // anything beyond the cut is irrelevant
        let got = truncate_upper(&long).unwrap();
        // Truncated prefix was all b'a' (0x61); incrementing the last byte gives 0x62.
        let mut expected = vec![b'a'; BOUND_TRUNCATE_LEN];
        *expected.last_mut().unwrap() = b'a' + 1;
        assert_eq!(got, expected);
        // Sanity: result must sort >= the original (real property this exists for).
        assert!(got.as_slice() >= long.as_slice());
    }

    #[test]
    fn truncate_upper_none_when_prefix_is_all_0xff() {
        let mut long = vec![0xFFu8; BOUND_TRUNCATE_LEN];
        long.push(0x00); // beyond cut, forces the truncation path
        assert_eq!(truncate_upper(&long), None);
    }

    #[test]
    fn truncate_upper_drops_trailing_0xff_before_incrementing() {
        let mut prefix = vec![b'a'; BOUND_TRUNCATE_LEN - 2];
        prefix.push(0xFF);
        prefix.push(0xFF);
        let mut long = prefix.clone();
        long.push(0x00); // beyond cut, forces the truncation path
        let got = truncate_upper(&long).unwrap();
        // Both trailing 0xFF bytes must be dropped, then the last real byte incremented.
        let mut expected = vec![b'a'; BOUND_TRUNCATE_LEN - 2];
        *expected.last_mut().unwrap() = b'a' + 1;
        assert_eq!(got, expected);
    }

    #[test]
    fn cmp_stat_bytes_int32_is_numeric_not_lexicographic() {
        // 1i32 LE = [1,0,0,0]; 256i32 LE = [0,1,0,0] — lexicographically [0,1,0,0] <
        // [1,0,0,0], but numerically 256 > 1. A byte-lexicographic compare would get
        // this backwards; the real (little-endian-aware) compare must not.
        let one = 1i32.to_le_bytes();
        let two_fifty_six = 256i32.to_le_bytes();
        assert_eq!(
            cmp_stat_bytes(PhysicalType::INT32, &one, &two_fifty_six),
            Ordering::Less
        );
    }

    #[test]
    fn cmp_stat_bytes_byte_array_is_lexicographic() {
        assert_eq!(
            cmp_stat_bytes(PhysicalType::BYTE_ARRAY, b"apple", b"banana"),
            Ordering::Less
        );
    }

    #[test]
    fn extract_column_stats_skips_vector_column_and_populates_others() {
        use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
        use ailake_file::AilakeFileWriter;
        use arrow_array::{Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let policy = VectorStoragePolicy {
            column_name: "embedding".into(),
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
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![10i32, 20, 30]))],
        )
        .unwrap();
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let bytes = AilakeFileWriter::new(policy)
            .write(&batch, &embeddings)
            .unwrap();

        let stats = extract_column_stats(&bytes, &["embedding"]).unwrap();

        // field_id 1 = "id" column; field_id 2 would be "embedding" — must be absent.
        assert!(stats.contains_key(&1));
        assert!(
            !stats.contains_key(&2),
            "vector column must be excluded from stats"
        );

        let id_stats = &stats[&1];
        assert_eq!(id_stats.value_count, 3);
        assert_eq!(id_stats.null_count, 0);

        use base64::Engine;
        let lower = base64::engine::general_purpose::STANDARD
            .decode(id_stats.lower_bound_b64.as_ref().unwrap())
            .unwrap();
        let upper = base64::engine::general_purpose::STANDARD
            .decode(id_stats.upper_bound_b64.as_ref().unwrap())
            .unwrap();
        assert_eq!(i32::from_le_bytes(lower.try_into().unwrap()), 10);
        assert_eq!(i32::from_le_bytes(upper.try_into().unwrap()), 30);
    }
}
