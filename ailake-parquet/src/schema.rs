// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::HashMap;
use std::sync::Arc;

use ailake_core::{VectorMetric, VectorPrecision};
use arrow_schema::{DataType, Field};

pub fn metric_str(m: VectorMetric) -> &'static str {
    match m {
        VectorMetric::Cosine => "cosine",
        VectorMetric::Euclidean => "euclidean",
        VectorMetric::DotProduct => "dotproduct",
    }
}

pub fn precision_str(p: VectorPrecision) -> &'static str {
    match p {
        VectorPrecision::F32 => "f32",
        VectorPrecision::F16 => "f16",
        VectorPrecision::I8 => "i8",
        VectorPrecision::Binary => "binary",
    }
}

/// Build Arrow field for the vector column. Physical type: FixedSizeBinary.
/// Field metadata carries ailake.* for discovery by SDK readers.
pub fn vector_field(
    name: &str,
    dim: u32,
    metric: VectorMetric,
    precision: VectorPrecision,
) -> Field {
    let byte_width = (dim as usize) * precision.bytes_per_element();
    let meta = HashMap::from([
        ("ailake.dim".to_string(), dim.to_string()),
        ("ailake.metric".to_string(), metric_str(metric).to_string()),
        (
            "ailake.precision".to_string(),
            precision_str(precision).to_string(),
        ),
    ]);
    Field::new(name, DataType::FixedSizeBinary(byte_width as i32), false).with_metadata(meta)
}

/// Build Arrow field for a multi-vector column: `List<FixedSizeBinary(bytes_per_vec)>`.
///
/// Physical layout: each row stores a variable-length list of fixed-size embeddings.
/// One document row → N chunk embeddings, no row duplication.
/// Standard Parquet readers see a List column of byte arrays (no error).
pub fn multi_vector_field(
    name: &str,
    dim: u32,
    metric: VectorMetric,
    precision: VectorPrecision,
) -> Field {
    let bytes_per_vec = (dim as usize) * precision.bytes_per_element();
    let inner = Field::new_list_field(DataType::FixedSizeBinary(bytes_per_vec as i32), false);
    let meta = HashMap::from([
        ("ailake.dim".to_string(), dim.to_string()),
        ("ailake.metric".to_string(), metric_str(metric).to_string()),
        (
            "ailake.precision".to_string(),
            precision_str(precision).to_string(),
        ),
        ("ailake.multi_vec".to_string(), "true".to_string()),
    ]);
    Field::new(name, DataType::List(Arc::new(inner)), true).with_metadata(meta)
}
