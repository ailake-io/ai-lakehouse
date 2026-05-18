use ailake_core::{VectorMetric, VectorPrecision};
use arrow_schema::{DataType, Field};
use std::collections::HashMap;

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
