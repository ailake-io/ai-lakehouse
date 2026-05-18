//! ailake-vec — vector data transformations
//!
//! No I/O. Pure computation: quantization, distance functions, centroid computation.

pub mod compress;
pub mod distance;
pub mod quantize;

pub use compress::BlockCompressor;
pub use distance::{compute_centroid_and_radius, cosine_distance, dot_product, euclidean_distance};
pub use quantize::{Quantizer, ScalingParams};
