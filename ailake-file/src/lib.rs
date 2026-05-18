//! ailake-file — unified file format
//!
//! Owns the combined Parquet + AI-Lake footer file.
//! The single file that Iceberg manifests point to.
//!
//! Layout: [PAR1][row groups][parquet footer][PAR1] [AILK header][centroid][HNSW][AILK trailer]
//!
//! See docs/specs/FILE_FORMAT.md for the binary specification.

pub mod footer;
pub mod reader;
pub mod writer;

pub use footer::{
    AilakeHeader, AilakeTrailer, DistanceMetric, Precision, AILAKE_FORMAT_VERSION, AILAKE_MAGIC,
    HEADER_SIZE, TRAILER_SIZE,
};
pub use reader::AilakeFileReader;
pub use writer::AilakeFileWriter;
