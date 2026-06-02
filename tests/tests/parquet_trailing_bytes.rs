// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verifies standard Parquet readers ignore the AI-Lake footer.

mod fixtures;

use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_file::AilakeFileWriter;

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

#[test]
fn ailake_file_is_valid_parquet() {
    let dim = 8u32;
    let (batch, embs) = fixtures::generate_batch(10, dim as usize);
    let writer = AilakeFileWriter::new(make_policy(dim));
    let file_bytes = writer.write(&batch, &embs).unwrap();

    // File must start and end with PAR1 — required by all Parquet readers
    assert_eq!(&file_bytes[..4], b"PAR1", "file must start with PAR1");
    assert_eq!(
        &file_bytes[file_bytes.len() - 4..],
        b"PAR1",
        "file must end with PAR1"
    );

    // AILK magic must appear inside (in the embedded AILK section, before the footer)
    let ailk_positions: Vec<usize> = file_bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"AILK")
        .map(|(i, _)| i)
        .collect();
    assert!(!ailk_positions.is_empty(), "AILK magic must appear in file");

    // AILK must NOT be the last 4 bytes (PAR1 must be)
    let last_ailk = *ailk_positions.last().unwrap();
    assert!(
        last_ailk < file_bytes.len() - 4,
        "AILK must not be the last 4 bytes — PAR1 must be last"
    );
}

#[test]
fn pyarrow_ignores_ailake_footer() {
    let dim = 8u32;
    let (batch, embs) = fixtures::generate_batch(5, dim as usize);
    let writer = AilakeFileWriter::new(make_policy(dim));
    let file_bytes = writer.write(&batch, &embs).unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &file_bytes).unwrap();

    let script = format!(
        r#"import pyarrow.parquet as pq; t = pq.read_table('{}'); assert len(t) == 5"#,
        tmp.path().display()
    );
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status()
        .expect("python3 not found");
    assert!(status.success(), "PyArrow failed to read AI-Lake file");
}
