//! Verifies standard Parquet readers ignore the AI-Lake footer.

mod fixtures;

use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter};

#[test]
fn ailake_footer_appended_after_par1() {
    let dim = 8u32;
    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
    };
    let (batch, embs) = fixtures::generate_batch(10, dim as usize);
    let writer = AilakeFileWriter::new(policy);
    let file_bytes = writer.write(&batch, &embs).unwrap();

    // File must contain PAR1 (Parquet magic) before the AILK trailer
    let par1_positions: Vec<usize> = file_bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| w == b"PAR1")
        .map(|(i, _)| i)
        .collect();
    // Must have at least 2 PAR1: header and footer of Parquet section
    assert!(par1_positions.len() >= 2, "expected at least 2 PAR1 markers");

    // AILK magic must come AFTER the last PAR1
    let last_par1 = *par1_positions.last().unwrap();
    assert!(last_par1 < file_bytes.len() - 4,
        "PAR1 is the last 4 bytes — AI-Lake footer not appended");
    assert_eq!(&file_bytes[file_bytes.len() - 4..], b"AILK");
}

#[test]
#[ignore = "requires python3 with pyarrow"]
fn pyarrow_ignores_ailake_footer() {
    let dim = 8u32;
    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
    };
    let (batch, embs) = fixtures::generate_batch(5, dim as usize);
    let writer = AilakeFileWriter::new(policy);
    let file_bytes = writer.write(&batch, &embs).unwrap();

    // Write to temp file and verify PyArrow can read it
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
