// SPDX-License-Identifier: MIT OR Apache-2.0
//! SIFT-1M dataset loader.
//!
//! Parses the texmex fvecs/ivecs binary format used by the original SIFT-1M
//! distribution (http://corpus-texmex.irisa.fr/).
//!
//! Wire format (little-endian):
//!   fvecs: [dim: u32][f32 × dim] repeated
//!   ivecs: [dim: u32][i32 × dim] repeated

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::Context;

pub struct Dataset {
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    /// ground_truth[query_idx] = sorted list of true nearest-neighbor row IDs
    pub ground_truth: Vec<Vec<u32>>,
    pub dim: usize,
}

impl Dataset {
    #[allow(dead_code)]
    pub fn base_count(&self) -> usize {
        self.base.len()
    }

    #[allow(dead_code)]
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }
}

/// Load SIFT-1M from `dir`. Expects:
///   sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs
///
/// `limit_base` truncates the base set (useful for quick smoke-tests).
pub fn load(dir: &Path, limit_base: Option<usize>) -> anyhow::Result<Dataset> {
    eprintln!("Loading base vectors from {} …", dir.display());
    let mut base = read_fvecs(&dir.join("sift_base.fvecs")).context("sift_base.fvecs")?;
    let dim = base.first().map(|v| v.len()).unwrap_or(128);

    if let Some(limit) = limit_base {
        base.truncate(limit);
        eprintln!("  truncated to {} vectors", base.len());
    }

    eprintln!("Loading query vectors …");
    let queries = read_fvecs(&dir.join("sift_query.fvecs")).context("sift_query.fvecs")?;

    eprintln!("Loading ground truth …");
    let ground_truth =
        read_ivecs(&dir.join("sift_groundtruth.ivecs")).context("sift_groundtruth.ivecs")?;

    eprintln!(
        "Dataset ready: {} base  |  {} queries  |  {} GT neighbors each",
        base.len(),
        queries.len(),
        ground_truth.first().map(|v| v.len()).unwrap_or(0),
    );

    Ok(Dataset {
        base,
        queries,
        ground_truth,
        dim,
    })
}

fn read_fvecs(path: &Path) -> anyhow::Result<Vec<Vec<f32>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::with_capacity(4 * 1024 * 1024, file);
    let mut vecs = Vec::new();

    loop {
        let dim = match read_u32_le(&mut r) {
            Ok(d) => d as usize,
            Err(e) if is_eof(&e) => break,
            Err(e) => return Err(e),
        };
        let mut bytes = vec![0u8; dim * 4];
        r.read_exact(&mut bytes)
            .with_context(|| format!("reading vector body at index {}", vecs.len()))?;
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        vecs.push(v);
    }
    Ok(vecs)
}

fn read_ivecs(path: &Path) -> anyhow::Result<Vec<Vec<u32>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::with_capacity(4 * 1024 * 1024, file);
    let mut vecs = Vec::new();

    loop {
        let dim = match read_u32_le(&mut r) {
            Ok(d) => d as usize,
            Err(e) if is_eof(&e) => break,
            Err(e) => return Err(e),
        };
        let mut bytes = vec![0u8; dim * 4];
        r.read_exact(&mut bytes)
            .with_context(|| format!("reading ivec body at index {}", vecs.len()))?;
        let v: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32)
            .collect();
        vecs.push(v);
    }
    Ok(vecs)
}

fn read_u32_le(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn is_eof(e: &anyhow::Error) -> bool {
    e.downcast_ref::<std::io::Error>()
        .map(|io| io.kind() == std::io::ErrorKind::UnexpectedEof)
        .unwrap_or(false)
}
