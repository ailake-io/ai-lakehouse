// SPDX-License-Identifier: MIT OR Apache-2.0
//! Serialize / deserialize a Tantivy `ManagedDirectory` to a self-contained zstd-compressed blob.
//!
//! Blob layout:
//!   MAGIC(4) | version(2 LE) | flags(2 LE) | num_files(4 LE)
//!   | [file_table: (name_len(4) name_bytes offset(8 LE) length(8 LE))*]
//!   | zstd_compressed_payload

use ailake_core::{AilakeError, AilakeResult};
use std::path::PathBuf;
use tantivy::Directory;

pub const BLOB_MAGIC: [u8; 4] = *b"AFTS";
const BLOB_VERSION: u16 = 1;
const FLAG_ZSTD: u16 = 0x0001;
/// Upper bound for a single FTS index payload after decompression.
/// Maximum uncompressed Tantivy payload accepted from a file (64 MiB).
/// This keeps a single corrupt or hostile index from forcing gigabytes of RAM.
pub const MAX_FTS_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Serialize all managed files from `dir` into a zstd-compressed blob.
///
/// `dir` is `Index::directory()` — a `ManagedDirectory` that tracks all live segment files.
pub fn dir_to_blob(dir: &tantivy::directory::ManagedDirectory) -> AilakeResult<Vec<u8>> {
    let managed = dir.list_managed_files();
    let mut paths: Vec<PathBuf> = managed.into_iter().collect();
    paths.sort();

    let mut file_entries: Vec<(String, Vec<u8>)> = Vec::with_capacity(paths.len());
    for path in &paths {
        let name = path.to_string_lossy().into_owned();
        let data = dir
            .atomic_read(path)
            .map_err(|e| AilakeError::Fts(format!("read '{name}': {e}")))?;
        file_entries.push((name, data));
    }

    // Build uncompressed payload with per-file offsets
    let mut payload: Vec<u8> = Vec::new();
    let mut table_entries: Vec<(String, u64, u64)> = Vec::with_capacity(file_entries.len());
    for (name, data) in &file_entries {
        let off = payload.len() as u64;
        let len = data.len() as u64;
        payload.extend_from_slice(data);
        table_entries.push((name.clone(), off, len));
    }

    let compressed = zstd::encode_all(&payload[..], 3)
        .map_err(|e| AilakeError::Fts(format!("zstd compress: {e}")))?;

    // Serialize header + file table
    let mut table_bytes: Vec<u8> = Vec::new();
    for (name, off, len) in &table_entries {
        let nb = name.as_bytes();
        table_bytes.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        table_bytes.extend_from_slice(nb);
        table_bytes.extend_from_slice(&off.to_le_bytes());
        table_bytes.extend_from_slice(&len.to_le_bytes());
    }

    let mut blob: Vec<u8> = Vec::with_capacity(12 + table_bytes.len() + compressed.len());
    blob.extend_from_slice(&BLOB_MAGIC);
    blob.extend_from_slice(&BLOB_VERSION.to_le_bytes());
    blob.extend_from_slice(&FLAG_ZSTD.to_le_bytes());
    blob.extend_from_slice(&(table_entries.len() as u32).to_le_bytes());
    blob.extend_from_slice(&table_bytes);
    blob.extend_from_slice(&compressed);

    Ok(blob)
}

/// Reconstruct a `RamDirectory` from a blob produced by `dir_to_blob`.
pub fn blob_to_ram_dir(blob: &[u8]) -> AilakeResult<tantivy::directory::RamDirectory> {
    if blob.len() < 12 {
        return Err(AilakeError::Fts("FTS blob too small".into()));
    }
    if blob[0..4] != BLOB_MAGIC {
        return Err(AilakeError::Fts(format!(
            "bad FTS magic: {:?}",
            &blob[0..4]
        )));
    }
    let version = u16::from_le_bytes([blob[4], blob[5]]);
    if version != BLOB_VERSION {
        return Err(AilakeError::Fts(format!(
            "unsupported FTS blob version: {version}"
        )));
    }
    let flags = u16::from_le_bytes([blob[6], blob[7]]);
    if flags & !FLAG_ZSTD != 0 {
        return Err(AilakeError::Fts(format!(
            "unsupported FTS blob flags: {flags:#06x}"
        )));
    }
    let num_files = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]) as usize;
    // Guard against crafted blobs that would cause excessive allocation.
    const MAX_FTS_FILES: usize = 65_536;
    if num_files > MAX_FTS_FILES {
        return Err(AilakeError::Fts(format!(
            "FTS blob claims {num_files} files (max {MAX_FTS_FILES})"
        )));
    }

    let mut pos = 12usize;
    let mut entries: Vec<(String, u64, u64)> = Vec::with_capacity(num_files);
    let mut expected_payload_len = 0u64;
    for _ in 0..num_files {
        if pos + 4 > blob.len() {
            return Err(AilakeError::Fts("truncated file table".into()));
        }
        let nl = u32::from_le_bytes(blob[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        // Use checked arithmetic to prevent overflow in the bounds expression.
        let end = pos
            .checked_add(nl)
            .and_then(|v| v.checked_add(16))
            .ok_or_else(|| AilakeError::Fts("file table entry length overflow".into()))?;
        if end > blob.len() {
            return Err(AilakeError::Fts("truncated filename or offsets".into()));
        }
        let name = std::str::from_utf8(&blob[pos..pos + nl])
            .map_err(|_| AilakeError::Fts("filename not UTF-8".into()))?
            .to_string();
        pos += nl;
        let off = u64::from_le_bytes(blob[pos..pos + 8].try_into().unwrap());
        let len = u64::from_le_bytes(blob[pos + 8..pos + 16].try_into().unwrap());
        pos += 16;
        expected_payload_len = expected_payload_len.max(
            off.checked_add(len)
                .ok_or_else(|| AilakeError::Fts("file table payload range overflow".into()))?,
        );
        entries.push((name, off, len));
    }

    if expected_payload_len > MAX_FTS_PAYLOAD_BYTES {
        return Err(AilakeError::Fts(format!(
            "FTS payload claims {expected_payload_len} bytes (max {MAX_FTS_PAYLOAD_BYTES})"
        )));
    }
    let expected_payload_len: usize = expected_payload_len
        .try_into()
        .map_err(|_| AilakeError::Fts("FTS payload length does not fit this platform".into()))?;
    let payload = if flags & FLAG_ZSTD != 0 {
        zstd::bulk::decompress(&blob[pos..], expected_payload_len)
            .map_err(|e| AilakeError::Fts(format!("zstd decompress: {e}")))?
    } else {
        blob[pos..].to_vec()
    };
    if payload.len() != expected_payload_len {
        return Err(AilakeError::Fts(format!(
            "FTS payload length mismatch: expected {expected_payload_len}, got {}",
            payload.len()
        )));
    }

    let dir = tantivy::directory::RamDirectory::create();
    for (name, off, len) in entries {
        let s: usize = off
            .try_into()
            .map_err(|_| AilakeError::Fts(format!("file '{name}' offset overflow")))?;
        let e: usize = s
            .checked_add(
                len.try_into()
                    .map_err(|_| AilakeError::Fts(format!("file '{name}' length overflow")))?,
            )
            .ok_or_else(|| AilakeError::Fts(format!("file '{name}' offset+length overflow")))?;
        if e > payload.len() {
            return Err(AilakeError::Fts(format!("file '{name}' out of bounds")));
        }
        dir.atomic_write(&PathBuf::from(&name), &payload[s..e])
            .map_err(|e| AilakeError::Fts(format!("write '{name}': {e}")))?;
    }

    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(num_files: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&BLOB_MAGIC);
        b.extend_from_slice(&BLOB_VERSION.to_le_bytes());
        b.extend_from_slice(&FLAG_ZSTD.to_le_bytes());
        b.extend_from_slice(&num_files.to_le_bytes());
        b
    }

    #[test]
    fn rejects_too_many_files() {
        let mut blob = make_header(65_537);
        blob.extend_from_slice(&[0u8; 64]);
        let err = blob_to_ram_dir(&blob).unwrap_err();
        assert!(
            err.to_string().contains("65537"),
            "error should mention count: {err}"
        );
    }

    #[test]
    fn rejects_truncated_blob() {
        let blob = make_header(1); // claims 1 file but no file table follows
        let err = blob_to_ram_dir(&blob).unwrap_err();
        assert!(err.to_string().contains("truncated"));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut blob = vec![0u8; 12];
        blob[0..4].copy_from_slice(b"XXXX");
        let err = blob_to_ram_dir(&blob).unwrap_err();
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_unsupported_blob_version_and_flags() {
        let mut blob = make_header(0);
        blob[4..6].copy_from_slice(&(BLOB_VERSION + 1).to_le_bytes());
        let err = blob_to_ram_dir(&blob).unwrap_err();
        assert!(err.to_string().contains("unsupported FTS blob version"));

        let mut blob = make_header(0);
        blob[6..8].copy_from_slice(&(FLAG_ZSTD | 0x0002).to_le_bytes());
        let err = blob_to_ram_dir(&blob).unwrap_err();
        assert!(err.to_string().contains("unsupported FTS blob flags"));
    }

    #[test]
    fn rejects_zstd_payload_larger_than_file_table_claim() {
        let mut blob = make_header(1);
        let name = b"index.bin";
        blob.extend_from_slice(&(name.len() as u32).to_le_bytes());
        blob.extend_from_slice(name);
        blob.extend_from_slice(&0u64.to_le_bytes());
        blob.extend_from_slice(&1u64.to_le_bytes());
        let oversized = vec![0u8; 1024 * 1024];
        blob.extend_from_slice(&zstd::encode_all(&oversized[..], 1).unwrap());

        let err = blob_to_ram_dir(&blob).unwrap_err();
        assert!(err.to_string().contains("zstd decompress"));
    }

    #[test]
    fn rejects_filename_len_overflow() {
        let mut blob = make_header(1);
        // nl = u32::MAX — pos + nl overflows
        blob.extend_from_slice(&u32::MAX.to_le_bytes());
        blob.extend_from_slice(&[0u8; 64]); // dummy remainder
        let err = blob_to_ram_dir(&blob).unwrap_err();
        assert!(
            err.to_string().contains("truncated") || err.to_string().contains("overflow"),
            "expected truncated/overflow error: {err}"
        );
    }
}
