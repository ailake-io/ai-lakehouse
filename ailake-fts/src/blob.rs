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
    let _version = u16::from_le_bytes([blob[4], blob[5]]);
    let flags = u16::from_le_bytes([blob[6], blob[7]]);
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
        entries.push((name, off, len));
    }

    let payload = if flags & FLAG_ZSTD != 0 {
        zstd::decode_all(&blob[pos..])
            .map_err(|e| AilakeError::Fts(format!("zstd decompress: {e}")))?
    } else {
        blob[pos..].to_vec()
    };

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
