#![no_main]

use libfuzzer_sys::fuzz_target;

use ailake_catalog::avro_manifest::{
    read_equality_delete_manifest, read_equality_delete_values, read_manifest_file,
    read_manifest_list, read_manifest_list_typed,
};

fuzz_target!(|data: &[u8]| {
    // Cap at 64 KiB — Avro headers may encode huge block counts from tiny
    // inputs, causing OOM. OSS-Fuzz treats OOM as crash; cap here prevents
    // false positives for a resource-exhaustion-only finding.
    if data.len() > 65536 {
        return;
    }
    let _ = read_manifest_file(data);
    let _ = read_manifest_list(data);
    let _ = read_manifest_list_typed(data);
    let _ = read_equality_delete_manifest(data);
    let _ = read_equality_delete_values(data);
});
