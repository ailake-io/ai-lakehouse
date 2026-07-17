#![no_main]

use libfuzzer_sys::fuzz_target;

use ailake_file::reader::AilakeFileReader;
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    let bytes = Bytes::copy_from_slice(data);
    let reader = AilakeFileReader::new(bytes, "embedding", 4);
    let _ = reader.verify_integrity();
    let _ = reader.ailk_offset();
    let _ = reader.read_header();
    let _ = reader.get_centroid();
    let _ = reader.load_any_index();
    if let Ok(offset) = reader.ailk_offset() {
        if offset < u64::MAX {
            let _ = reader.load_index();
        }
    }
});
