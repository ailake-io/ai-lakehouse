#![no_main]

use libfuzzer_sys::fuzz_target;

use ailake_file::footer::{AilakeHeader, AilakeTrailer, HEADER_SIZE, TRAILER_SIZE};
use ailake_file::parquet_footer_start;

fuzz_target!(|data: &[u8]| {
    if data.len() >= HEADER_SIZE {
        let arr: [u8; HEADER_SIZE] = data[..HEADER_SIZE].try_into().unwrap();
        let _ = AilakeHeader::from_bytes(&arr);
    }
    if data.len() >= TRAILER_SIZE {
        let arr: [u8; TRAILER_SIZE] = data[..TRAILER_SIZE].try_into().unwrap();
        let _ = AilakeTrailer::from_bytes(&arr);
    }
    let _ = parquet_footer_start(data);
});
