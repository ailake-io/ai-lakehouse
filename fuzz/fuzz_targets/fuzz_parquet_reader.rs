#![no_main]

use libfuzzer_sys::fuzz_target;

use ailake_parquet::ParquetVectorReader;
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    let bytes = Bytes::copy_from_slice(data);
    let reader = ParquetVectorReader::new(bytes, "embedding");
    let _ = reader.record_count();
    let _ = reader.kv_metadata("ailake.footer_offset");
});
