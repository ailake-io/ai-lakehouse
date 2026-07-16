#![no_main]

use libfuzzer_sys::fuzz_target;

use ailake_index::ivf_pq::IvfPqSerializer;
use ailake_index::mmap_loader::MmapLoader;

fuzz_target!(|data: &[u8]| {
    let _ = MmapLoader::from_bytes(data);
    let _ = IvfPqSerializer::from_bytes(data);
});
