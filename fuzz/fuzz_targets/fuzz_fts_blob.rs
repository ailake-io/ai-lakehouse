#![no_main]

use libfuzzer_sys::fuzz_target;

use ailake_fts::searcher::FtsSearcher;

fuzz_target!(|data: &[u8]| {
    let _ = FtsSearcher::from_blob(data);
});
