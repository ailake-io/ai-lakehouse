#![no_main]

use libfuzzer_sys::fuzz_target;
use std::ffi::{CStr, CString};

unsafe fn call_and_free(ptr: *mut std::ffi::c_char) {
    if !ptr.is_null() {
        let _ = CStr::from_ptr(ptr);
        ailake_jni::ailake_free_string(ptr);
    }
}

fuzz_target!(|data: &[u8]| {
    let prefix: Vec<u8> = data.iter().copied().take_while(|&b| b != 0).collect();
    if prefix.is_empty() {
        unsafe {
            call_and_free(ailake_jni::ailake_search_json(std::ptr::null()));
            call_and_free(ailake_jni::ailake_write_batch_json(std::ptr::null()));
            call_and_free(ailake_jni::ailake_scan_json(std::ptr::null()));
            call_and_free(ailake_jni::ailake_compact_json(std::ptr::null()));
        }
        return;
    }
    let Ok(c_input) = CString::new(prefix) else {
        return;
    };
    let ptr = c_input.as_ptr();

    unsafe {
        call_and_free(ailake_jni::ailake_search_json(ptr));
        call_and_free(ailake_jni::ailake_write_batch_json(ptr));
        call_and_free(ailake_jni::ailake_search_text_json(ptr));
        call_and_free(ailake_jni::ailake_search_multimodal_json(ptr));
        call_and_free(ailake_jni::ailake_scan_json(ptr));
        call_and_free(ailake_jni::ailake_delete_where_json(ptr));
        call_and_free(ailake_jni::ailake_evolve_schema_json(ptr));
        call_and_free(ailake_jni::ailake_compact_json(ptr));
        call_and_free(ailake_jni::ailake_write_batch_multi_json(ptr));
    }
});
