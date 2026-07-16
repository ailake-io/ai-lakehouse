// SPDX-License-Identifier: MIT OR Apache-2.0
//! Loom models for concurrent patterns in AI-Lake.
//!
//! Run: `LOOM_MAX_BRANCHES=10000 cargo test --features loom -p ailake-query -- loom_`

use loom::sync::{Arc, Mutex};
use loom::thread;
use loom::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// JNI table-lock pattern: 2 threads, 2 keys.
#[test]
fn jni_table_lock_model() {
    loom::model(|| {
        let global: Arc<Mutex<std::collections::HashMap<u64, Arc<Mutex<()>>>>> =
            Arc::new(Mutex::new(std::collections::HashMap::new()));
        let keys: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));

        let mut threads = Vec::new();
        for _ in 0..2 {
            let global = global.clone();
            let keys = keys.clone();
            threads.push(thread::spawn(move || {
                let k = (keys.fetch_add(1, Ordering::Relaxed) % 2) as u64;
                let lock = {
                    let mut map = global.lock().unwrap();
                    map.entry(k)
                        .or_insert_with(|| Arc::new(Mutex::new(())))
                        .clone()
                };
                let _guard = lock.lock().unwrap();
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
    });
}

/// Once-init flag pattern: 2 threads, AtomicBool guard.
#[test]
fn once_init_flag_model() {
    loom::model(|| {
        let ready = Arc::new(AtomicBool::new(false));
        let value: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
        let mut threads = Vec::new();

        for _ in 0..2 {
            let ready = ready.clone();
            let value = value.clone();
            threads.push(thread::spawn(move || {
                if !ready.load(Ordering::Acquire) {
                    let mut v = value.lock().unwrap();
                    if v.is_none() {
                        *v = Some(42);
                        ready.store(true, Ordering::Release);
                    }
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        let v = value.lock().unwrap();
        assert_eq!(*v, Some(42));
    });
}

/// Batch counter: 2 threads, AtomicU32 relaxado.
#[test]
fn writer_batch_counter_model() {
    loom::model(|| {
        let counter = Arc::new(AtomicU32::new(0));
        let mut threads = Vec::new();

        for _ in 0..2 {
            let c = counter.clone();
            threads.push(thread::spawn(move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        let final_val = counter.load(Ordering::Relaxed);
        assert_eq!(final_val, 2, "counter lost updates: got {final_val}");
    });
}
