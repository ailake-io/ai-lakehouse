// SPDX-License-Identifier: MIT OR Apache-2.0
//! Loom models for concurrent patterns in AI-Lake.
//!
//! Run: `LOOM_MAX_BRANCHES=10000 cargo test --features loom -p ailake-query -- loom_`

use loom::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

/// Models the per-key lock acquisition pattern used at `ailake-jni/src/lib.rs`
/// (lines 787, 1149, 1409, 2208, 2355, 2456: look up/insert a per-table `Mutex` in a
/// shared `HashMap`, then lock it) — proves that acquiring a per-key lock from a
/// shared map doesn't deadlock under concurrent access to the same or different keys.
///
/// **Known gap, not modeled here:** the real call sites take this `std::sync::Mutex`
/// and then call `rt().block_on(async { ... })` *while holding it* — blocking a
/// thread on a shared Tokio runtime from inside a std (non-async-aware) lock. Loom
/// models thread interleavings, not async runtime scheduling, so it cannot detect
/// runtime starvation from this pattern. Tracked as an open risk in
/// `docs/architecture/THREAT_MODEL.md`.
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

/// Generic hand-rolled "check flag, take lock, init, set flag" pattern — 2 threads
/// racing to initialize a shared value exactly once.
///
/// **Known gap, not modeled here:** this does NOT correspond to any real call site.
/// Actual once-init in this codebase (e.g. `ailake-jni/src/lib.rs:58,195`,
/// `ailake-index/src/hardware.rs:75`) uses `std::sync::OnceLock::get_or_init`, a
/// std-verified primitive, not this hand-rolled `AtomicBool` + `Mutex<Option<_>>`
/// combination. Kept as a general sanity check of the pattern in isolation, not as
/// evidence that any specific production call site is race-free.
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

/// Models the part-counter increment at `ailake-query/src/writer.rs:387`
/// (`part_counter.fetch_add(1, Ordering::SeqCst)`) — same `Ordering::SeqCst` as the
/// real code (not `Relaxed`), so the interleavings Loom explores here match what the
/// real counter actually does.
#[test]
fn writer_batch_counter_model() {
    loom::model(|| {
        let counter = Arc::new(AtomicU32::new(0));
        let mut threads = Vec::new();

        for _ in 0..2 {
            let c = counter.clone();
            threads.push(thread::spawn(move || {
                c.fetch_add(1, Ordering::SeqCst);
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        let final_val = counter.load(Ordering::SeqCst);
        assert_eq!(final_val, 2, "counter lost updates: got {final_val}");
    });
}
