#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
Writes a dim=8 AI-Lake fixture for the Spark/Trino/Flink JNI integration tests.

Embedding formula: row i → normalized([i*DIM+1, i*DIM+2, ..., i*DIM+DIM])
This matches the formula expected by AilakeNativeIntegrationTest in both
the Spark and Trino plugins (queryIdx=7 → nearest row must be row 7).

Output: {output_dir}/default/table/  (namespace=default, table=table)

Environment (same resolution order as write_fixture.py):
  AILAKE_LIB        — full path to libailake_jni.so
  AILAKE_NATIVE_LIB — alias for AILAKE_LIB
  AILAKE_LIB_PATH   — directory containing libailake_jni.so

Usage:
    python tests/fixtures/write_jni_fixture.py [output_dir]
    output_dir defaults to ./jni-fixture
"""

import sys
import os
import json
import ctypes
import math
import pathlib

DIM = 8
N = 16

# ── Load library ────────────────────────────────────────────────────────────────

def _load_lib():
    for env in ("AILAKE_LIB", "AILAKE_NATIVE_LIB"):
        explicit = os.environ.get(env)
        if explicit:
            return ctypes.CDLL(explicit)
    lib_dir = os.environ.get("AILAKE_LIB_PATH")
    if lib_dir:
        for name in ("libailake_jni.so", "ailake_jni.dll", "libailake_jni.dylib"):
            candidate = pathlib.Path(lib_dir) / name
            if candidate.exists():
                return ctypes.CDLL(str(candidate))
    try:
        return ctypes.CDLL("libailake_jni.so")
    except OSError:
        return None


lib = _load_lib()
if lib is None:
    print("FAIL: libailake_jni.so not found.")
    print("      Build with: cargo build --release -p ailake-jni")
    print("      Then set AILAKE_LIB=target/release/libailake_jni.so")
    sys.exit(1)

lib.ailake_write_batch_json.argtypes = [ctypes.c_char_p]
lib.ailake_write_batch_json.restype = ctypes.c_void_p
lib.ailake_free_string.argtypes = [ctypes.c_void_p]
lib.ailake_free_string.restype = None


def _call_write(req: dict) -> dict:
    ptr = lib.ailake_write_batch_json(json.dumps(req).encode())
    try:
        return json.loads(ctypes.string_at(ptr).decode())
    finally:
        lib.ailake_free_string(ptr)


def make_embedding(i: int) -> list:
    v = [float(i * DIM + j + 1) for j in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── Write fixture ────────────────────────────────────────────────────────────────

out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./jni-fixture")
out_dir.mkdir(parents=True, exist_ok=True)

warehouse = str(out_dir.resolve())
namespace = "default"
table = "table"

print(f"writing {N} rows (dim={DIM}) to {warehouse}/{namespace}/{table}/")

embeddings = [make_embedding(i) for i in range(N)]
ids = list(range(N))

resp = _call_write({
    "warehouse": warehouse,
    "namespace": namespace,
    "table": table,
    "vec_col": "embedding",
    "dim": DIM,
    "metric": "cosine",
    "precision": "f32",
    "ids": ids,
    "embeddings": embeddings,
})

assert resp.get("ok"), f"write_batch failed: {resp}"
print(f"committed: snapshot_id={resp['snapshot_id']}")
print(f"fixture ready at {out_dir}")
