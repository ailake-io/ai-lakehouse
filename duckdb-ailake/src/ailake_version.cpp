// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_version() → VARCHAR
//
// Returns the ailake-jni crate version statically linked into this extension
// binary (CARGO_PKG_VERSION of ailake-jni), for checking which native build a
// DuckDB session is actually running against.
//
// Unlike every other ailake_* function here, `ailake_version` returns a
// static string (see ailake-jni/src/lib.rs) — no ailake_free_string() call,
// no AilakeLib singleton involvement needed.
//
// Example:
//   SELECT ailake_version();

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"

using namespace duckdb;

extern "C" {
    const char *ailake_version();
}

static void AilakeVersionExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    result.SetVectorType(VectorType::CONSTANT_VECTOR);
    auto data = ConstantVector::GetData<string_t>(result);
    data[0] = StringVector::AddString(result, ailake_version());
}

void RegisterAilakeVersion(duckdb::ExtensionLoader &loader) {
    ScalarFunction fn(
        "ailake_version",
        {},
        LogicalType::VARCHAR,
        AilakeVersionExec
    );
    loader.RegisterFunction(fn);
}
