// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_compact(table_path[, min_files, target_size_bytes, max_files_per_pass,
//                 deferred, namespace, table_name]) → BIGINT
//
// Compacts small files in an AI-Lake table into a larger merged file.
// Returns the number of files compacted (0 = nothing eligible), -1 on error
// or if the native library isn't loaded.
//
// Parameters:
//   table_path          VARCHAR — table root path/URI
//   min_files           BIGINT  default 4          — min small files required to trigger
//   target_size_bytes   BIGINT  default 134217728   — target output file size (128 MiB)
//   max_files_per_pass  BIGINT  default 20          — bounds peak RAM / HNSW rebuild cost
//   deferred            BOOLEAN default false       — write merged Parquet immediately,
//                                                       build HNSW index in the background
//   namespace           VARCHAR default 'default'
//   table_name          VARCHAR default 'table'
//
// Example:
//   SELECT ailake_compact('file:///data/my_table', min_files := 2);

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeCompactExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v = args.data[0].GetValue(0);
    if (table_path_v.IsNull()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }
    if (!lib.is_compact_ready()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    std::string warehouse = StringValue::Get(table_path_v);

    int64_t min_files          = -1;
    int64_t target_size_bytes  = -1;
    int64_t max_files_per_pass = -1;
    bool    deferred           = false;
    std::string ns             = "default";
    std::string table_name     = "table";

    if ((idx_t)args.data.size() > 1 && !args.data[1].GetValue(0).IsNull())
        min_files          = BigIntValue::Get(args.data[1].GetValue(0));
    if ((idx_t)args.data.size() > 2 && !args.data[2].GetValue(0).IsNull())
        target_size_bytes  = BigIntValue::Get(args.data[2].GetValue(0));
    if ((idx_t)args.data.size() > 3 && !args.data[3].GetValue(0).IsNull())
        max_files_per_pass = BigIntValue::Get(args.data[3].GetValue(0));
    if ((idx_t)args.data.size() > 4 && !args.data[4].GetValue(0).IsNull())
        deferred           = BooleanValue::Get(args.data[4].GetValue(0));
    if ((idx_t)args.data.size() > 5 && !args.data[5].GetValue(0).IsNull())
        ns                 = StringValue::Get(args.data[5].GetValue(0));
    if ((idx_t)args.data.size() > 6 && !args.data[6].GetValue(0).IsNull())
        table_name         = StringValue::Get(args.data[6].GetValue(0));

    int64_t files_compacted = lib.compact(
        warehouse, table_name, min_files, target_size_bytes, max_files_per_pass, deferred, ns
    );
    result.SetValue(0, Value::BIGINT(files_compacted));
}

void RegisterAilakeCompact(duckdb::ExtensionLoader &loader) {
    ScalarFunctionSet fn_set("ailake_compact");

    // Arity 1: (table_path) — all defaults
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeCompactExec
    ));

    // Arity 2: + min_files BIGINT
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::BIGINT},
        LogicalType::BIGINT,
        AilakeCompactExec
    ));

    // Arity 3: + target_size_bytes BIGINT
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::BIGINT,
        AilakeCompactExec
    ));

    // Arity 4: + max_files_per_pass BIGINT
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::BIGINT,
        AilakeCompactExec
    ));

    // Arity 5: + deferred BOOLEAN
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
         LogicalType::BOOLEAN},
        LogicalType::BIGINT,
        AilakeCompactExec
    ));

    // Arity 6: + namespace VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
         LogicalType::BOOLEAN, LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeCompactExec
    ));

    // Arity 7: + table_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
         LogicalType::BOOLEAN, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeCompactExec
    ));

    loader.RegisterFunction(fn_set);
}
