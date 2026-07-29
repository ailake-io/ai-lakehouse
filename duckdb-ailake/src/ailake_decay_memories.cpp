// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_decay_memories(table_path, lambda
//     [, namespace, table_name, catalog_opts_json]) → BIGINT
//
// Recomputes recency_weight = exp(-lambda * days_since_access) from the
// last_accessed_at column across every data file, and commits a new snapshot
// via SnapshotOperation::Overwrite. Requires ailake.vector-dim / .vector-column
// / .vector-metric table properties (any table written with a vector column
// has them).
//
// Returns the number of files updated, -1 on error.
//
// Parameters:
//   table_path          VARCHAR — table root path/URI
//   lambda               FLOAT   — decay rate
//   namespace            VARCHAR default 'default'
//   table_name           VARCHAR default 'table'
//   catalog_opts_json    VARCHAR default '' — REST catalog config, see
//                         docs/guides/REST_CATALOG.md
//
// Example:
//   SELECT ailake_decay_memories('file:///data/agent_memory', 0.1);

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeDecayMemoriesExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v = args.data[0].GetValue(0);
    auto lambda_v       = args.data[1].GetValue(0);

    if (table_path_v.IsNull() || lambda_v.IsNull()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }
    if (!lib.is_decay_memories_ready()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    std::string warehouse = StringValue::Get(table_path_v);
    float       lambda     = FloatValue::Get(lambda_v);

    std::string ns          = "default";
    std::string table_name  = "table";
    std::string catalog_opts_json;

    if ((idx_t)args.data.size() > 2 && !args.data[2].GetValue(0).IsNull())
        ns                 = StringValue::Get(args.data[2].GetValue(0));
    if ((idx_t)args.data.size() > 3 && !args.data[3].GetValue(0).IsNull())
        table_name         = StringValue::Get(args.data[3].GetValue(0));
    if ((idx_t)args.data.size() > 4 && !args.data[4].GetValue(0).IsNull())
        catalog_opts_json  = StringValue::Get(args.data[4].GetValue(0));

    int64_t files_updated = lib.decay_memories(warehouse, table_name, lambda, ns, catalog_opts_json);
    result.SetValue(0, Value::BIGINT(files_updated));
}

void RegisterAilakeDecayMemories(duckdb::ExtensionLoader &loader) {
    ScalarFunctionSet fn_set("ailake_decay_memories");

    // Arity 2: (table_path, lambda) — all defaults
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::FLOAT},
        LogicalType::BIGINT,
        AilakeDecayMemoriesExec
    ));

    // Arity 3: + namespace VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::FLOAT, LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeDecayMemoriesExec
    ));

    // Arity 4: + table_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::FLOAT, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeDecayMemoriesExec
    ));

    // Arity 5: + catalog_opts_json VARCHAR — REST catalog config, see
    // docs/guides/REST_CATALOG.md. Empty/omitted = default Hadoop catalog.
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::FLOAT, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeDecayMemoriesExec
    ));

    loader.RegisterFunction(fn_set);
}
