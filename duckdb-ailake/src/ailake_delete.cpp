// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_delete_where(table_path, column, values) → BOOLEAN
//
// Performs an equality-delete on the given column for all listed values.
// Delegates to ailake_delete_where_json via the JNI shared library.
//
// Parameters:
//   table_path  VARCHAR        — table root path / URI
//   column      VARCHAR        — column name to match against
//   values      VARCHAR[]      — list of values to delete (LIST(VARCHAR))
//
// Returns TRUE on success, FALSE on any error or if lib not loaded.
//
// Example:
//   SELECT ailake_delete_where(
//       'file:///data/my_table',
//       'document_id',
//       ['doc-a', 'doc-b', 'doc-c']
//   );

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static std::vector<std::string> extract_varchar_list(const Value &v) {
    std::vector<std::string> out;
    if (v.IsNull()) return out;
    for (const auto &child : ListValue::GetChildren(v)) {
        if (!child.IsNull()) out.push_back(StringValue::Get(child));
    }
    return out;
}

static void AilakeDeleteWhereExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v = args.data[0].GetValue(0);
    auto column_v     = args.data[1].GetValue(0);
    auto values_v     = args.data[2].GetValue(0);

    if (table_path_v.IsNull() || column_v.IsNull() || values_v.IsNull()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }

    if (!lib.is_delete_ready()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }

    std::string warehouse = StringValue::Get(table_path_v);
    std::string column    = StringValue::Get(column_v);
    auto        values    = extract_varchar_list(values_v);

    if (values.empty()) {
        result.SetValue(0, Value::BOOLEAN(true));
        return;
    }

    bool ok = lib.delete_where(warehouse, "table", column, values);
    result.SetValue(0, Value::BOOLEAN(ok));
}

void RegisterAilakeDeleteWhere(duckdb::ExtensionLoader &loader) {
    ScalarFunction fn(
        "ailake_delete_where",
        {LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::VARCHAR)},
        LogicalType::BOOLEAN,
        AilakeDeleteWhereExec
    );
    loader.RegisterFunction( fn);
}
