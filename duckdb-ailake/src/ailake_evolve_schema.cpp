// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_evolve_schema(table_path, add_columns_json, rename_columns_json) → INTEGER
//
// Applies a metadata-only schema evolution to an AI-Lake table.
// Returns the new schema_id on success, -1 on any error.
//
// Parameters:
//   table_path          VARCHAR — table root path / URI
//   add_columns_json    VARCHAR — JSON array: [{"name":"col","type":"string","initial_default":null}]
//   rename_columns_json VARCHAR — JSON array: [{"from":"old_name","to":"new_name"}]
//
// Either add_columns_json or rename_columns_json may be '[]' or '' to skip.
//
// Example:
//   SELECT ailake_evolve_schema(
//       'file:///data/my_table',
//       '[{"name":"score","type":"float","initial_default":0.0}]',
//       '[{"from":"old_col","to":"new_col"}]'
//   );

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeEvolveSchemaExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v   = args.data[0].GetValue(0);
    auto add_cols_v     = args.data[1].GetValue(0);
    auto rename_cols_v  = args.data[2].GetValue(0);

    if (table_path_v.IsNull()) {
        result.SetValue(0, Value::INTEGER(-1));
        return;
    }

    if (!lib.is_evolve_ready()) {
        result.SetValue(0, Value::INTEGER(-1));
        return;
    }

    std::string warehouse       = StringValue::Get(table_path_v);
    std::string add_cols_json   = add_cols_v.IsNull()    ? "[]" : StringValue::Get(add_cols_v);
    std::string rename_cols_json = rename_cols_v.IsNull() ? "[]" : StringValue::Get(rename_cols_v);

    int32_t schema_id = lib.evolve_schema(warehouse, "table", add_cols_json, rename_cols_json);
    result.SetValue(0, Value::INTEGER(schema_id));
}

void RegisterAilakeEvolveSchema(duckdb::ExtensionLoader &loader) {
    ScalarFunction fn(
        "ailake_evolve_schema",
        {LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::INTEGER,
        AilakeEvolveSchemaExec
    );
    loader.RegisterFunction( fn);
}
