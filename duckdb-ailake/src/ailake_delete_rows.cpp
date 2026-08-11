// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_delete_rows(table_path, file, row_ids UINTEGER[]
//                     [, namespace, table_name, catalog_opts_json]) → BOOLEAN
//
// Position/DV delete: marks the given row_ids as deleted within a single
// named data file (path as returned by ailake_search's file_path column).
// Different from ailake_delete_where, which is an equality delete by column
// value across the whole table.
//
// Returns true on success, throws InvalidInputException on a real backend
// error.
//
// Parameters:
//   table_path         VARCHAR      — table root path/URI
//   file                VARCHAR      — data file path (relative to warehouse)
//   row_ids             UINTEGER[]   — row positions within that file
//   namespace           VARCHAR      default 'default'
//   table_name          VARCHAR      default 'table'
//   catalog_opts_json   VARCHAR      default '' — REST catalog config, see
//                                     docs/guides/REST_CATALOG.md
//
// Example:
//   SELECT ailake_delete_rows('file:///data/my_table',
//       'data/part-00001.parquet', [3, 7, 12]::UINTEGER[]);

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeDeleteRowsExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v = args.data[0].GetValue(0);
    auto file_v        = args.data[1].GetValue(0);
    auto row_ids_v      = args.data[2].GetValue(0);

    if (table_path_v.IsNull() || file_v.IsNull() || row_ids_v.IsNull()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }
    if (!lib.is_delete_rows_ready()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }

    std::string warehouse = StringValue::Get(table_path_v);
    std::string file       = StringValue::Get(file_v);

    std::vector<uint32_t> row_ids;
    for (auto &child : ListValue::GetChildren(row_ids_v)) {
        if (!child.IsNull())
            row_ids.push_back(UIntegerValue::Get(child));
    }

    std::string ns          = "default";
    std::string table_name  = "table";
    std::string catalog_opts_json;

    if ((idx_t)args.data.size() > 3 && !args.data[3].GetValue(0).IsNull())
        ns                 = StringValue::Get(args.data[3].GetValue(0));
    if ((idx_t)args.data.size() > 4 && !args.data[4].GetValue(0).IsNull())
        table_name         = StringValue::Get(args.data[4].GetValue(0));
    if ((idx_t)args.data.size() > 5 && !args.data[5].GetValue(0).IsNull())
        catalog_opts_json  = StringValue::Get(args.data[5].GetValue(0));

    bool ok = lib.delete_rows(warehouse, table_name, file, row_ids, ns, catalog_opts_json);
    result.SetValue(0, Value::BOOLEAN(ok));
}

void RegisterAilakeDeleteRows(duckdb::ExtensionLoader &loader) {
    ScalarFunctionSet fn_set("ailake_delete_rows");

    // Arity 3: (table_path, file, row_ids) — defaults for the rest
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::UINTEGER)},
        LogicalType::BOOLEAN,
        AilakeDeleteRowsExec
    ));

    // Arity 4: + namespace VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::UINTEGER),
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeDeleteRowsExec
    ));

    // Arity 5: + table_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::UINTEGER),
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeDeleteRowsExec
    ));

    // Arity 6: + catalog_opts_json VARCHAR — REST catalog config, see
    // docs/guides/REST_CATALOG.md. Empty/omitted = default Hadoop catalog.
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::UINTEGER),
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeDeleteRowsExec
    ));

    loader.RegisterFunction(fn_set);
}
