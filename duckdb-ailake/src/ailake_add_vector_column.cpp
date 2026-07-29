// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_add_vector_column(table_path, column, dim
//     [, metric, precision, pre_normalize, hnsw_m, hnsw_ef_construction,
//        namespace, table_name, catalog_opts_json]) → INTEGER
//
// Metadata-only schema evolution: adds a new vector column to an existing
// AI-Lake table (Phase 8 multimodal / migration workflows). Follow up with
// ailake_backfill_vector_column to populate it.
//
// Returns the new schema_id on success, -1 on error.
//
// Parameters:
//   table_path            VARCHAR — table root path/URI
//   column                 VARCHAR — new vector column name
//   dim                    INTEGER — vector dimension
//   metric                 VARCHAR default 'cosine'
//   precision              VARCHAR default 'f16'
//   pre_normalize          BOOLEAN default false
//   hnsw_m                 INTEGER default -1 (use native default)
//   hnsw_ef_construction   INTEGER default -1 (use native default)
//   namespace              VARCHAR default 'default'
//   table_name             VARCHAR default 'table'
//   catalog_opts_json      VARCHAR default '' — REST catalog config, see
//                           docs/guides/REST_CATALOG.md
//
// Example:
//   SELECT ailake_add_vector_column('file:///data/my_table',
//       'image_embedding', 512, metric := 'euclidean');

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeAddVectorColumnExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v = args.data[0].GetValue(0);
    auto column_v       = args.data[1].GetValue(0);
    auto dim_v          = args.data[2].GetValue(0);

    if (table_path_v.IsNull() || column_v.IsNull() || dim_v.IsNull()) {
        result.SetValue(0, Value::INTEGER(-1));
        return;
    }
    if (!lib.is_add_vector_column_ready()) {
        result.SetValue(0, Value::INTEGER(-1));
        return;
    }

    std::string warehouse = StringValue::Get(table_path_v);
    std::string column     = StringValue::Get(column_v);
    int         dim        = IntegerValue::Get(dim_v);

    std::string metric              = "cosine";
    std::string precision           = "f16";
    bool        pre_normalize       = false;
    int         hnsw_m              = -1;
    int         hnsw_ef_construction = -1;
    std::string ns                  = "default";
    std::string table_name          = "table";
    std::string catalog_opts_json;

    if ((idx_t)args.data.size() > 3 && !args.data[3].GetValue(0).IsNull())
        metric              = StringValue::Get(args.data[3].GetValue(0));
    if ((idx_t)args.data.size() > 4 && !args.data[4].GetValue(0).IsNull())
        precision           = StringValue::Get(args.data[4].GetValue(0));
    if ((idx_t)args.data.size() > 5 && !args.data[5].GetValue(0).IsNull())
        pre_normalize       = BooleanValue::Get(args.data[5].GetValue(0));
    if ((idx_t)args.data.size() > 6 && !args.data[6].GetValue(0).IsNull())
        hnsw_m              = IntegerValue::Get(args.data[6].GetValue(0));
    if ((idx_t)args.data.size() > 7 && !args.data[7].GetValue(0).IsNull())
        hnsw_ef_construction = IntegerValue::Get(args.data[7].GetValue(0));
    if ((idx_t)args.data.size() > 8 && !args.data[8].GetValue(0).IsNull())
        ns                  = StringValue::Get(args.data[8].GetValue(0));
    if ((idx_t)args.data.size() > 9 && !args.data[9].GetValue(0).IsNull())
        table_name          = StringValue::Get(args.data[9].GetValue(0));
    if ((idx_t)args.data.size() > 10 && !args.data[10].GetValue(0).IsNull())
        catalog_opts_json   = StringValue::Get(args.data[10].GetValue(0));

    int32_t new_schema_id = lib.add_vector_column(
        warehouse, ns, table_name, column, dim, metric, precision, pre_normalize,
        hnsw_m, hnsw_ef_construction, catalog_opts_json
    );
    result.SetValue(0, Value::INTEGER(new_schema_id));
}

void RegisterAilakeAddVectorColumn(duckdb::ExtensionLoader &loader) {
    ScalarFunctionSet fn_set("ailake_add_vector_column");

    // Arity 3: (table_path, column, dim) — all defaults
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 4: + metric VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 5: + precision VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 6: + pre_normalize BOOLEAN
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::BOOLEAN},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 7: + hnsw_m INTEGER
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::INTEGER},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 8: + hnsw_ef_construction INTEGER
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::INTEGER, LogicalType::INTEGER},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 9: + namespace VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::INTEGER, LogicalType::INTEGER,
         LogicalType::VARCHAR},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 10: + table_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::INTEGER, LogicalType::INTEGER,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    // Arity 11: + catalog_opts_json VARCHAR — REST catalog config, see
    // docs/guides/REST_CATALOG.md. Empty/omitted = default Hadoop catalog.
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::INTEGER, LogicalType::INTEGER,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::INTEGER,
        AilakeAddVectorColumnExec
    ));

    loader.RegisterFunction(fn_set);
}
