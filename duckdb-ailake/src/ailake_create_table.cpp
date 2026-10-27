// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_create_table(table_path, dim[, vector_column, metric, precision,
//                      format_version, hnsw_m, hnsw_ef_construction,
//                      pre_normalize, modality, partition_by, partition_value,
//                      partition_column_type, partition_fields_json,
//                      fts_columns, fts_tokenizer, embedding_model,
//                      namespace, table_name]) → BOOLEAN
//
// Creates an empty AI-Lake/Iceberg table with the given schema and policy.
// Returns true on success, throws InvalidInputException on error.
//
// Parameters:
//   table_path              VARCHAR  — table root path/URI
//   dim                     INTEGER  — vector dimension
//   vector_column           VARCHAR  default 'embedding'
//   metric                  VARCHAR  default 'cosine'
//   precision               VARCHAR  default 'f16'
//   format_version          INTEGER  default 2 (2 or 3)
//   hnsw_m                  INTEGER  default -1 (use native default)
//   hnsw_ef_construction    INTEGER  default -1 (use native default)
//   pre_normalize           BOOLEAN  default false
//   modality                VARCHAR  default ''
//   partition_by            VARCHAR  default ''
//   partition_value         VARCHAR  default ''
//   partition_column_type   VARCHAR  default ''
//   partition_fields_json   VARCHAR  default ''
//   fts_columns             VARCHAR  default ''
//   fts_tokenizer           VARCHAR  default ''
//   embedding_model         VARCHAR  default ''
//   namespace               VARCHAR  default 'default'
//   table_name              VARCHAR  default 'table'
//
// Example:
//   SELECT ailake_create_table('file:///data/my_table', 1536);
//   SELECT ailake_create_table('file:///data/my_table', 768,
//       vector_column := 'image_embedding', metric := 'euclidean');

#include "ailake_extension.hpp"
#include "duckdb_compat.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeCreateTableExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v = args.data[0].GetValue(0);
    auto dim_v        = args.data[1].GetValue(0);

    if (table_path_v.IsNull() || dim_v.IsNull()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }
    if (!lib.is_create_table_ready()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }

    std::string warehouse       = StringValue::Get(table_path_v);
    int         dim             = IntegerValue::Get(dim_v);

    std::string vector_column       = "embedding";
    std::string metric              = "cosine";
    std::string precision           = "f16";
    int         format_version      = 2;
    int         hnsw_m              = -1;
    int         hnsw_ef_construction = -1;
    bool        pre_normalize       = false;
    std::string modality            = "";
    std::string partition_by        = "";
    std::string partition_value     = "";
    std::string partition_column_type = "";
    std::string partition_fields_json  = "";
    std::string fts_columns         = "";
    std::string fts_tokenizer       = "";
    std::string embedding_model     = "";
    std::string ns                  = "default";
    std::string table_name          = "table";

    if ((idx_t)args.data.size() >  2 && !args.data[ 2].GetValue(0).IsNull())
        vector_column       = StringValue::Get(args.data[ 2].GetValue(0));
    if ((idx_t)args.data.size() >  3 && !args.data[ 3].GetValue(0).IsNull())
        metric              = StringValue::Get(args.data[ 3].GetValue(0));
    if ((idx_t)args.data.size() >  4 && !args.data[ 4].GetValue(0).IsNull())
        precision           = StringValue::Get(args.data[ 4].GetValue(0));
    if ((idx_t)args.data.size() >  5 && !args.data[ 5].GetValue(0).IsNull())
        format_version      = IntegerValue::Get(args.data[ 5].GetValue(0));
    if ((idx_t)args.data.size() >  6 && !args.data[ 6].GetValue(0).IsNull())
        hnsw_m              = IntegerValue::Get(args.data[ 6].GetValue(0));
    if ((idx_t)args.data.size() >  7 && !args.data[ 7].GetValue(0).IsNull())
        hnsw_ef_construction = IntegerValue::Get(args.data[ 7].GetValue(0));
    if ((idx_t)args.data.size() >  8 && !args.data[ 8].GetValue(0).IsNull())
        pre_normalize       = BooleanValue::Get(args.data[ 8].GetValue(0));
    if ((idx_t)args.data.size() >  9 && !args.data[ 9].GetValue(0).IsNull())
        modality            = StringValue::Get(args.data[ 9].GetValue(0));
    if ((idx_t)args.data.size() > 10 && !args.data[10].GetValue(0).IsNull())
        partition_by        = StringValue::Get(args.data[10].GetValue(0));
    if ((idx_t)args.data.size() > 11 && !args.data[11].GetValue(0).IsNull())
        partition_value     = StringValue::Get(args.data[11].GetValue(0));
    if ((idx_t)args.data.size() > 12 && !args.data[12].GetValue(0).IsNull())
        partition_column_type = StringValue::Get(args.data[12].GetValue(0));
    if ((idx_t)args.data.size() > 13 && !args.data[13].GetValue(0).IsNull())
        partition_fields_json  = StringValue::Get(args.data[13].GetValue(0));
    if ((idx_t)args.data.size() > 14 && !args.data[14].GetValue(0).IsNull())
        fts_columns         = StringValue::Get(args.data[14].GetValue(0));
    if ((idx_t)args.data.size() > 15 && !args.data[15].GetValue(0).IsNull())
        fts_tokenizer       = StringValue::Get(args.data[15].GetValue(0));
    if ((idx_t)args.data.size() > 16 && !args.data[16].GetValue(0).IsNull())
        embedding_model     = StringValue::Get(args.data[16].GetValue(0));
    if ((idx_t)args.data.size() > 17 && !args.data[17].GetValue(0).IsNull())
        ns                  = StringValue::Get(args.data[17].GetValue(0));
    if ((idx_t)args.data.size() > 18 && !args.data[18].GetValue(0).IsNull())
        table_name          = StringValue::Get(args.data[18].GetValue(0));
    // arity 20: catalog_opts_json VARCHAR — REST catalog config, see
    // docs/guides/REST_CATALOG.md. Empty/omitted = default Hadoop catalog.
    std::string catalog_opts_json;
    if ((idx_t)args.data.size() > 19 && !args.data[19].GetValue(0).IsNull())
        catalog_opts_json   = StringValue::Get(args.data[19].GetValue(0));

    bool ok = lib.create_table(
        warehouse, ns, table_name, vector_column, dim, metric, precision,
        format_version, hnsw_m, hnsw_ef_construction, pre_normalize,
        modality, partition_by, partition_value, partition_column_type,
        partition_fields_json, fts_columns, fts_tokenizer, embedding_model,
        catalog_opts_json
    );
    result.SetValue(0, Value::BOOLEAN(ok));
}

void RegisterAilakeCreateTable(duckdb::ExtensionLoader &loader) {
#if AILAKE_DUCKDB_V2
    // DuckDB 2 makes scalar-function arguments positional-only when they are
    // registered from LogicalType vectors; named FunctionParameters keep the
    // documented `name := value` SQL form working (same as ailake_compact).
    ScalarFunction function(
        Identifier("ailake_create_table"),
        {
            FunctionParameter(Identifier("table_path"), LogicalType::VARCHAR),
            FunctionParameter(Identifier("dim"), LogicalType::INTEGER),
            FunctionParameter(Identifier("vector_column"), LogicalType::VARCHAR, Value("embedding")),
            FunctionParameter(Identifier("metric"), LogicalType::VARCHAR, Value("cosine")),
            FunctionParameter(Identifier("precision"), LogicalType::VARCHAR, Value("f16")),
            FunctionParameter(Identifier("format_version"), LogicalType::INTEGER, Value::INTEGER(2)),
            FunctionParameter(Identifier("hnsw_m"), LogicalType::INTEGER, Value::INTEGER(-1)),
            FunctionParameter(Identifier("hnsw_ef_construction"), LogicalType::INTEGER, Value::INTEGER(-1)),
            FunctionParameter(Identifier("pre_normalize"), LogicalType::BOOLEAN, Value::BOOLEAN(false)),
            FunctionParameter(Identifier("modality"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("partition_by"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("partition_value"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("partition_column_type"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("partition_fields_json"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("fts_columns"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("fts_tokenizer"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("embedding_model"), LogicalType::VARCHAR, Value("")),
            FunctionParameter(Identifier("namespace"), LogicalType::VARCHAR, Value("default")),
            FunctionParameter(Identifier("table_name"), LogicalType::VARCHAR, Value("table")),
            FunctionParameter(Identifier("catalog_opts_json"), LogicalType::VARCHAR, Value("")),
        },
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    );
    AilakeMarkFallible(function);
    loader.RegisterFunction(function);
#else
    ScalarFunctionSet fn_set("ailake_create_table");

    // All arities from 2 to 19 share the same Exec function.
    // Arity 2: (table_path, dim) — bare minimum
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));

    // Arity 3: + vector_column VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    // Arity 19: full signature with namespace + table_name
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));
    // Arity 20: + catalog_opts_json VARCHAR — REST catalog config, see
    // docs/guides/REST_CATALOG.md.
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
         LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeCreateTableExec
    ));

    AilakeMarkFallible(fn_set);

    loader.RegisterFunction(fn_set);
#endif
}
