// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_migrate(table_path, old_column, new_column, embed_cmd
//     [, text_column, strategy, batch_size, model_name, model_version,
//        namespace, table_name, catalog_opts_json]) → BOOLEAN
//
// Re-embeds old_column into new_column by running embed_cmd (same
// stdin/stdout JSON protocol as ailake_backfill_vector_column) over
// text_column for every row, then atomically replaces the table's data files
// (strategy = 'atomic-replace', default) or writes new_column alongside
// old_column for a cutover window (strategy = 'dual-write-then-cutover').
//
// Returns true on success, throws InvalidInputException on error.
//
// Parameters:
//   table_path          VARCHAR — table root path/URI
//   old_column           VARCHAR — existing vector column to migrate from
//   new_column           VARCHAR — vector column to migrate to
//   embed_cmd            VARCHAR — shell command implementing the embed protocol
//   text_column          VARCHAR default 'chunk_text'
//   strategy             VARCHAR default 'atomic-replace'
//                         ('atomic-replace' | 'dual-write-then-cutover')
//   batch_size           BIGINT  default 10000
//   model_name           VARCHAR default '' — embedding model metadata (unset if empty)
//   model_version        VARCHAR default ''
//   namespace            VARCHAR default 'default'
//   table_name           VARCHAR default 'table'
//   catalog_opts_json    VARCHAR default '' — REST catalog config, see
//                         docs/guides/REST_CATALOG.md
//
// Example:
//   SELECT ailake_migrate('file:///data/my_table', 'embedding', 'embedding_v2',
//       'python3 embed_v2.py', model_name := 'text-embedding-3-large');

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeMigrateExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v  = args.data[0].GetValue(0);
    auto old_column_v    = args.data[1].GetValue(0);
    auto new_column_v    = args.data[2].GetValue(0);
    auto embed_cmd_v     = args.data[3].GetValue(0);

    if (table_path_v.IsNull() || old_column_v.IsNull() || new_column_v.IsNull() ||
        embed_cmd_v.IsNull()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }
    if (!lib.is_migrate_ready()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }

    std::string warehouse   = StringValue::Get(table_path_v);
    std::string old_column   = StringValue::Get(old_column_v);
    std::string new_column   = StringValue::Get(new_column_v);
    std::string embed_cmd    = StringValue::Get(embed_cmd_v);

    std::string text_column   = "chunk_text";
    std::string strategy      = "atomic-replace";
    int64_t     batch_size    = 10000;
    std::string model_name;
    std::string model_version;
    std::string ns            = "default";
    std::string table_name    = "table";
    std::string catalog_opts_json;

    if ((idx_t)args.data.size() > 4 && !args.data[4].GetValue(0).IsNull())
        text_column        = StringValue::Get(args.data[4].GetValue(0));
    if ((idx_t)args.data.size() > 5 && !args.data[5].GetValue(0).IsNull())
        strategy           = StringValue::Get(args.data[5].GetValue(0));
    if ((idx_t)args.data.size() > 6 && !args.data[6].GetValue(0).IsNull())
        batch_size         = BigIntValue::Get(args.data[6].GetValue(0));
    if ((idx_t)args.data.size() > 7 && !args.data[7].GetValue(0).IsNull())
        model_name         = StringValue::Get(args.data[7].GetValue(0));
    if ((idx_t)args.data.size() > 8 && !args.data[8].GetValue(0).IsNull())
        model_version      = StringValue::Get(args.data[8].GetValue(0));
    if ((idx_t)args.data.size() > 9 && !args.data[9].GetValue(0).IsNull())
        ns                 = StringValue::Get(args.data[9].GetValue(0));
    if ((idx_t)args.data.size() > 10 && !args.data[10].GetValue(0).IsNull())
        table_name         = StringValue::Get(args.data[10].GetValue(0));
    if ((idx_t)args.data.size() > 11 && !args.data[11].GetValue(0).IsNull())
        catalog_opts_json  = StringValue::Get(args.data[11].GetValue(0));

    bool ok = lib.migrate(
        warehouse, table_name, old_column, new_column, embed_cmd, text_column, strategy,
        batch_size, model_name, model_version, ns, catalog_opts_json
    );
    result.SetValue(0, Value::BOOLEAN(ok));
}

void RegisterAilakeMigrate(duckdb::ExtensionLoader &loader) {
    ScalarFunctionSet fn_set("ailake_migrate");

    // Arity 4: (table_path, old_column, new_column, embed_cmd) — all defaults
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 5: + text_column VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 6: + strategy VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 7: + batch_size BIGINT
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 8: + model_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 9: + model_version VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 10: + namespace VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 11: + table_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    // Arity 12: + catalog_opts_json VARCHAR — REST catalog config, see
    // docs/guides/REST_CATALOG.md. Empty/omitted = default Hadoop catalog.
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR,
         LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeMigrateExec
    ));

    loader.RegisterFunction(fn_set);
}
