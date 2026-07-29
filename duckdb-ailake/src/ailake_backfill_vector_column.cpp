// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_backfill_vector_column(table_path, column, text_column, embed_cmd
//     [, batch_size, namespace, table_name, catalog_opts_json]) → BOOLEAN
//
// Populates a vector column previously added via ailake_add_vector_column by
// running embed_cmd (a shell command reading JSON array-of-strings on stdin,
// writing JSON array-of-arrays-of-float on stdout — same protocol as the
// CLI's --embed-cmd) over text_column for every existing row.
//
// Returns true on success, throws InvalidInputException on error (including
// "column not found — call ailake_add_vector_column first").
//
// Parameters:
//   table_path        VARCHAR — table root path/URI
//   column              VARCHAR — vector column to backfill (must already exist)
//   text_column         VARCHAR — source text column for embedding
//   embed_cmd           VARCHAR — shell command implementing the embed protocol
//   batch_size          BIGINT  default 512
//   namespace           VARCHAR default 'default'
//   table_name          VARCHAR default 'table'
//   catalog_opts_json   VARCHAR default '' — REST catalog config, see
//                       docs/guides/REST_CATALOG.md
//
// Example:
//   SELECT ailake_add_vector_column('file:///data/my_table', 'v2', 1536);
//   SELECT ailake_backfill_vector_column('file:///data/my_table',
//       'v2', 'chunk_text', 'python3 embed.py');

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

static void AilakeBackfillVectorColumnExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v  = args.data[0].GetValue(0);
    auto column_v        = args.data[1].GetValue(0);
    auto text_column_v   = args.data[2].GetValue(0);
    auto embed_cmd_v     = args.data[3].GetValue(0);

    if (table_path_v.IsNull() || column_v.IsNull() || text_column_v.IsNull() ||
        embed_cmd_v.IsNull()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }
    if (!lib.is_backfill_vector_column_ready()) {
        result.SetValue(0, Value::BOOLEAN(false));
        return;
    }

    std::string warehouse    = StringValue::Get(table_path_v);
    std::string column        = StringValue::Get(column_v);
    std::string text_column   = StringValue::Get(text_column_v);
    std::string embed_cmd     = StringValue::Get(embed_cmd_v);

    int64_t     batch_size    = 512;
    std::string ns            = "default";
    std::string table_name    = "table";
    std::string catalog_opts_json;

    if ((idx_t)args.data.size() > 4 && !args.data[4].GetValue(0).IsNull())
        batch_size         = BigIntValue::Get(args.data[4].GetValue(0));
    if ((idx_t)args.data.size() > 5 && !args.data[5].GetValue(0).IsNull())
        ns                 = StringValue::Get(args.data[5].GetValue(0));
    if ((idx_t)args.data.size() > 6 && !args.data[6].GetValue(0).IsNull())
        table_name         = StringValue::Get(args.data[6].GetValue(0));
    if ((idx_t)args.data.size() > 7 && !args.data[7].GetValue(0).IsNull())
        catalog_opts_json  = StringValue::Get(args.data[7].GetValue(0));

    bool ok = lib.backfill_vector_column(
        warehouse, table_name, column, text_column, embed_cmd, batch_size, ns,
        catalog_opts_json
    );
    result.SetValue(0, Value::BOOLEAN(ok));
}

void RegisterAilakeBackfillVectorColumn(duckdb::ExtensionLoader &loader) {
    ScalarFunctionSet fn_set("ailake_backfill_vector_column");

    // Arity 4: (table_path, column, text_column, embed_cmd) — all defaults
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeBackfillVectorColumnExec
    ));

    // Arity 5: + batch_size BIGINT
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT},
        LogicalType::BOOLEAN,
        AilakeBackfillVectorColumnExec
    ));

    // Arity 6: + namespace VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeBackfillVectorColumnExec
    ));

    // Arity 7: + table_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeBackfillVectorColumnExec
    ));

    // Arity 8: + catalog_opts_json VARCHAR — REST catalog config, see
    // docs/guides/REST_CATALOG.md. Empty/omitted = default Hadoop catalog.
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
         LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::BOOLEAN,
        AilakeBackfillVectorColumnExec
    ));

    loader.RegisterFunction(fn_set);
}
