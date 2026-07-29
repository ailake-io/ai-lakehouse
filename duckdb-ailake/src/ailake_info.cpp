// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_info(table_path [, namespace := 'default', table_name := 'table',
//              catalog_opts_json := '']) →
//     TABLE(table VARCHAR, location VARCHAR, vector_column VARCHAR,
//           vector_dim VARCHAR, vector_metric VARCHAR, files BIGINT,
//           indexed_files BIGINT, failed_files BIGINT, foreign_files BIGINT,
//           foreign_file_paths VARCHAR[], rows BIGINT, size_bytes BIGINT,
//           snapshot_id BIGINT)
//
// Reports table metadata: current snapshot, file/row/size counts, index
// status breakdown, and "foreign" files — files written by a generic
// Iceberg engine (Spark/Trino OPTIMIZE, DuckDB) with no AI-Lake
// centroid/HNSW. Every query against a foreign file degrades to an O(N)
// flat scan until `ailake_compact` repairs it. Mirrors `ailake info
// --format json` exactly (same fields, same source data).
//
// snapshot_id is NULL when the table has never been committed to.
//
// Example:
//   SELECT * FROM ailake_info('file:///data/my_table');
//   SELECT foreign_files, foreign_file_paths FROM ailake_info('file:///data/my_table')
//     WHERE foreign_files > 0;

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Bind data ─────────────────────────────────────────────────────────────────

struct AilakeInfoBindData : public TableFunctionData {
    ailake::InfoResult result;
};

// ── Global state ──────────────────────────────────────────────────────────────

struct AilakeInfoGlobalState : public GlobalTableFunctionState {
    bool  emitted = false;
    idx_t MaxThreads() const override { return 1; }
};

// ── Bind (fetch data at bind time — single-row result) ──────────────────────

static unique_ptr<FunctionData> AilakeInfoBind(
    ClientContext                &context,
    TableFunctionBindInput       &input,
    vector<LogicalType>          &return_types,
    vector<string>               &names
) {
    auto data = make_uniq<AilakeInfoBindData>();

    std::string warehouse = StringValue::Get(input.inputs[0]);
    std::string ns         = "default";
    std::string table_name = "table";
    std::string catalog_opts_json;

    for (auto &named : input.named_parameters) {
        if (named.first == "namespace") {
            ns = StringValue::Get(named.second);
        } else if (named.first == "table_name") {
            table_name = StringValue::Get(named.second);
        } else if (named.first == "catalog_opts_json") {
            if (!named.second.IsNull())
                catalog_opts_json = StringValue::Get(named.second);
        }
    }

    names        = {"table", "location", "vector_column", "vector_dim", "vector_metric",
                     "files", "indexed_files", "failed_files", "foreign_files",
                     "foreign_file_paths", "rows", "size_bytes", "snapshot_id"};
    return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
                     LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT,
                     LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
                     LogicalType::LIST(LogicalType::VARCHAR), LogicalType::BIGINT,
                     LogicalType::BIGINT, LogicalType::BIGINT};

    auto &lib = ailake::AilakeLib::get();
    if (!lib.is_info_ready()) {
        return std::move(data);
    }

    data->result = lib.info(warehouse, table_name, ns, catalog_opts_json);
    if (!data->result.ok) {
        throw InvalidInputException(
            "ailake_info failed: " +
            (data->result.error.empty() ? "unknown error" : data->result.error)
        );
    }

    return std::move(data);
}

// ── Init ──────────────────────────────────────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> AilakeInfoInit(
    ClientContext          &context,
    TableFunctionInitInput &input
) {
    return make_uniq<AilakeInfoGlobalState>();
}

// ── Scan (single row) ────────────────────────────────────────────────────────

static void AilakeInfoScan(
    ClientContext      &context,
    TableFunctionInput &data_p,
    DataChunk          &output
) {
    auto &state = data_p.global_state->Cast<AilakeInfoGlobalState>();
    auto &bind  = data_p.bind_data->Cast<AilakeInfoBindData>();
    if (state.emitted || !bind.result.ok) {
        output.SetCardinality(0);
        return;
    }
    state.emitted = true;

    const auto &r = bind.result;
    output.SetValue(0, 0, Value(r.table));
    output.SetValue(1, 0, Value(r.location));
    output.SetValue(2, 0, Value(r.vector_column));
    output.SetValue(3, 0, Value(r.vector_dim));
    output.SetValue(4, 0, Value(r.vector_metric));
    output.SetValue(5, 0, Value::BIGINT(r.files));
    output.SetValue(6, 0, Value::BIGINT(r.indexed_files));
    output.SetValue(7, 0, Value::BIGINT(r.failed_files));
    output.SetValue(8, 0, Value::BIGINT(r.foreign_files));

    vector<Value> paths;
    paths.reserve(r.foreign_file_paths.size());
    for (auto &p : r.foreign_file_paths) paths.push_back(Value(p));
    output.SetValue(9, 0, Value::LIST(LogicalType::VARCHAR, paths));

    output.SetValue(10, 0, Value::BIGINT(r.rows));
    output.SetValue(11, 0, Value::BIGINT(r.size_bytes));
    output.SetValue(
        12, 0,
        r.has_snapshot_id ? Value::BIGINT(r.snapshot_id) : Value(LogicalType::BIGINT)
    );

    output.SetCardinality(1);
}

// ── Registration ──────────────────────────────────────────────────────────────

void RegisterAilakeInfo(duckdb::ExtensionLoader &loader) {
    TableFunction func(
        "ailake_info",
        {LogicalType::VARCHAR},
        AilakeInfoScan,
        AilakeInfoBind,
        AilakeInfoInit
    );

    func.named_parameters["namespace"]         = LogicalType::VARCHAR;
    func.named_parameters["table_name"]        = LogicalType::VARCHAR;
    func.named_parameters["catalog_opts_json"] = LogicalType::VARCHAR;

    loader.RegisterFunction(func);
}
