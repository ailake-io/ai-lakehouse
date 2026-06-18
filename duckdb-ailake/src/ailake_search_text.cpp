// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_search_text(table_path, query_text, top_k) → TABLE(row_id BIGINT, distance FLOAT, file_path VARCHAR)
//
// Pure BM25 full-text search — no embedding required.
// O(N) per call — intended for small/medium tables or offline ranking.
// Requires BM25 stats accumulated at write time via
// TableWriter(bm25_text_column=...) in Python or writer.with_bm25() in Rust.
//
// table_path   VARCHAR   — AI-Lake table root
// query_text   VARCHAR   — text query to score against
// top_k        INTEGER   — number of results (default 10)
//
// Optional named parameters:
// text_column      VARCHAR   default 'chunk_text'
// partition_filter VARCHAR   default '' (no filter)
//
// Returns: distance = negated BM25 score (lower = more relevant, consistent with
//          vector search convention). ORDER BY distance ASC = most relevant first.
//
// Example:
//   SELECT * FROM ailake_search_text(
//       'file:///data/my_table',
//       'rust programming language',
//       10,
//       text_column := 'chunk_text'
//   ) ORDER BY distance;

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Bind data ─────────────────────────────────────────────────────────────────

struct AilakeSearchTextBindData : public TableFunctionData {
    std::string warehouse;
    std::string table_name      = "table";
    std::string query_text;
    int         top_k           = 10;
    std::string text_column     = "chunk_text";
    std::string partition_filter;
};

// ── Global state ──────────────────────────────────────────────────────────────

struct AilakeSearchTextGlobalState : public GlobalTableFunctionState {
    std::vector<ailake::SearchRow> results;
    idx_t position = 0;

    idx_t MaxThreads() const override { return 1; }
};

// ── Bind ──────────────────────────────────────────────────────────────────────

static unique_ptr<FunctionData> AilakeSearchTextBind(
    ClientContext                &context,
    TableFunctionBindInput       &input,
    vector<LogicalType>          &return_types,
    vector<string>               &names
) {
    auto data = make_uniq<AilakeSearchTextBindData>();

    // arg 0: table_path VARCHAR
    data->warehouse = StringValue::Get(input.inputs[0]);

    // arg 1: query_text VARCHAR
    data->query_text = StringValue::Get(input.inputs[1]);
    if (data->query_text.empty()) {
        throw InvalidInputException("ailake_search_text: query_text must not be empty");
    }

    // arg 2: top_k INTEGER
    data->top_k = IntegerValue::Get(input.inputs[2]);
    if (data->top_k <= 0) {
        throw InvalidInputException("ailake_search_text: top_k must be > 0");
    }

    // named args (optional)
    for (auto &named : input.named_parameters) {
        if (named.first == "text_column") {
            if (!named.second.IsNull())
                data->text_column = StringValue::Get(named.second);
        } else if (named.first == "partition_filter") {
            if (!named.second.IsNull())
                data->partition_filter = StringValue::Get(named.second);
        } else if (named.first == "table_name") {
            data->table_name = StringValue::Get(named.second);
        }
    }

    return_types = {LogicalType::BIGINT, LogicalType::FLOAT, LogicalType::VARCHAR};
    names        = {"row_id", "distance", "file_path"};

    return std::move(data);
}

// ── Init ──────────────────────────────────────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> AilakeSearchTextInit(
    ClientContext          &context,
    TableFunctionInitInput &input
) {
    auto &bind  = input.bind_data->Cast<AilakeSearchTextBindData>();
    auto  state = make_uniq<AilakeSearchTextGlobalState>();

    auto &lib = ailake::AilakeLib::get();
    if (!lib.is_search_text_ready()) {
        // Graceful degradation — return empty result set.
        return std::move(state);
    }

    state->results = lib.search_text(
        bind.warehouse,
        bind.table_name,
        bind.query_text,
        bind.top_k,
        bind.text_column,
        bind.partition_filter
    );

    return std::move(state);
}

// ── Scan ──────────────────────────────────────────────────────────────────────

static void AilakeSearchTextScan(
    ClientContext      &context,
    TableFunctionInput &data_p,
    DataChunk          &output
) {
    auto &state = data_p.global_state->Cast<AilakeSearchTextGlobalState>();

    if (state.position >= state.results.size()) {
        output.SetCardinality(0);
        return;
    }

    idx_t count = MinValue<idx_t>(
        static_cast<idx_t>(STANDARD_VECTOR_SIZE),
        static_cast<idx_t>(state.results.size()) - state.position
    );

    auto *row_ids   = FlatVector::GetData<int64_t>(output.data[0]);
    auto *distances = FlatVector::GetData<float>(output.data[1]);

    for (idx_t i = 0; i < count; i++) {
        const auto &row  = state.results[state.position + i];
        row_ids[i]   = row.row_id;
        distances[i] = row.distance;
        FlatVector::GetData<string_t>(output.data[2])[i] =
            StringVector::AddString(output.data[2], row.file_path);
    }

    state.position += count;
    output.SetCardinality(count);
}

// ── Registration ──────────────────────────────────────────────────────────────

void RegisterAilakeSearchText(duckdb::DatabaseInstance &db) {
    TableFunction func(
        "ailake_search_text",
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER},
        AilakeSearchTextScan,
        AilakeSearchTextBind,
        AilakeSearchTextInit
    );

    func.named_parameters["text_column"]      = LogicalType::VARCHAR;
    func.named_parameters["partition_filter"] = LogicalType::VARCHAR;
    func.named_parameters["table_name"]       = LogicalType::VARCHAR;

    ExtensionUtil::RegisterFunction(db, func);
}
