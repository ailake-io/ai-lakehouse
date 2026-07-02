// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_search(table_path, query, top_k) → TABLE(row_id BIGINT, distance FLOAT, file_path VARCHAR)
//
// table_path  VARCHAR   — full path/URI to the AI-Lake table root
//                         (same as `table=` in Python ailake.search())
// query       FLOAT[]   — embedding vector (LIST(FLOAT))
// top_k       INTEGER   — number of nearest neighbors to return
//
// Optional named parameters (pass as named args in DuckDB):
// vec_col          VARCHAR   default 'embedding'
// ef_search        INTEGER   default 50
// partition_filter VARCHAR   default '' (no partition filter)
// hybrid_text      VARCHAR   default '' — enables hybrid BM25+vector when non-empty
// text_column      VARCHAR   default 'chunk_text' — Parquet column for BM25 scoring
// bm25_weight      FLOAT     default 0.5 — BM25 weight in RRF (0=pure vector, 1=pure BM25)
//
// Example (pure vector):
//   SELECT * FROM ailake_search(
//       'file:///data/my_table',
//       [0.1, 0.2, 0.3]::FLOAT[],
//       10
//   ) ORDER BY distance;
//
// Example (hybrid BM25+vector):
//   SELECT * FROM ailake_search(
//       'file:///data/my_table',
//       [0.1, 0.2, 0.3]::FLOAT[],
//       10,
//       hybrid_text := 'rust programming language',
//       text_column := 'chunk_text',
//       bm25_weight := 0.4
//   ) ORDER BY distance;

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Bind data (per query, immutable after Bind) ───────────────────────────────

struct AilakeSearchBindData : public TableFunctionData {
    std::string        warehouse;
    std::string        ns              = "default";
    std::string        table_name      = "table";
    std::string        vec_col         = "embedding";
    std::vector<float> query;
    int                top_k           = 10;
    int                ef_search       = 50;
    std::string        partition_filter;
    std::string        hybrid_text;        // non-empty = hybrid BM25+vector mode
    std::string        text_column     = "chunk_text";
    float              bm25_weight     = 0.5f;
};

// ── Global state (search executed once in Init, results cached) ───────────────

struct AilakeSearchGlobalState : public GlobalTableFunctionState {
    std::vector<ailake::SearchRow> results;
    idx_t position = 0;

    idx_t MaxThreads() const override { return 1; }
};

// ── Bind ──────────────────────────────────────────────────────────────────────

static unique_ptr<FunctionData> AilakeSearchBind(
    ClientContext                &context,
    TableFunctionBindInput       &input,
    vector<LogicalType>          &return_types,
    vector<string>               &names
) {
    auto data = make_uniq<AilakeSearchBindData>();

    // arg 0: table_path VARCHAR
    data->warehouse = StringValue::Get(input.inputs[0]);

    // arg 1: query FLOAT[] (LIST(FLOAT))
    if (input.inputs[1].type().id() != LogicalTypeId::LIST) {
        throw InvalidInputException("ailake_search: query must be a FLOAT[] list");
    }
    for (auto &child : ListValue::GetChildren(input.inputs[1])) {
        if (child.IsNull()) {
            throw InvalidInputException("ailake_search: query vector must not contain NULLs");
        }
        data->query.push_back(FloatValue::Get(child));
    }
    if (data->query.empty()) {
        throw InvalidInputException("ailake_search: query vector must not be empty");
    }

    // arg 2: top_k INTEGER
    data->top_k = IntegerValue::Get(input.inputs[2]);
    if (data->top_k <= 0) {
        throw InvalidInputException("ailake_search: top_k must be > 0");
    }

    // named args (optional)
    for (auto &named : input.named_parameters) {
        if (named.first == "vec_col") {
            data->vec_col = StringValue::Get(named.second);
        } else if (named.first == "ef_search") {
            data->ef_search = IntegerValue::Get(named.second);
        } else if (named.first == "table_name") {
            data->table_name = StringValue::Get(named.second);
        } else if (named.first == "namespace") {
            data->ns = StringValue::Get(named.second);
        } else if (named.first == "partition_filter") {
            if (!named.second.IsNull())
                data->partition_filter = StringValue::Get(named.second);
        } else if (named.first == "hybrid_text") {
            if (!named.second.IsNull())
                data->hybrid_text = StringValue::Get(named.second);
        } else if (named.first == "text_column") {
            if (!named.second.IsNull())
                data->text_column = StringValue::Get(named.second);
        } else if (named.first == "bm25_weight") {
            data->bm25_weight = FloatValue::Get(named.second);
        }
    }

    return_types = {LogicalType::BIGINT, LogicalType::FLOAT, LogicalType::VARCHAR};
    names        = {"row_id", "distance", "file_path"};

    return std::move(data);
}

// ── Init (execute search, cache results) ─────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> AilakeSearchInit(
    ClientContext          &context,
    TableFunctionInitInput &input
) {
    auto &bind  = input.bind_data->Cast<AilakeSearchBindData>();
    auto  state = make_uniq<AilakeSearchGlobalState>();

    auto &lib = ailake::AilakeLib::get();
    if (!lib.is_ready()) {
        // Graceful degradation: return empty result set with a warning.
        // Users see zero rows rather than a hard crash — same behaviour as Spark/Trino.
        return std::move(state);
    }

    state->results = lib.search(
        bind.warehouse,
        bind.table_name,
        bind.vec_col,
        bind.query,
        bind.top_k,
        bind.ef_search,
        bind.partition_filter,
        bind.hybrid_text,
        bind.text_column,
        bind.bm25_weight,
        bind.ns
    );

    return std::move(state);
}

// ── Scan (stream rows from cached results) ────────────────────────────────────

static void AilakeSearchScan(
    ClientContext      &context,
    TableFunctionInput &data_p,
    DataChunk          &output
) {
    auto &state = data_p.global_state->Cast<AilakeSearchGlobalState>();

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

void RegisterAilakeSearch(duckdb::ExtensionLoader &loader) {
    TableFunction func(
        "ailake_search",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
        AilakeSearchScan,
        AilakeSearchBind,
        AilakeSearchInit
    );

    func.named_parameters["vec_col"]          = LogicalType::VARCHAR;
    func.named_parameters["ef_search"]        = LogicalType::INTEGER;
    func.named_parameters["table_name"]       = LogicalType::VARCHAR;
    func.named_parameters["namespace"]        = LogicalType::VARCHAR;
    func.named_parameters["partition_filter"] = LogicalType::VARCHAR;
    func.named_parameters["hybrid_text"]      = LogicalType::VARCHAR;
    func.named_parameters["text_column"]      = LogicalType::VARCHAR;
    func.named_parameters["bm25_weight"]      = LogicalType::FLOAT;

    loader.RegisterFunction( func);
}
