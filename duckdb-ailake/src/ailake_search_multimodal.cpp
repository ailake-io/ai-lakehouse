// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_search_multimodal(table_path, queries, top_k)
//   → TABLE(row_id BIGINT, rrf_score FLOAT, file_path VARCHAR)
//
// queries is a STRUCT array: [{col VARCHAR, query FLOAT[], weight DOUBLE}, ...]
//
// Example:
//   SELECT * FROM ailake_search_multimodal(
//       'file:///data/media',
//       [{'col': 'embedding', 'query': [0.1, 0.2, ...]::FLOAT[], 'weight': 0.7},
//        {'col': 'image_embedding', 'query': [0.3, ...]::FLOAT[], 'weight': 0.3}],
//       20
//   ) ORDER BY rrf_score DESC;

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Bind data ─────────────────────────────────────────────────────────────────

struct AilakeMultimodalBindData : public TableFunctionData {
    std::string                        warehouse;
    std::string                        ns         = "default";
    std::string                        table_name = "table";
    std::vector<ailake::ModalQueryArg> queries;
    int                                top_k = 10;
    std::string                        partition_filter;
};

// ── Global state ──────────────────────────────────────────────────────────────

struct AilakeMultimodalGlobalState : public GlobalTableFunctionState {
    std::vector<ailake::MultimodalRow> results;
    idx_t position = 0;

    idx_t MaxThreads() const override { return 1; }
};

// ── Bind ──────────────────────────────────────────────────────────────────────

static unique_ptr<FunctionData> AilakeMultimodalBind(
    ClientContext          &context,
    TableFunctionBindInput &input,
    vector<LogicalType>    &return_types,
    vector<string>         &names
) {
    auto data = make_uniq<AilakeMultimodalBindData>();

    // arg 0: table_path VARCHAR
    data->warehouse = StringValue::Get(input.inputs[0]);

    // arg 1: queries — LIST(STRUCT(col VARCHAR, query FLOAT[], weight DOUBLE))
    if (input.inputs[1].type().id() != LogicalTypeId::LIST) {
        throw InvalidInputException("ailake_search_multimodal: queries must be a LIST of STRUCTs");
    }
    for (auto &elem : ListValue::GetChildren(input.inputs[1])) {
        if (elem.IsNull()) continue;
        if (elem.type().id() != LogicalTypeId::STRUCT) {
            throw InvalidInputException("ailake_search_multimodal: each query must be a STRUCT with {col, query, weight}");
        }
        auto &children = StructValue::GetChildren(elem);
        auto &keys     = StructType::GetChildTypes(elem.type());
        if (children.size() < 2) {
            throw InvalidInputException("ailake_search_multimodal: query STRUCT must have at least {col, query}");
        }

        ailake::ModalQueryArg arg;
        // Resolve by field name (keys may be in any order).
        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string &fname = keys[i].first;
            const Value       &fval  = children[i];
            if (fname == "col") {
                if (fval.type().id() != LogicalTypeId::VARCHAR) {
                    throw InvalidInputException("ailake_search_multimodal: query.col must be VARCHAR");
                }
                arg.col = StringValue::Get(fval);
            } else if (fname == "query") {
                if (fval.type().id() != LogicalTypeId::LIST) {
                    throw InvalidInputException("ailake_search_multimodal: query.query must be FLOAT[]");
                }
                for (auto &f : ListValue::GetChildren(fval)) {
                    if (!f.IsNull()) arg.query.push_back(FloatValue::Get(f));
                }
            } else if (fname == "weight") {
                // Struct type declares weight as DOUBLE, so fval is guaranteed DOUBLE.
                // Fallback handles explicit FLOAT cast or any other numeric type.
                if (fval.type().id() == LogicalTypeId::DOUBLE) {
                    arg.weight = static_cast<float>(DoubleValue::Get(fval));
                } else {
                    try {
                        arg.weight = FloatValue::Get(fval.DefaultCastAs(LogicalType::FLOAT));
                    } catch (...) {
                        arg.weight = 1.0f;
                    }
                }
            }
        }
        if (arg.col.empty() || arg.query.empty()) {
            throw InvalidInputException("ailake_search_multimodal: each STRUCT must have non-empty col and query");
        }
        data->queries.push_back(std::move(arg));
    }
    if (data->queries.empty()) {
        throw InvalidInputException("ailake_search_multimodal: queries must not be empty");
    }

    // arg 2: top_k INTEGER
    data->top_k = IntegerValue::Get(input.inputs[2]);
    if (data->top_k <= 0) {
        throw InvalidInputException("ailake_search_multimodal: top_k must be > 0");
    }

    // named args (optional)
    for (auto &named : input.named_parameters) {
        if (named.first == "partition_filter") {
            if (!named.second.IsNull())
                data->partition_filter = StringValue::Get(named.second);
        } else if (named.first == "table_name") {
            if (!named.second.IsNull())
                data->table_name = StringValue::Get(named.second);
        } else if (named.first == "namespace") {
            if (!named.second.IsNull())
                data->ns = StringValue::Get(named.second);
        }
    }

    return_types = {LogicalType::BIGINT, LogicalType::FLOAT, LogicalType::VARCHAR};
    names        = {"row_id", "rrf_score", "file_path"};

    return std::move(data);
}

// ── Init ──────────────────────────────────────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> AilakeMultimodalInit(
    ClientContext          &context,
    TableFunctionInitInput &input
) {
    auto &bind  = input.bind_data->Cast<AilakeMultimodalBindData>();
    auto  state = make_uniq<AilakeMultimodalGlobalState>();

    auto &lib = ailake::AilakeLib::get();
    if (!lib.is_multimodal_ready()) {
        return std::move(state); // graceful degradation — zero rows
    }

    state->results = lib.search_multimodal(
        bind.warehouse,
        bind.table_name,
        bind.queries,
        bind.top_k,
        bind.partition_filter,
        bind.ns
    );

    return std::move(state);
}

// ── Scan ──────────────────────────────────────────────────────────────────────

static void AilakeMultimodalScan(
    ClientContext      &context,
    TableFunctionInput &data_p,
    DataChunk          &output
) {
    auto &state = data_p.global_state->Cast<AilakeMultimodalGlobalState>();

    if (state.position >= state.results.size()) {
        output.SetCardinality(0);
        return;
    }

    idx_t count = MinValue<idx_t>(
        static_cast<idx_t>(STANDARD_VECTOR_SIZE),
        static_cast<idx_t>(state.results.size()) - state.position
    );

    auto *row_ids    = FlatVector::GetData<int64_t>(output.data[0]);
    auto *rrf_scores = FlatVector::GetData<float>(output.data[1]);

    for (idx_t i = 0; i < count; i++) {
        const auto &row  = state.results[state.position + i];
        row_ids[i]    = row.row_id;
        rrf_scores[i] = row.rrf_score;
        FlatVector::GetData<string_t>(output.data[2])[i] =
            StringVector::AddString(output.data[2], row.file_path);
    }

    state.position += count;
    output.SetCardinality(count);
}

// ── Registration ──────────────────────────────────────────────────────────────

void RegisterAilakeSearchMultimodal(duckdb::ExtensionLoader &loader) {
    // weight is declared DOUBLE so SQL literal 1.0 (DOUBLE) matches without coercion.
    // queries arg: LIST(STRUCT(col VARCHAR, query FLOAT[], weight DOUBLE))
    auto struct_type = LogicalType::STRUCT({
        {"col",    LogicalType::VARCHAR},
        {"query",  LogicalType::LIST(LogicalType::FLOAT)},
        {"weight", LogicalType::DOUBLE},
    });

    TableFunction func(
        "ailake_search_multimodal",
        {LogicalType::VARCHAR, LogicalType::LIST(struct_type), LogicalType::INTEGER},
        AilakeMultimodalScan,
        AilakeMultimodalBind,
        AilakeMultimodalInit
    );

    func.named_parameters["partition_filter"] = LogicalType::VARCHAR;
    func.named_parameters["table_name"]       = LogicalType::VARCHAR;
    func.named_parameters["namespace"]        = LogicalType::VARCHAR;

    loader.RegisterFunction( func);
}
