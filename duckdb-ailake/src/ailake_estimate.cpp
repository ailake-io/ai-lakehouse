// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_estimate(rows, dim [, hnsw_m := 16, pq_m := -1]) →
//     TABLE(mode VARCHAR, vectors_bytes BIGINT, index_bytes BIGINT,
//           total_bytes BIGINT, reduction_vs_f32_hnsw DOUBLE,
//           recall VARCHAR, note VARCHAR)
//
// Pure storage/index-size math (no I/O, no warehouse, no catalog) — the same
// formula as `ailake estimate` (CLI) / ailake-py's estimate(). Returns one
// row per storage/precision mode (F32, F16, I8, F16+IVF-PQ, I8+IVF-PQ,
// PQ-only) so callers can compare footprints without writing any data.
//
// Parameters:
//   rows    BIGINT  — row count to estimate for
//   dim     INTEGER — vector dimension
//   hnsw_m  INTEGER default 16 — HNSW M parameter used for index size estimate
//   pq_m    INTEGER default -1 — IVF-PQ sub-quantizer count (-1 = auto: dim/32, clamped [8, dim])
//
// Example:
//   SELECT * FROM ailake_estimate(100000000, 1536);

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Bind data ─────────────────────────────────────────────────────────────────

struct AilakeEstimateBindData : public TableFunctionData {
    ailake::EstimateResult result;
};

// ── Global state ──────────────────────────────────────────────────────────────

struct AilakeEstimateGlobalState : public GlobalTableFunctionState {
    idx_t position = 0;
    idx_t MaxThreads() const override { return 1; }
};

// ── Bind (fetch data + derive schema) ────────────────────────────────────────

static unique_ptr<FunctionData> AilakeEstimateBind(
    ClientContext                &context,
    TableFunctionBindInput       &input,
    vector<LogicalType>          &return_types,
    vector<string>               &names
) {
    auto data = make_uniq<AilakeEstimateBindData>();

    uint64_t rows = static_cast<uint64_t>(BigIntValue::Get(input.inputs[0]));
    int      dim  = IntegerValue::Get(input.inputs[1]);
    if (dim <= 0) {
        throw InvalidInputException("ailake_estimate: dim must be > 0");
    }

    int hnsw_m = 16;
    int pq_m   = -1;
    for (auto &named : input.named_parameters) {
        if (named.first == "hnsw_m") {
            hnsw_m = IntegerValue::Get(named.second);
        } else if (named.first == "pq_m") {
            pq_m = IntegerValue::Get(named.second);
        }
    }

    names        = {"mode", "vectors_bytes", "index_bytes", "total_bytes",
                     "reduction_vs_f32_hnsw", "recall", "note"};
    return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT,
                     LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::VARCHAR,
                     LogicalType::VARCHAR};

    auto &lib = ailake::AilakeLib::get();
    if (!lib.is_estimate_ready()) {
        return std::move(data);
    }

    data->result = lib.estimate(rows, dim, hnsw_m, pq_m);
    if (!data->result.ok) {
        throw InvalidInputException(
            "ailake_estimate failed: " +
            (data->result.error.empty() ? "unknown error" : data->result.error)
        );
    }

    return std::move(data);
}

// ── Init ──────────────────────────────────────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> AilakeEstimateInit(
    ClientContext          &context,
    TableFunctionInitInput &input
) {
    return make_uniq<AilakeEstimateGlobalState>();
}

// ── Scan (stream rows from pre-computed data) ─────────────────────────────────

static void AilakeEstimateScan(
    ClientContext      &context,
    TableFunctionInput &data_p,
    DataChunk          &output
) {
    auto &state = data_p.global_state->Cast<AilakeEstimateGlobalState>();
    auto &bind  = data_p.bind_data->Cast<AilakeEstimateBindData>();

    const auto &rows  = bind.result.rows;
    const idx_t total = rows.size();

    if (state.position >= total) {
        output.SetCardinality(0);
        return;
    }

    idx_t count = MinValue<idx_t>(
        static_cast<idx_t>(STANDARD_VECTOR_SIZE),
        total - state.position
    );

    for (idx_t i = 0; i < count; ++i) {
        const auto &row = rows[state.position + i];
        output.SetValue(0, i, Value(row.mode));
        output.SetValue(1, i, Value::BIGINT(row.vectors_bytes));
        output.SetValue(2, i, Value::BIGINT(row.index_bytes));
        output.SetValue(3, i, Value::BIGINT(row.total_bytes));
        output.SetValue(4, i, Value::DOUBLE(row.reduction_vs_f32_hnsw));
        output.SetValue(5, i, Value(row.recall));
        output.SetValue(6, i, Value(row.note));
    }

    state.position += count;
    output.SetCardinality(count);
}

// ── Registration ──────────────────────────────────────────────────────────────

void RegisterAilakeEstimate(duckdb::ExtensionLoader &loader) {
    TableFunction func(
        "ailake_estimate",
        {LogicalType::BIGINT, LogicalType::INTEGER},
        AilakeEstimateScan,
        AilakeEstimateBind,
        AilakeEstimateInit
    );

    func.named_parameters["hnsw_m"] = LogicalType::INTEGER;
    func.named_parameters["pq_m"]   = LogicalType::INTEGER;

    loader.RegisterFunction(func);
}
