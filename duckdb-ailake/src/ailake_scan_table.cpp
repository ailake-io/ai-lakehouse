// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_scan(table_path, query, top_k) → TABLE(<all_parquet_cols>, _distance FLOAT)
//
// Performs vector search + full row fetch in one call. Unlike ailake_search()
// which returns only (row_id, distance, file_path) pointers, ailake_scan()
// returns all Parquet columns alongside _distance — no JOIN required.
//
// The full scan result is fetched at bind time and cached in BindData, so LIMIT
// does not reduce Rust-side I/O: use top_k to control how many rows are fetched.
//
// Example:
//   SELECT id, chunk_text, _distance
//   FROM ailake_scan('file:///data/my_table', [0.1, 0.2, 0.3]::FLOAT[], 10)
//   ORDER BY _distance;

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Bind data ─────────────────────────────────────────────────────────────────

struct AilakeScanBindData : public TableFunctionData {
    // Query params (stored for potential re-execution).
    std::string        warehouse;
    std::string        table_name = "table";
    std::string        vec_col    = "embedding";
    std::vector<float> query;
    int                top_k      = 10;
    int                ef_search  = 50;

    // Pre-fetched result (populated at bind time).
    ailake::ScanResult result;
};

// ── Global state ──────────────────────────────────────────────────────────────

struct AilakeScanGlobalState : public GlobalTableFunctionState {
    idx_t position = 0;
    idx_t MaxThreads() const override { return 1; }
};

// ── Bind (fetch data + derive schema) ────────────────────────────────────────

static unique_ptr<FunctionData> AilakeScanBind(
    ClientContext                &context,
    TableFunctionBindInput       &input,
    vector<LogicalType>          &return_types,
    vector<string>               &names
) {
    auto data = make_uniq<AilakeScanBindData>();

    // arg 0: table_path VARCHAR
    data->warehouse = StringValue::Get(input.inputs[0]);

    // arg 1: query FLOAT[]
    if (input.inputs[1].type().id() != LogicalTypeId::LIST) {
        throw InvalidInputException("ailake_scan: query must be a FLOAT[] list");
    }
    for (auto &child : ListValue::GetChildren(input.inputs[1])) {
        if (child.IsNull()) {
            throw InvalidInputException("ailake_scan: query vector must not contain NULLs");
        }
        data->query.push_back(FloatValue::Get(child));
    }
    if (data->query.empty()) {
        throw InvalidInputException("ailake_scan: query vector must not be empty");
    }

    // arg 2: top_k INTEGER
    data->top_k = IntegerValue::Get(input.inputs[2]);
    if (data->top_k <= 0) {
        throw InvalidInputException("ailake_scan: top_k must be > 0");
    }

    // named args
    for (auto &named : input.named_parameters) {
        if (named.first == "vec_col") {
            data->vec_col = StringValue::Get(named.second);
        } else if (named.first == "ef_search") {
            data->ef_search = IntegerValue::Get(named.second);
        } else if (named.first == "table_name") {
            data->table_name = StringValue::Get(named.second);
        }
    }

    auto &lib = ailake::AilakeLib::get();
    if (!lib.is_scan_ready()) {
        // Graceful degradation: zero-row empty schema.
        return_types = {LogicalType::FLOAT};
        names        = {"_distance"};
        return std::move(data);
    }

    // Fetch everything now — schema is needed to set return_types.
    data->result = lib.scan(
        data->warehouse,
        data->table_name,
        data->vec_col,
        data->query,
        data->top_k,
        data->ef_search
    );

    if (!data->result.ok || data->result.columns.empty()) {
        // Empty result or error: single _distance column so DuckDB has something.
        return_types = {LogicalType::FLOAT};
        names        = {"_distance"};
        return std::move(data);
    }

    // Map ScanColType → DuckDB LogicalType.
    for (auto &col : data->result.columns) {
        names.push_back(col.name);
        switch (col.type) {
            case ailake::ScanColType::INT64:
                return_types.push_back(LogicalType::BIGINT);
                break;
            case ailake::ScanColType::FLOAT32:
                return_types.push_back(LogicalType::FLOAT);
                break;
            case ailake::ScanColType::FLOAT64:
                return_types.push_back(LogicalType::DOUBLE);
                break;
            case ailake::ScanColType::VARCHAR:
                return_types.push_back(LogicalType::VARCHAR);
                break;
            case ailake::ScanColType::BOOL:
                return_types.push_back(LogicalType::BOOLEAN);
                break;
            case ailake::ScanColType::LIST_FLOAT32:
                return_types.push_back(LogicalType::LIST(LogicalType::FLOAT));
                break;
            default:
                return_types.push_back(LogicalType::VARCHAR);
                break;
        }
    }

    return std::move(data);
}

// ── Init ──────────────────────────────────────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> AilakeScanInit(
    ClientContext          &context,
    TableFunctionInitInput &input
) {
    return make_uniq<AilakeScanGlobalState>();
}

// ── Scan (stream rows from pre-fetched data) ──────────────────────────────────

static void AilakeScanScan(
    ClientContext      &context,
    TableFunctionInput &data_p,
    DataChunk          &output
) {
    auto &state = data_p.global_state->Cast<AilakeScanGlobalState>();
    auto &bind  = data_p.bind_data->Cast<AilakeScanBindData>();

    const auto &result = bind.result;
    const idx_t total  = static_cast<idx_t>(result.num_rows);

    if (state.position >= total || result.columns.empty()) {
        output.SetCardinality(0);
        return;
    }

    idx_t count = MinValue<idx_t>(
        static_cast<idx_t>(STANDARD_VECTOR_SIZE),
        total - state.position
    );

    for (idx_t col_idx = 0; col_idx < result.columns.size(); ++col_idx) {
        const auto &sc  = result.columns[col_idx];
        auto       &vec = output.data[col_idx];
        auto       &validity = FlatVector::Validity(vec);

        switch (sc.type) {
            case ailake::ScanColType::INT64: {
                auto *dst = FlatVector::GetData<int64_t>(vec);
                for (idx_t i = 0; i < count; ++i) {
                    idx_t src = state.position + i;
                    if (sc.is_null[src]) {
                        validity.SetInvalid(i);
                    } else {
                        dst[i] = sc.int_vals[src];
                    }
                }
                break;
            }
            case ailake::ScanColType::FLOAT32: {
                auto *dst = FlatVector::GetData<float>(vec);
                for (idx_t i = 0; i < count; ++i) {
                    idx_t src = state.position + i;
                    if (sc.is_null[src]) {
                        validity.SetInvalid(i);
                    } else {
                        dst[i] = sc.float_vals[src];
                    }
                }
                break;
            }
            case ailake::ScanColType::FLOAT64: {
                auto *dst = FlatVector::GetData<double>(vec);
                for (idx_t i = 0; i < count; ++i) {
                    idx_t src = state.position + i;
                    if (sc.is_null[src]) {
                        validity.SetInvalid(i);
                    } else {
                        dst[i] = sc.double_vals[src];
                    }
                }
                break;
            }
            case ailake::ScanColType::VARCHAR: {
                for (idx_t i = 0; i < count; ++i) {
                    idx_t src = state.position + i;
                    if (sc.is_null[src]) {
                        validity.SetInvalid(i);
                    } else {
                        FlatVector::GetData<string_t>(vec)[i] =
                            StringVector::AddString(vec, sc.str_vals[src]);
                    }
                }
                break;
            }
            case ailake::ScanColType::BOOL: {
                auto *dst = FlatVector::GetData<bool>(vec);
                for (idx_t i = 0; i < count; ++i) {
                    idx_t src = state.position + i;
                    if (sc.is_null[src]) {
                        validity.SetInvalid(i);
                    } else {
                        dst[i] = sc.bool_vals[src];
                    }
                }
                break;
            }
            case ailake::ScanColType::LIST_FLOAT32: {
                // Populate DuckDB LIST(FLOAT) using ListVector API.
                idx_t total_child = 0;
                for (idx_t i = 0; i < count; ++i) {
                    idx_t src = state.position + i;
                    total_child += sc.is_null[src] ? 0 : sc.list_vals[src].size();
                }
                ListVector::Reserve(vec, total_child);

                auto *list_entries = FlatVector::GetData<list_entry_t>(vec);
                auto &child_vec    = ListVector::GetEntry(vec);
                auto *child_data   = FlatVector::GetData<float>(child_vec);
                idx_t child_offset = 0;

                for (idx_t i = 0; i < count; ++i) {
                    idx_t src = state.position + i;
                    if (sc.is_null[src]) {
                        validity.SetInvalid(i);
                        list_entries[i] = {child_offset, 0};
                    } else {
                        const auto &floats = sc.list_vals[src];
                        list_entries[i] = {child_offset, floats.size()};
                        for (float f : floats) {
                            child_data[child_offset++] = f;
                        }
                    }
                }
                ListVector::SetListSize(vec, child_offset);
                break;
            }
            default:
                break;
        }
    }

    state.position += count;
    output.SetCardinality(count);
}

// ── Registration ──────────────────────────────────────────────────────────────

void RegisterAilakeScan(duckdb::ExtensionLoader &loader) {
    TableFunction func(
        "ailake_scan",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
        AilakeScanScan,
        AilakeScanBind,
        AilakeScanInit
    );

    func.named_parameters["vec_col"]    = LogicalType::VARCHAR;
    func.named_parameters["ef_search"]  = LogicalType::INTEGER;
    func.named_parameters["table_name"] = LogicalType::VARCHAR;

    loader.RegisterFunction( func);
}
