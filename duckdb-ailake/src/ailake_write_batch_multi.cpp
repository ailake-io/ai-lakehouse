// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_write_batch_multi(table_path, ids, vector_columns) → BIGINT
//
// Writes a batch of rows with N independent vector columns (Phase 8
// multimodal — e.g. text + image embeddings on the same row), each getting
// its own HNSW section in the same AI-Lake file. Searchable via
// ailake_search_multimodal's RRF fusion. Returns snapshot_id on success,
// -1 on failure.
//
// Parameters:
//   table_path      VARCHAR   — table root path/URI
//   ids             BIGINT[]  — row identifiers
//   vector_columns  LIST(STRUCT(col VARCHAR, dim INTEGER, embeddings FLOAT[][],
//                                metric VARCHAR, precision VARCHAR, modality VARCHAR))
//                     — one entry per vector column; first entry is primary
//                       (used for geometric pruning in the manifest)
//   namespace       VARCHAR default 'default'
//   table_name      VARCHAR default 'table'
//   format_version  INTEGER default 2 (2 or 3)
//   deferred        BOOLEAN default false — persist Parquet immediately, build
//                     all HNSW indexes in the background
//
// Example:
//   SELECT ailake_write_batch_multi(
//       'file:///data/media',
//       [0, 1]::BIGINT[],
//       [
//           {'col': 'embedding',       'dim': 4, 'metric': 'cosine', 'precision': 'f16',
//            'embeddings': [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]::FLOAT[][]},
//           {'col': 'image_embedding', 'dim': 2, 'metric': 'cosine', 'precision': 'f16',
//            'embeddings': [[0.9, 1.0], [1.1, 1.2]]::FLOAT[][]}
//       ]
//   );

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<int64_t> extract_bigint_list(const Value &v) {
    std::vector<int64_t> out;
    if (v.IsNull()) return out;
    for (const auto &child : ListValue::GetChildren(v)) {
        if (!child.IsNull()) out.push_back(BigIntValue::Get(child));
    }
    return out;
}

static std::vector<std::vector<float>> extract_float_list_list(const Value &v) {
    std::vector<std::vector<float>> out;
    if (v.IsNull()) return out;
    for (const auto &inner : ListValue::GetChildren(v)) {
        std::vector<float> row;
        if (!inner.IsNull()) {
            for (const auto &f : ListValue::GetChildren(inner)) {
                if (!f.IsNull()) row.push_back(FloatValue::Get(f));
            }
        }
        out.push_back(std::move(row));
    }
    return out;
}

// Parses vector_columns: LIST(STRUCT(col, dim, embeddings, metric, precision, modality)).
static std::vector<ailake::VectorColSpecArg> extract_vector_columns(const Value &v) {
    std::vector<ailake::VectorColSpecArg> out;
    if (v.IsNull() || v.type().id() != LogicalTypeId::LIST) return out;

    for (auto &elem : ListValue::GetChildren(v)) {
        if (elem.IsNull() || elem.type().id() != LogicalTypeId::STRUCT) continue;

        ailake::VectorColSpecArg spec;
        auto &children = StructValue::GetChildren(elem);
        auto &keys     = StructType::GetChildTypes(elem.type());
        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string &fname = keys[i].first;
            const Value       &fval  = children[i];
            if (fval.IsNull()) continue;
            if (fname == "col") {
                spec.col = StringValue::Get(fval);
            } else if (fname == "dim") {
                spec.dim = IntegerValue::Get(fval);
            } else if (fname == "embeddings") {
                spec.embeddings = extract_float_list_list(fval);
            } else if (fname == "metric") {
                spec.metric = StringValue::Get(fval);
            } else if (fname == "precision") {
                spec.precision = StringValue::Get(fval);
            } else if (fname == "modality") {
                spec.modality = StringValue::Get(fval);
            }
        }
        if (!spec.col.empty() && !spec.embeddings.empty()) {
            out.push_back(std::move(spec));
        }
    }
    return out;
}

// ── Execution ─────────────────────────────────────────────────────────────────

static void AilakeWriteBatchMultiExec(
    DataChunk       &args,
    ExpressionState &state,
    Vector          &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v = args.data[0].GetValue(0);
    auto ids_v        = args.data[1].GetValue(0);
    auto cols_v       = args.data[2].GetValue(0);

    if (table_path_v.IsNull() || ids_v.IsNull() || cols_v.IsNull()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    if (!lib.is_write_multi_ready()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    std::string warehouse       = StringValue::Get(table_path_v);
    auto        ids             = extract_bigint_list(ids_v);
    auto        vector_columns  = extract_vector_columns(cols_v);

    if (ids.empty() || vector_columns.empty()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }
    for (const auto &vc : vector_columns) {
        if (vc.embeddings.size() != ids.size()) {
            result.SetValue(0, Value::BIGINT(-1));
            return;
        }
    }

    std::string ns         = "default";
    std::string table_name = "table";
    int         format_version = 2;
    bool        deferred       = false;
    if ((idx_t)args.data.size() > 3 && !args.data[3].GetValue(0).IsNull())
        ns             = StringValue::Get(args.data[3].GetValue(0));
    if ((idx_t)args.data.size() > 4 && !args.data[4].GetValue(0).IsNull())
        table_name     = StringValue::Get(args.data[4].GetValue(0));
    if ((idx_t)args.data.size() > 5 && !args.data[5].GetValue(0).IsNull())
        format_version = IntegerValue::Get(args.data[5].GetValue(0));
    if ((idx_t)args.data.size() > 6 && !args.data[6].GetValue(0).IsNull())
        deferred       = BooleanValue::Get(args.data[6].GetValue(0));

    int64_t snap = lib.write_batch_multi(
        warehouse, ns, table_name, ids, vector_columns, format_version, deferred
    );
    result.SetValue(0, Value::BIGINT(snap));
}

// ── Registration ──────────────────────────────────────────────────────────────

void RegisterAilakeWriteBatchMulti(duckdb::ExtensionLoader &loader) {
    auto struct_type = LogicalType::STRUCT({
        {"col",        LogicalType::VARCHAR},
        {"dim",        LogicalType::INTEGER},
        {"embeddings", LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT))},
        {"metric",     LogicalType::VARCHAR},
        {"precision",  LogicalType::VARCHAR},
        {"modality",   LogicalType::VARCHAR},
    });

    ScalarFunctionSet fn_set("ailake_write_batch_multi");

    // Arity 3: (table_path, ids, vector_columns) — namespace='default', table_name='table'
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(struct_type)},
        LogicalType::BIGINT,
        AilakeWriteBatchMultiExec
    ));

    // Arity 4: + namespace VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(struct_type),
         LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeWriteBatchMultiExec
    ));

    // Arity 5: + table_name VARCHAR
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(struct_type),
         LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeWriteBatchMultiExec
    ));

    // Arity 6: + format_version INTEGER (2 or 3)
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(struct_type),
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::INTEGER},
        LogicalType::BIGINT,
        AilakeWriteBatchMultiExec
    ));

    // Arity 7: + deferred BOOLEAN
    fn_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(struct_type),
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::INTEGER,
         LogicalType::BOOLEAN},
        LogicalType::BIGINT,
        AilakeWriteBatchMultiExec
    ));

    loader.RegisterFunction(fn_set);
}
