// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// ailake_write_batch(table_path, ids, embeddings[, vec_col, metric, precision]) → BIGINT
//
// Writes a batch of rows and their embeddings to an AI-Lake table.
// Returns snapshot_id on success, -1 on failure.
//
// Parameters:
//   table_path  VARCHAR           — table root path/URI
//   ids         BIGINT[]          — row identifiers (LIST(BIGINT))
//   embeddings  FLOAT[][]         — embedding vectors (LIST(LIST(FLOAT)))
//   vec_col     VARCHAR default 'embedding'
//   metric      VARCHAR default 'cosine'   (cosine | euclidean | dot)
//   precision   VARCHAR default 'f16'      (f32 | f16 | i8)
//   namespace   VARCHAR default 'default'
//   table_name  VARCHAR default 'table'
//
// Example:
//   SELECT ailake_write_batch(
//       'file:///data/my_table',
//       [0, 1, 2]::BIGINT[],
//       [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]::FLOAT[][]
//   );

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"

using namespace duckdb;

// ── Helpers to extract LIST values ───────────────────────────────────────────

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

// ── Execution ─────────────────────────────────────────────────────────────────

static void AilakeWriteExec(
    DataChunk      &args,
    ExpressionState &state,
    Vector         &result
) {
    auto &lib = ailake::AilakeLib::get();

    // Process row 0 of the chunk (write_batch is a single-call operation).
    // Using ConstantVector since callers pass literals.
    auto table_path = args.data[0].GetValue(0);
    auto ids_val    = args.data[1].GetValue(0);
    auto emb_val    = args.data[2].GetValue(0);

    if (table_path.IsNull() || ids_val.IsNull() || emb_val.IsNull()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    std::string warehouse = StringValue::Get(table_path);
    auto ids              = extract_bigint_list(ids_val);
    auto embeddings       = extract_float_list_list(emb_val);

    if (ids.empty() || embeddings.empty() || ids.size() != embeddings.size()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    // Optional named arg values are injected via bound function state — read
    // from the scalar function bind info stored in ExpressionState.
    // Since DuckDB scalar functions don't support named parameters directly,
    // we use overloaded arities instead (see registration below).
    int dim = static_cast<int>(embeddings[0].size());

    if (!lib.is_ready()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    int64_t snap = lib.write_batch(
        warehouse,
        "default",
        "table",
        "embedding",
        dim,
        "cosine",
        "f16",
        ids,
        embeddings
    );
    result.SetValue(0, Value::BIGINT(snap));
}

// Overload that accepts explicit vec_col, metric, precision, namespace, table_name.
static void AilakeWriteExecFull(
    DataChunk      &args,
    ExpressionState &state,
    Vector         &result
) {
    auto &lib = ailake::AilakeLib::get();

    auto table_path_v  = args.data[0].GetValue(0);
    auto ids_v         = args.data[1].GetValue(0);
    auto emb_v         = args.data[2].GetValue(0);
    auto vec_col_v     = args.data[3].GetValue(0);
    auto metric_v      = args.data[4].GetValue(0);
    auto precision_v   = args.data[5].GetValue(0);

    if (table_path_v.IsNull() || ids_v.IsNull() || emb_v.IsNull()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    std::string warehouse    = StringValue::Get(table_path_v);
    std::string vec_col      = vec_col_v.IsNull()    ? "embedding" : StringValue::Get(vec_col_v);
    std::string metric       = metric_v.IsNull()     ? "cosine"    : StringValue::Get(metric_v);
    std::string precision    = precision_v.IsNull()  ? "f16"       : StringValue::Get(precision_v);

    // Optional partition args (arity 7 and 8)
    std::string partition_by, partition_value;
    if ((idx_t)args.data.size() > 6 && !args.data[6].GetValue(0).IsNull())
        partition_by    = StringValue::Get(args.data[6].GetValue(0));
    if ((idx_t)args.data.size() > 7 && !args.data[7].GetValue(0).IsNull())
        partition_value = StringValue::Get(args.data[7].GetValue(0));

    auto ids        = extract_bigint_list(ids_v);
    auto embeddings = extract_float_list_list(emb_v);

    if (ids.empty() || embeddings.empty() || ids.size() != embeddings.size()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    int dim = static_cast<int>(embeddings[0].size());

    if (!lib.is_ready()) {
        result.SetValue(0, Value::BIGINT(-1));
        return;
    }

    int64_t snap = lib.write_batch(
        warehouse,
        "default",
        "table",
        vec_col,
        dim,
        metric,
        precision,
        ids,
        embeddings,
        partition_by,
        partition_value
    );
    result.SetValue(0, Value::BIGINT(snap));
}

// ── Registration ──────────────────────────────────────────────────────────────

void RegisterAilakeWrite(duckdb::DatabaseInstance &db) {
    ScalarFunctionSet write_set("ailake_write_batch");

    // Arity 3: (table_path, ids, embeddings) — defaults: embedding / cosine / f16
    write_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT))},
        LogicalType::BIGINT,
        AilakeWriteExec
    ));

    // Arity 6: (table_path, ids, embeddings, vec_col, metric, precision)
    write_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT)),
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeWriteExecFull
    ));

    // Arity 7: (table_path, ids, embeddings, vec_col, metric, precision, partition_by)
    write_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT)),
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeWriteExecFull
    ));

    // Arity 8: (table_path, ids, embeddings, vec_col, metric, precision, partition_by, partition_value)
    write_set.AddFunction(ScalarFunction(
        {LogicalType::VARCHAR,
         LogicalType::LIST(LogicalType::BIGINT),
         LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT)),
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::VARCHAR,
         LogicalType::VARCHAR},
        LogicalType::BIGINT,
        AilakeWriteExecFull
    ));

    ExtensionUtil::RegisterFunction(db, write_set);
}
