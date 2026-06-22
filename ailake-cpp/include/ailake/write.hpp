// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Write-side operations for AI-Lake tables (Phase O).
//
// The C++ SDK is a read-only reader. Write operations that require Rust
// business logic (equality delete, schema evolution) are delegated to the
// `ailake` CLI binary:
//
//   Priority 1: AILAKE_BIN env var     — path to a specific `ailake` binary
//   Priority 2: `ailake` in PATH       — system-wide install (searches via popen)
//
// Both functions throw std::runtime_error when no binary is available or the
// CLI exits with a non-zero code.
#pragma once

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#  define AILAKE_POPEN  _popen
#  define AILAKE_PCLOSE _pclose
#else
#  include <cstdio>
#  include <sys/wait.h>
#  define AILAKE_POPEN  popen
#  define AILAKE_PCLOSE pclose
#endif

namespace ailake {

// AddColumnReq describes a column addition for evolve_schema.
struct AddColumnReq {
    std::string name;
    std::string type;            // Iceberg type: "string", "int", "long", "float", ...
    std::string initial_default; // JSON literal (null, 0, 0.0, "unknown"); empty → null
};

// RenameColumnReq describes a column rename for evolve_schema.
struct RenameColumnReq {
    std::string from;
    std::string to;
};

namespace detail {

// Return the ailake binary path: AILAKE_BIN env or "ailake" (from PATH).
inline std::string resolve_bin() {
    if (const char* env = std::getenv("AILAKE_BIN")) {
        if (env[0] != '\0') return env;
    }
    return "ailake";
}

// Run a command via popen and return its combined stdout output.
// Throws std::runtime_error on popen failure or non-zero exit.
inline std::string run_cmd(const std::string& cmd) {
    std::string output;
    FILE* pipe = AILAKE_POPEN((cmd + " 2>&1").c_str(), "r");
    if (!pipe) throw std::runtime_error("ailake: popen failed: " + cmd);
    char buf[256];
    while (std::fgets(buf, sizeof(buf), pipe)) output += buf;
    int rc = AILAKE_PCLOSE(pipe);
#ifndef _WIN32
    // pclose() returns a wait-status on POSIX — extract actual exit code.
    int exit_code = (WIFEXITED(rc)) ? WEXITSTATUS(rc) : rc;
#else
    int exit_code = rc;
#endif
    if (exit_code != 0) throw std::runtime_error("ailake CLI failed (exit " + std::to_string(exit_code) + "):\n" + output);
    return output;
}

// Shell-escape a single argument (POSIX: wrap in single-quotes, escape embeds).
inline std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
}

} // namespace detail

// WriteBatchOptions controls optional parameters for write_batch.
struct WriteBatchOptions {
    std::string vec_col;              // embedding column name (default "embedding")
    std::string metric;               // cosine | euclidean | dot (default "cosine")
    std::string precision;            // f32 | f16 | i8 (default "f16")
    std::string embedding_model;      // optional model label
    std::string partition_by;         // single-column partition key
    std::string partition_value;      // single-column partition value
    int         format_version = 2;   // Iceberg format version (2 or 3)
    std::vector<std::string> fts_columns;  // text columns for Tantivy FTS
    std::string fts_tokenizer;        // Tantivy tokenizer (default "default")
    int         hnsw_m = 0;           // HNSW M (0 = use table default)
    int         hnsw_ef_construction = 0; // HNSW ef_construction (0 = use table default)
    bool        pre_normalize = false;// normalize vectors to unit L2 at write time
    bool        deferred = false;     // build index asynchronously
};

// write_batch inserts a Parquet file into an AI-Lake table via the `ailake` CLI.
//
// `parquet_file` must be a local path to a Parquet file whose column
// `opts.vec_col` holds the embedding vectors. The table is created if it does
// not exist (same behaviour as `ailake insert`).
//
// Throws std::runtime_error if the CLI binary is not found or exits non-zero.
inline void write_batch(
    const std::string&    warehouse,
    const std::string&    table_id,      // "namespace.table"
    const std::string&    parquet_file,
    const WriteBatchOptions& opts = {})
{
    std::string bin = detail::resolve_bin();
    std::string vec_col = opts.vec_col.empty() ? "embedding" : opts.vec_col;

    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " insert " + detail::shell_quote(table_id)
        + " " + detail::shell_quote(parquet_file)
        + " --embeddings " + detail::shell_quote(vec_col);

    if (!opts.metric.empty())
        cmd += " --metric " + detail::shell_quote(opts.metric);
    if (!opts.precision.empty())
        cmd += " --precision " + detail::shell_quote(opts.precision);
    if (!opts.embedding_model.empty())
        cmd += " --embedding-model " + detail::shell_quote(opts.embedding_model);
    if (!opts.partition_by.empty())
        cmd += " --partition-by " + detail::shell_quote(opts.partition_by);
    if (!opts.partition_value.empty())
        cmd += " --partition-value " + detail::shell_quote(opts.partition_value);
    if (opts.format_version != 0 && opts.format_version != 2)
        cmd += " --format-version " + std::to_string(opts.format_version);
    if (!opts.fts_columns.empty()) {
        std::string cols;
        for (size_t i = 0; i < opts.fts_columns.size(); ++i) {
            if (i > 0) cols += ',';
            cols += opts.fts_columns[i];
        }
        cmd += " --fts-columns " + detail::shell_quote(cols);
        if (!opts.fts_tokenizer.empty() && opts.fts_tokenizer != "default")
            cmd += " --fts-tokenizer " + detail::shell_quote(opts.fts_tokenizer);
    }
    if (opts.hnsw_m > 0)
        cmd += " --hnsw-m " + std::to_string(opts.hnsw_m);
    if (opts.hnsw_ef_construction > 0)
        cmd += " --hnsw-ef " + std::to_string(opts.hnsw_ef_construction);
    if (opts.pre_normalize)
        cmd += " --pre-normalize";
    if (opts.deferred)
        cmd += " --deferred";

    detail::run_cmd(cmd);
}

// delete_where logically deletes all rows where `column` equals any value in
// `values`. Writes an Iceberg equality delete file via the `ailake` CLI.
//
// No data files are rewritten; deleted rows are masked at scan time.
// `warehouse` is the table root path (--store arg), `table_id` is "namespace.table".
inline void delete_where(
    const std::string&              warehouse,
    const std::string&              table_id,
    const std::string&              column,
    const std::vector<std::string>& values)
{
    if (values.empty()) return;

    std::string bin = detail::resolve_bin();

    // Build comma-separated value list.
    std::string vals;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) vals += ',';
        vals += values[i];
    }

    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " delete-where " + detail::shell_quote(table_id)
        + " --col "  + detail::shell_quote(column)
        + " --vals " + detail::shell_quote(vals);

    detail::run_cmd(cmd);
}

// evolve_schema applies a metadata-only schema evolution to the table.
// Returns the new schema_id (-1 if not parseable from CLI output).
//
// add_cols and rename_cols may be empty if only one operation is desired.
inline int evolve_schema(
    const std::string&                 warehouse,
    const std::string&                 table_id,
    const std::vector<AddColumnReq>&   add_cols,
    const std::vector<RenameColumnReq>& rename_cols)
{
    if (add_cols.empty() && rename_cols.empty()) return 0;

    std::string bin = detail::resolve_bin();
    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " evolve " + detail::shell_quote(table_id);

    for (const auto& ac : add_cols) {
        cmd += " --add " + detail::shell_quote(ac.name + ":" + ac.type);
        if (!ac.initial_default.empty())
            cmd += " --initial-default " + detail::shell_quote(ac.initial_default);
    }
    for (const auto& rc : rename_cols) {
        cmd += " --rename " + detail::shell_quote(rc.from + ":" + rc.to);
    }

    std::string out = detail::run_cmd(cmd);

    // Parse "new_schema_id: N" from output.
    int schema_id = -1;
    auto pos = out.find("new_schema_id:");
    if (pos != std::string::npos) {
        pos += 14;
        while (pos < out.size() && (out[pos] == ' ' || out[pos] == '\t')) ++pos;
        try { schema_id = std::stoi(out.substr(pos)); } catch (...) {}
    }
    return schema_id;
}

} // namespace ailake
