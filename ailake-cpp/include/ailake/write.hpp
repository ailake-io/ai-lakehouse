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
#include <map>
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

// Appends --catalog/--rest-* CLI flags built from a catalog_opts map (see
// WriteBatchOptions::catalog_opts / CompactOptions::catalog_opts) to `cmd`.
// Empty map = no flags added, unchanged default Hadoop-catalog behavior.
// Recognized keys: "catalog", "rest-uri", "rest-prefix", "rest-warehouse",
// "rest-auth", "rest-token", "rest-oauth-token-endpoint",
// "rest-oauth-client-id", "rest-oauth-client-secret", "rest-oauth-scope" —
// unrecognized keys are ignored. See docs/guides/REST_CATALOG.md.
inline void append_catalog_args(std::string& cmd, const std::map<std::string, std::string>& opts) {
    static const std::vector<std::pair<std::string, std::string>> flag_order = {
        {"catalog", "--catalog"},
        {"rest-uri", "--rest-uri"},
        {"rest-prefix", "--rest-prefix"},
        {"rest-warehouse", "--rest-warehouse"},
        {"rest-auth", "--rest-auth"},
        {"rest-token", "--rest-token"},
        {"rest-oauth-token-endpoint", "--rest-oauth-token-endpoint"},
        {"rest-oauth-client-id", "--rest-oauth-client-id"},
        {"rest-oauth-client-secret", "--rest-oauth-client-secret"},
        {"rest-oauth-scope", "--rest-oauth-scope"},
    };
    for (const auto& kf : flag_order) {
        auto it = opts.find(kf.first);
        if (it != opts.end() && !it->second.empty()) {
            cmd += " " + kf.second + " " + shell_quote(it->second);
        }
    }
}

} // namespace detail

// CreateTableOptions controls optional parameters for create_table.
struct CreateTableOptions {
    std::string metric;                    // cosine | euclidean | dot (default "cosine")
    std::string precision;                  // f32 | f16 | i8 (default "f16")
    std::string column;                     // vector column name (default "embedding")
    bool        pre_normalize = false;      // normalize vectors to unit L2 at write time
    int         hnsw_m = 0;                 // HNSW M (0 = CLI default, 16)
    int         hnsw_ef_construction = 0;   // HNSW ef_construction (0 = CLI default, 150)
    bool        pq_only = false;            // omit raw vector column from Parquet files
    bool        ivf_residual = false;       // encode (vec - coarse_centroid) per IVF cell
    std::string modality;                   // text | image | audio | video; empty = unset
    int         format_version = 2;         // Iceberg format version (2 or 3)
    std::vector<std::string> fts_columns;   // text columns for Tantivy FTS
    std::string fts_tokenizer;              // Tantivy tokenizer (default "default")
    // Selects/configures a non-Hadoop catalog backend — see
    // WriteBatchOptions::catalog_opts for accepted keys and
    // docs/guides/REST_CATALOG.md. Empty = default Hadoop catalog.
    std::map<std::string, std::string> catalog_opts;
};

// create_table creates an empty AI-Lake table with the given vector schema
// and policy via the `ailake create` CLI. Unlike write_batch (which
// auto-creates a table on first insert with default policy), this is the
// only way to set pq_only/ivf_residual/modality or HNSW tuning before any
// data is written.
//
// Throws std::runtime_error if the CLI binary is not found or exits non-zero.
inline void create_table(
    const std::string&       warehouse,
    const std::string&       table_id,      // "namespace.table"
    int                       dim,
    const CreateTableOptions& opts = {})
{
    std::string bin = detail::resolve_bin();

    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " create " + detail::shell_quote(table_id)
        + " --dim " + std::to_string(dim);

    if (!opts.metric.empty())
        cmd += " --metric " + detail::shell_quote(opts.metric);
    if (!opts.precision.empty())
        cmd += " --precision " + detail::shell_quote(opts.precision);
    if (!opts.column.empty())
        cmd += " --column " + detail::shell_quote(opts.column);
    if (opts.pre_normalize)
        cmd += " --pre-normalize";
    if (opts.hnsw_m > 0)
        cmd += " --hnsw-m " + std::to_string(opts.hnsw_m);
    if (opts.hnsw_ef_construction > 0)
        cmd += " --hnsw-ef " + std::to_string(opts.hnsw_ef_construction);
    if (opts.pq_only)
        cmd += " --pq-only";
    if (opts.ivf_residual)
        cmd += " --ivf-residual";
    if (!opts.modality.empty())
        cmd += " --modality " + detail::shell_quote(opts.modality);
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
    detail::append_catalog_args(cmd, opts.catalog_opts);

    detail::run_cmd(cmd);
}

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
    bool        deferred = false;     // build index asynchronously; not combinable
                                       // with batch_id (deferred writes don't yet
                                       // carry an idempotency tag — the CLI rejects
                                       // both set, same as ailake insert itself)
    // Idempotency key — a retry with the same batch_id is a no-op if already
    // committed (safe for at-least-once callers, e.g. Airflow retries). Empty
    // = no idempotency check, every call writes a new batch.
    std::string batch_id;
    // Multi-column Iceberg partition spec as a JSON array, e.g.
    // '[{"column":"topic_id","transform":"identity","column_type":"int"}]' —
    // this header has no JSON dependency, so build the string yourself (same
    // convention as fts_columns' comma-joining below). Takes precedence over
    // partition_by/partition_value when set.
    std::string partition_fields_json;
    // Selects/configures a non-Hadoop catalog backend (e.g. REST Catalog —
    // Polaris, Unity Catalog, BigLake, S3 Tables, Nessie, Gravitino). Empty =
    // default Hadoop-style catalog, unchanged behavior. Keys: "catalog",
    // "rest-uri", "rest-prefix", "rest-warehouse", "rest-auth", "rest-token",
    // "rest-oauth-token-endpoint", "rest-oauth-client-id",
    // "rest-oauth-client-secret", "rest-oauth-scope". See docs/guides/REST_CATALOG.md.
    std::map<std::string, std::string> catalog_opts;
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
    if (!opts.partition_fields_json.empty())
        cmd += " --partition-fields " + detail::shell_quote(opts.partition_fields_json);
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
    if (!opts.batch_id.empty())
        cmd += " --batch-id " + detail::shell_quote(opts.batch_id);
    if (opts.hnsw_m > 0)
        cmd += " --hnsw-m " + std::to_string(opts.hnsw_m);
    if (opts.hnsw_ef_construction > 0)
        cmd += " --hnsw-ef " + std::to_string(opts.hnsw_ef_construction);
    if (opts.pre_normalize)
        cmd += " --pre-normalize";
    if (opts.deferred)
        cmd += " --deferred";
    detail::append_catalog_args(cmd, opts.catalog_opts);

    detail::run_cmd(cmd);
}

// VectorColSpec describes one vector column in a multi-column (Phase 8
// multimodal) write — e.g. text + image embeddings on the same row.
struct VectorColSpec {
    std::string column;
    int         dim = 0;
    std::string metric = "cosine";
    std::string modality; // optional: text | image | audio | video
};

// write_batch_multi inserts a Parquet file with N independent vector columns
// via `ailake insert --vector-cols` (Phase 8 multimodal write). Each column
// gets its own HNSW index in the AILK footer.
//
// Throws std::runtime_error if `vector_cols` is empty, the CLI binary is not
// found, or the CLI exits non-zero.
inline void write_batch_multi(
    const std::string&    warehouse,
    const std::string&    table_id,      // "namespace.table"
    const std::string&    parquet_file,
    const std::vector<VectorColSpec>& vector_cols,
    const WriteBatchOptions& opts = {})
{
    if (vector_cols.empty())
        throw std::runtime_error("ailake: write_batch_multi requires at least one VectorColSpec");

    std::string bin = detail::resolve_bin();

    std::string spec;
    for (size_t i = 0; i < vector_cols.size(); ++i) {
        if (i > 0) spec += ',';
        const auto& c = vector_cols[i];
        spec += c.column + ":" + std::to_string(c.dim) + ":"
              + (c.metric.empty() ? "cosine" : c.metric);
        if (!c.modality.empty()) spec += ":" + c.modality;
    }

    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " insert " + detail::shell_quote(table_id)
        + " " + detail::shell_quote(parquet_file)
        + " --vector-cols " + detail::shell_quote(spec);

    // Multi-column mode hardcodes F16 precision and ignores --metric/--precision
    // (metric travels per-column in --vector-cols) — same contract as the CLI's
    // own Insert handler and ailake-go's WriteBatch multi-column branch.
    if (!opts.partition_by.empty())
        cmd += " --partition-by " + detail::shell_quote(opts.partition_by);
    if (!opts.partition_value.empty())
        cmd += " --partition-value " + detail::shell_quote(opts.partition_value);
    if (!opts.partition_fields_json.empty())
        cmd += " --partition-fields " + detail::shell_quote(opts.partition_fields_json);
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
    if (!opts.batch_id.empty())
        cmd += " --batch-id " + detail::shell_quote(opts.batch_id);
    if (opts.deferred)
        cmd += " --deferred";
    detail::append_catalog_args(cmd, opts.catalog_opts);

    detail::run_cmd(cmd);
}

// delete_where logically deletes all rows where `column` equals any value in
// `values`. Writes an Iceberg equality delete file via the `ailake` CLI.
//
// No data files are rewritten; deleted rows are masked at scan time.
// `warehouse` is the table root path (--store arg), `table_id` is "namespace.table".
//
// catalog_opts selects/configures a non-Hadoop catalog backend (e.g. REST
// Catalog) — see WriteBatchOptions::catalog_opts for accepted keys and
// docs/guides/REST_CATALOG.md. Regression: this was the last write-path
// function here (with evolve_schema below) never wired to
// append_catalog_args, even though every other write-path function
// (create_table, write_batch(_multi), compact, decay_memories, migrate,
// delete_rows, add_vector_column, backfill_vector_column) already was.
inline void delete_where(
    const std::string&              warehouse,
    const std::string&              table_id,
    const std::string&              column,
    const std::vector<std::string>& values,
    const std::map<std::string, std::string>& catalog_opts = {})
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
    detail::append_catalog_args(cmd, catalog_opts);

    detail::run_cmd(cmd);
}

// evolve_schema applies a metadata-only schema evolution to the table.
// Returns the new schema_id (-1 if not parseable from CLI output).
//
// add_cols and rename_cols may be empty if only one operation is desired.
// catalog_opts — see delete_where's doc comment above.
inline int evolve_schema(
    const std::string&                 warehouse,
    const std::string&                 table_id,
    const std::vector<AddColumnReq>&   add_cols,
    const std::vector<RenameColumnReq>& rename_cols,
    const std::map<std::string, std::string>& catalog_opts = {})
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
    detail::append_catalog_args(cmd, catalog_opts);

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

// CompactOptions controls optional parameters for compact.
struct CompactOptions {
    int64_t target_size = 0;        // bytes, 0 = CLI default (512 MiB)
    int     min_files = 0;          // 0 = CLI default (4)
    int     max_files_per_pass = 0; // 0 = CLI default (20)
    bool    deferred = false;
    // Selects/configures a non-Hadoop catalog backend — see
    // WriteBatchOptions::catalog_opts for accepted keys and
    // docs/guides/REST_CATALOG.md. Empty = default Hadoop catalog.
    std::map<std::string, std::string> catalog_opts;
};

// compact merges small files in an AI-Lake table via `ailake compact`.
// Returns the number of files compacted (0 = nothing eligible).
inline int compact(
    const std::string&    warehouse,
    const std::string&    table_id,
    const CompactOptions& opts = {})
{
    std::string bin = detail::resolve_bin();
    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " compact " + detail::shell_quote(table_id)
        + " --format json";

    if (opts.target_size > 0)
        cmd += " --target-size " + std::to_string(opts.target_size);
    if (opts.min_files > 0)
        cmd += " --min-files " + std::to_string(opts.min_files);
    if (opts.max_files_per_pass > 0)
        cmd += " --max-files-per-pass " + std::to_string(opts.max_files_per_pass);
    if (opts.deferred)
        cmd += " --deferred";
    detail::append_catalog_args(cmd, opts.catalog_opts);

    std::string out = detail::run_cmd(cmd);

    // Parse "files_compacted":N from JSON output (no JSON dependency in this
    // header — same substring-parse style as evolve_schema's new_schema_id).
    int files_compacted = 0;
    std::string key = "\"files_compacted\":";
    auto pos = out.find(key);
    if (pos != std::string::npos) {
        pos += key.size();
        while (pos < out.size() && out[pos] == ' ') ++pos;
        try { files_compacted = std::stoi(out.substr(pos)); } catch (...) {}
    }
    return files_compacted;
}

// decay_memories recomputes recency weights (exp(-λ×days_since_access)) across
// all memory files in the table via `ailake decay-memories` (Phase 9
// agent-memory schema). Returns the number of files updated (0 = nothing changed).
inline int decay_memories(
    const std::string& warehouse,
    const std::string& table_id,
    float               lambda,
    const std::map<std::string, std::string>& catalog_opts = {})
{
    std::string bin = detail::resolve_bin();
    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " decay-memories " + detail::shell_quote(table_id)
        + " --lambda " + std::to_string(lambda)
        + " --format json";
    detail::append_catalog_args(cmd, catalog_opts);

    std::string out = detail::run_cmd(cmd);

    // Parse "files_updated":N from JSON output (no JSON dependency in this
    // header — same substring-parse style as compact's files_compacted).
    int files_updated = 0;
    std::string key = "\"files_updated\":";
    auto pos = out.find(key);
    if (pos != std::string::npos) {
        pos += key.size();
        while (pos < out.size() && out[pos] == ' ') ++pos;
        try { files_updated = std::stoi(out.substr(pos)); } catch (...) {}
    }
    return files_updated;
}

// MigrateOptions controls optional parameters for migrate.
struct MigrateOptions {
    std::string old_column;     // existing embedding column (default "embedding")
    std::string new_column;     // migrated column name (default "embedding_v2")
    std::string text_column;    // raw-text column to re-embed (default "chunk_text")
    std::string strategy;       // "atomic-replace" | "dual-write-then-cutover" (default)
    int         batch_size = 0; // texts per embed-cmd call (0 = CLI default, 512)
    std::string model_name;     // stored in ailake.embedding-model after migration
    std::string model_version;  // appended to model_name as "<name>@<version>"
    std::map<std::string, std::string> catalog_opts;
};

// migrate re-embeds a table's vector column via an external embed command
// (`ailake migrate` CLI). embed_cmd is a shell command that reads a JSON
// array of strings from stdin and writes a JSON array of float arrays to
// stdout. Throws std::runtime_error if the CLI binary is not found or exits
// non-zero.
inline void migrate(
    const std::string&   warehouse,
    const std::string&   table_id,
    const std::string&   embed_cmd,
    const MigrateOptions& opts = {})
{
    std::string bin = detail::resolve_bin();
    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " migrate " + detail::shell_quote(table_id)
        + " --embed-cmd " + detail::shell_quote(embed_cmd);

    if (!opts.old_column.empty())
        cmd += " --old-column " + detail::shell_quote(opts.old_column);
    if (!opts.new_column.empty())
        cmd += " --new-column " + detail::shell_quote(opts.new_column);
    if (!opts.text_column.empty())
        cmd += " --text-column " + detail::shell_quote(opts.text_column);
    if (!opts.strategy.empty())
        cmd += " --strategy " + detail::shell_quote(opts.strategy);
    if (opts.batch_size > 0)
        cmd += " --batch-size " + std::to_string(opts.batch_size);
    if (!opts.model_name.empty())
        cmd += " --model-name " + detail::shell_quote(opts.model_name);
    if (!opts.model_version.empty())
        cmd += " --model-version " + detail::shell_quote(opts.model_version);
    detail::append_catalog_args(cmd, opts.catalog_opts);

    detail::run_cmd(cmd);
}

// delete_rows marks rows as deleted in a V3 table using Iceberg Deletion
// Vectors via `ailake delete-rows`. `file` is the Parquet data file path as
// reported by list_files (e.g. "data/part-00001.parquet"). Requires the
// table to have been created with format_version=3 — the CLI raises a clear
// error on a V2 table rather than corrupting it; use delete_where (equality
// predicate) for V2 tables. No-op when row_positions is empty.
inline void delete_rows(
    const std::string&        warehouse,
    const std::string&        table_id,
    const std::string&        file,
    const std::vector<int>&   row_positions,
    const std::map<std::string, std::string>& catalog_opts = {})
{
    if (row_positions.empty()) return;

    std::string bin = detail::resolve_bin();
    std::string rows;
    for (size_t i = 0; i < row_positions.size(); ++i) {
        if (i > 0) rows += ',';
        rows += std::to_string(row_positions[i]);
    }

    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " delete-rows " + detail::shell_quote(table_id)
        + " --file " + detail::shell_quote(file)
        + " --rows " + detail::shell_quote(rows);
    detail::append_catalog_args(cmd, catalog_opts);

    detail::run_cmd(cmd);
}

// AddVectorColumnOptions controls optional parameters for add_vector_column.
struct AddVectorColumnOptions {
    std::string metric;                  // cosine | euclidean | dot (default "cosine")
    std::string precision;                // f32 | f16 | i8 (default "f16")
    bool        pre_normalize = false;
    int         hnsw_m = 0;               // 0 = CLI default (16)
    int         hnsw_ef_construction = 0; // 0 = CLI default (150)
    std::map<std::string, std::string> catalog_opts;
};

// add_vector_column adds a new vector column to an existing table schema (no
// data files rewritten) via `ailake add-vector-column`. Old files return
// null for the new column until backfill_vector_column is run.
inline void add_vector_column(
    const std::string&             warehouse,
    const std::string&             table_id,
    const std::string&             column,
    int                             dim,
    const AddVectorColumnOptions&   opts = {})
{
    std::string bin = detail::resolve_bin();
    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " add-vector-column " + detail::shell_quote(table_id)
        + " --column " + detail::shell_quote(column)
        + " --dim " + std::to_string(dim);

    if (!opts.metric.empty())
        cmd += " --metric " + detail::shell_quote(opts.metric);
    if (!opts.precision.empty())
        cmd += " --precision " + detail::shell_quote(opts.precision);
    if (opts.pre_normalize)
        cmd += " --pre-normalize";
    if (opts.hnsw_m > 0)
        cmd += " --hnsw-m " + std::to_string(opts.hnsw_m);
    if (opts.hnsw_ef_construction > 0)
        cmd += " --hnsw-ef " + std::to_string(opts.hnsw_ef_construction);
    detail::append_catalog_args(cmd, opts.catalog_opts);

    detail::run_cmd(cmd);
}

// BackfillVectorColumnOptions controls optional parameters for backfill_vector_column.
struct BackfillVectorColumnOptions {
    std::string text_column;  // raw-text column to embed (default "chunk_text")
    int         batch_size = 0; // texts per embed-cmd call (0 = CLI default, 512)
    std::map<std::string, std::string> catalog_opts;
};

// backfill_vector_column backfills a new vector column in all existing files
// (rewriting each file with new embeddings) via `ailake
// backfill-vector-column`. `column` must already exist via
// add_vector_column. Idempotent: files already containing the column are
// skipped.
inline void backfill_vector_column(
    const std::string&                  warehouse,
    const std::string&                  table_id,
    const std::string&                  column,
    const std::string&                  embed_cmd,
    const BackfillVectorColumnOptions&   opts = {})
{
    std::string bin = detail::resolve_bin();
    std::string cmd = detail::shell_quote(bin)
        + " --store " + detail::shell_quote(warehouse)
        + " backfill-vector-column " + detail::shell_quote(table_id)
        + " --column " + detail::shell_quote(column)
        + " --embed-cmd " + detail::shell_quote(embed_cmd);

    if (!opts.text_column.empty())
        cmd += " --text-column " + detail::shell_quote(opts.text_column);
    if (opts.batch_size > 0)
        cmd += " --batch-size " + std::to_string(opts.batch_size);
    detail::append_catalog_args(cmd, opts.catalog_opts);

    detail::run_cmd(cmd);
}

// EstimateOptions controls optional parameters for estimate.
struct EstimateOptions {
    int hnsw_m = 0; // 0 = CLI default (16)
    int pq_m = 0;   // 0 = CLI default (dim/32, clamped to [8, dim])
};

// estimate computes storage-usage estimates before writing (pure math, no
// I/O, no warehouse/catalog involved) via `ailake estimate`. `rows` supports
// K/M/B suffixes (e.g. "1M", "500K"). Returns the raw JSON response string —
// this header has no JSON dependency (same convention as compact/
// decay_memories' own substring parsing); parse with whatever JSON library
// your application already uses if you need structured access to the
// per-mode estimates.
inline std::string estimate(
    const std::string&      rows,
    int                      dim,
    const EstimateOptions&   opts = {})
{
    std::string bin = detail::resolve_bin();
    std::string cmd = detail::shell_quote(bin)
        + " estimate"
        + " --rows " + detail::shell_quote(rows)
        + " --dim " + std::to_string(dim)
        + " --format json";

    if (opts.hnsw_m > 0)
        cmd += " --hnsw-m " + std::to_string(opts.hnsw_m);
    if (opts.pq_m > 0)
        cmd += " --pq-m " + std::to_string(opts.pq_m);

    return detail::run_cmd(cmd);
}

} // namespace ailake
