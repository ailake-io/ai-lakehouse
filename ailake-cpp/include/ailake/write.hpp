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
    if (rc != 0) throw std::runtime_error("ailake CLI failed (exit " + std::to_string(rc) + "):\n" + output);
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
