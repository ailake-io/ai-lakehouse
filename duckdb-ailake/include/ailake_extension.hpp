// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// Shared types and AilakeLib singleton used by search and write functions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
    #define AILAKE_DLOPEN(path)         LoadLibrary(path)
    #define AILAKE_DLSYM(h, sym)        GetProcAddress((HMODULE)(h), sym)
    #define AILAKE_DLCLOSE(h)           FreeLibrary((HMODULE)(h))
    #define AILAKE_LIB_NAME             "ailake_jni.dll"
#else
    #include <dlfcn.h>
    #define AILAKE_DLOPEN(path)         dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
    #define AILAKE_DLSYM(h, sym)        dlsym(h, sym)
    #define AILAKE_DLCLOSE(h)           dlclose(h)
    #define AILAKE_LIB_NAME             "libailake_jni.so"
#endif

namespace ailake {

struct SearchRow {
    int64_t     row_id;
    float       distance;
    std::string file_path;
};

// One query arm for cross-modal RRF search.
struct ModalQueryArg {
    std::string        col;
    std::vector<float> query;
    float              weight = 1.0f;
};

// One result row from ailake_search_multimodal.
struct MultimodalRow {
    int64_t     row_id;
    float       rrf_score;
    std::string file_path;
};

// ── ailake_scan column types ──────────────────────────────────────────────────

enum class ScanColType { INT64, FLOAT32, FLOAT64, VARCHAR, BOOL, LIST_FLOAT32, UNKNOWN };

// Pre-parsed column data for ailake_scan results (one of the value vectors is active).
struct ScanColumn {
    std::string  name;
    ScanColType  type = ScanColType::UNKNOWN;

    // Active member depends on type:
    std::vector<int64_t>              int_vals;    // INT64
    std::vector<float>                float_vals;  // FLOAT32
    std::vector<double>               double_vals; // FLOAT64
    std::vector<std::string>          str_vals;    // VARCHAR
    std::vector<bool>                 bool_vals;   // BOOL
    std::vector<std::vector<float>>   list_vals;   // LIST_FLOAT32
    std::vector<bool>                 is_null;     // parallel null bitmap (all types)
};

struct ScanResult {
    bool                     ok        = false;
    std::string              error;
    int64_t                  num_rows  = 0;
    std::vector<ScanColumn>  columns;
};

// ── AilakeLib singleton ───────────────────────────────────────────────────────

// Singleton holding dlopen handle and resolved C-ABI function pointers.
// Thread-safe after Load() completes.
class AilakeLib {
public:
    using search_fn_t     = char *(*)(const char *);
    using multimodal_fn_t = char *(*)(const char *);
    using scan_fn_t       = char *(*)(const char *);
    using write_fn_t      = char *(*)(const char *);
    using free_fn_t       = void (*)(char *);

    static AilakeLib &get();

    // Load libailake_jni.so. If lib_path is empty, searches LD_LIBRARY_PATH.
    // Safe to call multiple times — no-ops after first successful load.
    bool load(const std::string &lib_path = "");

    bool is_ready()            const { return search_fn_     != nullptr; }
    bool is_multimodal_ready() const { return multimodal_fn_ != nullptr; }
    bool is_scan_ready()       const { return scan_fn_       != nullptr; }

    // Execute ailake_search_json. Returns empty on any error.
    std::vector<SearchRow> search(
        const std::string        &warehouse,
        const std::string        &table_name,
        const std::string        &vec_col,
        const std::vector<float> &query,
        int                       top_k,
        int                       ef_search = 50
    ) const;

    // Execute ailake_scan_json. Returns pre-parsed columnar data.
    ScanResult scan(
        const std::string        &warehouse,
        const std::string        &table_name,
        const std::string        &vec_col,
        const std::vector<float> &query,
        int                       top_k,
        int                       ef_search = 50
    ) const;

    // Execute ailake_search_multimodal_json. Returns empty on any error.
    std::vector<MultimodalRow> search_multimodal(
        const std::string                 &warehouse,
        const std::string                 &table_name,
        const std::vector<ModalQueryArg>  &queries,
        int                                top_k
    ) const;

    // Execute ailake_write_batch_json. Returns snapshot_id or -1 on error.
    int64_t write_batch(
        const std::string              &warehouse,
        const std::string              &ns,
        const std::string              &table_name,
        const std::string              &vec_col,
        int                             dim,
        const std::string              &metric,
        const std::string              &precision,
        const std::vector<int64_t>     &ids,
        const std::vector<std::vector<float>> &embeddings
    ) const;

private:
    AilakeLib() = default;

    void          *handle_        = nullptr;
    search_fn_t    search_fn_     = nullptr;
    multimodal_fn_t multimodal_fn_= nullptr;
    scan_fn_t      scan_fn_       = nullptr;
    write_fn_t     write_fn_      = nullptr;
    free_fn_t      free_fn_       = nullptr;
};

// Escape a string value for embedding in a JSON literal.
inline std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    out += '"';
    return out;
}

} // namespace ailake
