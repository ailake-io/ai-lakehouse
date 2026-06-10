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

// Singleton holding dlopen handle and resolved C-ABI function pointers.
// Thread-safe after Load() completes.
class AilakeLib {
public:
    using search_fn_t = char *(*)(const char *);
    using write_fn_t  = char *(*)(const char *);
    using free_fn_t   = void (*)(char *);

    static AilakeLib &get();

    // Load libailake_jni.so. If lib_path is empty, searches LD_LIBRARY_PATH.
    // Safe to call multiple times — no-ops after first successful load.
    bool load(const std::string &lib_path = "");

    bool is_ready() const { return search_fn_ != nullptr; }

    // Execute ailake_search_json. Returns empty on any error.
    std::vector<SearchRow> search(
        const std::string        &warehouse,
        const std::string        &table_name,
        const std::string        &vec_col,
        const std::vector<float> &query,
        int                       top_k,
        int                       ef_search = 50
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

    void        *handle_     = nullptr;
    search_fn_t  search_fn_  = nullptr;
    write_fn_t   write_fn_   = nullptr;
    free_fn_t    free_fn_    = nullptr;
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
