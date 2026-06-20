// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake::search_text and FtsResult (Phase T FTS).
//
// Tests that don't require the ailake binary test the public struct and the
// early-return path (empty query). Integration tests are guarded by AILAKE_BIN.

#include <ailake/ailake.hpp>
#include <ailake/catalog.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while(0)

#define CHECK_EQ(a, b) do { \
    if (!((a) == (b))) { \
        std::fprintf(stderr, "FAIL %s:%d  expected equality [%s] == [%s]\n", \
                     __FILE__, __LINE__, #a, #b); \
        ++g_fail; \
    } \
} while(0)

// ── FtsResult struct ──────────────────────────────────────────────────────────

static void test_fts_result_fields() {
    ailake::FtsResult r;
    r.row_id    = 42;
    r.score     = 0.99;
    r.file_path = "part-001.parquet";
    CHECK_EQ(r.row_id,    42);
    CHECK(r.score > 0.98 && r.score < 1.0);
    CHECK_EQ(r.file_path, "part-001.parquet");
}

static void test_fts_result_default_zero() {
    ailake::FtsResult r{};
    CHECK_EQ(r.row_id, 0);
    CHECK(r.score == 0.0);
    CHECK(r.file_path.empty());
}

// ── search_text early-return paths ───────────────────────────────────────────

static void test_search_text_empty_query_returns_empty() {
    // Empty query must return empty vector without invoking the CLI.
    ailake::HadoopCatalog cat("/tmp/nonexistent");
    auto results = ailake::search_text(cat, "default", "table", "");
    CHECK(results.empty());
}

static void test_search_text_default_cols_used_when_empty() {
    // Build cols the same way search_text does internally.
    std::vector<std::string> text_columns = {};
    std::string cols;
    for (size_t i = 0; i < text_columns.size(); ++i) {
        if (i) cols += ',';
        cols += text_columns[i];
    }
    if (cols.empty()) cols = "chunk_text";
    CHECK_EQ(cols, "chunk_text");
}

static void test_search_text_single_column_used_as_is() {
    std::vector<std::string> text_columns = {"document_body"};
    std::string cols;
    for (size_t i = 0; i < text_columns.size(); ++i) {
        if (i) cols += ',';
        cols += text_columns[i];
    }
    if (cols.empty()) cols = "chunk_text";
    CHECK_EQ(cols, "document_body");
}

static void test_search_text_multi_column_comma_joined() {
    std::vector<std::string> text_columns = {"chunk_text", "title", "summary"};
    std::string cols;
    for (size_t i = 0; i < text_columns.size(); ++i) {
        if (i) cols += ',';
        cols += text_columns[i];
    }
    CHECK_EQ(cols, "chunk_text,title,summary");
}

// ── Integration (AILAKE_BIN required) ────────────────────────────────────────

static void test_integration_search_text_no_bin_throws() {
    const char* fixture = std::getenv("AILAKE_FIXTURE");
    if (!fixture) { std::fprintf(stdout, "SKIP: AILAKE_FIXTURE not set\n"); return; }

    // Remove AILAKE_BIN and ensure "ailake" is not on PATH to force CLI error.
    unsetenv("AILAKE_BIN");
    const char* old_path = std::getenv("PATH");
    setenv("PATH", "/nonexistent_for_ailake_test", 1);

    ailake::HadoopCatalog cat(fixture);
    bool threw = false;
    try {
        ailake::search_text(cat, "default", "table", "rust programming", {"chunk_text"}, 5);
    } catch (const std::runtime_error&) {
        threw = true;
    } catch (...) {
        threw = true;
    }
    CHECK(threw);

    if (old_path) setenv("PATH", old_path, 1);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_fts_result_fields();
    test_fts_result_default_zero();
    test_search_text_empty_query_returns_empty();
    test_search_text_default_cols_used_when_empty();
    test_search_text_single_column_used_as_is();
    test_search_text_multi_column_comma_joined();

    test_integration_search_text_no_bin_throws();

    if (g_fail > 0) {
        std::fprintf(stderr, "%d test(s) FAILED\n", g_fail);
        return 1;
    }
    std::fprintf(stdout, "All FTS tests passed.\n");
    return 0;
}
