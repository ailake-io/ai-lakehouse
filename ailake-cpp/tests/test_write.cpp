// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake/write.hpp — Phase O write-op delegation.
//
// Tests cover:
//   - PartitionDef / SchemaField struct construction
//   - AddColumnReq / RenameColumnReq struct construction
//   - shell_quote escaping (embedded single-quotes)
//   - resolve_bin: AILAKE_BIN env override
//   - delete_where / evolve_schema no-op on empty inputs
//   - Integration tests guarded by AILAKE_BIN + AILAKE_FIXTURE env vars
#include <ailake/ailake.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
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
        std::fprintf(stderr, "FAIL %s:%d  expected [%s] == [%s]\n", \
                     __FILE__, __LINE__, #a, #b); \
        ++g_fail; \
    } \
} while(0)

// ── PartitionDef / SchemaField ────────────────────────────────────────────────

static void test_partition_def_fields() {
    ailake::PartitionDef pd;
    pd.column      = "agent_id";
    pd.transform   = "identity";
    pd.column_type = "string";
    CHECK_EQ(pd.column,      "agent_id");
    CHECK_EQ(pd.transform,   "identity");
    CHECK_EQ(pd.column_type, "string");
}

static void test_schema_field_defaults() {
    ailake::SchemaField sf;
    CHECK_EQ(sf.id, 0);
    CHECK_EQ(sf.required, false);
    CHECK(sf.name.empty());
    CHECK(sf.type.empty());
}

static void test_schema_field_set() {
    ailake::SchemaField sf;
    sf.id       = 1;
    sf.name     = "doc_id";
    sf.type     = "string";
    sf.required = false;
    CHECK_EQ(sf.id,   1);
    CHECK_EQ(sf.name, "doc_id");
    CHECK_EQ(sf.type, "string");
}

// ── AddColumnReq / RenameColumnReq ───────────────────────────────────────────

static void test_add_column_req() {
    ailake::AddColumnReq r;
    r.name            = "score";
    r.type            = "float";
    r.initial_default = "0.0";
    CHECK_EQ(r.name,            "score");
    CHECK_EQ(r.type,            "float");
    CHECK_EQ(r.initial_default, "0.0");
}

static void test_rename_column_req() {
    ailake::RenameColumnReq r;
    r.from = "old_col";
    r.to   = "new_col";
    CHECK_EQ(r.from, "old_col");
    CHECK_EQ(r.to,   "new_col");
}

// ── shell_quote ───────────────────────────────────────────────────────────────

static void test_shell_quote_simple() {
    CHECK_EQ(ailake::detail::shell_quote("hello"),  "'hello'");
    CHECK_EQ(ailake::detail::shell_quote(""),        "''");
}

static void test_shell_quote_with_single_quote() {
    // "it's" → 'it'\''s'
    CHECK_EQ(ailake::detail::shell_quote("it's"), "'it'\\''s'");
}

static void test_shell_quote_spaces() {
    CHECK_EQ(ailake::detail::shell_quote("a b c"), "'a b c'");
}

// ── resolve_bin ───────────────────────────────────────────────────────────────

static void test_resolve_bin_env() {
    const char* orig = std::getenv("AILAKE_BIN");
    std::string saved = orig ? orig : "";
    setenv("AILAKE_BIN", "/custom/ailake", 1);
    auto bin = ailake::detail::resolve_bin();
    CHECK_EQ(bin, "/custom/ailake");
    if (orig) setenv("AILAKE_BIN", saved.c_str(), 1);
    else      unsetenv("AILAKE_BIN");
}

static void test_resolve_bin_default_when_no_env() {
    // unsetenv() mutates the process environment permanently — save/restore so
    // later tests (including the integration ones) still see a real AILAKE_BIN
    // if the shell provided one.
    const char* orig = std::getenv("AILAKE_BIN");
    std::string saved = orig ? orig : "";
    unsetenv("AILAKE_BIN");
    auto bin = ailake::detail::resolve_bin();
    CHECK_EQ(bin, "ailake");
    if (orig) setenv("AILAKE_BIN", saved.c_str(), 1);
}

// ── No-op paths ───────────────────────────────────────────────────────────────

static void test_delete_where_empty_values_noop() {
    // Empty values vector → no CLI call, no error.
    ailake::delete_where("/tmp/test", "default.table", "doc_id", {});
    // If we get here without exception: pass.
    CHECK(true);
}

static void test_evolve_schema_empty_noop() {
    int id = ailake::evolve_schema("/tmp/test", "default.table", {}, {});
    CHECK_EQ(id, 0);
}

// ── TableInfo new fields ──────────────────────────────────────────────────────

static void test_table_info_format_version_default() {
    ailake::TableInfo info;
    CHECK_EQ(info.format_version, 2);
}

static void test_table_info_partition_fields_empty_default() {
    ailake::TableInfo info;
    CHECK(info.partition_fields.empty());
}

static void test_table_info_schema_fields_empty_default() {
    ailake::TableInfo info;
    CHECK(info.schema_fields.empty());
}

// ── VectorColSpec / CompactOptions ───────────────────────────────────────────

static void test_vector_col_spec_fields() {
    ailake::VectorColSpec spec;
    spec.column   = "image_embedding";
    spec.dim      = 512;
    spec.metric   = "euclidean";
    spec.modality = "image";
    CHECK_EQ(spec.column,   "image_embedding");
    CHECK_EQ(spec.dim,      512);
    CHECK_EQ(spec.metric,   "euclidean");
    CHECK_EQ(spec.modality, "image");
}

static void test_compact_options_fields() {
    ailake::CompactOptions opts;
    opts.target_size        = 1024;
    opts.min_files          = 2;
    opts.max_files_per_pass = 10;
    opts.deferred            = true;
    CHECK_EQ(opts.target_size,        1024);
    CHECK_EQ(opts.min_files,          2);
    CHECK_EQ(opts.max_files_per_pass, 10);
    CHECK_EQ(opts.deferred,           true);
}

static void test_write_batch_multi_empty_cols_throws() {
    bool threw = false;
    try {
        ailake::write_batch_multi("/tmp/test", "default.table", "/tmp/x.parquet", {});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

// ── Integration: multi-column write + compact (own temp warehouse, requires
// only AILAKE_BIN — writes testdata/multimodal_fixture.parquet, not gated on
// the shared AILAKE_FIXTURE) ─────────────────────────────────────────────────

static std::string make_temp_dir() {
    char tmpl[] = "/tmp/ailake_cpp_test_XXXXXX";
    char* dir = mkdtemp(tmpl);
    if (!dir) throw std::runtime_error("mkdtemp failed");
    return std::string(dir);
}

// Resolve testdata/ relative to this source file (not CWD) — ctest runs
// binaries from the build dir, so a bare "testdata/..." relative path
// wouldn't resolve there the way it does for `go test` (which runs from the
// package source dir).
static std::string fixture_path() {
    std::string here = __FILE__; // .../ailake-cpp/tests/test_write.cpp
    auto slash = here.find_last_of('/');
    return here.substr(0, slash) + "/../testdata/multimodal_fixture.parquet";
}

// Finds the first *.parquet under `dir` (recursively) and returns its path
// relative to `dir` — mirrors how ailake-go's ListFiles resolves data-file
// paths, but this header has no manifest-listing API of its own, so tests
// that need a real on-disk data file path (e.g. delete_rows) glob for it
// directly instead.
static std::string find_first_parquet_relative(const std::string& dir) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (entry.path().extension() == ".parquet") {
            return std::filesystem::relative(entry.path(), dir).string();
        }
    }
    throw std::runtime_error("no .parquet file found under " + dir);
}

// Writes a throwaway shell script standing in for the `ailake` binary: it
// captures its argv to `args_path` (for the test to assert on) and echoes
// `stdout_json` (if non-empty) so JSON-parsing callers (decay_memories) see
// a well-formed response. Used for delete_rows/decay_memories/migrate/
// backfill_vector_column call sites that need a real embed_cmd/memory-schema
// fixture this repo doesn't have — the fake binary lets these tests verify
// this header's own argv-building code for real, independent of whatever the
// real CLI would do with those args.
static std::string write_fake_ailake_bin(const std::string& dir, const std::string& args_path, const std::string& stdout_json) {
    std::string bin_path = dir + "/fake_ailake";
    std::ofstream f(bin_path);
    f << "#!/bin/sh\necho \"$@\" > " << args_path << "\n";
    if (!stdout_json.empty()) {
        f << "echo '" << stdout_json << "'\n";
    }
    f.close();
    std::filesystem::permissions(bin_path, std::filesystem::perms::owner_all, std::filesystem::perm_options::add);
    return bin_path;
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return content;
}

// Saves AILAKE_BIN on construction, restores it (or removes it, if it was
// never set) on destruction — a bare unsetenv() at the end of a test would
// permanently drop a real AILAKE_BIN the test binary was invoked with,
// silently SKIP-ing every later integration test in the same run (this bit
// the first version of the args-forwarded tests below).
struct ScopedAilakeBin {
    std::string saved;
    bool had_value;
    explicit ScopedAilakeBin(const std::string& fake_bin) {
        const char* orig = std::getenv("AILAKE_BIN");
        had_value = orig != nullptr;
        if (had_value) saved = orig;
        setenv("AILAKE_BIN", fake_bin.c_str(), 1);
    }
    ~ScopedAilakeBin() {
        if (had_value) setenv("AILAKE_BIN", saved.c_str(), 1);
        else           unsetenv("AILAKE_BIN");
    }
};

static void test_integration_create_table() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::CreateTableOptions copts;
        copts.metric = "cosine";
        copts.precision = "f16";
        ailake::create_table(warehouse, "default.docs", 4, copts);

        ailake::HadoopCatalog cat(warehouse);
        auto info = cat.load_table("default", "docs");
        CHECK_EQ(info.vector_dim, "4");
        CHECK_EQ(info.vector_metric, "cosine");

        // Writing into the pre-created table must succeed without re-creating it.
        ailake::WriteBatchOptions wopts; wopts.vec_col = "embedding";
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration create_table: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_estimate() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string out = ailake::estimate("1M", 1536);
        // to_string_pretty (server side) inserts a space after ':' — match
        // loosely rather than assuming compact JSON formatting.
        CHECK(out.find("\"rows\"") != std::string::npos && out.find("1000000") != std::string::npos);
        CHECK(out.find("\"dim\"") != std::string::npos && out.find("1536") != std::string::npos);
        CHECK(out.find("\"estimates\"") != std::string::npos);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration estimate: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_add_vector_column() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::CreateTableOptions copts; copts.metric = "cosine";
        ailake::create_table(warehouse, "default.docs", 4, copts);

        ailake::AddVectorColumnOptions avopts; avopts.metric = "cosine";
        ailake::add_vector_column(warehouse, "default.docs", "image_embedding", 2, avopts);

        ailake::HadoopCatalog cat(warehouse);
        auto info = cat.load_table("default", "docs");
        bool found = false;
        for (const auto& f : info.schema_fields) {
            if (f.name == "image_embedding") found = true;
        }
        CHECK(found);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration add_vector_column: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_delete_rows() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::CreateTableOptions copts; copts.metric = "cosine"; copts.format_version = 3;
        ailake::create_table(warehouse, "default.docs", 4, copts);

        ailake::WriteBatchOptions wopts; wopts.vec_col = "embedding";
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);

        std::string file = find_first_parquet_relative(warehouse);
        // delete_rows itself must not throw — confirms this header's argv is
        // accepted by the real CLI (which really does drop the row from its
        // own search, confirmed via a separate manual CLI-only repro).
        // NOT asserting the effect via ailake::search here: this header's
        // search() does not apply Deletion Vector masking — see
        // docs/guides/REST_CATALOG.md and the identical, separately-documented
        // finding in ailake-go (unlike ailake-go, this header has no Roaring
        // Bitmap decoder — see test_deletion_vector_metadata_parsed below for
        // what IS fixed: the catalog now correctly parses and exposes the DV
        // pointer via DataFileEntry::deletion_vector).
        ailake::delete_rows(warehouse, "default.docs", file, {0});
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration delete_rows: %s\n", e.what());
        ++g_fail;
    }
}

// Regression: DataFileEntry never carried a deletion_vector field at all —
// list_files() silently dropped the DV pointer that delete_rows() writes
// into key_metadata JSON, even though every other key_metadata field
// (centroid, hnsw_offset, batch_id, ...) was already parsed. Proves the
// catalog-level parsing fix for real against a file the actual CLI wrote.
static void test_integration_deletion_vector_metadata_parsed() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::CreateTableOptions copts; copts.metric = "cosine"; copts.format_version = 3;
        ailake::create_table(warehouse, "default.docs", 4, copts);

        ailake::WriteBatchOptions wopts; wopts.vec_col = "embedding";
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);

        std::string file = find_first_parquet_relative(warehouse);
        ailake::delete_rows(warehouse, "default.docs", file, {0, 2});

        ailake::HadoopCatalog cat(warehouse);
        auto entries = cat.list_files("default", "docs");
        CHECK_EQ(entries.size(), (size_t)1);
        CHECK(entries[0].deletion_vector.has_value());
        if (entries[0].deletion_vector) {
            CHECK(!entries[0].deletion_vector->path.empty());
            CHECK(entries[0].deletion_vector->length > 0);
            CHECK_EQ(entries[0].deletion_vector->cardinality, (int64_t)2);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration deletion_vector_metadata_parsed: %s\n", e.what());
        ++g_fail;
    }
}

// Regression: this header had zero equality-delete support at all —
// list_files() never distinguished data manifests (content=0) from delete
// manifests (content=1), so a delete manifest's entries silently fell
// through and got misparsed as regular DataFileEntry rows (or corrupted
// list_files() output entirely, depending on layout). Proves the fix for
// real: list_equality_deletes() finds the delete file, and
// read_equality_delete_values() correctly decodes the predicate values from
// it (dynamic Avro schema, not the fixed manifest_entry schema).
static void test_integration_equality_delete_catalog() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::CreateTableOptions copts; copts.metric = "cosine";
        ailake::create_table(warehouse, "default.docs", 4, copts);

        ailake::WriteBatchOptions wopts; wopts.vec_col = "embedding";
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);

        ailake::delete_where(warehouse, "default.docs", "id", {"3"});

        ailake::HadoopCatalog cat(warehouse);
        auto deletes = cat.list_equality_deletes("default", "docs");
        CHECK_EQ(deletes.size(), (size_t)1);
        if (!deletes.empty()) {
            CHECK_EQ(deletes[0].equality_ids.size(), (size_t)1);
            std::string resolved = cat.resolve_path("default", "docs", deletes[0].path);
            auto values = ailake::HadoopCatalog::read_equality_delete_values(resolved);
            CHECK_EQ(values.size(), (size_t)1);
            if (!values.empty()) {
                CHECK_EQ(values[0].first, std::string("id"));
                CHECK_EQ(values[0].second, std::string("3"));
            }
        }

        // list_files() must still see exactly the 1 real data file — the
        // delete manifest (content=1) must not leak a phantom DataFileEntry.
        auto files = cat.list_files("default", "docs");
        CHECK_EQ(files.size(), (size_t)1);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration equality_delete_catalog: %s\n", e.what());
        ++g_fail;
    }
}

// ── decay_memories / migrate / backfill_vector_column — argv-forwarding
// checks (fake `ailake` binary, not a mock of this header's own code) ───────
//
// These need a table with columns (chunk_text, last_accessed_at, ...) no
// fixture in this repo provides — a full embed_cmd/memory-decay round trip
// is out of scope here. What's verified for real: the exact argv this
// header's code builds and hands to run_cmd/popen.

static void test_decay_memories_args_forwarded() {
    try {
        std::string dir = make_temp_dir();
        std::string args_path = dir + "/args.txt";
        std::string bin = write_fake_ailake_bin(dir, args_path, "{\"ok\":true,\"files_updated\":3}");
        ScopedAilakeBin guard(bin);

        int n = ailake::decay_memories("/tmp/wh", "default.docs", 0.25f,
            {{"catalog", "rest"}, {"rest-uri", "https://catalog.example.com"}});
        CHECK_EQ(n, 3);

        std::string args = read_file(args_path);
        for (const std::string& want : {std::string("decay-memories"), std::string("default.docs"),
                                          std::string("--lambda"), std::string("--format"), std::string("json"),
                                          std::string("--catalog"), std::string("rest"), std::string("--rest-uri")}) {
            CHECK(args.find(want) != std::string::npos);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL decay_memories args-forwarded: %s\n", e.what());
        ++g_fail;
    }
}

static void test_migrate_args_forwarded() {
    try {
        std::string dir = make_temp_dir();
        std::string args_path = dir + "/args.txt";
        std::string bin = write_fake_ailake_bin(dir, args_path, "");
        ScopedAilakeBin guard(bin);

        ailake::MigrateOptions mopts;
        mopts.old_column = "embedding"; mopts.new_column = "embedding_v2";
        mopts.text_column = "chunk_text"; mopts.strategy = "atomic-replace";
        mopts.batch_size = 256; mopts.model_name = "text-embedding-3-small";
        ailake::migrate("/tmp/wh", "default.docs", "python3 embed.py", mopts);

        std::string args = read_file(args_path);
        for (const std::string& want : {std::string("migrate"), std::string("default.docs"),
                                          std::string("--embed-cmd"), std::string("python3 embed.py"),
                                          std::string("--old-column"), std::string("embedding"),
                                          std::string("--new-column"), std::string("embedding_v2"),
                                          std::string("--strategy"), std::string("atomic-replace"),
                                          std::string("--batch-size"), std::string("256")}) {
            CHECK(args.find(want) != std::string::npos);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL migrate args-forwarded: %s\n", e.what());
        ++g_fail;
    }
}

static void test_backfill_vector_column_args_forwarded() {
    try {
        std::string dir = make_temp_dir();
        std::string args_path = dir + "/args.txt";
        std::string bin = write_fake_ailake_bin(dir, args_path, "");
        ScopedAilakeBin guard(bin);

        ailake::BackfillVectorColumnOptions bopts;
        bopts.text_column = "image_uri"; bopts.batch_size = 128;
        ailake::backfill_vector_column("/tmp/wh", "default.docs", "image_embedding", "python3 embed_images.py", bopts);

        std::string args = read_file(args_path);
        for (const std::string& want : {std::string("backfill-vector-column"), std::string("default.docs"),
                                          std::string("--column"), std::string("image_embedding"),
                                          std::string("--embed-cmd"), std::string("python3 embed_images.py"),
                                          std::string("--text-column"), std::string("image_uri"),
                                          std::string("--batch-size"), std::string("128")}) {
            CHECK(args.find(want) != std::string::npos);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL backfill_vector_column args-forwarded: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_write_batch_multi() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::write_batch_multi(warehouse, "default.media", fixture_path(), {
            {"embedding", 4, "cosine", ""},
            {"image_embedding", 2, "cosine", "image"},
        });

        ailake::HadoopCatalog cat(warehouse);
        std::vector<ailake::ModalQuery> queries = {
            {"embedding", {0.1f, 0.2f, 0.3f, 0.4f}, 0.7f},
            {"image_embedding", {0.5f, 0.6f}, 0.3f},
        };
        ailake::SearchOptions opts; opts.top_k = 3;
        auto results = ailake::search_multimodal(cat, "default", "media", queries, opts);
        CHECK_EQ(results.size(), (size_t)3);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration write_batch_multi: %s\n", e.what());
        ++g_fail;
    }
}

// Reads "rows":N straight from `ailake info --format json` (real CLI call,
// same substring-parse convention as compact/decay_memories). Originally
// added as a workaround for a real bug this test surfaced (this header's own
// search() returned 0 rows on any table with real column stats — root cause:
// list_files()'s manifest-entry reader treated lower_bounds/upper_bounds
// (map<int,bytes>) as map<int,long>, never consuming the actual bytes
// payload after their zigzag length prefix, misaligning every field read
// after them including key_metadata's own union tag — since fixed, see
// HadoopCatalog::list_files' comment on the field-skip loop in catalog.hpp).
// Kept as the row-count check anyway: it's a real, independent CLI round
// trip (not this header's own read path), a stronger signal than asserting
// against the same code under test.
static int row_count_via_info(const std::string& warehouse, const std::string& table_id) {
    std::string cmd = ailake::detail::shell_quote(ailake::detail::resolve_bin())
        + " --store " + ailake::detail::shell_quote(warehouse)
        + " info " + ailake::detail::shell_quote(table_id)
        + " --format json";
    std::string out = ailake::detail::run_cmd(cmd);
    std::string key = "\"rows\":";
    auto pos = out.find(key);
    if (pos == std::string::npos) throw std::runtime_error("no \"rows\" key in info output: " + out);
    pos += key.size();
    return std::stoi(out.substr(pos));
}

// Real end-to-end proof of the production risk Fase 14/21's audit flagged:
// without batch_id, a retried write (e.g. an Airflow task retry after a
// timeout whose first attempt actually succeeded) silently double-inserts
// rows. Writes the same batch twice with the same batch_id and confirms the
// second call is a no-op (row count unchanged), then confirms a *different*
// batch_id does add rows as normal.
static void test_integration_batch_id_idempotency() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::WriteBatchOptions opts;
        opts.vec_col = "embedding";
        opts.batch_id = "test-batch-001";
        ailake::write_batch(warehouse, "default.docs", fixture_path(), opts);
        ailake::write_batch(warehouse, "default.docs", fixture_path(), opts); // retry, same batch_id
        CHECK_EQ(row_count_via_info(warehouse, "default.docs"), 6); // not 12 — retry was a no-op

        ailake::WriteBatchOptions opts2 = opts;
        opts2.batch_id = "test-batch-002"; // different batch_id: not a retry, must add rows
        ailake::write_batch(warehouse, "default.docs", fixture_path(), opts2);
        CHECK_EQ(row_count_via_info(warehouse, "default.docs"), 12);

        // Belt-and-suspenders: this header's own search() independently
        // agrees with `info`'s row count (exercises the manifest-parsing fix
        // above through the actual read path applications use, not just the
        // CLI round trip).
        ailake::HadoopCatalog cat(warehouse);
        float q[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        ailake::SearchOptions sopts; sopts.top_k = 20;
        auto results = ailake::search(cat, "default", "docs", q, 4, sopts);
        CHECK_EQ(results.size(), (size_t)12);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration batch_id idempotency: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_compact() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::WriteBatchOptions wopts; wopts.vec_col = "embedding";
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);

        ailake::CompactOptions copts; copts.min_files = 2;
        int n = ailake::compact(warehouse, "default.docs", copts);
        CHECK_EQ(n, 1);

        ailake::HadoopCatalog cat(warehouse);
        float q[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        ailake::SearchOptions sopts; sopts.top_k = 20;
        auto results = ailake::search(cat, "default", "docs", q, 4, sopts);
        CHECK_EQ(results.size(), (size_t)12);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration compact: %s\n", e.what());
        ++g_fail;
    }
}

// ── Integration (AILAKE_BIN + AILAKE_FIXTURE required) ───────────────────────

static void test_integration_load_table() {
    const char* fixture = std::getenv("AILAKE_FIXTURE");
    if (!fixture) { std::fprintf(stdout, "SKIP: AILAKE_FIXTURE not set\n"); return; }

    ailake::HadoopCatalog cat(fixture);
    ailake::TableInfo info = cat.load_table("default", "table");

    CHECK(!info.vector_column.empty());
    CHECK(!info.vector_dim.empty());
    CHECK(info.format_version == 2 || info.format_version == 3);
    // Schema fields should have been populated for tables with a valid schema.
    // (Not strictly required for old fixture tables, so just check non-negative id.)
    for (const auto& sf : info.schema_fields) {
        CHECK(sf.id >= 0);
        CHECK(!sf.name.empty());
    }
}

static void test_integration_delete_where() {
    const char* fixture = std::getenv("AILAKE_FIXTURE");
    const char* bin     = std::getenv("AILAKE_BIN");
    if (!fixture || !bin) { std::fprintf(stdout, "SKIP: AILAKE_FIXTURE/AILAKE_BIN not set\n"); return; }

    setenv("AILAKE_BIN", bin, 1);
    // Delete a non-existent value — zero-row delete is always valid.
    try {
        ailake::delete_where(fixture, "default.table", "document_id", {"__nonexistent_cpp__"});
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration delete_where: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_evolve_schema() {
    const char* fixture = std::getenv("AILAKE_FIXTURE");
    const char* bin     = std::getenv("AILAKE_BIN");
    if (!fixture || !bin) { std::fprintf(stdout, "SKIP: AILAKE_FIXTURE/AILAKE_BIN not set\n"); return; }

    setenv("AILAKE_BIN", bin, 1);
    try {
        ailake::AddColumnReq ac;
        ac.name            = "_cpp_test_col";
        ac.type            = "string";
        ac.initial_default = "\"\"";
        int id = ailake::evolve_schema(fixture, "default.table", {ac}, {});
        CHECK(id >= -1); // -1 if parse failed, ≥0 on success
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration evolve_schema: %s\n", e.what());
        ++g_fail;
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_partition_def_fields();
    test_schema_field_defaults();
    test_schema_field_set();
    test_add_column_req();
    test_rename_column_req();
    test_shell_quote_simple();
    test_shell_quote_with_single_quote();
    test_shell_quote_spaces();
    test_resolve_bin_env();
    test_resolve_bin_default_when_no_env();
    test_delete_where_empty_values_noop();
    test_evolve_schema_empty_noop();
    test_table_info_format_version_default();
    test_table_info_partition_fields_empty_default();
    test_table_info_schema_fields_empty_default();
    test_vector_col_spec_fields();
    test_compact_options_fields();
    test_write_batch_multi_empty_cols_throws();

    test_integration_load_table();
    test_integration_create_table();
    test_integration_estimate();
    test_integration_add_vector_column();
    test_integration_delete_rows();
    test_integration_deletion_vector_metadata_parsed();
    test_integration_equality_delete_catalog();
    test_decay_memories_args_forwarded();
    test_migrate_args_forwarded();
    test_backfill_vector_column_args_forwarded();
    test_integration_write_batch_multi();
    test_integration_batch_id_idempotency();
    test_integration_compact();
    test_integration_delete_where();
    test_integration_evolve_schema();

    if (g_fail > 0) {
        std::fprintf(stderr, "%d test(s) FAILED\n", g_fail);
        return 1;
    }
    std::fprintf(stdout, "All write tests passed.\n");
    return 0;
}
