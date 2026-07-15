// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Iceberg HadoopCatalog reader for AI-Lake tables (C++ implementation).
// Reads version-hint.text → metadata.json → Avro manifest list → manifest file.
//
// Avro OCF parsing: minimal hand-rolled decoder (no external deps).
// JSON parsing: nlohmann/json (header-only, fetched via CMake FetchContent).
//
// For production use, link with Apache Arrow C++ which includes both Parquet
// and Iceberg catalog readers; this implementation targets embedded/CLI usage.
#pragma once

#include "footer.hpp"
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal JSON parser — used only if nlohmann/json is not available.
// Define AILAKE_USE_NLOHMANN_JSON and include nlohmann/json.hpp before this
// header to use the full JSON library.
#ifndef AILAKE_USE_NLOHMANN_JSON
#  include <sstream>
#  include <map>
#endif

namespace ailake {

// ---------------------------------------------------------------------------
// PartitionDef / SchemaField — Phase K / N
// ---------------------------------------------------------------------------

// PartitionDef mirrors ailake_core::PartitionDef.
// transform is "identity" or "truncate[W]" (e.g. "truncate[4]").
struct PartitionDef {
    std::string column;
    std::string transform;
    std::string column_type; // Iceberg type: "string", "int", "long", ...
};

// SchemaField mirrors one field in the Iceberg table schema.
struct SchemaField {
    int         id       = 0;
    std::string name;
    std::string type;    // Iceberg primitive type string
    bool        required = false;
};

// ---------------------------------------------------------------------------
// DataFileEntry — mirrors ailake_catalog::provider::DataFileEntry
// ---------------------------------------------------------------------------

// Secondary vector column index entry (Phase 8 multi-column tables).
struct ExtraVectorIndex {
    std::string column;
    uint32_t    dim          = 0;
    uint64_t    hnsw_offset  = 0;
    uint64_t    hnsw_len     = 0;
    std::string centroid_b64; // base64-encoded F32 centroid (may be empty)
    float       radius       = 0.f;
};

struct DataFileEntry {
    std::string path;
    uint64_t    record_count   = 0;
    uint64_t    file_size_bytes= 0;
    std::vector<float> centroid;
    float       radius         = 0.f;
    std::optional<uint64_t> hnsw_offset;
    std::optional<uint64_t> hnsw_len;
    std::string vector_column;
    uint32_t    vector_dim     = 0;
    std::vector<ExtraVectorIndex> extra_vector_indexes; // secondary columns (Phase 8)
    std::string index_status;   // "ready" | "indexing"
    std::string batch_id;
    std::string embedding_model; // "<name>" or "<name>@<version>"; empty if not set
    std::string partition_value; // agent_id or other partition value (Phase 9)
};

// ---------------------------------------------------------------------------
// FileSearchResult (declared here, used in ailake.hpp)
// ---------------------------------------------------------------------------

struct FileSearchResult {
    uint64_t    row_id;
    float       distance;
    std::string file_path;
};

// ---------------------------------------------------------------------------
// TableInfo — mirrors JSON output of "ailake info --format json"
// ---------------------------------------------------------------------------

struct TableInfo {
    std::string table;
    std::string location;
    std::string vector_column;
    std::string vector_dim;
    std::string vector_metric;
    std::string embedding_model; // "<name>" or "<name>@<version>"; empty if not set
    int         files            = 0;
    int         indexed_files    = 0;
    uint64_t    rows             = 0;
    uint64_t    size_bytes       = 0;
    std::optional<int64_t> snapshot_id;
    int format_version           = 2;  // 2 or 3
    std::vector<PartitionDef> partition_fields; // empty for unpartitioned tables
    std::vector<SchemaField>  schema_fields;    // current schema fields
};

// ---------------------------------------------------------------------------
// metric_from_str helper (used in ailake.hpp)
// ---------------------------------------------------------------------------

inline Metric metric_from_str(const std::string& s) {
    if (s == "euclidean") return Metric::Euclidean;
    if (s == "dotproduct" || s == "dot") return Metric::DotProduct;
    if (s == "normalized_cosine") return Metric::NormalizedCosine;
    return Metric::Cosine;
}

// ---------------------------------------------------------------------------
// Minimal Avro OCF reader (data-only, no schema evolution)
// ---------------------------------------------------------------------------

namespace detail {

// Read a zigzag-encoded varint from a stream.
//
// Throws std::runtime_error on EOF/read failure instead of spinning forever:
// without this check, a failed s.read() leaves the loop-local `b` byte
// uninitialized, and if that stack garbage happens to have bit 0x80 set,
// every subsequent read also fails silently (the stream is already in a
// failed state) and the loop never terminates — a real infinite hang, not
// just a wrong result.
inline int64_t read_zigzag(std::istream& s) {
    uint64_t raw = 0;
    int shift = 0;
    while (true) {
        uint8_t b;
        if (!s.read(reinterpret_cast<char*>(&b), 1))
            throw std::runtime_error("avro: unexpected EOF reading zigzag varint");
        raw |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
    }
    return (int64_t)((raw >> 1) ^ -(int64_t)(raw & 1));
}

// Read a bytes/string field (zigzag length + raw bytes).
inline std::string read_avro_bytes(std::istream& s) {
    int64_t n = read_zigzag(s);
    if (n <= 0) return {};
    std::string out(n, '\0');
    s.read(out.data(), n);
    return out;
}

// Skip N bytes.
inline void skip(std::istream& s, std::streamsize n) {
    s.seekg(n, std::ios::cur);
}

// Read Avro OCF container header, return schema JSON string.
inline std::string read_ocf_header(std::istream& s, std::array<uint8_t,16>& sync_marker) {
    // Magic: "Obj\x01"
    char magic[4]; s.read(magic, 4);
    if (magic[0]!='O'||magic[1]!='b'||magic[2]!='j'||magic[3]!=1)
        throw std::runtime_error("avro: bad OCF magic");

    // Meta map: zigzag block count + [string, bytes] pairs
    std::string schema;
    for (;;) {
        int64_t count = read_zigzag(s);
        if (count == 0) break;
        if (count < 0) { read_zigzag(s); count = -count; } // block size
        for (int64_t i = 0; i < count; ++i) {
            std::string key = read_avro_bytes(s);
            std::string val = read_avro_bytes(s);
            if (key == "avro.schema") schema = val;
        }
    }
    // Sync marker (16 bytes)
    s.read(reinterpret_cast<char*>(sync_marker.data()), 16);
    return schema;
}

} // namespace detail

// ---------------------------------------------------------------------------
// HadoopCatalog
// ---------------------------------------------------------------------------

class HadoopCatalog {
public:
    explicit HadoopCatalog(std::string warehouse) : warehouse_(std::move(warehouse)) {}

    const std::string& warehouse() const { return warehouse_; }

    // ns/tbl are accepted for API stability but not used in the join: the
    // Rust catalog writer always stores data-file paths (DataFileEntry::path)
    // relative to the warehouse ROOT, already including namespace/table
    // (e.g. "default/docs/data/part-00000.parquet") — joining table_dir(ns,
    // tbl) again on top would double-prefix namespace/table.
    std::string resolve_path(const std::string& ns, const std::string& tbl,
                              const std::string& rel) const {
        (void)ns;
        (void)tbl;
        return resolve_warehouse_path(rel);
    }

    TableInfo load_table(const std::string& ns, const std::string& tbl) const {
        auto dir  = table_dir(ns, tbl);
        auto meta = read_metadata_json(dir);

        TableInfo info;
        info.table    = ns + "." + tbl;
        info.location = dir;

        // Helper: extract first string value after key in JSON text.
        auto get_str_in = [](const std::string& json, const std::string& key) -> std::string {
            auto pos = json.find("\"" + key + "\":");
            if (pos == std::string::npos) return {};
            pos = json.find('"', pos + key.size() + 3);
            if (pos == std::string::npos) return {};
            auto end = json.find('"', pos + 1);
            if (end == std::string::npos) return {};
            return json.substr(pos + 1, end - pos - 1);
        };
        // Helper: extract first integer value after key.
        auto get_int_in = [](const std::string& json, const std::string& key) -> std::optional<int64_t> {
            auto pos = json.find("\"" + key + "\":");
            if (pos == std::string::npos) return {};
            pos += key.size() + 3;
            while (pos < json.size() && (json[pos]==' '||json[pos]=='\n'||json[pos]=='\r')) ++pos;
            if (pos >= json.size() || json[pos] == 'n') return {};
            try { return std::stoll(json.substr(pos)); } catch (...) { return {}; }
        };

        info.vector_column   = get_str_in(meta, "ailake.vector-column");
        info.vector_dim      = get_str_in(meta, "ailake.vector-dim");
        info.vector_metric   = get_str_in(meta, "ailake.vector-metric");
        info.embedding_model = get_str_in(meta, "ailake.embedding-model");

        if (auto v = get_int_in(meta, "current-snapshot-id")) info.snapshot_id = *v;
        if (auto v = get_int_in(meta, "format-version"))      info.format_version = (int)*v;

        // --- Parse current schema fields ---
        int current_schema_id = -1;
        if (auto v = get_int_in(meta, "current-schema-id")) current_schema_id = (int)*v;

        // Find the "schemas" array and iterate over JSON objects until schema-id matches.
        {
            auto schemas_pos = meta.find("\"schemas\":");
            if (schemas_pos != std::string::npos) {
                auto bracket = meta.find('[', schemas_pos);
                if (bracket != std::string::npos) {
                    // Walk schema objects: find each '{' ... '}' block.
                    size_t p = bracket + 1;
                    while (p < meta.size()) {
                        auto obj_start = meta.find('{', p);
                        if (obj_start == std::string::npos) break;
                        // Find matching closing brace (depth-aware).
                        size_t depth = 1; size_t q = obj_start + 1;
                        while (q < meta.size() && depth > 0) {
                            if (meta[q] == '{') ++depth;
                            else if (meta[q] == '}') --depth;
                            ++q;
                        }
                        std::string schema_obj = meta.substr(obj_start, q - obj_start);
                        if (auto sid = get_int_in(schema_obj, "schema-id")) {
                            if ((int)*sid == current_schema_id) {
                                // Parse fields array inside this schema object.
                                auto fa = schema_obj.find("\"fields\":");
                                if (fa != std::string::npos) {
                                    auto fb = schema_obj.find('[', fa);
                                    if (fb != std::string::npos) {
                                        size_t fp = fb + 1;
                                        while (fp < schema_obj.size()) {
                                            auto fo = schema_obj.find('{', fp);
                                            if (fo == std::string::npos) break;
                                            size_t fd = 1; size_t fq = fo + 1;
                                            while (fq < schema_obj.size() && fd > 0) {
                                                if (schema_obj[fq]=='{') ++fd;
                                                else if (schema_obj[fq]=='}') --fd;
                                                ++fq;
                                            }
                                            std::string fobj = schema_obj.substr(fo, fq - fo);
                                            SchemaField sf;
                                            if (auto id = get_int_in(fobj, "id")) sf.id = (int)*id;
                                            sf.name = get_str_in(fobj, "name");
                                            sf.type = get_str_in(fobj, "type");
                                            // required: look for "required":true
                                            sf.required = (fobj.find("\"required\":true") != std::string::npos);
                                            if (!sf.name.empty()) info.schema_fields.push_back(sf);
                                            fp = fq;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        p = q;
                    }
                }
            }
        }

        // Build field-id → type map for partition spec resolution.
        std::map<int, std::string> field_type_by_id;
        for (const auto& sf : info.schema_fields) field_type_by_id[sf.id] = sf.type;

        // --- Parse default partition spec fields ---
        int default_spec_id = -1;
        if (auto v = get_int_in(meta, "default-spec-id")) default_spec_id = (int)*v;

        {
            auto specs_pos = meta.find("\"partition-specs\":");
            if (specs_pos != std::string::npos) {
                auto bracket = meta.find('[', specs_pos);
                if (bracket != std::string::npos) {
                    size_t p = bracket + 1;
                    while (p < meta.size()) {
                        auto obj_start = meta.find('{', p);
                        if (obj_start == std::string::npos) break;
                        size_t depth = 1; size_t q = obj_start + 1;
                        while (q < meta.size() && depth > 0) {
                            if (meta[q]=='{') ++depth;
                            else if (meta[q]=='}') --depth;
                            ++q;
                        }
                        std::string spec_obj = meta.substr(obj_start, q - obj_start);
                        if (auto sid = get_int_in(spec_obj, "spec-id")) {
                            if ((int)*sid == default_spec_id) {
                                auto fa = spec_obj.find("\"fields\":");
                                if (fa != std::string::npos) {
                                    auto fb = spec_obj.find('[', fa);
                                    if (fb != std::string::npos) {
                                        size_t fp = fb + 1;
                                        while (fp < spec_obj.size()) {
                                            auto fo = spec_obj.find('{', fp);
                                            if (fo == std::string::npos) break;
                                            size_t fd = 1; size_t fq = fo + 1;
                                            while (fq < spec_obj.size() && fd > 0) {
                                                if (spec_obj[fq]=='{') ++fd;
                                                else if (spec_obj[fq]=='}') --fd;
                                                ++fq;
                                            }
                                            std::string fobj = spec_obj.substr(fo, fq - fo);
                                            PartitionDef pd;
                                            pd.column    = get_str_in(fobj, "name");
                                            pd.transform = get_str_in(fobj, "transform");
                                            if (auto src = get_int_in(fobj, "source-id")) {
                                                auto it = field_type_by_id.find((int)*src);
                                                if (it != field_type_by_id.end()) pd.column_type = it->second;
                                            }
                                            if (pd.column_type.empty()) pd.column_type = "string";
                                            if (!pd.column.empty()) info.partition_fields.push_back(pd);
                                            fp = fq;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        p = q;
                    }
                }
            }
        }

        return info;
    }

    std::vector<DataFileEntry> list_files(const std::string& ns, const std::string& tbl) const {
        auto dir  = table_dir(ns, tbl);
        auto meta = read_metadata_json(dir);

        // Find current snapshot → manifest-list path
        int64_t current_snap = 0;
        std::string snap_key = "\"current-snapshot-id\":";
        auto snap_pos = meta.find(snap_key);
        if (snap_pos != std::string::npos) {
            snap_pos += snap_key.size();
            current_snap = std::stoll(meta.substr(snap_pos));
        }

        // Extract manifest-list for current snapshot from the "snapshots" array.
        // Must scope the search to the specific object whose "snapshot-id"
        // equals current_snap — naively searching for the numeric snapshot-id
        // string anywhere in the file finds its FIRST occurrence, which is the
        // top-level "current-snapshot-id" field itself, and then the next
        // "manifest-list" key after that is just the array's first snapshot
        // (usually a stale pre-compaction/pre-replace one), not the current one.
        std::string manifest_list;
        {
            auto snaps_pos = meta.find("\"snapshots\":");
            if (snaps_pos != std::string::npos) {
                auto bracket = meta.find('[', snaps_pos);
                if (bracket != std::string::npos) {
                    size_t p = bracket + 1;
                    std::string id_key = "\"snapshot-id\":";
                    while (p < meta.size()) {
                        auto obj_start = meta.find('{', p);
                        if (obj_start == std::string::npos) break;
                        size_t depth = 1; size_t q2 = obj_start + 1;
                        while (q2 < meta.size() && depth > 0) {
                            if (meta[q2] == '{') ++depth;
                            else if (meta[q2] == '}') --depth;
                            ++q2;
                        }
                        std::string obj = meta.substr(obj_start, q2 - obj_start);
                        auto id_pos = obj.find(id_key);
                        int64_t obj_snap_id = -1;
                        if (id_pos != std::string::npos) {
                            auto vpos = id_pos + id_key.size();
                            while (vpos < obj.size() && obj[vpos] == ' ') ++vpos;
                            try { obj_snap_id = std::stoll(obj.substr(vpos)); } catch (...) {}
                        }
                        if (obj_snap_id == current_snap) {
                            const std::string ml_key = "\"manifest-list\":";
                            auto mq = obj.find(ml_key);
                            if (mq != std::string::npos) {
                                mq += ml_key.size();
                                auto s = obj.find('"', mq); auto e = obj.find('"', s+1);
                                if (s != std::string::npos && e != std::string::npos)
                                    manifest_list = obj.substr(s+1, e-s-1);
                            }
                            break;
                        }
                        p = q2;
                        // Stop once past the snapshots array's closing bracket.
                        auto arr_end = meta.find(']', bracket);
                        if (arr_end != std::string::npos && p > arr_end) break;
                    }
                }
            }
        }
        if (manifest_list.empty()) return {};

        // Resolve manifest-list path. Like data-file paths, manifest-list and
        // manifest-file paths are stored relative to the warehouse ROOT
        // (e.g. "default/docs/metadata/snap-....avro", already including
        // namespace/table) — join against warehouse_, not the table dir,
        // or namespace/table gets double-prefixed. Also handles an absolute
        // file:// URI (ailake-py writer) — see resolve_warehouse_path().
        manifest_list = resolve_warehouse_path(manifest_list);

        auto manifest_paths = read_manifest_list(manifest_list);
        std::vector<DataFileEntry> all;
        for (auto& mp : manifest_paths) {
            mp = resolve_warehouse_path(mp);
            auto es = read_manifest_file(mp);
            all.insert(all.end(), es.begin(), es.end());
        }
        return all;
    }

private:
    std::string warehouse_;

    // Resolves a path emitted by the Rust catalog writer against the
    // warehouse root. See ailake-go's resolveWarehousePath (catalog.go) for
    // the identical fix on the Go side — same root cause, same three cases:
    //   - an absolute file:// URI — ailake-py's local_catalog_store always
    //     writes warehouse_uri as file://<absolute path> (required for
    //     Trino's Iceberg connector, see ailake-py/src/lib.rs), so
    //     metadata.json written by the Python SDK stores manifest-list this
    //     way. The old `rel[0] == '/'` check didn't recognize this as
    //     absolute (a file:// URI starts with 'f'), so it got string-joined
    //     onto warehouse_, producing a corrupted double-prefixed path
    //     (confirmed: warehouse "/a/b" + "file:///a/b/x.avro" via
    //     `warehouse_ + "/" + path` yields "/a/b/file:///a/b/x.avro", not
    //     "/a/b/x.avro"). Scheme is stripped, remainder used as-is.
    //   - a plain OS-absolute path (leading '/') — used as-is.
    //   - relative to the warehouse root (the common case for the Rust
    //     CLI/JNI writer) — joined onto warehouse_.
    std::string resolve_warehouse_path(const std::string& path) const {
        static const std::string scheme = "file://";
        if (path.compare(0, scheme.size(), scheme) == 0) {
            return path.substr(scheme.size());
        }
        if (!path.empty() && path[0] == '/') return path;
        return warehouse_ + "/" + path;
    }

    // Matches ailake-catalog's HadoopCatalog::table_root() exactly: flat
    // "<warehouse>/<namespace>/<table>", no Hive-style ".db" suffix.
    std::string table_dir(const std::string& ns, const std::string& tbl) const {
        return warehouse_ + "/" + ns + "/" + tbl;
    }

    static std::string read_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("ailake: cannot open " + path);
        std::streamsize sz = f.tellg();
        f.seekg(0);
        std::string s(sz, '\0');
        f.read(s.data(), sz);
        return s;
    }

    std::string read_metadata_json(const std::string& dir) const {
        std::string hint_path = dir + "/metadata/version-hint.text";
        auto version = read_file(hint_path);
        while (!version.empty() && (version.back()=='\n'||version.back()=='\r'))
            version.pop_back();
        return read_file(dir + "/metadata/v" + version + ".metadata.json");
    }

    static std::vector<std::string> read_manifest_list(const std::string& path) {
        std::ifstream s(path, std::ios::binary);
        if (!s) throw std::runtime_error("ailake: cannot open manifest list " + path);
        std::array<uint8_t,16> sync;
        detail::read_ocf_header(s, sync);

        std::vector<std::string> paths;
        for (;;) {
            // write_avro_container never writes a trailing count=0 terminator —
            // the file simply ends after the last block's sync marker (matching
            // apache-avro's own Reader, which treats EOF here as clean end-of-
            // stream — see the comment in ailake-catalog/src/avro_raw.rs). A
            // clean EOF right at the start of a block read is therefore a
            // normal loop exit, not an error; read_zigzag still throws on a
            // genuinely truncated/mid-varint EOF.
            if (s.peek() == std::char_traits<char>::eof()) break;
            int64_t obj_count = detail::read_zigzag(s);
            if (obj_count == 0) break;
            // Per the Avro OCF spec, a data block is always `count, byte_size,
            // objects..., sync` — byte_size is present unconditionally, not only
            // when count is negative (that's the array/map block convention used
            // elsewhere in this same file for the metadata map, which has no
            // byte_size at all — a different encoding). write_avro_container
            // (ailake-catalog/src/avro_raw.rs) always writes a positive count
            // followed by byte_size for its single data block.
            if (obj_count < 0) obj_count = -obj_count;
            detail::read_zigzag(s); // byte_size (unused — objects are read by schema below)
            for (int64_t i = 0; i < obj_count; ++i) {
                // manifest_file record — first field is manifest_path (string)
                std::string mp = detail::read_avro_bytes(s);
                paths.push_back(mp);
                // Skip remaining fields (length, spec_id, content, seq, min_seq,
                // snap_id, added, existing, deleted, added_rows, existing_rows,
                // deleted_rows, partitions array)
                // We do this by consuming the rest of the record bytes using
                // the schema; for simplicity skip by reading a raw block length.
                // NOTE: This simplified approach works only if manifest_path is
                // the first field — which it is in the AI-Lake Avro schema.
                // Skip: manifest_length(i64), partition_spec_id(i32), content(i32),
                //        sequence_number(i64), min_sequence_number(i64),
                //        added_snapshot_id(i64), added_data_files_count(i32),
                //        existing_data_files_count(i32), deleted_data_files_count(i32),
                //        added_rows_count(i64), existing_rows_count(i64),
                //        deleted_rows_count(i64), partitions(array of records)
                detail::read_zigzag(s); // manifest_length
                detail::read_zigzag(s); // partition_spec_id
                detail::read_zigzag(s); // content
                detail::read_zigzag(s); // sequence_number
                detail::read_zigzag(s); // min_sequence_number
                detail::read_zigzag(s); // added_snapshot_id
                detail::read_zigzag(s); // added_data_files_count
                detail::read_zigzag(s); // existing_data_files_count
                detail::read_zigzag(s); // deleted_data_files_count
                detail::read_zigzag(s); // added_rows_count
                detail::read_zigzag(s); // existing_rows_count
                detail::read_zigzag(s); // deleted_rows_count
                // partitions: block count = 0 (empty array)
                detail::read_zigzag(s);
            }
            // Sync marker
            s.seekg(16, std::ios::cur);
        }
        return paths;
    }

    static std::vector<DataFileEntry> read_manifest_file(const std::string& path) {
        std::ifstream s(path, std::ios::binary);
        if (!s) throw std::runtime_error("ailake: cannot open manifest " + path);
        std::array<uint8_t,16> sync;
        detail::read_ocf_header(s, sync);

        std::vector<DataFileEntry> entries;
        for (;;) {
            // See the matching comment in read_manifest_list: a clean EOF right
            // at the start of a block read is a normal loop exit here, since
            // write_avro_container never writes a trailing count=0 terminator.
            if (s.peek() == std::char_traits<char>::eof()) break;
            int64_t obj_count = detail::read_zigzag(s);
            if (obj_count == 0) break;
            // See the matching comment in read_manifest_list: byte_size always
            // follows count in an OCF data block, regardless of sign.
            if (obj_count < 0) obj_count = -obj_count;
            detail::read_zigzag(s); // byte_size (unused)
            for (int64_t i = 0; i < obj_count; ++i) {
                // manifest_entry schema:
                // status(int), snapshot_id(union<null,long>), sequence_number(union<null,long>),
                // file_sequence_number(union<null,long>), data_file(record)
                detail::read_zigzag(s); // status
                // union: 0=null, 1=long
                auto tag = detail::read_zigzag(s);
                if (tag == 1) detail::read_zigzag(s);  // snapshot_id
                tag = detail::read_zigzag(s);
                if (tag == 1) detail::read_zigzag(s);  // sequence_number
                tag = detail::read_zigzag(s);
                if (tag == 1) detail::read_zigzag(s);  // file_sequence_number

                // data_file record:
                // content(int), file_path(string), file_format(string),
                // partition(record = empty), record_count(long), file_size_in_bytes(long),
                // column_sizes(union null/array), value_counts(union null/array),
                // null_value_counts(union null/array), nan_value_counts(union null/array),
                // lower_bounds(union null/array), upper_bounds(union null/array),
                // key_metadata(union null/bytes), split_offsets(union null/array),
                // equality_ids(union null/array), sort_order_id(union null/int)

                DataFileEntry e;
                detail::read_zigzag(s);           // content
                e.path         = detail::read_avro_bytes(s); // file_path
                detail::read_avro_bytes(s);        // file_format (ignore)
                // partition r102: empty record → no fields
                e.record_count    = (uint64_t)detail::read_zigzag(s);
                e.file_size_bytes = (uint64_t)detail::read_zigzag(s);

                // Skip map fields (column_sizes, value_counts, null_value_counts, nan_value_counts,
                // lower_bounds, upper_bounds) — each is union null/array-of-map-records
                // union tag 0 = null; tag 1 = array follows
                for (int field = 0; field < 6; ++field) {
                    int64_t t = detail::read_zigzag(s);
                    if (t != 0) {
                        // array of records; read blocks until count=0
                        for (;;) {
                            int64_t bc = detail::read_zigzag(s);
                            if (bc == 0) break;
                            if (bc < 0) { detail::read_zigzag(s); bc = -bc; }
                            for (int64_t r = 0; r < bc; ++r) {
                                detail::read_zigzag(s); // key (int)
                                detail::read_zigzag(s); // value (long or bytes len)
                                // For lower/upper_bounds, value is bytes
                                // For column_sizes etc., value is long — already read above
                                // We skip by just reading zigzag; works for int/long fields.
                                // For bytes fields (bounds), zigzag gave length, skip that many.
                                // This is fragile but sufficient for AI-Lake's manifest structure.
                            }
                        }
                    }
                }

                // key_metadata: union null/bytes — this is our AI-Lake extension JSON
                {
                    int64_t t = detail::read_zigzag(s);
                    if (t != 0) {
                        std::string km = detail::read_avro_bytes(s);
                        parse_key_metadata(km, e);
                    }
                }

                // split_offsets: union null/array<long>
                {
                    int64_t t = detail::read_zigzag(s);
                    if (t != 0) {
                        for (;;) {
                            int64_t bc = detail::read_zigzag(s);
                            if (bc == 0) break;
                            if (bc < 0) { detail::read_zigzag(s); bc = -bc; }
                            for (int64_t r = 0; r < bc; ++r) detail::read_zigzag(s);
                        }
                    }
                }
                // equality_ids: union null/array<int>
                {
                    int64_t t = detail::read_zigzag(s);
                    if (t != 0) {
                        for (;;) {
                            int64_t bc = detail::read_zigzag(s);
                            if (bc == 0) break;
                            if (bc < 0) { detail::read_zigzag(s); bc = -bc; }
                            for (int64_t r = 0; r < bc; ++r) detail::read_zigzag(s);
                        }
                    }
                }
                // sort_order_id: union null/int
                {
                    int64_t t = detail::read_zigzag(s);
                    if (t != 0) detail::read_zigzag(s);
                }
                // first_row_id: union null/long (V3 row lineage). The Rust writer
                // (write_manifest_file) always encodes this field, null for V2
                // tables — skipping it here was the last field the reader was
                // missing, misaligning every subsequent record/block read by
                // one field.
                {
                    int64_t t = detail::read_zigzag(s);
                    if (t != 0) detail::read_zigzag(s);
                }

                entries.push_back(std::move(e));
            }
            s.seekg(16, std::ios::cur); // sync marker
        }
        return entries;
    }

    // Parse the JSON stored in key_metadata bytes into DataFileEntry fields.
    // Uses a simple string search approach (no JSON library dependency).
    static void parse_key_metadata(const std::string& json, DataFileEntry& e) {
        auto get_str = [&](const std::string& key) -> std::string {
            auto pos = json.find("\"" + key + "\":");
            if (pos == std::string::npos) return {};
            pos = json.find('"', pos + key.size() + 3);
            if (pos == std::string::npos) return {};
            auto end = json.find('"', pos + 1);
            if (end == std::string::npos) return {};
            return json.substr(pos + 1, end - pos - 1);
        };
        auto get_num = [&](const std::string& key) -> std::optional<uint64_t> {
            auto pos = json.find("\"" + key + "\":");
            if (pos == std::string::npos) return {};
            pos += key.size() + 3;
            while (pos < json.size() && json[pos] == ' ') ++pos;
            if (json[pos] == 'n') return {}; // null
            return (uint64_t)std::stoull(json.substr(pos));
        };
        auto get_f32 = [&](const std::string& key) -> float {
            auto pos = json.find("\"" + key + "\":");
            if (pos == std::string::npos) return 0.f;
            pos += key.size() + 3;
            while (pos < json.size() && json[pos] == ' ') ++pos;
            if (pos >= json.size() || json[pos] == 'n') return 0.f;
            try { return std::stof(json.substr(pos)); } catch (...) { return 0.f; }
        };

        auto cb64 = get_str("centroid_b64");
        if (!cb64.empty()) {
            // Base64 decode → F32 centroid. `centroid_b64` holds ONLY the
            // centroid (dim floats) — radius is a separate top-level JSON
            // field (AilakeEntryExt::radius in avro_manifest.rs), not packed
            // into the same blob.
            auto bytes = base64_decode(cb64);
            size_t n = bytes.size() / 4;
            e.centroid.resize(n);
            for (size_t i = 0; i < n; ++i) {
                uint32_t bits = (uint8_t)bytes[i*4]
                    | ((uint32_t)(uint8_t)bytes[i*4+1] << 8)
                    | ((uint32_t)(uint8_t)bytes[i*4+2] << 16)
                    | ((uint32_t)(uint8_t)bytes[i*4+3] << 24);
                float v; std::memcpy(&v, &bits, 4);
                e.centroid[i] = v;
            }
        }
        e.radius = get_f32("radius");

        e.hnsw_offset  = get_num("hnsw_offset");
        e.hnsw_len     = get_num("hnsw_len");
        auto vc = get_str("vector_column");
        if (!vc.empty()) e.vector_column = vc;
        auto vd = get_num("vector_dim");
        if (vd) e.vector_dim = (uint32_t)*vd;
        e.index_status    = get_str("index_status");
        e.batch_id        = get_str("batch_id");
        e.embedding_model = get_str("embedding_model");
        e.partition_value = get_str("partition_value");

        // Parse extra_vector_indexes array (Phase 8 multi-column tables)
        {
            auto arr_pos = json.find("\"extra_vector_indexes\":");
            if (arr_pos != std::string::npos) {
                auto bracket = json.find('[', arr_pos);
                auto bracket_end = json.find(']', arr_pos);
                if (bracket != std::string::npos && bracket_end != std::string::npos) {
                    std::string arr = json.substr(bracket + 1, bracket_end - bracket - 1);
                    size_t obj_start = arr.find('{');
                    while (obj_start != std::string::npos) {
                        auto obj_end = arr.find('}', obj_start);
                        if (obj_end == std::string::npos) break;
                        std::string obj = arr.substr(obj_start, obj_end - obj_start + 1);

                        auto xi_str = [&](const std::string& key) -> std::string {
                            auto p = obj.find("\"" + key + "\":");
                            if (p == std::string::npos) return {};
                            p = obj.find('"', p + key.size() + 3);
                            if (p == std::string::npos) return {};
                            auto e2 = obj.find('"', p + 1);
                            if (e2 == std::string::npos) return {};
                            return obj.substr(p + 1, e2 - p - 1);
                        };
                        auto xi_u64 = [&](const std::string& key) -> uint64_t {
                            auto p = obj.find("\"" + key + "\":");
                            if (p == std::string::npos) return 0;
                            p += key.size() + 3;
                            while (p < obj.size() && obj[p] == ' ') ++p;
                            if (p >= obj.size() || obj[p] == 'n') return 0;
                            try { return std::stoull(obj.substr(p)); } catch (...) { return 0; }
                        };
                        auto xi_f32 = [&](const std::string& key) -> float {
                            auto p = obj.find("\"" + key + "\":");
                            if (p == std::string::npos) return 0.f;
                            p += key.size() + 3;
                            while (p < obj.size() && obj[p] == ' ') ++p;
                            if (p >= obj.size() || obj[p] == 'n') return 0.f;
                            try { return std::stof(obj.substr(p)); } catch (...) { return 0.f; }
                        };

                        ExtraVectorIndex xi;
                        xi.column      = xi_str("column");
                        xi.dim         = (uint32_t)xi_u64("dim");
                        xi.hnsw_offset = xi_u64("hnsw_offset");
                        xi.hnsw_len    = xi_u64("hnsw_len");
                        xi.centroid_b64= xi_str("centroid_b64");
                        xi.radius      = xi_f32("radius");
                        if (!xi.column.empty()) e.extra_vector_indexes.push_back(std::move(xi));

                        obj_start = arr.find('{', obj_end + 1);
                    }
                }
            }
        }
    }

    static std::string base64_decode(const std::string& in) {
        static const std::string b64c =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(in.size() * 3 / 4);
        uint32_t val = 0; int bits = 0;
        for (unsigned char c : in) {
            auto p = b64c.find(c);
            if (p == std::string::npos) continue;
            val = (val << 6) | (uint32_t)p;
            bits += 6;
            if (bits >= 8) { bits -= 8; out += (char)((val >> bits) & 0xFF); }
        }
        return out;
    }
};

} // namespace ailake
