# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
cmake_minimum_required(VERSION 3.20)
project(duckdb_ailake VERSION 0.1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── DuckDB version (must match the duckdb Python package used in tests) ────────
set(DUCKDB_VERSION "v1.5.3" CACHE STRING "DuckDB version to build against")

# ── Fetch DuckDB source (headers only — do NOT add_subdirectory) ──────────────
include(FetchContent)

FetchContent_Declare(duckdb
    GIT_REPOSITORY https://github.com/duckdb/duckdb.git
    GIT_TAG        ${DUCKDB_VERSION}
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(duckdb)
if(NOT duckdb_POPULATED)
    FetchContent_Populate(duckdb)
endif()

# ── nlohmann/json (single header, response parsing) ───────────────────────────
FetchContent_Declare(nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Extension shared library ──────────────────────────────────────────────────
add_library(duckdb_ailake MODULE
    src/ailake_extension.cpp
    src/ailake_search.cpp
    src/ailake_search_multimodal.cpp
    src/ailake_search_text.cpp
    src/ailake_write.cpp
    src/ailake_scan_table.cpp
    src/ailake_delete.cpp
    src/ailake_evolve_schema.cpp
    src/ailake_write_batch_multi.cpp
    src/ailake_compact.cpp
)

target_include_directories(duckdb_ailake PRIVATE
    include
    ${duckdb_SOURCE_DIR}/src/include
)

target_link_libraries(duckdb_ailake PRIVATE
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS}
)

# ABI note: match the ABI of the host duckdb Python package.
# Anaconda/pip wheels on modern Linux use C++11 ABI (default, no flag needed).
# Older manylinux wheels use _GLIBCXX_USE_CXX11_ABI=0 — add back if loading fails
# with "undefined symbol: ...B5cxx11..." vs "...ToStringEv" (without B5cxx11).

# Extensions must NOT link against duckdb — symbols resolved from the host process at dlopen time.
if(APPLE)
    target_link_options(duckdb_ailake PRIVATE -undefined dynamic_lookup)
elseif(UNIX)
    target_link_options(duckdb_ailake PRIVATE -Wl,--allow-shlib-undefined)
endif()

set_target_properties(duckdb_ailake PROPERTIES
    PREFIX      ""
    OUTPUT_NAME "ailake"
    SUFFIX      ".duckdb_extension"
)

# ── Post-build: append DuckDB 1.0+ extension metadata ────────────────────────
# DuckDB 1.0+ requires a 512-byte metadata block at the end of every extension
# binary (8×32-byte fields + 256-byte signature), even for unsigned extensions.
# Without it DuckDB refuses to load the file regardless of allow_unsigned_extensions.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(TARGET duckdb_ailake POST_BUILD
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/append_extension_metadata.py"
            "$<TARGET_FILE:duckdb_ailake>"
            "${DUCKDB_VERSION}"
            "${CMAKE_PROJECT_VERSION}"
    COMMENT "Appending DuckDB extension metadata (512 bytes)"
    VERBATIM
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS duckdb_ailake DESTINATION lib/duckdb_extensions)
