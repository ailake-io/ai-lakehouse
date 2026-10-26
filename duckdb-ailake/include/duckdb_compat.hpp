// SPDX-License-Identifier: MIT OR Apache-2.0
#pragma once

#include "duckdb.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/function/table_function.hpp"

#include <initializer_list>

#if AILAKE_DUCKDB_V2
using AilakeBindNames = duckdb::vector<duckdb::Identifier>;
#define AILAKE_FLAT_VECTOR_DATA(type, vector) duckdb::FlatVector::GetDataMutable<type>(vector)
#else
using AilakeBindNames = duckdb::vector<duckdb::string>;
#define AILAKE_FLAT_VECTOR_DATA(type, vector) duckdb::FlatVector::GetData<type>(vector)
#endif

inline duckdb::TableFunction AilakeMakeTableFunction(
    const char *name,
    std::initializer_list<duckdb::LogicalType> arguments,
    duckdb::table_function_t function,
    duckdb::table_function_bind_t bind,
    duckdb::table_function_init_global_t init_global
) {
#if AILAKE_DUCKDB_V2
    duckdb::FunctionSignature signature;
    for (const auto &argument : arguments) {
        signature.AddParameter(argument);
    }
    return duckdb::TableFunction(duckdb::Identifier(name), std::move(signature), function, bind, init_global);
#else
    return duckdb::TableFunction(name, arguments, function, bind, init_global);
#endif
}

inline void AilakeAddNamedParameter(
    duckdb::TableFunction &function,
    const char *name,
    duckdb::LogicalType type,
    duckdb::optional<duckdb::Value> default_value = {}
) {
#if AILAKE_DUCKDB_V2
    function.GetSignature().AddKeywordOnly(duckdb::Identifier(name), std::move(type), std::move(default_value));
#else
    (void)default_value;
    function.named_parameters[name] = std::move(type);
#endif
}
