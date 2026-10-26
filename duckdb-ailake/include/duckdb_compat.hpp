// SPDX-License-Identifier: MIT OR Apache-2.0
#pragma once

#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

#if AILAKE_DUCKDB_V2
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#endif

#include <initializer_list>

#if AILAKE_DUCKDB_V2
using AilakeBindNames = duckdb::vector<duckdb::Identifier>;
#define AILAKE_FLAT_VECTOR_DATA(type, vector) duckdb::FlatVector::GetDataMutable<type>(vector)
#define AILAKE_FLAT_VECTOR_VALIDITY(vector) duckdb::FlatVector::ValidityMutable(vector)
#define AILAKE_LIST_VECTOR_ENTRY(vector) (*duckdb::ListVector::GetChildMutable(vector))
#else
using AilakeBindNames = duckdb::vector<duckdb::string>;
#define AILAKE_FLAT_VECTOR_DATA(type, vector) duckdb::FlatVector::GetData<type>(vector)
#define AILAKE_FLAT_VECTOR_VALIDITY(vector) duckdb::FlatVector::Validity(vector)
#define AILAKE_LIST_VECTOR_ENTRY(vector) duckdb::ListVector::GetEntry(vector)
#endif

inline AilakeBindNames::value_type AilakeMakeBindName(const std::string &name) {
#if AILAKE_DUCKDB_V2
    return duckdb::Identifier(duckdb::string(name));
#else
    return name;
#endif
}

#if AILAKE_DUCKDB_V2
inline const duckdb::string &AilakeGetIdentifierName(const duckdb::Identifier &name) {
    return name.GetIdentifierName();
}
#else
inline const duckdb::string &AilakeGetIdentifierName(const duckdb::string &name) {
    return name;
}
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
