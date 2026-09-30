/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#ifndef AILAKE_FFI_H
#define AILAKE_FFI_H

#include <stdint.h>

#if defined(_WIN32)
#  define AILAKE_API __declspec(dllimport)
#else
#  define AILAKE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Static string owned by the library; never pass it to ailake_free_string. */
AILAKE_API const char *ailake_version(void);

/* Stable C-ABI contract version. */
AILAKE_API uint32_t ailake_ffi_abi_version(void);

/* Every returned JSON string must be released exactly once. */
AILAKE_API void ailake_free_string(char *ptr);

/* Preferred versioned JSON envelope APIs. */
AILAKE_API char *ailake_search_json(const char *request_json);
AILAKE_API char *ailake_write_batch_json(const char *request_json);
AILAKE_API char *ailake_write_batch_multi_json(const char *request_json);
AILAKE_API char *ailake_search_text_json(const char *request_json);
AILAKE_API char *ailake_search_multimodal_json(const char *request_json);
AILAKE_API char *ailake_scan_json(const char *request_json);
AILAKE_API char *ailake_info_json(const char *request_json);
AILAKE_API char *ailake_delete_where_json(const char *request_json);
AILAKE_API char *ailake_evolve_schema_json(const char *request_json);
AILAKE_API char *ailake_compact_json(const char *request_json);

/*
 * Kof-native borrowed adapters. Kof copies a returned C string into its own
 * managed String, so these functions release the normal JSON allocation
 * internally. The pointer is valid until the next adapter call on the same
 * thread and must not be passed to ailake_free_string.
 */
AILAKE_API const char *ailake_kof_search_json(const char *request_json);
AILAKE_API const char *ailake_kof_write_batch_json(const char *request_json);
AILAKE_API const char *ailake_kof_info_json(const char *request_json);
AILAKE_API const char *ailake_kof_compact_json(const char *request_json);

/* Arrow IPC stream input. Maximum accepted payload is 512 MiB. */
AILAKE_API char *ailake_write_batch_ipc(const uint8_t *ipc_bytes,
                                        int64_t ipc_len,
                                        const char *opts_json);

/* Legacy raw-vector entry point. Prefer ailake_search_json for new clients. */
AILAKE_API char *ailake_vector_search_json(const char *table_uri,
                                           const float *query_ptr,
                                           uint32_t query_len,
                                           uint32_t top_k);

#ifdef __cplusplus
}
#endif

#endif /* AILAKE_FFI_H */
