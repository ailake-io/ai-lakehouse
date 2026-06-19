// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Phase 9 agent memory schema types.
//
// These structs document the expected column layout for agent-memory tables.
// They mirror ailake_core::schema::{ToolCallSchema, EpisodicMemorySchema} in Rust.
#pragma once

#include <cstdint>
#include <string>

namespace ailake {

// ── ToolCallSchema ────────────────────────────────────────────────────────────

/// Outcome of a tool call.
enum class ToolCallOutcome { Success, Failure, Timeout };

/// Column layout for agent tool-call history tables (Phase 9).
/// Write rows of this type to enable semantic search over historical tool invocations:
/// "when did tool X fail in contexts similar to this one?"
struct ToolCallSchema {
    // Identity
    std::string agent_id;
    std::string session_id;
    uint32_t    step_index      = 0;

    // Tool invocation
    std::string     tool_name;
    std::string     tool_input_json;
    std::string     tool_output_json;
    ToolCallOutcome outcome       = ToolCallOutcome::Success;
    uint32_t        latency_ms    = 0;

    // LlmContextSchema fields (inherited)
    std::string chunk_id;
    std::string document_id;
    uint32_t    chunk_index      = 0;
    std::string chunk_text;
    std::string document_title;
    std::string section_path;
    std::string preceding_context;
    std::string following_context;
    std::string document_summary;
    std::string chunk_summary;
    std::string source_uri;
    int64_t     created_at_ms    = 0; // Unix timestamp in milliseconds

    // Embedding vector populated at write time.
};

// ── EpisodicMemorySchema ──────────────────────────────────────────────────────

/// Column layout for episodic agent memory tables (Phase 9).
/// recency_weight decays over time: exp(-λ * days_since_access).
/// Use ailake::Agent.recall() for hybrid scoring (distance × recency × importance).
struct EpisodicMemorySchema {
    // Recency and importance signals
    float    recency_weight    = 1.0f; // starts at 1.0, decays via exp(-λ*t)
    uint32_t access_count      = 0;   // incremented on each recall hit
    int64_t  last_accessed_ms  = 0;   // Unix timestamp in milliseconds
    float    importance_score  = 1.0f; // agent-defined; 1.0 = neutral

    // LlmContextSchema fields (inherited)
    std::string agent_id;
    std::string session_id;
    std::string chunk_id;
    std::string document_id;
    uint32_t    chunk_index    = 0;
    std::string chunk_text;
    std::string document_title;
    std::string section_path;
    std::string preceding_context;
    std::string following_context;
    std::string document_summary;
    std::string chunk_summary;
    std::string source_uri;
    int64_t     created_at_ms  = 0; // Unix timestamp in milliseconds

    // Embedding vector populated at write time.
};

} // namespace ailake
