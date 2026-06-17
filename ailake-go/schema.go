// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Phase 9 agent memory schemas.
//
// These structs document the expected column layout for agent-memory tables.
// Use them as templates when creating tables via TableWriter or as typed
// views when reading back results from Search().
package ailake

import "time"

// ToolCallSchema documents the column layout for agent tool-call history tables.
// Mirrors ailake_core::schema::ToolCallSchema in Rust.
//
// Usage: create a table with these columns and write ToolCallSchema records to
// enable vector search over historical tool-call contexts.
type ToolCallSchema struct {
	// Identity
	AgentID   string `json:"agent_id"`
	SessionID string `json:"session_id"`
	StepIndex uint32 `json:"step_index"`

	// Tool invocation
	ToolName       string `json:"tool_name"`
	ToolInputJSON  string `json:"tool_input_json"`
	ToolOutputJSON string `json:"tool_output_json"`
	Outcome        string `json:"outcome"` // "success" | "failure" | "timeout"
	LatencyMs      uint32 `json:"latency_ms"`

	// LlmContextSchema fields (inherited)
	ChunkID         string    `json:"chunk_id"`
	DocumentID      string    `json:"document_id"`
	ChunkIndex      uint32    `json:"chunk_index"`
	ChunkText       string    `json:"chunk_text"`
	DocumentTitle   string    `json:"document_title"`
	SectionPath     string    `json:"section_path"`
	PrecedingCtx    string    `json:"preceding_context"`
	FollowingCtx    string    `json:"following_context"`
	DocumentSummary string    `json:"document_summary"`
	ChunkSummary    string    `json:"chunk_summary"`
	SourceURI       string    `json:"source_uri"`
	CreatedAt       time.Time `json:"created_at"`

	// Embedding vector (stored as FIXED_LEN_BYTE_ARRAY F16 in Parquet)
	// Omitted here — populated at write time by the embedding function.
}

// EpisodicMemorySchema documents the column layout for episodic agent memory tables.
// Mirrors ailake_core::schema::EpisodicMemorySchema in Rust.
//
// The recency_weight field decays over time via exp(-λ * days_since_access).
// Use Agent.recall() which applies hybrid scoring automatically.
type EpisodicMemorySchema struct {
	// Recency and importance signals
	RecencyWeight  float32   `json:"recency_weight"`   // starts at 1.0, decays via exp(-λ*t)
	AccessCount    uint32    `json:"access_count"`     // incremented on each recall hit
	LastAccessedAt time.Time `json:"last_accessed_at"` // updated on each recall hit
	ImportanceScore float32  `json:"importance_score"` // agent-defined; 1.0 = neutral

	// LlmContextSchema fields (inherited)
	AgentID         string    `json:"agent_id"`
	SessionID       string    `json:"session_id"`
	ChunkID         string    `json:"chunk_id"`
	DocumentID      string    `json:"document_id"`
	ChunkIndex      uint32    `json:"chunk_index"`
	ChunkText       string    `json:"chunk_text"`
	DocumentTitle   string    `json:"document_title"`
	SectionPath     string    `json:"section_path"`
	PrecedingCtx    string    `json:"preceding_context"`
	FollowingCtx    string    `json:"following_context"`
	DocumentSummary string    `json:"document_summary"`
	ChunkSummary    string    `json:"chunk_summary"`
	SourceURI       string    `json:"source_uri"`
	CreatedAt       time.Time `json:"created_at"`

	// Embedding vector (stored as FIXED_LEN_BYTE_ARRAY F16 in Parquet)
	// Omitted here — populated at write time by the embedding function.
}
