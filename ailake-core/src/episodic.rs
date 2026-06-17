// SPDX-License-Identifier: MIT OR Apache-2.0
//! Phase 9 — Episodic memory schema and recency scoring primitives.
//!
//! Provides zero-I/O building blocks for agent memory tables:
//! - `EpisodicMemorySchema` marker + `episodic_columns` constants
//! - `RecencyConfig` / `recency_weight` — exponential time-decay
//! - `hybrid_score` — fuses distance, recency, and importance into one ranking signal
//!
//! These are pure-math functions with no async, no I/O, no external deps.

use serde::{Deserialize, Serialize};

// ── Column name constants ─────────────────────────────────────────────────────

/// Canonical column names for episodic memory tables.
///
/// Include these alongside `llm_columns::*` in the Arrow schema of any
/// `EpisodicMemorySchema` table. The AI-Lake SDK reads columns by name.
pub mod episodic_columns {
    /// Recency weight in [0.0, 1.0] — decays exponentially with time since last access.
    /// Recomputed by `MemoryDecayJob`; initial value is 1.0 at write time.
    /// Formula: `exp(-λ * days_since_access)` where λ = `RecencyConfig::lambda`.
    pub const RECENCY_WEIGHT: &str = "recency_weight";

    /// Number of times this memory chunk has been retrieved by `recall()`.
    /// Higher access count signals relevance; used by agents for importance inference.
    pub const ACCESS_COUNT: &str = "access_count";

    /// Timestamp of the most recent retrieval (Unix seconds, Int64).
    /// Updated in-place via logical delete + reinsert on each `recall()` hit.
    pub const LAST_ACCESSED_AT: &str = "last_accessed_at";

    /// Agent-assigned importance score in [0.0, 1.0].
    /// Set at `remember()` time; never decays automatically (unlike `recency_weight`).
    /// Agents use this to pin critical memories (e.g. user preferences, hard constraints).
    pub const IMPORTANCE_SCORE: &str = "importance_score";

    /// UUID of the agent instance that owns this memory chunk.
    /// Matches `tool_call_columns::AGENT_ID` for cross-table joins.
    pub const AGENT_ID: &str = "agent_id";

    /// UUID of the conversation / task session this memory was created in.
    pub const SESSION_ID: &str = "session_id";

    /// Timestamp when this memory was first written (Unix seconds, Int64).
    pub const CREATED_AT: &str = "created_at";
}

// ── EpisodicMemorySchema marker ───────────────────────────────────────────────

/// Marker struct for episodic agent memory tables (Phase 9).
/// Actual schema is enforced by column names in `episodic_columns` module.
///
/// An episodic memory table extends `LlmContextSchema` with recency and
/// importance signals, enabling hybrid scoring during recall:
///
/// ```text
/// -- From llm_columns::* (required baseline)
/// chunk_id:          Utf8
/// chunk_text:        Utf8
/// embedding:         FixedSizeBinary(N)  -- F16, cosine
///
/// -- From episodic_columns::* (Phase 9 extensions)
/// agent_id:          Utf8       -- UUID string
/// session_id:        Utf8       -- UUID string
/// created_at:        Int64      -- Unix seconds
/// recency_weight:    Float32    -- exp(-λ * days_since_access), updated by MemoryDecayJob
/// access_count:      UInt32     -- incremented on each recall() hit
/// last_accessed_at:  Int64      -- Unix seconds, updated on recall
/// importance_score:  Float32    -- agent-assigned [0.0, 1.0]
/// ```
///
/// **Hybrid scoring**: after HNSW retrieval, re-rank results by
/// `hybrid_score(distance, recency_weight, importance_score)`. Memories
/// that are semantically similar AND recently accessed AND flagged important
/// rank highest.
///
/// **Recommended setup**:
/// - One HNSW over `embedding` (text, cosine, dim=1536).
/// - Partition by `agent_id` via `VectorStoragePolicy` hidden partitioning.
/// - Run `MemoryDecayJob` daily to update `recency_weight` via compaction.
pub struct EpisodicMemorySchema;

// ── Recency decay ─────────────────────────────────────────────────────────────

/// Parameters for exponential time-decay of memory recency.
///
/// The decay formula is: `recency_weight = exp(-lambda * days_since_access)`
///
/// Common presets:
/// | Half-life | lambda  |
/// |-----------|---------|
/// | 1 day     | 0.693   |
/// | 1 week    | 0.099   |
/// | 1 month   | 0.023   |
/// | 3 months  | 0.0077  |
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RecencyConfig {
    /// Decay rate λ. Must be > 0. Typical range: 0.005–1.0.
    pub lambda: f32,
}

impl RecencyConfig {
    /// λ ≈ ln(2)/7 — half-life of 7 days. Good default for conversational agents.
    pub const WEEKLY_DECAY: Self = Self { lambda: 0.099 };

    /// λ ≈ ln(2)/30 — half-life of 30 days. Good for long-term knowledge bases.
    pub const MONTHLY_DECAY: Self = Self { lambda: 0.023 };

    /// λ ≈ ln(2)/1 — half-life of 1 day. Aggressive decay for short-term sessions.
    pub const DAILY_DECAY: Self = Self { lambda: 0.693 };

    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}

impl Default for RecencyConfig {
    fn default() -> Self {
        Self::WEEKLY_DECAY
    }
}

/// Compute the recency weight for a memory chunk.
///
/// Returns a value in (0.0, 1.0]:
/// - 1.0 when `days_since_access == 0` (accessed right now)
/// - 0.5 at the half-life (days = ln(2) / lambda)
/// - Approaches 0 for very old, never-accessed memories
///
/// `days_since_access` may be fractional (e.g. 0.5 = 12 hours).
/// Negative values are clamped to 0 (future timestamps treated as "now").
#[inline]
pub fn recency_weight(days_since_access: f32, cfg: &RecencyConfig) -> f32 {
    let days = days_since_access.max(0.0);
    (-cfg.lambda * days).exp()
}

// ── Hybrid scoring ────────────────────────────────────────────────────────────

/// Compute the hybrid ranking score for a retrieved memory chunk.
///
/// Fuses three signals into a single ascending score (lower = better rank,
/// consistent with AI-Lake's distance-ascending convention):
///
/// ```text
/// hybrid_score = distance / (recency_weight * importance_score)
/// ```
///
/// Rationale:
/// - `distance` is HNSW cosine distance in [0.0, 2.0] (lower = more similar).
/// - Dividing by `recency_weight * importance_score` boosts recent/important
///   memories (high values shrink the score → rise in rank).
/// - Result approaches `distance` as recency and importance → 1.0 (neutral).
/// - Safeguard: denominator clamped to `f32::EPSILON` to avoid division by zero.
///
/// **Usage**: call after HNSW top-k retrieval, before returning results to the agent.
///
/// # Arguments
/// - `distance`: HNSW cosine distance for this result (from `SearchResult.distance`)
/// - `recency_weight`: value from `episodic_columns::RECENCY_WEIGHT` column (or computed via `recency_weight()`)
/// - `importance_score`: value from `episodic_columns::IMPORTANCE_SCORE` column
#[inline]
pub fn hybrid_score(distance: f32, recency_weight: f32, importance_score: f32) -> f32 {
    let denom = (recency_weight * importance_score).max(f32::EPSILON);
    distance / denom
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recency_weight_at_zero_days() {
        let w = recency_weight(0.0, &RecencyConfig::WEEKLY_DECAY);
        assert!((w - 1.0).abs() < 1e-6, "expected 1.0 at day 0, got {w}");
    }

    #[test]
    fn recency_weight_half_life() {
        // At half-life = ln(2)/lambda days, weight should be ~0.5
        let cfg = RecencyConfig::WEEKLY_DECAY;
        let half_life_days = std::f32::consts::LN_2 / cfg.lambda;
        let w = recency_weight(half_life_days, &cfg);
        assert!((w - 0.5).abs() < 0.01, "expected ~0.5 at half-life, got {w}");
    }

    #[test]
    fn recency_weight_negative_clamped() {
        let w = recency_weight(-5.0, &RecencyConfig::WEEKLY_DECAY);
        assert!((w - 1.0).abs() < 1e-6, "negative days should clamp to 0");
    }

    #[test]
    fn hybrid_score_neutral_signals() {
        // recency=1.0, importance=1.0 → hybrid_score == distance
        let d = 0.3_f32;
        let s = hybrid_score(d, 1.0, 1.0);
        assert!((s - d).abs() < 1e-6);
    }

    #[test]
    fn hybrid_score_old_memory_ranks_worse() {
        // Low recency (old memory) → larger score → worse rank
        let d = 0.3_f32;
        let old_memory = hybrid_score(d, 0.3, 1.0);   // recency 0.3 = accessed long ago
        let recent_memory = hybrid_score(d, 1.0, 1.0); // recency 1.0 = just accessed
        assert!(old_memory > recent_memory, "old memory should rank lower (higher score)");
    }

    #[test]
    fn hybrid_score_unimportant_ranks_worse() {
        // Low importance → larger score → worse rank
        let d = 0.3_f32;
        let unimportant = hybrid_score(d, 1.0, 0.2);
        let important = hybrid_score(d, 1.0, 1.0);
        assert!(unimportant > important, "unimportant memory should rank lower");
    }

    #[test]
    fn hybrid_score_zero_denom_no_panic() {
        // Should not divide by zero
        let s = hybrid_score(0.5, 0.0, 0.0);
        assert!(s.is_finite());
    }
}
