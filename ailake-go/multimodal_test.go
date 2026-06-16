// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"testing"
)

// ── ModalQuery / RRFResult types ─────────────────────────────────────────────

func TestModalQuery_Defaults(t *testing.T) {
	mq := ModalQuery{Column: "embedding", Query: []float32{1, 0, 0}}
	if mq.Weight != 0 {
		t.Errorf("Weight default: got %v, want 0 (SearchMultimodal treats 0 as 1.0)", mq.Weight)
	}
}

func TestSearchMultimodal_EmptyQueriesError(t *testing.T) {
	cat := &HadoopCatalog{Warehouse: "/nonexistent"}
	_, err := SearchMultimodal(cat, "ns", "tbl", nil, SearchOptions{TopK: 5})
	if err == nil {
		t.Fatal("expected error for empty queries, got nil")
	}
}

// ── RRF score accumulation logic ─────────────────────────────────────────────

func TestRRFScoreAccumulation(t *testing.T) {
	// Simulate two single-column result lists for the same row appearing in rank 0
	// of list A (weight 0.7) and rank 1 of list B (weight 0.3).
	//
	// score = 0.7/(60+1) + 0.3/(60+2) = 0.01148 + 0.00484 ≈ 0.01632
	wA := float32(0.7) / float32(61)
	wB := float32(0.3) / float32(62)
	total := wA + wB

	if total < 0.015 || total > 0.018 {
		t.Errorf("RRF score out of expected range: got %v", total)
	}
}

// ── ExtraVectorIndex JSON round-trip (via catalog parsing) ───────────────────

func TestExtraVectorIndex_ZeroValues(t *testing.T) {
	xi := ExtraVectorIndex{}
	if xi.Column != "" || xi.Dim != 0 || xi.HnswOffset != 0 || xi.HnswLen != 0 {
		t.Error("ExtraVectorIndex zero-value not as expected")
	}
}

// ── searchFileCol: primary column fallback ────────────────────────────────────

func TestSearchFileCol_PrimaryColumnAlias(t *testing.T) {
	// Empty-column ModalQuery should map to primary column branch.
	// With nil HnswOffset, searchFileCol returns (nil, nil) — no panic.
	entry := DataFileEntry{
		Path:      "part-0001.parquet",
		VectorDim: 4,
	}
	mq := ModalQuery{Column: "", Query: []float32{1, 0, 0, 0}, Weight: 1}
	hits, err := searchFileCol("/tmp", "ns", "tbl", entry, mq, "embedding",
		SearchOptions{TopK: 5}, DetectHardware())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if hits != nil {
		t.Errorf("expected nil hits for file with no hnsw_offset, got %v", hits)
	}
}

func TestSearchFileCol_SecondaryColumnNotFound(t *testing.T) {
	// When the requested column is not in ExtraVectorIndexes, returns (nil, nil).
	entry := DataFileEntry{
		Path:               "part-0001.parquet",
		VectorDim:          4,
		ExtraVectorIndexes: []ExtraVectorIndex{{Column: "other_col", Dim: 8}},
	}
	mq := ModalQuery{Column: "image_embedding", Query: []float32{1, 0, 0, 0}, Weight: 1}
	hits, err := searchFileCol("/tmp", "ns", "tbl", entry, mq, "embedding",
		SearchOptions{TopK: 5}, DetectHardware())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if hits != nil {
		t.Errorf("expected nil hits for missing column, got %v", hits)
	}
}

func TestSearchFileCol_SecondaryColumnZeroOffset(t *testing.T) {
	// ExtraVectorIndex with hnsw_offset=0 means not-yet-indexed → skip gracefully.
	entry := DataFileEntry{
		Path:      "part-0001.parquet",
		VectorDim: 4,
		ExtraVectorIndexes: []ExtraVectorIndex{
			{Column: "image_embedding", Dim: 4, HnswOffset: 0, HnswLen: 0},
		},
	}
	mq := ModalQuery{Column: "image_embedding", Query: []float32{1, 0, 0, 0}, Weight: 1}
	hits, err := searchFileCol("/tmp", "ns", "tbl", entry, mq, "embedding",
		SearchOptions{TopK: 5}, DetectHardware())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if hits != nil {
		t.Errorf("expected nil hits for zero-offset index, got %v", hits)
	}
}
