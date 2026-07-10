// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"os"
	"strings"
	"testing"
)

// ── resolveBin ────────────────────────────────────────────────────────────────

func TestResolveBin_MissingEnvAndPath(t *testing.T) {
	orig := os.Getenv("AILAKE_BIN")
	os.Unsetenv("AILAKE_BIN")
	defer os.Setenv("AILAKE_BIN", orig)

	// PATH should not contain `ailake` in unit test environments.
	// This is best-effort; if `ailake` is actually installed the test is skipped.
	bin, err := resolveBin()
	if err == nil {
		t.Skipf("ailake binary found at %q; skipping no-binary test", bin)
	}
	if !strings.Contains(err.Error(), "no CLI binary") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestResolveBin_InvalidEnv(t *testing.T) {
	orig := os.Getenv("AILAKE_BIN")
	os.Setenv("AILAKE_BIN", "/tmp/ailake_does_not_exist_12345")
	defer os.Setenv("AILAKE_BIN", orig)

	_, err := resolveBin()
	if err == nil {
		t.Fatal("expected error for non-existent AILAKE_BIN, got nil")
	}
}

// ── AddColumnReq / RenameColumnReq ───────────────────────────────────────────

func TestAddColumnReq_Fields(t *testing.T) {
	req := AddColumnReq{Name: "score", Type: "float", InitialDefault: "0.0"}
	if req.Name != "score" || req.Type != "float" || req.InitialDefault != "0.0" {
		t.Errorf("unexpected field values: %+v", req)
	}
}

func TestRenameColumnReq_Fields(t *testing.T) {
	req := RenameColumnReq{From: "old_col", To: "new_col"}
	if req.From != "old_col" || req.To != "new_col" {
		t.Errorf("unexpected field values: %+v", req)
	}
}

// ── DeleteWhere / EvolveSchema no-op paths ───────────────────────────────────

func TestDeleteWhere_EmptyValues_Noop(t *testing.T) {
	catalog := &HadoopCatalog{Warehouse: "/tmp/test"}
	err := DeleteWhere(catalog, "default", "table", "doc_id", nil)
	if err != nil {
		t.Errorf("empty values: expected nil error, got %v", err)
	}
}

func TestEvolveSchema_NoCols_Noop(t *testing.T) {
	catalog := &HadoopCatalog{Warehouse: "/tmp/test"}
	id, err := EvolveSchema(catalog, "default", "table", nil, nil)
	if err != nil {
		t.Errorf("no cols: expected nil error, got %v", err)
	}
	if id != 0 {
		t.Errorf("no cols: expected 0 schema_id, got %d", id)
	}
}

// ── Integration tests (require AILAKE_BIN + AILAKE_FIXTURE) ──────────────────

func TestDeleteWhereIntegration(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set")
	}
	bin := os.Getenv("AILAKE_BIN")
	if bin == "" {
		t.Skip("AILAKE_BIN not set")
	}

	catalog := &HadoopCatalog{Warehouse: fixtureDir}
	// Delete a non-existent value — should succeed (zero-row delete is valid).
	err := DeleteWhere(catalog, "default", "table", "document_id", []string{"__nonexistent_doc__"})
	if err != nil {
		t.Errorf("DeleteWhere: %v", err)
	}
}

func TestEvolveSchemaIntegration(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set")
	}
	bin := os.Getenv("AILAKE_BIN")
	if bin == "" {
		t.Skip("AILAKE_BIN not set")
	}

	catalog := &HadoopCatalog{Warehouse: fixtureDir}
	schemaID, err := EvolveSchema(
		catalog, "default", "table",
		[]AddColumnReq{{Name: "_go_test_col", Type: "string", InitialDefault: `""`}},
		nil,
	)
	if err != nil {
		t.Errorf("EvolveSchema: %v", err)
	}
	if schemaID < 0 {
		t.Errorf("EvolveSchema: expected non-negative schema_id, got %d", schemaID)
	}
}

// ── VectorColSpec / CompactOptions ───────────────────────────────────────────

func TestVectorColSpec_Fields(t *testing.T) {
	spec := VectorColSpec{Column: "image_embedding", Dim: 512, Metric: "euclidean", Modality: "image"}
	if spec.Column != "image_embedding" || spec.Dim != 512 || spec.Metric != "euclidean" || spec.Modality != "image" {
		t.Errorf("unexpected field values: %+v", spec)
	}
}

func TestCompactOptions_Fields(t *testing.T) {
	opts := CompactOptions{TargetSize: 1024, MinFiles: 2, MaxFilesPerPass: 10, Deferred: true}
	if opts.TargetSize != 1024 || opts.MinFiles != 2 || opts.MaxFilesPerPass != 10 || !opts.Deferred {
		t.Errorf("unexpected field values: %+v", opts)
	}
}

// ── Integration tests: multi-column write + compact (own temp warehouse,
// require only AILAKE_BIN — no shared AILAKE_FIXTURE needed since these
// write their own data via testdata/multimodal_fixture.parquet) ────────────

func TestWriteBatchMultiColumnIntegration(t *testing.T) {
	bin := os.Getenv("AILAKE_BIN")
	if bin == "" {
		t.Skip("AILAKE_BIN not set")
	}

	catalog := &HadoopCatalog{Warehouse: t.TempDir()}
	err := WriteBatch(catalog, "default", "media", "testdata/multimodal_fixture.parquet", WriteBatchOptions{
		VectorCols: []VectorColSpec{
			{Column: "embedding", Dim: 4, Metric: "cosine"},
			{Column: "image_embedding", Dim: 2, Metric: "cosine", Modality: "image"},
		},
	})
	if err != nil {
		t.Fatalf("WriteBatch (multi-column): %v", err)
	}

	results, err := SearchMultimodal(catalog, "default", "media", []ModalQuery{
		{Column: "embedding", Query: []float32{0.1, 0.2, 0.3, 0.4}, Weight: 0.7},
		{Column: "image_embedding", Query: []float32{0.5, 0.6}, Weight: 0.3},
	}, SearchOptions{TopK: 3})
	if err != nil {
		t.Fatalf("SearchMultimodal: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("SearchMultimodal: expected 3 results, got %d", len(results))
	}
}

func TestCompactIntegration(t *testing.T) {
	bin := os.Getenv("AILAKE_BIN")
	if bin == "" {
		t.Skip("AILAKE_BIN not set")
	}

	catalog := &HadoopCatalog{Warehouse: t.TempDir()}
	opts := WriteBatchOptions{VecCol: "embedding"}
	if err := WriteBatch(catalog, "default", "docs", "testdata/multimodal_fixture.parquet", opts); err != nil {
		t.Fatalf("WriteBatch (batch 1): %v", err)
	}
	if err := WriteBatch(catalog, "default", "docs", "testdata/multimodal_fixture.parquet", opts); err != nil {
		t.Fatalf("WriteBatch (batch 2): %v", err)
	}

	filesCompacted, err := Compact(catalog, "default", "docs", CompactOptions{MinFiles: 2})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if filesCompacted != 1 {
		t.Errorf("Compact: expected 1 file compacted, got %d", filesCompacted)
	}

	results, err := Search(catalog, "default", "docs", []float32{0.1, 0.2, 0.3, 0.4}, SearchOptions{TopK: 20})
	if err != nil {
		t.Fatalf("Search after compact: %v", err)
	}
	if len(results) != 12 {
		t.Errorf("Search after compact: expected 12 rows searchable (2 batches x 6 rows), got %d", len(results))
	}
}
