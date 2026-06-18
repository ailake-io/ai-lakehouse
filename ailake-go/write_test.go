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
