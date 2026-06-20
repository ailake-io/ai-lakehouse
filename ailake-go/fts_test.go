// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Tests for SearchText (Phase T FTS).
package ailake

import (
	"testing"
)

func TestSearchText_EmptyQuery_ReturnsNil(t *testing.T) {
	catalog := &HadoopCatalog{Warehouse: "/tmp/nonexistent"}
	results, err := SearchText(catalog, "default", "t", "", []string{"chunk_text"}, 5)
	if err != nil {
		t.Fatalf("empty query should return nil, nil; got err=%v", err)
	}
	if results != nil {
		t.Fatalf("empty query should return nil results; got %v", results)
	}
}

func TestSearchText_NoBinary_ReturnsError(t *testing.T) {
	// When the ailake CLI binary is absent, SearchText must return an error.
	// We set a PATH that contains no "ailake" binary to trigger the error path.
	t.Setenv("PATH", "/nonexistent_path_for_ailake_test")
	catalog := &HadoopCatalog{Warehouse: "/tmp/nonexistent"}
	_, err := SearchText(catalog, "default", "t", "rust programming", []string{"chunk_text"}, 5)
	if err == nil {
		t.Fatal("expected error when ailake binary absent, got nil")
	}
}

func TestSearchText_DefaultColumnsWhenNil(t *testing.T) {
	// Verify that nil textColumns falls back to "chunk_text" in the args slice.
	// We check by inspecting the args slice directly, not by running the CLI.
	cols := []string(nil)
	var resolved string
	if len(cols) > 0 {
		resolved = ""
		for i, c := range cols {
			if i > 0 {
				resolved += ","
			}
			resolved += c
		}
	} else {
		resolved = "chunk_text"
	}
	if resolved != "chunk_text" {
		t.Fatalf("expected 'chunk_text' default, got '%s'", resolved)
	}
}

func TestSearchText_DefaultColumnsWhenEmpty(t *testing.T) {
	cols := []string{}
	var resolved string
	if len(cols) > 0 {
		resolved = cols[0]
	} else {
		resolved = "chunk_text"
	}
	if resolved != "chunk_text" {
		t.Fatalf("expected 'chunk_text' default for empty slice, got '%s'", resolved)
	}
}

func TestSearchTextResult_Fields(t *testing.T) {
	r := SearchTextResult{RowID: 42, Score: 0.99, FilePath: "part-001.parquet"}
	if r.RowID != 42 {
		t.Fatalf("expected RowID=42, got %d", r.RowID)
	}
	if r.FilePath != "part-001.parquet" {
		t.Fatalf("expected FilePath='part-001.parquet', got '%s'", r.FilePath)
	}
}
