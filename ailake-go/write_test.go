// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"os"
	"path/filepath"
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

func TestAppendCatalogArgs_EmptyIsNoop(t *testing.T) {
	args := appendCatalogArgs([]string{"--store", "/tmp/wh"}, nil)
	if len(args) != 2 {
		t.Errorf("expected no flags appended for nil opts, got %v", args)
	}
	args = appendCatalogArgs([]string{"--store", "/tmp/wh"}, map[string]string{})
	if len(args) != 2 {
		t.Errorf("expected no flags appended for empty opts, got %v", args)
	}
}

func TestAppendCatalogArgs_Rest(t *testing.T) {
	args := appendCatalogArgs(nil, map[string]string{
		"catalog":    "rest",
		"rest-uri":   "https://catalog.example.com",
		"rest-auth":  "bearer",
		"rest-token": "tok123",
	})
	want := []string{
		"--catalog", "rest",
		"--rest-uri", "https://catalog.example.com",
		"--rest-auth", "bearer",
		"--rest-token", "tok123",
	}
	if len(args) != len(want) {
		t.Fatalf("expected %v, got %v", want, args)
	}
	for i := range want {
		if args[i] != want[i] {
			t.Fatalf("expected %v, got %v", want, args)
		}
	}
}

func TestAppendCatalogArgs_UnknownKeyIgnored(t *testing.T) {
	args := appendCatalogArgs(nil, map[string]string{"not-a-real-key": "x"})
	if len(args) != 0 {
		t.Errorf("expected unrecognized key to be ignored, got %v", args)
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

func TestCreateTableIntegration(t *testing.T) {
	bin := os.Getenv("AILAKE_BIN")
	if bin == "" {
		t.Skip("AILAKE_BIN not set")
	}

	catalog := &HadoopCatalog{Warehouse: t.TempDir()}
	err := CreateTable(catalog, "default", "docs", 4, CreateTableOptions{
		Metric:    "cosine",
		Precision: "f16",
		PQOnly:    false,
	})
	if err != nil {
		t.Fatalf("CreateTable: %v", err)
	}

	info, err := catalog.LoadTable("default", "docs")
	if err != nil {
		t.Fatalf("LoadTable after CreateTable: %v", err)
	}
	if info.VectorDim != "4" {
		t.Errorf("expected vector dim '4', got %q", info.VectorDim)
	}
	if info.VectorMetric != "cosine" {
		t.Errorf("expected metric 'cosine', got %q", info.VectorMetric)
	}

	// Writing into the pre-created table must succeed without re-creating it.
	if err := WriteBatch(catalog, "default", "docs", "testdata/multimodal_fixture.parquet", WriteBatchOptions{}); err != nil {
		t.Fatalf("WriteBatch into pre-created table: %v", err)
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

// ── Estimate (no warehouse/I-O — pure math) ──────────────────────────────────

func TestEstimateIntegration(t *testing.T) {
	if os.Getenv("AILAKE_BIN") == "" {
		t.Skip("AILAKE_BIN not set")
	}

	result, err := Estimate("1M", 1536, EstimateOptions{})
	if err != nil {
		t.Fatalf("Estimate: %v", err)
	}
	if result.Rows != 1_000_000 {
		t.Errorf("expected 1,000,000 rows, got %d", result.Rows)
	}
	if result.Dim != 1536 {
		t.Errorf("expected dim 1536, got %d", result.Dim)
	}
	if len(result.Estimates) == 0 {
		t.Error("expected non-empty Estimates")
	}
}

// ── AddVectorColumn / DeleteRows (own temp warehouse) ────────────────────────

func TestAddVectorColumnIntegration(t *testing.T) {
	if os.Getenv("AILAKE_BIN") == "" {
		t.Skip("AILAKE_BIN not set")
	}

	catalog := &HadoopCatalog{Warehouse: t.TempDir()}
	if err := CreateTable(catalog, "default", "docs", 4, CreateTableOptions{Metric: "cosine"}); err != nil {
		t.Fatalf("CreateTable: %v", err)
	}
	if err := AddVectorColumn(catalog, "default", "docs", "image_embedding", 2, AddVectorColumnOptions{Metric: "cosine"}); err != nil {
		t.Fatalf("AddVectorColumn: %v", err)
	}

	info, err := catalog.LoadTable("default", "docs")
	if err != nil {
		t.Fatalf("LoadTable after AddVectorColumn: %v", err)
	}
	found := false
	for _, f := range info.SchemaFields {
		if f.Name == "image_embedding" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected schema to contain 'image_embedding' after AddVectorColumn, got %+v", info.SchemaFields)
	}
}

func TestDeleteRowsIntegration(t *testing.T) {
	if os.Getenv("AILAKE_BIN") == "" {
		t.Skip("AILAKE_BIN not set")
	}

	catalog := &HadoopCatalog{Warehouse: t.TempDir()}
	if err := CreateTable(catalog, "default", "docs", 4, CreateTableOptions{Metric: "cosine", FormatVersion: 3}); err != nil {
		t.Fatalf("CreateTable (format-version 3): %v", err)
	}
	if err := WriteBatch(catalog, "default", "docs", "testdata/multimodal_fixture.parquet", WriteBatchOptions{}); err != nil {
		t.Fatalf("WriteBatch: %v", err)
	}

	files, err := catalog.ListFiles("default", "docs")
	if err != nil {
		t.Fatalf("ListFiles: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("expected at least one data file after WriteBatch")
	}

	if err := DeleteRows(catalog, "default", "docs", files[0].Path, []int{0}, nil); err != nil {
		t.Fatalf("DeleteRows: %v", err)
	}

	// Regression fix (delete.go): this package's Search/Scan now applies
	// Deletion Vectors (position deletes, V3) the same way the real `ailake`
	// CLI's own search does — DeleteRows above wrote row position 0 of
	// files[0] into a Puffin DV, and searchFile() masks it via
	// bm.Contains(rowID) before returning hits. 1 of 6 rows is gone.
	results, err := Search(catalog, "default", "docs", []float32{0.1, 0.2, 0.3, 0.4}, SearchOptions{TopK: 20})
	if err != nil {
		t.Fatalf("Search after DeleteRows: %v", err)
	}
	if len(results) != 5 {
		t.Errorf("expected 5 of 6 rows visible after DeleteRows masked 1 via its Deletion Vector, got %d", len(results))
	}
	for _, r := range results {
		if r.FilePath == files[0].Path && r.RowID == 0 {
			t.Errorf("expected row 0 of %s to be masked by the Deletion Vector, but it was returned", files[0].Path)
		}
	}
}

func TestDeleteWhereAppliedInNativeScan(t *testing.T) {
	if os.Getenv("AILAKE_BIN") == "" {
		t.Skip("AILAKE_BIN not set")
	}

	// Regression: this package's native Scan (Search + FetchRows) never
	// applied equality deletes at all — see delete.go. testdata/multimodal_fixture.parquet
	// has 6 rows with a plain int64 "id" column (0..5), dim=4 "embedding".
	catalog := &HadoopCatalog{Warehouse: t.TempDir()}
	if err := CreateTable(catalog, "default", "docs", 4, CreateTableOptions{Metric: "cosine"}); err != nil {
		t.Fatalf("CreateTable: %v", err)
	}
	if err := WriteBatch(catalog, "default", "docs", "testdata/multimodal_fixture.parquet", WriteBatchOptions{}); err != nil {
		t.Fatalf("WriteBatch: %v", err)
	}
	if err := DeleteWhere(catalog, "default", "docs", "id", []string{"3"}); err != nil {
		t.Fatalf("DeleteWhere: %v", err)
	}

	rows, err := Scan(catalog, "default", "docs", []float32{0.1, 0.2, 0.3, 0.4}, SearchOptions{TopK: 20})
	if err != nil {
		t.Fatalf("Scan after DeleteWhere: %v", err)
	}
	if len(rows) != 5 {
		t.Errorf("expected 5 of 6 rows visible after DeleteWhere masked id=3, got %d", len(rows))
	}
	for _, r := range rows {
		if id, ok := r.Fields["id"].(int64); ok && id == 3 {
			t.Errorf("expected id=3 to be masked by the equality delete, but it was returned")
		}
	}
}

// ── DecayMemories / Migrate / BackfillVectorColumn — CLI arg-forwarding checks
// (own fake `ailake` binary that echoes its argv to a file, not a mock of
// this package's own code) ───────────────────────────────────────────────────
//
// These 3 operations need a table with columns (chunk_text, last_accessed_at,
// ...) that no fixture in this repo currently provides — round-tripping a
// real embed_cmd/memory-decay pass is out of scope here. What IS verified for
// real: the exact argv this package's Go code builds and hands to
// exec.Command, by pointing AILAKE_BIN at a throwaway shell script instead of
// the real CLI.

func writeFakeAilakeBin(t *testing.T, dir string, extraStdout string) (binPath, argsPath string) {
	t.Helper()
	binPath = filepath.Join(dir, "fake_ailake")
	argsPath = filepath.Join(dir, "args.txt")
	script := "#!/bin/sh\necho \"$@\" > " + argsPath + "\n"
	if extraStdout != "" {
		script += "echo '" + extraStdout + "'\n"
	}
	if err := os.WriteFile(binPath, []byte(script), 0o755); err != nil {
		t.Fatalf("writing fake ailake binary: %v", err)
	}
	return binPath, argsPath
}

func TestDecayMemories_ArgsForwarded(t *testing.T) {
	dir := t.TempDir()
	bin, argsPath := writeFakeAilakeBin(t, dir, `{"ok":true,"files_updated":3}`)
	orig := os.Getenv("AILAKE_BIN")
	os.Setenv("AILAKE_BIN", bin)
	defer os.Setenv("AILAKE_BIN", orig)

	catalog := &HadoopCatalog{Warehouse: "/tmp/wh"}
	n, err := DecayMemories(catalog, "default", "docs", 0.25, map[string]string{"catalog": "rest", "rest-uri": "https://catalog.example.com"})
	if err != nil {
		t.Fatalf("DecayMemories: %v", err)
	}
	if n != 3 {
		t.Errorf("expected files_updated=3 from fake response, got %d", n)
	}
	argsRaw, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("reading captured args: %v", err)
	}
	args := string(argsRaw)
	for _, want := range []string{"decay-memories", "default.docs", "--lambda 0.25", "--format json", "--catalog rest", "--rest-uri https://catalog.example.com"} {
		if !strings.Contains(args, want) {
			t.Errorf("expected args to contain %q, got: %s", want, args)
		}
	}
}

func TestMigrate_ArgsForwarded(t *testing.T) {
	dir := t.TempDir()
	bin, argsPath := writeFakeAilakeBin(t, dir, "")
	orig := os.Getenv("AILAKE_BIN")
	os.Setenv("AILAKE_BIN", bin)
	defer os.Setenv("AILAKE_BIN", orig)

	catalog := &HadoopCatalog{Warehouse: "/tmp/wh"}
	err := Migrate(catalog, "default", "docs", "python3 embed.py", MigrateOptions{
		OldColumn: "embedding", NewColumn: "embedding_v2", TextColumn: "chunk_text",
		Strategy: "atomic-replace", BatchSize: 256, ModelName: "text-embedding-3-small",
	})
	if err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	argsRaw, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("reading captured args: %v", err)
	}
	args := string(argsRaw)
	for _, want := range []string{
		"migrate", "default.docs", "--embed-cmd python3 embed.py",
		"--old-column embedding", "--new-column embedding_v2", "--text-column chunk_text",
		"--strategy atomic-replace", "--batch-size 256", "--model-name text-embedding-3-small",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("expected args to contain %q, got: %s", want, args)
		}
	}
}

func TestBackfillVectorColumn_ArgsForwarded(t *testing.T) {
	dir := t.TempDir()
	bin, argsPath := writeFakeAilakeBin(t, dir, "")
	orig := os.Getenv("AILAKE_BIN")
	os.Setenv("AILAKE_BIN", bin)
	defer os.Setenv("AILAKE_BIN", orig)

	catalog := &HadoopCatalog{Warehouse: "/tmp/wh"}
	err := BackfillVectorColumn(catalog, "default", "docs", "image_embedding", "python3 embed_images.py", BackfillVectorColumnOptions{
		TextColumn: "image_uri", BatchSize: 128,
	})
	if err != nil {
		t.Fatalf("BackfillVectorColumn: %v", err)
	}
	argsRaw, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("reading captured args: %v", err)
	}
	args := string(argsRaw)
	for _, want := range []string{
		"backfill-vector-column", "default.docs", "--column image_embedding",
		"--embed-cmd python3 embed_images.py", "--text-column image_uri", "--batch-size 128",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("expected args to contain %q, got: %s", want, args)
		}
	}
}

// TestDeleteWhere_CatalogOptsForwarded and TestEvolveSchema_CatalogOptsForwarded
// verify the fix for the gap documented in docs/guides/REST_CATALOG.md's
// "Known limitations": DeleteWhere/EvolveSchema were the two remaining
// write-path functions never wired to appendCatalogArgs (Fase 19 only wired
// WriteBatch/Compact) — every other write-path function (WriteBatch, Compact,
// DecayMemories, Migrate, DeleteRows, AddVectorColumn, BackfillVectorColumn,
// CreateTable) already forwarded --catalog/--rest-*.

func TestDeleteWhere_CatalogOptsForwarded(t *testing.T) {
	dir := t.TempDir()
	bin, argsPath := writeFakeAilakeBin(t, dir, "")
	orig := os.Getenv("AILAKE_BIN")
	os.Setenv("AILAKE_BIN", bin)
	defer os.Setenv("AILAKE_BIN", orig)

	catalog := &HadoopCatalog{Warehouse: "/tmp/wh"}
	err := DeleteWhere(catalog, "default", "docs", "doc_id", []string{"a", "b"},
		map[string]string{"catalog": "rest", "rest-uri": "https://catalog.example.com"})
	if err != nil {
		t.Fatalf("DeleteWhere: %v", err)
	}
	argsRaw, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("reading captured args: %v", err)
	}
	args := string(argsRaw)
	for _, want := range []string{
		"delete-where", "default.docs", "--col doc_id", "--vals a,b",
		"--catalog rest", "--rest-uri https://catalog.example.com",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("expected args to contain %q, got: %s", want, args)
		}
	}
}

func TestDeleteWhere_CatalogOptsOmittedStillWorks(t *testing.T) {
	// Backward compatibility: the variadic param must be safely omittable by
	// existing callers compiled against the pre-Fase-19-fix signature.
	dir := t.TempDir()
	bin, argsPath := writeFakeAilakeBin(t, dir, "")
	orig := os.Getenv("AILAKE_BIN")
	os.Setenv("AILAKE_BIN", bin)
	defer os.Setenv("AILAKE_BIN", orig)

	catalog := &HadoopCatalog{Warehouse: "/tmp/wh"}
	if err := DeleteWhere(catalog, "default", "docs", "doc_id", []string{"a"}); err != nil {
		t.Fatalf("DeleteWhere: %v", err)
	}
	argsRaw, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("reading captured args: %v", err)
	}
	if strings.Contains(string(argsRaw), "--catalog") {
		t.Errorf("expected no --catalog flag when catalogOpts omitted, got: %s", string(argsRaw))
	}
}

func TestEvolveSchema_CatalogOptsForwarded(t *testing.T) {
	dir := t.TempDir()
	bin, argsPath := writeFakeAilakeBin(t, dir, "new_schema_id: 2")
	orig := os.Getenv("AILAKE_BIN")
	os.Setenv("AILAKE_BIN", bin)
	defer os.Setenv("AILAKE_BIN", orig)

	catalog := &HadoopCatalog{Warehouse: "/tmp/wh"}
	schemaID, err := EvolveSchema(catalog, "default", "docs",
		[]AddColumnReq{{Name: "source", Type: "string"}}, nil,
		map[string]string{"catalog": "rest", "rest-uri": "https://catalog.example.com"})
	if err != nil {
		t.Fatalf("EvolveSchema: %v", err)
	}
	if schemaID != 2 {
		t.Errorf("expected schema_id=2 from fake response, got %d", schemaID)
	}
	argsRaw, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("reading captured args: %v", err)
	}
	args := string(argsRaw)
	for _, want := range []string{
		"evolve", "default.docs", "--add source:string",
		"--catalog rest", "--rest-uri https://catalog.example.com",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("expected args to contain %q, got: %s", want, args)
		}
	}
}
