// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"math"
	"os"
	"testing"
)

// ── DecodeF16Vector / f16ToF32 ────────────────────────────────────────────────

func TestDecodeF16Vector_KnownValues(t *testing.T) {
	// F16 encoding (little-endian):
	//   1.0  = 0x3C00
	//   0.5  = 0x3800
	//  -1.0  = 0xBC00
	//   0.0  = 0x0000
	raw := []byte{
		0x00, 0x3C, // 1.0
		0x00, 0x38, // 0.5
		0x00, 0xBC, // -1.0
		0x00, 0x00, // 0.0
	}
	want := []float32{1.0, 0.5, -1.0, 0.0}
	got := DecodeF16Vector(raw, 4)
	if len(got) != len(want) {
		t.Fatalf("len: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Errorf("[%d]: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestDecodeF16Vector_TooShort(t *testing.T) {
	if got := DecodeF16Vector([]byte{0x00, 0x3C}, 4); got != nil {
		t.Errorf("expected nil for short input, got %v", got)
	}
}

func TestDecodeF16Vector_Zeros(t *testing.T) {
	raw := make([]byte, 6) // dim=3, all zeros = 0.0
	got := DecodeF16Vector(raw, 3)
	if len(got) != 3 {
		t.Fatalf("len: got %d, want 3", len(got))
	}
	for i, v := range got {
		if v != 0.0 {
			t.Errorf("[%d]: got %v, want 0.0", i, v)
		}
	}
}

// ── metricFromString ──────────────────────────────────────────────────────────

func TestMetricFromString(t *testing.T) {
	cases := []struct {
		input string
		want  uint8
	}{
		{"cosine", MetricCosine},
		{"", MetricCosine},
		{"unknown", MetricCosine},
		{"euclidean", MetricEuclidean},
		{"dotproduct", MetricDotProduct},
		{"dot", MetricDotProduct},
		{"dot_product", MetricCosine}, // not a recognised token → default cosine
		{"normalized_cosine", MetricNormalizedCosine},
	}
	for _, c := range cases {
		if got := metricFromString(c.input); got != c.want {
			t.Errorf("metricFromString(%q): got %d, want %d", c.input, got, c.want)
		}
	}
}

// ── KV hint parsers ───────────────────────────────────────────────────────────

func TestF32FromKVHint(t *testing.T) {
	cases := []struct {
		s    string
		want float32
	}{
		{"0.5", 0.5},
		{"1.0", 1.0},
		{"", 0.0},
		{"bad", 0.0},
	}
	for _, c := range cases {
		got := f32FromKVHint(c.s)
		if math.Abs(float64(got-c.want)) > 1e-6 {
			t.Errorf("f32FromKVHint(%q): got %v, want %v", c.s, got, c.want)
		}
	}
}

func TestU64FromKVHint(t *testing.T) {
	cases := []struct {
		s    string
		want uint64
	}{
		{"42", 42},
		{"0", 0},
		{"", 0},
		{"bad", 0},
	}
	for _, c := range cases {
		if got := u64FromKVHint(c.s); got != c.want {
			t.Errorf("u64FromKVHint(%q): got %d, want %d", c.s, got, c.want)
		}
	}
}

// ── PartitionFilter pruning (unit, no fixture) ────────────────────────────────

func TestPartitionPruning_MatchingFilter(t *testing.T) {
	entries := []DataFileEntry{
		{Path: "a.parquet", PartitionValue: "agent-A"},
		{Path: "b.parquet", PartitionValue: "agent-B"},
		{Path: "c.parquet", PartitionValue: "agent-A"},
	}
	filter := "agent-A"
	var pruned []DataFileEntry
	for _, e := range entries {
		if e.PartitionValue == filter {
			pruned = append(pruned, e)
		}
	}
	if len(pruned) != 2 {
		t.Fatalf("expected 2 entries matching agent-A, got %d", len(pruned))
	}
	for _, e := range pruned {
		if e.PartitionValue != "agent-A" {
			t.Errorf("pruned entry has wrong PartitionValue: %q", e.PartitionValue)
		}
	}
}

func TestPartitionPruning_EmptyFilterKeepsAll(t *testing.T) {
	entries := []DataFileEntry{
		{Path: "a.parquet", PartitionValue: "agent-A"},
		{Path: "b.parquet", PartitionValue: "agent-B"},
		{Path: "c.parquet", PartitionValue: ""},
	}
	filter := ""
	var pruned []DataFileEntry
	if filter != "" {
		for _, e := range entries {
			if e.PartitionValue == filter {
				pruned = append(pruned, e)
			}
		}
	} else {
		pruned = entries
	}
	if len(pruned) != len(entries) {
		t.Errorf("empty filter: expected %d entries, got %d", len(entries), len(pruned))
	}
}

func TestPartitionPruning_NonMatchingFilterYieldsEmpty(t *testing.T) {
	entries := []DataFileEntry{
		{Path: "a.parquet", PartitionValue: "agent-A"},
		{Path: "b.parquet", PartitionValue: "agent-B"},
	}
	filter := "agent-C"
	var pruned []DataFileEntry
	for _, e := range entries {
		if e.PartitionValue == filter {
			pruned = append(pruned, e)
		}
	}
	if len(pruned) != 0 {
		t.Errorf("expected 0 entries for non-matching filter, got %d", len(pruned))
	}
}

func TestSearchOptions_PartitionFilter_Field(t *testing.T) {
	opts := SearchOptions{TopK: 5, PartitionFilter: "agent-A"}
	if opts.PartitionFilter != "agent-A" {
		t.Errorf("PartitionFilter: got %q, want %q", opts.PartitionFilter, "agent-A")
	}
}

// ── Search integration (requires AILAKE_FIXTURE) ──────────────────────────────

func TestSearchIntegration(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set")
	}
	catalog := &HadoopCatalog{Warehouse: fixtureDir}

	query := makeTestQuery(128)
	results, err := Search(catalog, "default", "table", query, SearchOptions{TopK: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search returned 0 results")
	}
	if len(results) > 10 {
		t.Errorf("Search returned %d results, want <= 10", len(results))
	}
	for i, r := range results {
		if r.Distance < 0 {
			t.Errorf("result %d: negative distance %v", i, r.Distance)
		}
		if i > 0 && r.Distance < results[i-1].Distance {
			t.Errorf("result %d: not sorted (dist %v < prev %v)", i, r.Distance, results[i-1].Distance)
		}
		if r.FilePath == "" {
			t.Errorf("result %d: empty FilePath", i)
		}
	}
}

// ── Catalog integration (requires AILAKE_FIXTURE) ─────────────────────────────

func TestLoadTableIntegration(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set")
	}
	catalog := &HadoopCatalog{Warehouse: fixtureDir}

	info, err := catalog.LoadTable("default", "table")
	if err != nil {
		t.Fatalf("LoadTable: %v", err)
	}
	if info.VectorColumn == "" {
		t.Error("VectorColumn: empty")
	}
	if info.VectorDim == "" {
		t.Error("VectorDim: empty")
	}
	if info.SnapshotID == nil {
		t.Error("SnapshotID: nil")
	}
	if info.EmbeddingModel == "" {
		t.Error("EmbeddingModel: empty (expected fixture-model@v1 from write_fixture.py)")
	} else if info.EmbeddingModel != "fixture-model@v1" {
		t.Errorf("EmbeddingModel: got %q, want %q", info.EmbeddingModel, "fixture-model@v1")
	}
}

func TestListFilesIntegration(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set")
	}
	catalog := &HadoopCatalog{Warehouse: fixtureDir}

	entries, err := catalog.ListFiles("default", "table")
	if err != nil {
		t.Fatalf("ListFiles: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("ListFiles: returned 0 entries")
	}
	for i, e := range entries {
		if e.Path == "" {
			t.Errorf("entry %d: empty Path", i)
		}
		if e.RecordCount == 0 {
			t.Errorf("entry %d: RecordCount=0", i)
		}
		if e.EmbeddingModel == "" {
			t.Errorf("entry %d: EmbeddingModel empty (expected fixture-model@v1)", i)
		} else if e.EmbeddingModel != "fixture-model@v1" {
			t.Errorf("entry %d: EmbeddingModel=%q, want %q", i, e.EmbeddingModel, "fixture-model@v1")
		}
	}
}

func TestSearchDimMismatchIntegration(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set")
	}
	catalog := &HadoopCatalog{Warehouse: fixtureDir}

	// Fixture table uses dim=128; query with wrong dim should return error.
	wrongDimQuery := makeTestQuery(64)
	_, err := Search(catalog, "default", "table", wrongDimQuery, SearchOptions{TopK: 5})
	if err == nil {
		t.Fatal("Search with wrong dim: expected error, got nil")
	}
	t.Logf("Search dim mismatch error (expected): %v", err)
}

// ── Shared helpers ────────────────────────────────────────────────────────────

// makeTestQuery creates a unit-length query vector of the given dimension.
func makeTestQuery(dim int) []float32 {
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i) / float32(dim)
	}
	var norm float32
	for _, v := range q {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range q {
			q[i] /= norm
		}
	}
	return q
}
