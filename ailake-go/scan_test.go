// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"math"
	"os"
	"testing"

	parquetgo "github.com/parquet-go/parquet-go"
)

// ── Helper tests ───────────────────────────────────────────────────────────────

func TestResolveWarehousePath(t *testing.T) {
	cases := []struct {
		name      string
		warehouse string
		path      string
		want      string
	}{
		{"relative joins onto warehouse", "/data", "file.parquet", "/data/file.parquet"},
		{"warehouse with trailing slash", "/data/", "file.parquet", "/data/file.parquet"},
		{"empty warehouse", "", "file.parquet", "file.parquet"},
		{"already OS-absolute, used as-is", "/data", "/other/file.parquet", "/other/file.parquet"},
		// The regression case: ailake-py's local_catalog_store always writes
		// warehouse_uri as file://<absolute path> (Trino Iceberg-connector
		// compatibility), so metadata.json's manifest-list can be an absolute
		// file:// URI. Before this fix, resolveWarehousePath's predecessors
		// (filepath.IsAbs / isAbsPath) didn't recognize "file://" as absolute,
		// so this got joined onto warehouse — filepath.Join then normalized
		// "/data" + "file:///abs/path/snap-1.avro" into the corrupted
		// "/data/file:/abs/path/snap-1.avro", not "/abs/path/snap-1.avro".
		{
			"absolute file:// URI, scheme stripped and used as-is",
			"/data/go_client_test",
			"file:///home/thiago/data/go_client_test/default/table/metadata/snap-1.avro",
			"/home/thiago/data/go_client_test/default/table/metadata/snap-1.avro",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := resolveWarehousePath(c.warehouse, c.path); got != c.want {
				t.Errorf("resolveWarehousePath(%q, %q) = %q, want %q", c.warehouse, c.path, got, c.want)
			}
		})
	}
}

// ── parquetRowToFields unit tests ─────────────────────────────────────────────

func TestParquetRowToFieldsBasicTypes(t *testing.T) {
	row := parquetgo.Row{
		parquetgo.ValueOf(int64(42)).Level(0, 0, 0),
		parquetgo.ValueOf(int32(7)).Level(0, 0, 1),
		parquetgo.ValueOf(float32(1.5)).Level(0, 0, 2),
		parquetgo.ValueOf(float64(2.5)).Level(0, 0, 3),
		parquetgo.ValueOf(true).Level(0, 0, 4),
		parquetgo.ValueOf("hello").Level(0, 0, 5),
	}
	cols := []string{"id", "count", "score_f32", "score_f64", "flag", "text"}

	fields := parquetRowToFields(row, cols, "", 0)

	if v, ok := fields["id"].(int64); !ok || v != 42 {
		t.Errorf("id: got %v (%T), want int64(42)", fields["id"], fields["id"])
	}
	if v, ok := fields["count"].(int64); !ok || v != 7 {
		t.Errorf("count: got %v, want int64(7)", fields["count"])
	}
	if v, ok := fields["score_f32"].(float64); !ok || math.Abs(v-1.5) > 1e-5 {
		t.Errorf("score_f32: got %v, want ~1.5", fields["score_f32"])
	}
	if v, ok := fields["score_f64"].(float64); !ok || math.Abs(v-2.5) > 1e-9 {
		t.Errorf("score_f64: got %v, want 2.5", fields["score_f64"])
	}
	if v, ok := fields["flag"].(bool); !ok || !v {
		t.Errorf("flag: got %v, want true", fields["flag"])
	}
	if v, ok := fields["text"].(string); !ok || v != "hello" {
		t.Errorf("text: got %v, want 'hello'", fields["text"])
	}
}

func TestParquetRowToFieldsVectorDecoded(t *testing.T) {
	// Encode a 2-dim F16 vector: [1.0, 0.5]
	dim := 2
	raw := make([]byte, dim*2)
	// 1.0 in F16 = 0x3C00
	raw[0], raw[1] = 0x00, 0x3C
	// 0.5 in F16 = 0x3800
	raw[2], raw[3] = 0x00, 0x38

	row := parquetgo.Row{
		parquetgo.FixedLenByteArrayValue(raw).Level(0, 0, 0),
	}
	cols := []string{"embedding"}

	fields := parquetRowToFields(row, cols, "embedding", uint32(dim))

	vec, ok := fields["embedding"].([]float32)
	if !ok {
		t.Fatalf("embedding: got %T, want []float32", fields["embedding"])
	}
	if len(vec) != dim {
		t.Fatalf("embedding len: got %d, want %d", len(vec), dim)
	}
	if math.Abs(float64(vec[0])-1.0) > 1e-3 {
		t.Errorf("embedding[0]: got %v, want ~1.0", vec[0])
	}
	if math.Abs(float64(vec[1])-0.5) > 1e-3 {
		t.Errorf("embedding[1]: got %v, want ~0.5", vec[1])
	}
}

func TestParquetRowToFieldsNullSkipped(t *testing.T) {
	// Null value should not appear in Fields map.
	null := parquetgo.Value{}.Level(0, 0, 0) // zero Value is null
	row := parquetgo.Row{null}
	fields := parquetRowToFields(row, []string{"col"}, "", 0)
	if _, exists := fields["col"]; exists {
		t.Errorf("null value should not appear in Fields map")
	}
}

func TestFetchRowsEmpty(t *testing.T) {
	rows, err := FetchRows(nil, "/tmp", "embedding", 128)
	if err != nil {
		t.Fatalf("FetchRows(nil): unexpected error: %v", err)
	}
	if rows != nil {
		t.Errorf("FetchRows(nil): expected nil, got %v", rows)
	}
}

// ── Integration test (requires AILAKE_FIXTURE env var) ────────────────────────

// TestScanIntegration verifies end-to-end Scan() against the compat fixture.
//
// Set AILAKE_FIXTURE to the fixture directory written by write_fixture.py.
// The fixture uses namespace="default", table="table", dim=128.
//
// Skip when AILAKE_FIXTURE is not set (unit-test-only runs).
func TestScanIntegration(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set — skipping integration test")
	}

	catalog := &HadoopCatalog{Warehouse: fixtureDir}

	const dim = 128
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i) / float32(dim)
	}
	// Normalize to unit length (cosine-safe).
	var norm float32
	for _, v := range query {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	for i := range query {
		query[i] /= norm
	}

	const topK = 10
	rows, err := Scan(catalog, "default", "table", query, SearchOptions{TopK: topK})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}

	if len(rows) == 0 {
		t.Fatal("Scan returned 0 rows")
	}
	if len(rows) > topK {
		t.Errorf("Scan returned %d rows, want <= %d", len(rows), topK)
	}

	// Distances must be non-negative and ascending.
	for i, r := range rows {
		if r.Distance < 0 {
			t.Errorf("row %d: negative distance %v", i, r.Distance)
		}
		if len(r.Fields) == 0 {
			t.Errorf("row %d: Fields map is empty — Parquet rows not fetched", i)
		}
		if i > 0 && r.Distance < rows[i-1].Distance {
			t.Errorf("row %d: distance %v < previous %v (not sorted)", i, r.Distance, rows[i-1].Distance)
		}
	}

	// Each ScanRow must have more data than a bare FileSearchResult.
	// Check at least one non-vector field exists.
	for i, r := range rows {
		hasScalar := false
		for _, v := range r.Fields {
			switch v.(type) {
			case int64, float64, string, bool:
				hasScalar = true
			}
		}
		if !hasScalar {
			t.Errorf("row %d: no scalar fields in Fields map (keys: %v)", i, fieldKeys(r.Fields))
		}
	}
}

// TestScanVsSearchConsistency checks that Scan and Search return the same row ID set.
// Order is normalised by (distance, rowID) before comparison because unstable sort
// can break ties differently across two independent HNSW traversals.
func TestScanVsSearchConsistency(t *testing.T) {
	fixtureDir := os.Getenv("AILAKE_FIXTURE")
	if fixtureDir == "" {
		t.Skip("AILAKE_FIXTURE not set — skipping integration test")
	}

	catalog := &HadoopCatalog{Warehouse: fixtureDir}

	const dim = 128
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i%7) / 7.0
	}

	opts := SearchOptions{TopK: 5}

	scanRows, err := Scan(catalog, "default", "table", query, opts)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	searchRows, err := Search(catalog, "default", "table", query, opts)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(scanRows) != len(searchRows) {
		t.Fatalf("Scan=%d rows, Search=%d rows — must match", len(scanRows), len(searchRows))
	}

	// Build rowID sets — ties in distance can produce different orderings across
	// two independent HNSW traversals with unstable sort, so compare sets not slices.
	scanIDs := make(map[uint64]float32, len(scanRows))
	for _, r := range scanRows {
		scanIDs[r.RowID] = r.Distance
	}
	for _, r := range searchRows {
		dist, ok := scanIDs[r.RowID]
		if !ok {
			t.Errorf("Search row %d not in Scan results", r.RowID)
			continue
		}
		if math.Abs(float64(dist-r.Distance)) > 1e-4 {
			t.Errorf("rowID=%d distance mismatch: Scan=%v Search=%v", r.RowID, dist, r.Distance)
		}
	}
}

func fieldKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
