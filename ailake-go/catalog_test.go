// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

// ── asInt64 ───────────────────────────────────────────────────────────────────

func TestAsInt64(t *testing.T) {
	cases := []struct {
		input any
		want  int64
	}{
		{int64(42), 42},
		{int32(7), 7},
		{float64(3.9), 3},   // truncates
		{"99", 99},
		{nil, 0},
		{"bad", 0},
	}
	for _, c := range cases {
		if got := asInt64(c.input); got != c.want {
			t.Errorf("asInt64(%v %T): got %d, want %d", c.input, c.input, got, c.want)
		}
	}
}

// ── decodeCentroid ────────────────────────────────────────────────────────────

func TestDecodeCentroid_Valid(t *testing.T) {
	// Rust encodes only the vector floats (dim*4 bytes). Radius is a separate JSON field.
	dim := 2
	buf := make([]byte, dim*4)
	vecs := []float32{1.0, -0.5}
	for i, v := range vecs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	b64 := base64.StdEncoding.EncodeToString(buf)

	vec, err := decodeCentroid(b64)
	if err != nil {
		t.Fatalf("decodeCentroid: %v", err)
	}
	if len(vec) != dim {
		t.Fatalf("vec len: got %d, want %d", len(vec), dim)
	}
	if math.Abs(float64(vec[0]-1.0)) > 1e-6 {
		t.Errorf("vec[0]: got %v, want 1.0", vec[0])
	}
	if math.Abs(float64(vec[1]+0.5)) > 1e-6 {
		t.Errorf("vec[1]: got %v, want -0.5", vec[1])
	}
}

func TestDecodeCentroid_BadBase64(t *testing.T) {
	if _, err := decodeCentroid("!!!not-base64!!!"); err == nil {
		t.Error("decodeCentroid bad base64: expected error, got nil")
	}
}

func TestDecodeCentroid_TooShort(t *testing.T) {
	// 3 bytes — not a multiple of 4.
	b64 := base64.StdEncoding.EncodeToString([]byte{1, 2, 3})
	if _, err := decodeCentroid(b64); err == nil {
		t.Error("decodeCentroid not multiple of 4: expected error, got nil")
	}
}

// ── HadoopCatalog.tableDir ────────────────────────────────────────────────────

func TestTableDir(t *testing.T) {
	c := &HadoopCatalog{Warehouse: "/data/warehouse"}
	got := c.tableDir("default", "docs")
	want := "/data/warehouse/default/docs"
	if got != want {
		t.Errorf("tableDir: got %q, want %q", got, want)
	}
}

// ── ExtraVectorIndex — ailakeEntryExt JSON unmarshal ─────────────────────────

func unmarshalExt(t *testing.T, jsonStr string) ailakeEntryExt {
	t.Helper()
	var ext ailakeEntryExt
	if err := json.Unmarshal([]byte(jsonStr), &ext); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	return ext
}

func TestExtraVectorIndexes_Valid(t *testing.T) {
	dim := 2
	buf := make([]byte, dim*4)
	vecs := []float32{0.1, -0.9}
	for i, v := range vecs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	centroidB64 := base64.StdEncoding.EncodeToString(buf)

	ext := unmarshalExt(t, `{
		"extra_vector_indexes": [
			{
				"column": "context_embedding",
				"dim": 2,
				"hnsw_offset": 131072,
				"hnsw_len": 65536,
				"centroid_b64": "`+centroidB64+`",
				"radius": 0.42
			}
		]
	}`)

	if len(ext.ExtraVectorIndexes) != 1 {
		t.Fatalf("len: got %d, want 1", len(ext.ExtraVectorIndexes))
	}
	xi := ext.ExtraVectorIndexes[0]
	if xi.Column != "context_embedding" {
		t.Errorf("Column: got %q", xi.Column)
	}
	if xi.Dim != 2 {
		t.Errorf("Dim: got %d, want 2", xi.Dim)
	}
	if xi.HnswOffset != 131072 {
		t.Errorf("HnswOffset: got %d, want 131072", xi.HnswOffset)
	}
	if xi.HnswLen != 65536 {
		t.Errorf("HnswLen: got %d, want 65536", xi.HnswLen)
	}
	if xi.Radius == nil || math.Abs(float64(*xi.Radius-0.42)) > 1e-5 {
		t.Errorf("Radius: got %v, want ~0.42", xi.Radius)
	}
	if xi.CentroidB64 == nil || *xi.CentroidB64 != centroidB64 {
		t.Error("CentroidB64 mismatch")
	}
}

func TestExtraVectorIndexes_Empty(t *testing.T) {
	ext := unmarshalExt(t, `{"hnsw_offset": 4096}`)
	if len(ext.ExtraVectorIndexes) != 0 {
		t.Errorf("got %d entries, want 0", len(ext.ExtraVectorIndexes))
	}
}

func TestExtraVectorIndexes_Multiple(t *testing.T) {
	ext := unmarshalExt(t, `{
		"extra_vector_indexes": [
			{"column": "col_a", "dim": 4, "hnsw_offset": 1000, "hnsw_len": 500, "centroid_b64": "", "radius": 0.1},
			{"column": "col_b", "dim": 8, "hnsw_offset": 2000, "hnsw_len": 1000, "centroid_b64": "", "radius": 0.2}
		]
	}`)
	if len(ext.ExtraVectorIndexes) != 2 {
		t.Fatalf("len: got %d, want 2", len(ext.ExtraVectorIndexes))
	}
	if ext.ExtraVectorIndexes[0].Column != "col_a" {
		t.Errorf("entry[0].Column: got %q", ext.ExtraVectorIndexes[0].Column)
	}
	if ext.ExtraVectorIndexes[1].HnswOffset != 2000 {
		t.Errorf("entry[1].HnswOffset: got %d, want 2000", ext.ExtraVectorIndexes[1].HnswOffset)
	}
}

// ── PartitionValue in ailakeEntryExt (Phase 9) ────────────────────────────────

func TestAilakeEntryExt_PartitionValue_Parsed(t *testing.T) {
	ext := unmarshalExt(t, `{"hnsw_offset": 4096, "partition_value": "agent-A"}`)
	if ext.PartitionValue == nil {
		t.Fatal("PartitionValue: expected non-nil, got nil")
	}
	if *ext.PartitionValue != "agent-A" {
		t.Errorf("PartitionValue: got %q, want %q", *ext.PartitionValue, "agent-A")
	}
}

func TestAilakeEntryExt_PartitionValue_Missing_IsNil(t *testing.T) {
	ext := unmarshalExt(t, `{"hnsw_offset": 4096}`)
	if ext.PartitionValue != nil {
		t.Errorf("PartitionValue: expected nil when absent, got %q", *ext.PartitionValue)
	}
}

func TestAilakeEntryExt_PartitionValue_EmptyString_IsNonNil(t *testing.T) {
	ext := unmarshalExt(t, `{"partition_value": ""}`)
	if ext.PartitionValue == nil {
		t.Fatal("PartitionValue: expected non-nil pointer for empty string field")
	}
	if *ext.PartitionValue != "" {
		t.Errorf("PartitionValue: got %q, want empty string", *ext.PartitionValue)
	}
}
