// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"encoding/base64"
	"encoding/binary"
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
