// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// buildHeader constructs a valid 64-byte AILK header buffer.
func buildHeader(version, flags uint16, dim uint32, precision, metric uint8,
	recordCount, centroidOffset, centroidLen, hnswOffset, hnswLen uint64,
) []byte {
	buf := make([]byte, HeaderSize)
	buf[0], buf[1], buf[2], buf[3] = 'A', 'I', 'L', 'K'
	binary.LittleEndian.PutUint16(buf[4:], version)
	binary.LittleEndian.PutUint16(buf[6:], flags)
	binary.LittleEndian.PutUint32(buf[8:], dim)
	buf[12] = precision
	buf[13] = metric
	binary.LittleEndian.PutUint64(buf[16:], recordCount)
	binary.LittleEndian.PutUint64(buf[24:], centroidOffset)
	binary.LittleEndian.PutUint64(buf[32:], centroidLen)
	binary.LittleEndian.PutUint64(buf[40:], hnswOffset)
	binary.LittleEndian.PutUint64(buf[48:], hnswLen)
	return buf
}

func TestParseHeaderBytes_Valid(t *testing.T) {
	buf := buildHeader(1, 0, 1536, PrecisionF16, MetricCosine, 50000, 4096, 12288, 16384, 4194304)
	h, err := ParseHeaderBytes(buf)
	if err != nil {
		t.Fatalf("ParseHeaderBytes: unexpected error: %v", err)
	}
	if h.FormatVersion != 1 {
		t.Errorf("FormatVersion: got %d, want 1", h.FormatVersion)
	}
	if h.Dim != 1536 {
		t.Errorf("Dim: got %d, want 1536", h.Dim)
	}
	if h.Precision != PrecisionF16 {
		t.Errorf("Precision: got %d, want F16(%d)", h.Precision, PrecisionF16)
	}
	if h.DistanceMetric != MetricCosine {
		t.Errorf("DistanceMetric: got %d, want Cosine(%d)", h.DistanceMetric, MetricCosine)
	}
	if h.RecordCount != 50000 {
		t.Errorf("RecordCount: got %d, want 50000", h.RecordCount)
	}
	if h.HnswOffset != 16384 {
		t.Errorf("HnswOffset: got %d, want 16384", h.HnswOffset)
	}
	if h.HnswLen != 4194304 {
		t.Errorf("HnswLen: got %d, want 4194304", h.HnswLen)
	}
	if h.IsIvfPq() {
		t.Error("IsIvfPq: expected false for flags=0")
	}
}

func TestParseHeaderBytes_IvfPqFlag(t *testing.T) {
	buf := buildHeader(1, FlagIndexIvfPq, 768, PrecisionF32, MetricEuclidean, 1000, 0, 0, 0, 0)
	h, err := ParseHeaderBytes(buf)
	if err != nil {
		t.Fatalf("ParseHeaderBytes: %v", err)
	}
	if !h.IsIvfPq() {
		t.Error("IsIvfPq: expected true when FlagIndexIvfPq set")
	}
}

func TestParseHeaderBytes_BadMagic(t *testing.T) {
	buf := buildHeader(1, 0, 128, PrecisionF16, MetricCosine, 100, 0, 0, 0, 0)
	buf[0] = 'X' // corrupt magic
	if _, err := ParseHeaderBytes(buf); err == nil {
		t.Error("ParseHeaderBytes with bad magic: expected error, got nil")
	}
}

func TestParseHeaderBytes_UnsupportedVersion(t *testing.T) {
	buf := buildHeader(99, 0, 128, PrecisionF16, MetricCosine, 100, 0, 0, 0, 0)
	if _, err := ParseHeaderBytes(buf); err == nil {
		t.Error("ParseHeaderBytes version=99: expected error, got nil")
	}
}

func TestParseHeaderBytes_TooShort(t *testing.T) {
	if _, err := ParseHeaderBytes([]byte{1, 2, 3}); err == nil {
		t.Error("ParseHeaderBytes with short buf: expected error, got nil")
	}
}

func TestParseHeader_Reader(t *testing.T) {
	buf := buildHeader(1, 0, 256, PrecisionI8, MetricDotProduct, 200, 512, 1024, 2048, 65536)
	h, err := ParseHeader(bytes.NewReader(buf))
	if err != nil {
		t.Fatalf("ParseHeader: %v", err)
	}
	if h.Dim != 256 {
		t.Errorf("Dim: got %d, want 256", h.Dim)
	}
	if h.Precision != PrecisionI8 {
		t.Errorf("Precision: got %d, want I8(%d)", h.Precision, PrecisionI8)
	}
}

func TestParseTrailerBytes_Valid(t *testing.T) {
	buf := make([]byte, TrailerSize)
	binary.LittleEndian.PutUint64(buf[0:], 12345678)  // FooterOffset
	binary.LittleEndian.PutUint64(buf[8:], 4194304)   // FooterLen
	binary.LittleEndian.PutUint16(buf[16:], 1)         // FormatVersion
	binary.LittleEndian.PutUint16(buf[18:], 0)         // Flags
	buf[20], buf[21], buf[22], buf[23] = 'A', 'I', 'L', 'K'

	tr, err := ParseTrailerBytes(buf)
	if err != nil {
		t.Fatalf("ParseTrailerBytes: %v", err)
	}
	if tr.FooterOffset != 12345678 {
		t.Errorf("FooterOffset: got %d, want 12345678", tr.FooterOffset)
	}
	if tr.FooterLen != 4194304 {
		t.Errorf("FooterLen: got %d, want 4194304", tr.FooterLen)
	}
	if tr.FormatVersion != 1 {
		t.Errorf("FormatVersion: got %d, want 1", tr.FormatVersion)
	}
}

func TestParseTrailerBytes_BadMagic(t *testing.T) {
	buf := make([]byte, TrailerSize)
	buf[20], buf[21], buf[22], buf[23] = 'X', 'X', 'X', 'X'
	if _, err := ParseTrailerBytes(buf); err == nil {
		t.Error("ParseTrailerBytes bad magic: expected error")
	}
}

func TestParseCentroid_Valid(t *testing.T) {
	dim := uint32(3)
	buf := make([]byte, (dim+1)*4)
	// Vector: [1.0, 2.0, 3.0], Radius: 0.5
	vecs := []float32{1.0, 2.0, 3.0}
	for i, v := range vecs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	binary.LittleEndian.PutUint32(buf[dim*4:], math.Float32bits(0.5))

	vec, radius, err := ParseCentroid(buf, dim)
	if err != nil {
		t.Fatalf("ParseCentroid: %v", err)
	}
	if len(vec) != int(dim) {
		t.Fatalf("vec len: got %d, want %d", len(vec), dim)
	}
	for i, want := range vecs {
		if math.Abs(float64(vec[i]-want)) > 1e-6 {
			t.Errorf("vec[%d]: got %v, want %v", i, vec[i], want)
		}
	}
	if math.Abs(float64(radius-0.5)) > 1e-6 {
		t.Errorf("radius: got %v, want 0.5", radius)
	}
}

func TestParseCentroid_TooShort(t *testing.T) {
	if _, _, err := ParseCentroid([]byte{1, 2, 3}, 2); err == nil {
		t.Error("ParseCentroid too short: expected error")
	}
}
