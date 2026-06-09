// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Package ailake implements the AI-Lake file format reader.
//
// Binary layout reference: docs/specs/FILE_FORMAT.md
package ailake

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// Magic bytes at the start of every AILK section.
var ailakeMagic = [4]byte{'A', 'I', 'L', 'K'}

const (
	HeaderSize  = 64
	TrailerSize = 24

	// flags bit 0: IVF-PQ. Default (flags = 0): HNSW.
	FlagIndexIvfPq uint16 = 0x0001

	// precision values
	PrecisionF32    uint8 = 0
	PrecisionF16    uint8 = 1
	PrecisionI8     uint8 = 2
	PrecisionBinary uint8 = 3

	// distance metric values
	MetricCosine            uint8 = 0
	MetricEuclidean         uint8 = 1
	MetricDotProduct        uint8 = 2
	MetricNormalizedCosine  uint8 = 3
)

// AilakeHeader is the 64-byte header at the start of every AILK section.
type AilakeHeader struct {
	FormatVersion  uint16
	Flags          uint16 // bit 0: 0=HNSW, 1=IVF-PQ
	Dim            uint32
	Precision      uint8
	DistanceMetric uint8
	RecordCount    uint64
	CentroidOffset uint64
	CentroidLen    uint64
	HnswOffset     uint64 // offset to index blob relative to AILK section start
	HnswLen        uint64 // length of index blob
}

// IsIvfPq reports whether the index blob is IVF-PQ (vs HNSW).
func (h *AilakeHeader) IsIvfPq() bool { return h.Flags&FlagIndexIvfPq != 0 }

// AilakeTrailer is the 24-byte trailer at the end of every AILK section.
type AilakeTrailer struct {
	FooterOffset  uint64 // absolute byte offset of this AILK header in the file
	FooterLen     uint64 // total byte length of this AILK section
	FormatVersion uint16
	Flags         uint16
}

// ParseHeader parses a 64-byte AILK header from r.
func ParseHeader(r io.Reader) (*AilakeHeader, error) {
	buf := make([]byte, HeaderSize)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, fmt.Errorf("ailake: read header: %w", err)
	}
	return ParseHeaderBytes(buf)
}

// ParseHeaderBytes parses a 64-byte AILK header from a byte slice.
func ParseHeaderBytes(buf []byte) (*AilakeHeader, error) {
	if len(buf) < HeaderSize {
		return nil, errors.New("ailake: header too short")
	}
	magic := [4]byte{buf[0], buf[1], buf[2], buf[3]}
	if magic != ailakeMagic {
		return nil, fmt.Errorf("ailake: bad magic %v, expected AILK", magic)
	}
	h := &AilakeHeader{
		FormatVersion:  binary.LittleEndian.Uint16(buf[4:6]),
		Flags:          binary.LittleEndian.Uint16(buf[6:8]),
		Dim:            binary.LittleEndian.Uint32(buf[8:12]),
		Precision:      buf[12],
		DistanceMetric: buf[13],
		// buf[14:16] reserved
		RecordCount:    binary.LittleEndian.Uint64(buf[16:24]),
		CentroidOffset: binary.LittleEndian.Uint64(buf[24:32]),
		CentroidLen:    binary.LittleEndian.Uint64(buf[32:40]),
		HnswOffset:     binary.LittleEndian.Uint64(buf[40:48]),
		HnswLen:        binary.LittleEndian.Uint64(buf[48:56]),
		// buf[56:64] reserved
	}
	if h.FormatVersion != 1 {
		return nil, fmt.Errorf("ailake: unsupported format version %d", h.FormatVersion)
	}
	return h, nil
}

// ParseTrailerBytes parses a 24-byte AILK trailer from a byte slice.
func ParseTrailerBytes(buf []byte) (*AilakeTrailer, error) {
	if len(buf) < TrailerSize {
		return nil, errors.New("ailake: trailer too short")
	}
	magic := [4]byte{buf[20], buf[21], buf[22], buf[23]}
	if magic != ailakeMagic {
		return nil, fmt.Errorf("ailake: trailer bad magic %v", magic)
	}
	return &AilakeTrailer{
		FooterOffset:  binary.LittleEndian.Uint64(buf[0:8]),
		FooterLen:     binary.LittleEndian.Uint64(buf[8:16]),
		FormatVersion: binary.LittleEndian.Uint16(buf[16:18]),
		Flags:         binary.LittleEndian.Uint16(buf[18:20]),
	}, nil
}

// ParseCentroid parses the centroid blob (dim×4 bytes F32 LE + 4-byte radius F32 LE).
// Returns (centroid vector, radius).
func ParseCentroid(buf []byte, dim uint32) ([]float32, float32, error) {
	expected := int(dim)*4 + 4
	if len(buf) < expected {
		return nil, 0, fmt.Errorf("ailake: centroid blob too short: got %d, need %d", len(buf), expected)
	}
	vec := make([]float32, dim)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(buf[i*4:])
		vec[i] = math32FromBits(bits)
	}
	radiusBits := binary.LittleEndian.Uint32(buf[int(dim)*4:])
	return vec, math32FromBits(radiusBits), nil
}
