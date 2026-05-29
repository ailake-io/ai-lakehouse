// SPDX-License-Identifier: MIT OR Apache-2.0
// Minimal bincode v1 (little-endian, fixed-int) decoder.
//
// Spec: https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md
// Rules used by AI-Lake:
//   - usize serialized as u64 (8 bytes LE)
//   - Vec<T>: u64 length prefix + T * length
//   - Option<T>: 0x00 (None) or 0x01 + T (Some)
//   - Primitives: native width, LE
package ailake

import (
	"encoding/binary"
	"errors"
	"math"
)

type bincodeReader struct {
	buf []byte
	pos int
}

func newBincodeReader(b []byte) *bincodeReader { return &bincodeReader{buf: b} }

func (r *bincodeReader) remaining() int { return len(r.buf) - r.pos }

func (r *bincodeReader) readN(n int) ([]byte, error) {
	if r.remaining() < n {
		return nil, errors.New("bincode: unexpected EOF")
	}
	out := r.buf[r.pos : r.pos+n]
	r.pos += n
	return out, nil
}

func (r *bincodeReader) readU8() (uint8, error) {
	b, err := r.readN(1)
	return b[0], err
}

func (r *bincodeReader) readU32() (uint32, error) {
	b, err := r.readN(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (r *bincodeReader) readU64() (uint64, error) {
	b, err := r.readN(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

// readUsize reads a Rust usize (always serialized as u64 in bincode v1).
func (r *bincodeReader) readUsize() (uint64, error) { return r.readU64() }

func (r *bincodeReader) readF32() (float32, error) {
	bits, err := r.readU32()
	return math32FromBits(bits), err
}

func (r *bincodeReader) readF32Slice() ([]float32, error) {
	n, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	for i := range out {
		if out[i], err = r.readF32(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (r *bincodeReader) readF32Slice2D() ([][]float32, error) {
	n, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, n)
	for i := range out {
		if out[i], err = r.readF32Slice(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (r *bincodeReader) readU64Slice() ([]uint64, error) {
	n, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	out := make([]uint64, n)
	for i := range out {
		if out[i], err = r.readU64(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (r *bincodeReader) readU64Slice2D() ([][]uint64, error) {
	n, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	out := make([][]uint64, n)
	for i := range out {
		if out[i], err = r.readU64Slice(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (r *bincodeReader) readU64Slice3D() ([][]uint64, error) {
	// Vec<Vec<Vec<usize>>> — innermost usize decoded as u64
	outer, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	// Flatten layer × node × neighbour into a ragged slice-of-slices
	// where each inner element is one node's neighbour list at a given layer.
	// We represent as [layer0_node0, layer0_node1, ..., layer1_node0, ...]
	// with a separate shape index. For simplicity we collapse to flat pairs.
	// Caller accesses neighbors[layerIdx][nodeIdx] = []uint64.
	type layerSlice = [][]uint64
	result := make([]layerSlice, outer)
	for i := range result {
		result[i], err = r.readU64Slice2D()
		if err != nil {
			return nil, err
		}
	}
	// Re-pack for the HnswSnapshot layout: neighbors[node] = [][]uint64 per layer.
	// We swap axes: input is [layer][node][neighbor] → output [node][layer][neighbor].
	// But to avoid complexity we keep the raw 3D structure as a flat encoding.
	// The actual field in HnswSnapshot is neighbors: Vec<Vec<Vec<usize>>> which is
	// indexed as neighbors[node_idx][layer_idx] = neighbor_indices.
	// So outer dimension is node count.
	_ = result
	return nil, errors.New("use readNeighbors instead")
}

// readNeighbors reads Vec<Vec<Vec<usize>>> where [node][layer] = []usize neighbors.
func (r *bincodeReader) readNeighbors() ([][][]uint64, error) {
	nodeCount, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	out := make([][][]uint64, nodeCount)
	for i := range out {
		layerCount, err := r.readUsize()
		if err != nil {
			return nil, err
		}
		out[i] = make([][]uint64, layerCount)
		for j := range out[i] {
			if out[i][j], err = r.readU64Slice(); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func (r *bincodeReader) readOptionU64() (uint64, bool, error) {
	tag, err := r.readU8()
	if err != nil {
		return 0, false, err
	}
	if tag == 0 {
		return 0, false, nil
	}
	v, err := r.readU64()
	return v, true, err
}

func (r *bincodeReader) readU8Slice2D() ([][]byte, error) {
	n, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	out := make([][]byte, n)
	for i := range out {
		inner, err := r.readUsize()
		if err != nil {
			return nil, err
		}
		out[i], err = r.readN(int(inner))
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// math32FromBits converts IEEE 754 bits to float32.
func math32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}
