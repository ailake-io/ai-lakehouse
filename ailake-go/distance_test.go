// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"math"
	"testing"
)

func TestCosineDistance(t *testing.T) {
	// Identical unit vectors → distance = 0
	a := []float32{1, 0, 0}
	if d := cosineDistance(a, a); math.Abs(float64(d)) > 1e-6 {
		t.Errorf("cosineDistance(a, a): got %v, want ~0", d)
	}

	// Orthogonal → distance = 1
	b := []float32{0, 1, 0}
	if d := cosineDistance(a, b); math.Abs(float64(d)-1.0) > 1e-5 {
		t.Errorf("cosineDistance(a, b orthogonal): got %v, want ~1", d)
	}

	// Opposite → distance = 2
	c := []float32{-1, 0, 0}
	if d := cosineDistance(a, c); math.Abs(float64(d)-2.0) > 1e-5 {
		t.Errorf("cosineDistance(a, -a): got %v, want ~2", d)
	}
}

func TestEuclideanDistance(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 0, 0}
	if d := euclideanDistance(a, b); math.Abs(float64(d)-1.0) > 1e-6 {
		t.Errorf("euclideanDistance: got %v, want 1.0", d)
	}

	// Pythagorean triple: (3,4,0) → dist=5
	c := []float32{3, 4, 0}
	if d := euclideanDistance(a, c); math.Abs(float64(d)-5.0) > 1e-5 {
		t.Errorf("euclideanDistance(0,3-4-0): got %v, want 5.0", d)
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	want := float32(1*4 + 2*5 + 3*6) // 32
	if got := dotProduct(a, b); math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("dotProduct: got %v, want %v", got, want)
	}

	// Self dot product of unit vector = 1
	u := []float32{1, 0, 0}
	if got := dotProduct(u, u); math.Abs(float64(got)-1.0) > 1e-6 {
		t.Errorf("dotProduct(unit, unit): got %v, want 1", got)
	}
}

func TestNormalizeL2(t *testing.T) {
	v := []float32{3, 4, 0}
	n := normalizeL2(v)
	// L2 norm of result must be 1
	var sum float32
	for _, x := range n {
		sum += x * x
	}
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("normalizeL2: L2 norm of result = %v, want ~1.0", sum)
	}
	// Direction preserved: ratio n[0]/n[1] = 3/4
	if math.Abs(float64(n[0]/n[1])-0.75) > 1e-5 {
		t.Errorf("normalizeL2: direction changed — ratio got %v, want 0.75", n[0]/n[1])
	}
}

func TestNormalizedCosineDistance(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	// For unit vectors, NormalizedCosine = 1 - dot(a,b) = 1 - 0 = 1
	d := normalizedCosineDistance(a, b)
	if math.Abs(float64(d)-1.0) > 1e-5 {
		t.Errorf("normalizedCosineDistance(orthogonal): got %v, want ~1", d)
	}

	// Same vector → 1 - 1 = 0
	d2 := normalizedCosineDistance(a, a)
	if math.Abs(float64(d2)) > 1e-6 {
		t.Errorf("normalizedCosineDistance(same): got %v, want ~0", d2)
	}
}

func TestDistanceByMetric(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}

	cases := []struct {
		metric uint8
		a, b   []float32
		wantGT float32 // distance must be > this (non-zero for orthogonal)
	}{
		{MetricCosine, a, b, 0.9},
		{MetricEuclidean, a, b, 0.9},
		{MetricDotProduct, a, b, -0.1}, // dot=0, 1-dot=1 → gt -0.1 passes
		{MetricNormalizedCosine, a, b, 0.9},
	}
	for _, c := range cases {
		d := distanceByMetric(c.metric, c.a, c.b)
		if d <= c.wantGT {
			t.Errorf("distanceByMetric(%d, a, b): got %v, want > %v", c.metric, d, c.wantGT)
		}
	}
}
