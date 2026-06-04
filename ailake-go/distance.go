// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import "math"

func cosineDistance(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if sim > 1.0 {
		sim = 1.0
	}
	return float32(1.0 - sim)
}

func euclideanDistance(a, b []float32) float32 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return float32(math.Sqrt(sum))
}

func dotProduct(a, b []float32) float32 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}

func normalizedCosineDistance(a, b []float32) float32 {
	return 1.0 - dotProduct(a, b)
}

func normalizeL2(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum < 1e-12 {
		return v
	}
	inv := float32(1.0 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func distanceByMetric(metric uint8, query, centroid []float32) float32 {
	switch metric {
	case MetricEuclidean:
		return euclideanDistance(query, centroid)
	case MetricDotProduct:
		return -dotProduct(query, centroid)
	case MetricNormalizedCosine:
		return normalizedCosineDistance(query, centroid)
	default:
		return cosineDistance(query, centroid)
	}
}
