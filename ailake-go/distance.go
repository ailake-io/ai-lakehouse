// SPDX-License-Identifier: MIT OR Apache-2.0
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

func distanceByMetric(metric uint8, query, centroid []float32) float32 {
	switch metric {
	case MetricEuclidean:
		return euclideanDistance(query, centroid)
	case MetricDotProduct:
		return -dotProduct(query, centroid)
	default:
		return cosineDistance(query, centroid)
	}
}
