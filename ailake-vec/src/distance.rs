use ailake_core::{Centroid, VectorMetric};

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dispatch(metric: VectorMetric) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        VectorMetric::Cosine => cosine_distance,
        VectorMetric::Euclidean => euclidean_distance,
        // For DotProduct, higher = more similar, so negate for distance
        VectorMetric::DotProduct => |a, b| -dot_product(a, b),
    }
}

pub fn compute_centroid_and_radius(vectors: &[Vec<f32>], metric: VectorMetric) -> Centroid {
    if vectors.is_empty() {
        return Centroid {
            values: vec![],
            radius: 0.0,
            metric,
        };
    }
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let centroid: Vec<f32> = (0..dim)
        .map(|i| vectors.iter().map(|v| v[i]).sum::<f32>() / n)
        .collect();
    let dist_fn = dispatch(metric);
    let radius = vectors
        .iter()
        .map(|v| dist_fn(&centroid, v))
        .fold(0.0_f32, f32::max);
    Centroid {
        values: centroid,
        radius,
        metric,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!(cosine_distance(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn centroid_single() {
        let v = vec![vec![1.0, 2.0, 3.0]];
        let c = compute_centroid_and_radius(&v, VectorMetric::Cosine);
        assert_eq!(c.values, vec![1.0, 2.0, 3.0]);
        assert!(c.radius < 1e-6, "radius should be ~0 but was {}", c.radius);
    }

    #[test]
    fn centroid_two_points() {
        let vs = vec![vec![0.0, 0.0], vec![2.0, 2.0]];
        let c = compute_centroid_and_radius(&vs, VectorMetric::Euclidean);
        assert!((c.values[0] - 1.0).abs() < 1e-6);
        assert!((c.values[1] - 1.0).abs() < 1e-6);
        assert!(c.radius > 0.0);
    }
}
