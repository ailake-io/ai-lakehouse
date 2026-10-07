//! Dependency-free benchmark for the hot vector distance kernels.
//! Run with: `cargo bench -p ailake-vec --bench distance`.

use std::env;
use std::hint::black_box;
use std::time::Instant;

const SAMPLES: usize = 7;
type DistanceKernel = fn(&[f32], &[f32]) -> f32;

struct BenchResult {
    kernel: &'static str,
    dim: usize,
    iterations: usize,
    median_nanos_per_op: f64,
    min_nanos_per_op: f64,
    max_nanos_per_op: f64,
    checksum: f32,
}

fn run_case(
    kernel: &'static str,
    distance: fn(&[f32], &[f32]) -> f32,
    dim: usize,
    iterations: usize,
) -> BenchResult {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut checksum = 0.0f32;
    // Warm up dispatch and caches before collecting timed samples.
    for _ in 0..iterations.min(2_000) {
        checksum += black_box(distance(black_box(&a), black_box(&b)));
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        for _ in 0..iterations {
            checksum += black_box(distance(black_box(&a), black_box(&b)));
        }
        samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    BenchResult {
        kernel,
        dim,
        iterations,
        median_nanos_per_op: samples[SAMPLES / 2],
        min_nanos_per_op: samples[0],
        max_nanos_per_op: samples[SAMPLES - 1],
        checksum,
    }
}

fn main() {
    let json = env::args().any(|arg| arg == "--json");
    let cases: [(&str, DistanceKernel); 3] = [
        ("cosine", ailake_vec::cosine_distance),
        ("euclidean", ailake_vec::euclidean_distance),
        ("dot_product", ailake_vec::dot_product),
    ];
    let mut results = Vec::new();
    for (kernel, distance) in cases {
        for (dim, iterations) in [(128, 100_000), (768, 25_000), (1_536, 12_500)] {
            results.push(run_case(kernel, distance, dim, iterations));
        }
    }

    if json {
        print!(r#"{{"benchmark":"ailake-vec-distance","results":["#);
        for (index, result) in results.iter().enumerate() {
            if index > 0 {
                print!(",");
            }
            print!(
                r#"{{"kernel":"{}","dim":{},"iterations":{},"samples":{},"nanos_per_op":{:.4},"min_nanos_per_op":{:.4},"max_nanos_per_op":{:.4},"checksum":{:.6}}}"#,
                result.kernel,
                result.dim,
                result.iterations,
                SAMPLES,
                result.median_nanos_per_op,
                result.min_nanos_per_op,
                result.max_nanos_per_op,
                result.checksum
            );
        }
        println!("]}}");
    } else {
        for result in results {
            println!(
                "{} dim={:4} median={:10.2} ns/op range=[{:.2}, {:.2}] samples={} checksum={:.4}",
                result.kernel,
                result.dim,
                result.median_nanos_per_op,
                result.min_nanos_per_op,
                result.max_nanos_per_op,
                SAMPLES,
                result.checksum
            );
        }
    }
}
