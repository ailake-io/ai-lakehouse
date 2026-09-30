//! Small dependency-free benchmark for the hot vector distance kernels.
//! Run with: `cargo bench -p ailake-vec --bench distance`.

use std::env;
use std::hint::black_box;
use std::time::Instant;

struct BenchResult {
    dim: usize,
    iterations: usize,
    nanos_per_op: f64,
    checksum: f32,
}

fn run_case(dim: usize, iterations: usize) -> BenchResult {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let started = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..iterations {
        checksum += black_box(ailake_vec::cosine_distance(black_box(&a), black_box(&b)));
    }
    let elapsed = started.elapsed();
    let nanos_per_op = elapsed.as_nanos() as f64 / iterations as f64;
    BenchResult {
        dim,
        iterations,
        nanos_per_op,
        checksum,
    }
}

fn main() {
    let json = env::args().any(|arg| arg == "--json");
    let results: Vec<BenchResult> = [(128, 100_000), (768, 25_000), (1_536, 12_500)]
        .into_iter()
        .map(|(dim, iterations)| run_case(dim, iterations))
        .collect();

    if json {
        print!(r#"{{"benchmark":"ailake-vec-distance","results":["#);
        for (index, result) in results.iter().enumerate() {
            if index > 0 {
                print!(",");
            }
            print!(
                r#"{{"dim":{},"iterations":{},"nanos_per_op":{:.4},"checksum":{:.6}}}"#,
                result.dim, result.iterations, result.nanos_per_op, result.checksum
            );
        }
        println!("]}}");
    } else {
        for result in results {
            println!(
                "cosine dim={:4} iterations={:7} ns/op={:10.2} checksum={:.4}",
                result.dim, result.iterations, result.nanos_per_op, result.checksum
            );
        }
    }
}
