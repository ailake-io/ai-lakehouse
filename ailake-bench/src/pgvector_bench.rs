// SPDX-License-Identifier: MIT OR Apache-2.0
//! pgvector HNSW benchmark on SIFT-1M (128D L2).
//!
//! Requires PostgreSQL with the pgvector extension (≥ 0.5.0 for HNSW support).
//! Example connection string: "host=localhost user=postgres password=postgres dbname=postgres"

use std::fmt::Write as FmtWrite;
use std::time::Instant;

use anyhow::Context;
use bytes::Bytes;
use futures::SinkExt;
use pgvector::Vector;
use tokio_postgres::NoTls;

use crate::{bench_result::BenchResult, dataset::Dataset, metrics};

const COPY_CHUNK: usize = 10_000;

pub async fn run(
    ds: &Dataset,
    pg_url: &str,
    top_k: usize,
    hnsw_m: u32,
    hnsw_ef_construction: u32,
    ef_search: u32,
) -> anyhow::Result<BenchResult> {
    let (client, conn) = tokio_postgres::connect(pg_url, NoTls)
        .await
        .context("connect to PostgreSQL — is pgvector installed?")?;
    tokio::spawn(async move {
        let _ = conn.await;
    });

    client
        .batch_execute(
            "CREATE EXTENSION IF NOT EXISTS vector; \
             DROP TABLE IF EXISTS ailake_bench_sift1m;",
        )
        .await
        .context("setup")?;
    client
        .execute(
            &format!(
                "CREATE TABLE ailake_bench_sift1m \
                 (id bigint PRIMARY KEY, embedding vector({}))",
                ds.dim
            ),
            &[],
        )
        .await
        .context("create table")?;

    // ── Write phase (text COPY) ───────────────────────────────────────────────
    eprintln!("\npgvector write phase (COPY, {} vectors) …", ds.base.len());
    let write_start = Instant::now();

    let sink = client
        .copy_in("COPY ailake_bench_sift1m (id, embedding) FROM STDIN")
        .await
        .context("copy_in")?;
    // Pin<Box<T>>: Unpin because Box<T>: Unpin — lets us call SinkExt methods directly.
    let mut sink = Box::pin(sink);

    let total = ds.base.len();
    let mut buf = String::with_capacity(COPY_CHUNK * (20 + ds.dim * 9));
    for (i, vec) in ds.base.iter().enumerate() {
        write!(buf, "{}\t[", i).unwrap();
        for (j, &f) in vec.iter().enumerate() {
            if j > 0 {
                buf.push(',');
            }
            write!(buf, "{f}").unwrap();
        }
        buf.push_str("]\n");

        if (i + 1) % COPY_CHUNK == 0 || i + 1 == total {
            sink.send(Bytes::from(std::mem::take(&mut buf)))
                .await
                .context("COPY send")?;
            buf.reserve(COPY_CHUNK * (20 + ds.dim * 9));
            eprint!("\r  {}/{total} vectors …", i + 1);
        }
    }
    sink.close().await.context("COPY close")?;
    eprintln!("\r  {total}/{total} vectors written");

    let write_elapsed = write_start.elapsed();
    let write_vec_per_sec = total as f64 / write_elapsed.as_secs_f64();
    eprintln!(
        "  write: {:.1}s  {:.0} vec/s",
        write_elapsed.as_secs_f64(),
        write_vec_per_sec
    );

    // ── Index build ───────────────────────────────────────────────────────────
    eprintln!("  Building HNSW index (m={hnsw_m}, ef_construction={hnsw_ef_construction}) …");
    let index_start = Instant::now();
    client
        .execute(
            &format!(
                "CREATE INDEX ON ailake_bench_sift1m \
                 USING hnsw (embedding vector_l2_ops) \
                 WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction})"
            ),
            &[],
        )
        .await
        .context("CREATE INDEX hnsw")?;
    let index_elapsed = index_start.elapsed();
    eprintln!("  index built in {:.1}s", index_elapsed.as_secs_f64());

    // ── Warm-up ───────────────────────────────────────────────────────────────
    let load_start = Instant::now();
    client
        .execute(&format!("SET hnsw.ef_search = {ef_search}"), &[])
        .await
        .context("SET ef_search")?;
    let warmup_v = Vector::from(ds.queries[0].clone());
    client
        .query(
            &format!(
                "SELECT id FROM ailake_bench_sift1m \
                 ORDER BY embedding <-> $1 LIMIT {top_k}"
            ),
            &[&warmup_v],
        )
        .await
        .ok();
    let load_elapsed = load_start.elapsed();

    // ── Search phase ──────────────────────────────────────────────────────────
    eprintln!("\npgvector search phase (top_k={top_k}, ef_search={ef_search}) …");

    let search_stmt = client
        .prepare(&format!(
            "SELECT id FROM ailake_bench_sift1m \
             ORDER BY embedding <-> $1 LIMIT {top_k}"
        ))
        .await
        .context("prepare search stmt")?;

    let num_queries = ds.queries.len();
    let mut latencies_us = Vec::with_capacity(num_queries);
    let mut recall_sum = 0.0f64;
    let search_wall_start = Instant::now();

    for (qi, query) in ds.queries.iter().enumerate() {
        let v = Vector::from(query.clone());
        let t0 = Instant::now();
        let rows = client
            .query(&search_stmt, &[&v])
            .await
            .context("search query")?;
        latencies_us.push(t0.elapsed().as_micros() as u64);

        let result_ids: Vec<u32> = rows.iter().map(|r| r.get::<_, i64>(0) as u32).collect();
        recall_sum += metrics::recall_at_k(&result_ids, &ds.ground_truth[qi], top_k);

        if (qi + 1) % 1_000 == 0 {
            eprint!("\r  {}/{num_queries} queries …", qi + 1);
        }
    }
    eprintln!("\r  {num_queries}/{num_queries} queries done");

    let search_wall_ns = search_wall_start.elapsed().as_nanos() as u64;
    let lat = metrics::LatencyStats::compute(&mut latencies_us, search_wall_ns);

    client
        .execute("DROP TABLE IF EXISTS ailake_bench_sift1m", &[])
        .await
        .ok();

    Ok(BenchResult {
        engine: format!("pgvector HNSW (m={hnsw_m}, ef_c={hnsw_ef_construction})"),
        write_secs: write_elapsed.as_secs_f64(),
        write_vec_per_sec,
        index_build_secs: index_elapsed.as_secs_f64(),
        load_secs: load_elapsed.as_secs_f64(),
        recall: recall_sum / num_queries as f64,
        qps: lat.qps,
        mean_ms: lat.mean_ms,
        p50_ms: lat.p50_ms,
        p95_ms: lat.p95_ms,
        p99_ms: lat.p99_ms,
    })
}
