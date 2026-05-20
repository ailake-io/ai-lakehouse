use std::cell::RefCell;
use std::collections::BinaryHeap;

use ailake_core::{RowId, VectorMetric};
use ailake_vec::{cosine_distance, dot_product, euclidean_distance};
use rand::Rng;
use rayon::prelude::*;

// ── Visited tracker (generation-based bitmap) ─────────────────────────────────

thread_local! {
    static VISITED: RefCell<VisitedTracker> = RefCell::new(VisitedTracker::default());
}

#[derive(Default)]
struct VisitedTracker {
    gen: Vec<u32>,
    current: u32,
}

impl VisitedTracker {
    #[inline]
    fn prepare(&mut self, n: usize) {
        if self.gen.len() < n {
            self.gen.resize(n, 0);
        }
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            self.gen.fill(0);
            self.current = 1;
        }
    }

    #[inline(always)]
    fn visit(&mut self, idx: usize) -> bool {
        let slot = unsafe { self.gen.get_unchecked_mut(idx) };
        if *slot == self.current { false } else { *slot = self.current; true }
    }
}

// ── Config / Builder ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub max_elements: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, max_elements: 1_000_000 }
    }
}

pub struct HnswBuilder {
    pub(crate) config: HnswConfig,
    pub(crate) metric: VectorMetric,
    pub(crate) dim: u32,
    pub(crate) vectors: Vec<(RowId, Vec<f32>)>,
}

impl HnswBuilder {
    pub fn new(dim: u32, metric: VectorMetric, config: HnswConfig) -> Self {
        Self { config, metric, dim, vectors: Vec::new() }
    }

    pub fn insert(&mut self, row_id: RowId, vector: Vec<f32>) {
        self.vectors.push((row_id, vector));
    }

    /// Build HNSW graph over all inserted vectors (Algorithm 1, Malkov & Yashunin 2018).
    pub fn build(self) -> HnswIndex {
        let n = self.vectors.len();
        let dim = self.dim as usize;

        if n == 0 {
            return HnswIndex {
                config: self.config,
                metric: self.metric,
                dim: self.dim,
                flat_vecs: vec![],
                row_ids: vec![],
                neighbors: vec![],
                node_levels: vec![],
                entry_point: None,
                max_layer: 0,
            };
        }

        // Flatten vectors into contiguous storage for cache-friendly distance ops
        let mut flat_vecs: Vec<f32> = Vec::with_capacity(n * dim);
        let mut row_ids: Vec<u64> = Vec::with_capacity(n);
        for (id, v) in &self.vectors {
            row_ids.push(id.as_u64());
            flat_vecs.extend_from_slice(v);
        }

        let m = self.config.m;
        let ef_c = self.config.ef_construction;
        let ml = 1.0_f64 / (m as f64).ln();
        let metric = self.metric;

        let mut rng = rand::thread_rng();
        let node_levels: Vec<usize> = (0..n).map(|_| random_level(&mut rng, ml)).collect();

        let mut neighbors: Vec<Vec<Vec<usize>>> = node_levels
            .iter()
            .map(|&l| vec![Vec::new(); l + 1])
            .collect();

        let mut entry_point: Option<usize> = None;
        let mut max_layer: usize = 0;

        let mut tracker = VisitedTracker::default();
        tracker.prepare(n);

        for i in 0..n {
            let l = node_levels[i];
            let q = &flat_vecs[i * dim..(i + 1) * dim];

            let ep = match entry_point {
                None => { entry_point = Some(i); max_layer = l; continue; }
                Some(ep) => ep,
            };

            let mut eps: Vec<usize> = vec![ep];
            for lc in (l + 1..=max_layer).rev() {
                tracker.prepare(i + 1);
                let w = search_layer(q, &eps, 1, lc, &flat_vecs, dim, &neighbors, &node_levels, metric, &mut tracker);
                eps = vec![w[0].1];
            }

            for lc in (0..=l.min(max_layer)).rev() {
                let m_lc = if lc == 0 { 2 * m } else { m };
                tracker.prepare(i + 1);
                let w = search_layer(q, &eps, ef_c, lc, &flat_vecs, dim, &neighbors, &node_levels, metric, &mut tracker);

                let selected: Vec<usize> = w.iter().take(m_lc).map(|&(_, nb)| nb).collect();
                neighbors[i][lc] = selected.clone();

                for nb in selected {
                    neighbors[nb][lc].push(i);
                    let m_max = if lc == 0 { 2 * m } else { m };
                    if neighbors[nb][lc].len() > m_max {
                        let nb_vec = &flat_vecs[nb * dim..(nb + 1) * dim];
                        prune_connections(&mut neighbors[nb][lc], nb_vec, &flat_vecs, dim, m_max, metric);
                    }
                }

                eps = w.iter().map(|&(_, idx)| idx).collect();
            }

            if l > max_layer {
                entry_point = Some(i);
                max_layer = l;
            }
        }

        HnswIndex { config: self.config, metric, dim: self.dim, flat_vecs, row_ids, neighbors, node_levels, entry_point, max_layer }
    }
}

// ── Index ─────────────────────────────────────────────────────────────────────

pub struct HnswIndex {
    pub(crate) config: HnswConfig,
    pub(crate) metric: VectorMetric,
    pub(crate) dim: u32,
    /// Contiguous vector storage: flat_vecs[i*dim..(i+1)*dim] = vector i.
    pub(crate) flat_vecs: Vec<f32>,
    /// Row IDs parallel to flat_vecs.
    pub(crate) row_ids: Vec<u64>,
    pub(crate) neighbors: Vec<Vec<Vec<usize>>>,
    pub(crate) node_levels: Vec<usize>,
    pub(crate) entry_point: Option<usize>,
    pub(crate) max_layer: usize,
}

impl HnswIndex {
    #[inline(always)]
    fn vec_at(&self, idx: usize) -> &[f32] {
        let d = self.dim as usize;
        &self.flat_vecs[idx * d..(idx + 1) * d]
    }

    pub fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<(RowId, f32)> {
        #[cfg(feature = "gpu")]
        if let Some(r) = crate::gpu::try_gpu_search(
            query, &self.row_ids, &self.flat_vecs, self.dim as usize, self.metric, top_k,
        ) {
            return r;
        }

        if self.neighbors.is_empty() {
            return self.brute_force(query, top_k);
        }

        let n = self.row_ids.len();
        VISITED.with(|cell| {
            let mut tracker = cell.borrow_mut();
            self.hnsw_search(query, top_k, ef, &mut tracker, n)
        })
    }

    fn hnsw_search(&self, query: &[f32], top_k: usize, ef: usize, tracker: &mut VisitedTracker, n: usize) -> Vec<(RowId, f32)> {
        let ep = match self.entry_point { Some(ep) => ep, None => return vec![] };
        let dim = self.dim as usize;
        let mut eps = vec![ep];

        for lc in (1..=self.max_layer).rev() {
            tracker.prepare(n);
            let w = search_layer(query, &eps, 1, lc, &self.flat_vecs, dim, &self.neighbors, &self.node_levels, self.metric, tracker);
            eps = vec![w[0].1];
        }

        tracker.prepare(n);
        let w = search_layer(query, &eps, ef.max(top_k), 0, &self.flat_vecs, dim, &self.neighbors, &self.node_levels, self.metric, tracker);

        w.into_iter().take(top_k).map(|(d, idx)| (RowId::new(self.row_ids[idx]), d)).collect()
    }

    fn brute_force(&self, query: &[f32], top_k: usize) -> Vec<(RowId, f32)> {
        let metric = self.metric;
        let dim = self.dim as usize;
        let n = self.row_ids.len();
        let mut results: Vec<(RowId, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v = &self.flat_vecs[i * dim..(i + 1) * dim];
                (RowId::new(self.row_ids[i]), dist(metric, query, v))
            })
            .collect();
        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    pub fn node_count(&self) -> u64 { self.row_ids.len() as u64 }
    pub fn metric(&self) -> VectorMetric { self.metric }
    pub fn dim(&self) -> u32 { self.dim }
}

// ── Heap types ────────────────────────────────────────────────────────────────

#[derive(PartialEq)]
struct MaxEntry { dist: f32, idx: usize }
impl Eq for MaxEntry {}
impl PartialOrd for MaxEntry {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for MaxEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(std::cmp::Ordering::Equal).then_with(|| o.idx.cmp(&self.idx))
    }
}

#[derive(PartialEq)]
struct MinEntry { neg_dist: f32, idx: usize }
impl Eq for MinEntry {}
impl PartialOrd for MinEntry {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for MinEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.neg_dist.partial_cmp(&o.neg_dist).unwrap_or(std::cmp::Ordering::Equal).then_with(|| o.idx.cmp(&self.idx))
    }
}

// ── Algorithm 2: SEARCH-LAYER ─────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn search_layer(
    q: &[f32],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
    flat_vecs: &[f32],
    dim: usize,
    neighbors: &[Vec<Vec<usize>>],
    node_levels: &[usize],
    metric: VectorMetric,
    tracker: &mut VisitedTracker,
) -> Vec<(f32, usize)> {
    let mut cands: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(ef * 2);
    let mut w: BinaryHeap<MaxEntry> = BinaryHeap::with_capacity(ef + 1);

    for &ep in entry_points {
        if tracker.visit(ep) {
            let d = dist(metric, q, &flat_vecs[ep * dim..(ep + 1) * dim]);
            cands.push(MinEntry { neg_dist: -d, idx: ep });
            w.push(MaxEntry { dist: d, idx: ep });
        }
    }

    while let Some(c) = cands.pop() {
        let c_dist = -c.neg_dist;
        let f_dist = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
        if c_dist > f_dist { break; }

        if c.idx >= node_levels.len() || layer > node_levels[c.idx] { continue; }

        for &nb in &neighbors[c.idx][layer] {
            if tracker.visit(nb) {
                let d = dist(metric, q, &flat_vecs[nb * dim..(nb + 1) * dim]);
                let f_dist = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
                if d < f_dist || w.len() < ef {
                    cands.push(MinEntry { neg_dist: -d, idx: nb });
                    w.push(MaxEntry { dist: d, idx: nb });
                    if w.len() > ef { w.pop(); }
                }
            }
        }
    }

    let mut result: Vec<(f32, usize)> = w.into_iter().map(|e| (e.dist, e.idx)).collect();
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ── Neighbor pruning ──────────────────────────────────────────────────────────

fn prune_connections(conn: &mut Vec<usize>, node_vec: &[f32], flat_vecs: &[f32], dim: usize, m_max: usize, metric: VectorMetric) {
    let mut sorted: Vec<(f32, usize)> = conn
        .iter()
        .map(|&nb| (dist(metric, node_vec, &flat_vecs[nb * dim..(nb + 1) * dim]), nb))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    *conn = sorted.into_iter().take(m_max).map(|(_, nb)| nb).collect();
}

// ── Distance ──────────────────────────────────────────────────────────────────

#[inline(always)]
fn dist(metric: VectorMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        VectorMetric::Cosine => cosine_distance(a, b),
        VectorMetric::Euclidean => euclidean_distance(a, b),
        VectorMetric::DotProduct => -dot_product(a, b),
    }
}

fn random_level(rng: &mut impl Rng, ml: f64) -> usize {
    let r: f64 = rng.gen::<f64>().max(f64::EPSILON);
    (-r.ln() * ml).floor() as usize
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(vecs: Vec<Vec<f32>>) -> HnswIndex {
        let mut b = HnswBuilder::new(vecs[0].len() as u32, VectorMetric::Cosine, Default::default());
        for (i, v) in vecs.into_iter().enumerate() {
            b.insert(RowId::new(i as u64), v);
        }
        b.build()
    }

    #[test]
    fn top1_is_exact_match() {
        let idx = make_index(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);
        let r = idx.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, RowId::new(0));
        assert!(r[0].1 < 1e-5);
    }

    #[test]
    fn top_k_returns_k() {
        let idx = make_index(vec![vec![1.0, 0.0], vec![0.8, 0.2], vec![0.0, 1.0], vec![-1.0, 0.0]]);
        assert_eq!(idx.search(&[1.0, 0.0], 2, 50).len(), 2);
    }

    #[test]
    fn node_count() {
        let idx = make_index(vec![vec![1.0, 0.0]; 5]);
        assert_eq!(idx.node_count(), 5);
    }

    #[test]
    fn large_index_recall() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let dim = 16;
        let vecs: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let mut gt: Vec<(f32, usize)> = vecs.iter().enumerate()
            .map(|(i, v)| (cosine_distance(&query, v), i))
            .collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let gt_ids: std::collections::HashSet<usize> = gt.iter().take(10).map(|&(_, i)| i).collect();

        let mut b = HnswBuilder::new(dim as u32, VectorMetric::Cosine, HnswConfig { m: 16, ef_construction: 200, max_elements: 1000 });
        for (i, v) in vecs.into_iter().enumerate() { b.insert(RowId::new(i as u64), v); }
        let idx = b.build();

        let results = idx.search(&query, 10, 50);
        let found: std::collections::HashSet<usize> = results.iter().map(|(id, _)| id.as_u64() as usize).collect();
        let recall = found.intersection(&gt_ids).count() as f64 / gt_ids.len() as f64;
        assert!(recall >= 0.8, "recall@10={recall:.2} < 0.8");
    }
}
