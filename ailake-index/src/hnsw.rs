// SPDX-License-Identifier: MIT OR Apache-2.0
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    RwLock,
};

use ailake_core::{RowId, VectorMetric};
use ailake_vec::{
    cosine_distance, cosine_distance_f16, dot_product, dot_product_f16, euclidean_distance,
    euclidean_distance_f16, normalize_l2, normalized_cosine_distance,
    normalized_cosine_distance_f16,
};
use half::f16;
use rand::Rng;
use rayon::prelude::*;

// ── Prefetch helper ───────────────────────────────────────────────────────────

/// Prefetch the cache line at `ptr` into L1.
///
/// On x86_64: `_mm_prefetch` with T0 hint — drops silently on inaccessible
/// addresses (Intel SDM §12.4.6), so safe to call speculatively.
/// On all other targets: no-op.
#[inline(always)]
fn prefetch_l1(ptr: *const f32) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: _mm_prefetch never faults; processor discards hint on bad addr.
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

// ── Distance trait — compile-time metric dispatch ────────────────────────────
//
// Zero-sized structs for each metric. Making `search_layer`, `build_serial`, etc.
// generic over `M: DistFn` lets the compiler inline the distance function and
// eliminates the per-call `match metric { … }` branch from the hot loop.

trait DistFn: Copy + 'static {
    const METRIC: VectorMetric;
    fn dist(a: &[f32], b: &[f32]) -> f32;
    fn dist_f16(a: &[f32], b: &[f16]) -> f32;
}

#[derive(Clone, Copy)]
struct CosineDist;
#[derive(Clone, Copy)]
struct EuclideanDist;
#[derive(Clone, Copy)]
struct DotProductDist;

impl DistFn for CosineDist {
    const METRIC: VectorMetric = VectorMetric::Cosine;
    #[inline(always)]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        cosine_distance(a, b)
    }
    #[inline(always)]
    fn dist_f16(a: &[f32], b: &[f16]) -> f32 {
        cosine_distance_f16(a, b)
    }
}

impl DistFn for EuclideanDist {
    const METRIC: VectorMetric = VectorMetric::Euclidean;
    #[inline(always)]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        euclidean_distance(a, b)
    }
    #[inline(always)]
    fn dist_f16(a: &[f32], b: &[f16]) -> f32 {
        euclidean_distance_f16(a, b)
    }
}

impl DistFn for DotProductDist {
    const METRIC: VectorMetric = VectorMetric::DotProduct;
    #[inline(always)]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        -dot_product(a, b)
    }
    #[inline(always)]
    fn dist_f16(a: &[f32], b: &[f16]) -> f32 {
        -dot_product_f16(a, b)
    }
}

/// Pre-normalized cosine: 1 - dot(a, b). No sqrt — requires unit-length vectors.
/// ~2× faster than CosineDist in the HNSW edge-traversal hot loop.
#[derive(Clone, Copy)]
struct NormalizedCosineDist;
impl DistFn for NormalizedCosineDist {
    const METRIC: VectorMetric = VectorMetric::NormalizedCosine;
    #[inline(always)]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        normalized_cosine_distance(a, b)
    }
    #[inline(always)]
    fn dist_f16(a: &[f32], b: &[f16]) -> f32 {
        normalized_cosine_distance_f16(a, b)
    }
}

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
        debug_assert!(
            idx < self.gen.len(),
            "visit idx={idx} out of bounds (len={})",
            self.gen.len()
        );
        let slot = unsafe { self.gen.get_unchecked_mut(idx) };
        if *slot == self.current {
            false
        } else {
            *slot = self.current;
            true
        }
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
        Self {
            m: 16,
            ef_construction: 150,
            max_elements: 1_000_000,
        }
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
        Self {
            config,
            metric,
            dim,
            vectors: Vec::new(),
        }
    }

    pub fn insert(&mut self, row_id: RowId, vector: Vec<f32>) {
        self.vectors.push((row_id, vector));
    }

    /// Build HNSW graph (Algorithm 1). Parallel on multi-core when n ≥ 500.
    /// Dispatches on metric once here; all inner functions are monomorphic.
    pub fn build(self) -> HnswIndex {
        let parallel = rayon::current_num_threads() > 1 && self.vectors.len() >= 500;
        match self.metric {
            VectorMetric::Cosine => {
                if parallel {
                    self.build_parallel_typed::<CosineDist>()
                } else {
                    self.build_serial_typed::<CosineDist>()
                }
            }
            VectorMetric::Euclidean => {
                if parallel {
                    self.build_parallel_typed::<EuclideanDist>()
                } else {
                    self.build_serial_typed::<EuclideanDist>()
                }
            }
            VectorMetric::DotProduct => {
                if parallel {
                    self.build_parallel_typed::<DotProductDist>()
                } else {
                    self.build_serial_typed::<DotProductDist>()
                }
            }
            VectorMetric::NormalizedCosine => {
                if parallel {
                    self.build_parallel_typed::<NormalizedCosineDist>()
                } else {
                    self.build_serial_typed::<NormalizedCosineDist>()
                }
            }
        }
    }

    fn build_serial_typed<M: DistFn>(self) -> HnswIndex {
        let n = self.vectors.len();
        let dim = self.dim as usize;

        if n == 0 {
            return HnswIndex {
                config: self.config,
                metric: M::METRIC,
                dim: self.dim,
                flat_vecs: vec![],
                flat_vecs_f16: None,
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
                None => {
                    entry_point = Some(i);
                    max_layer = l;
                    continue;
                }
                Some(ep) => ep,
            };

            let mut eps: Vec<usize> = vec![ep];
            for lc in (l + 1..=max_layer).rev() {
                tracker.prepare(i + 1);
                let w = search_layer::<M>(
                    q,
                    &eps,
                    1,
                    lc,
                    &flat_vecs,
                    None,
                    dim,
                    &neighbors,
                    &node_levels,
                    &mut tracker,
                );
                eps = vec![w[0].1];
            }

            for lc in (0..=l.min(max_layer)).rev() {
                let m_lc = if lc == 0 { 2 * m } else { m };
                tracker.prepare(i + 1);
                let w = search_layer::<M>(
                    q,
                    &eps,
                    ef_c,
                    lc,
                    &flat_vecs,
                    None,
                    dim,
                    &neighbors,
                    &node_levels,
                    &mut tracker,
                );

                let selected = select_neighbors_heuristic::<M>(&w, &flat_vecs, dim, m_lc, true);
                neighbors[i][lc] = selected.clone();

                for nb in selected {
                    neighbors[nb][lc].push(i);
                    let m_max = if lc == 0 { 2 * m } else { m };
                    if neighbors[nb][lc].len() > m_max {
                        let nb_vec = &flat_vecs[nb * dim..(nb + 1) * dim];
                        prune_connections::<M>(
                            &mut neighbors[nb][lc],
                            nb_vec,
                            &flat_vecs,
                            dim,
                            m_max,
                        );
                    }
                }

                eps = w.iter().map(|&(_, idx)| idx).collect();
            }

            if l > max_layer {
                entry_point = Some(i);
                max_layer = l;
            }
        }

        HnswIndex {
            config: self.config,
            metric: M::METRIC,
            dim: self.dim,
            flat_vecs,
            flat_vecs_f16: None,
            row_ids,
            neighbors,
            node_levels,
            entry_point,
            max_layer,
        }
    }

    /// Parallel build using per-(node,layer) RwLock — same algorithm as build_serial
    /// but with relaxed insertion ordering (like hnswlib multithreaded mode).
    fn build_parallel_typed<M: DistFn>(self) -> HnswIndex {
        let n = self.vectors.len();
        let dim = self.dim as usize;
        let m = self.config.m;
        let ef_c = self.config.ef_construction;
        let ml = 1.0_f64 / (m as f64).ln();

        let mut flat_vecs: Vec<f32> = Vec::with_capacity(n * dim);
        let mut row_ids: Vec<u64> = Vec::with_capacity(n);
        for (id, v) in &self.vectors {
            row_ids.push(id.as_u64());
            flat_vecs.extend_from_slice(v);
        }

        let mut rng = rand::thread_rng();
        let node_levels: Vec<usize> = (0..n).map(|_| random_level(&mut rng, ml)).collect();

        // One RwLock per (node, layer) — readers search concurrently; writers prune.
        let par_nb: Vec<Vec<RwLock<Vec<usize>>>> = node_levels
            .iter()
            .map(|&l| (0..=l).map(|_| RwLock::new(Vec::new())).collect())
            .collect();

        // Bootstrap: node 0 is the entry point.
        let entry_pt: RwLock<Option<usize>> = RwLock::new(Some(0));
        let max_layer_atom: AtomicUsize = AtomicUsize::new(node_levels[0]);

        // Insert nodes 1..n in parallel (rayon scoped — no 'static needed).
        (1..n).into_par_iter().for_each(|i| {
            let l = node_levels[i];
            let q = &flat_vecs[i * dim..(i + 1) * dim];

            let ep = entry_pt.read().unwrap().unwrap_or(0);
            let cur_max = max_layer_atom.load(Ordering::Relaxed);
            let mut eps = vec![ep];

            // Greedy descent above insertion layer (ef=1).
            for lc in (l + 1..=cur_max).rev() {
                let w =
                    search_layer_par::<M>(q, &eps, 1, lc, &flat_vecs, dim, &par_nb, &node_levels);
                if let Some(&(_, best)) = w.first() {
                    eps = vec![best];
                }
            }

            // Insert at layers 0..=min(l, cur_max).
            for lc in (0..=l.min(cur_max)).rev() {
                let m_lc = if lc == 0 { 2 * m } else { m };
                let w = search_layer_par::<M>(
                    q,
                    &eps,
                    ef_c,
                    lc,
                    &flat_vecs,
                    dim,
                    &par_nb,
                    &node_levels,
                );
                let selected = select_neighbors_heuristic::<M>(&w, &flat_vecs, dim, m_lc, true);

                if lc < par_nb[i].len() {
                    *par_nb[i][lc].write().unwrap() = selected.clone();
                }
                for &nb in &selected {
                    if lc < par_nb[nb].len() {
                        let mut nblist = par_nb[nb][lc].write().unwrap();
                        nblist.push(i);
                        let m_max = if lc == 0 { 2 * m } else { m };
                        if nblist.len() > m_max {
                            let nb_vec = &flat_vecs[nb * dim..(nb + 1) * dim];
                            prune_connections::<M>(&mut nblist, nb_vec, &flat_vecs, dim, m_max);
                        }
                    }
                }
                eps = w.iter().map(|&(_, idx)| idx).collect();
            }

            // Promote entry point if this node has a higher layer.
            if l > max_layer_atom.load(Ordering::Relaxed) {
                let mut ep_w = entry_pt.write().unwrap();
                let cur = max_layer_atom.load(Ordering::Relaxed);
                if l > cur {
                    max_layer_atom.store(l, Ordering::SeqCst);
                    *ep_w = Some(i);
                }
            }
        });

        let neighbors: Vec<Vec<Vec<usize>>> = par_nb
            .into_iter()
            .map(|node| {
                node.into_iter()
                    .map(|lk| lk.into_inner().unwrap())
                    .collect()
            })
            .collect();

        let final_ep = *entry_pt.read().unwrap();
        let final_max = max_layer_atom.load(Ordering::SeqCst);

        HnswIndex {
            config: self.config,
            metric: M::METRIC,
            dim: self.dim,
            flat_vecs,
            flat_vecs_f16: None,
            row_ids,
            neighbors,
            node_levels,
            entry_point: final_ep,
            max_layer: final_max,
        }
    }
}

// ── Index ─────────────────────────────────────────────────────────────────────

pub struct HnswIndex {
    pub(crate) config: HnswConfig,
    pub(crate) metric: VectorMetric,
    pub(crate) dim: u32,
    /// Contiguous F32 vector storage: flat_vecs[i*dim..(i+1)*dim] = vector i.
    pub(crate) flat_vecs: Vec<f32>,
    /// F16-quantized mirror of flat_vecs. Populated by `quantize_to_f16()`.
    /// When present, search uses F16 distances (less cache pressure, ~half bandwidth).
    /// F32 vectors are retained for brute-force fallback and callers that need them.
    pub(crate) flat_vecs_f16: Option<Vec<f16>>,
    /// Row IDs parallel to flat_vecs.
    pub(crate) row_ids: Vec<u64>,
    pub(crate) neighbors: Vec<Vec<Vec<usize>>>,
    pub(crate) node_levels: Vec<usize>,
    pub(crate) entry_point: Option<usize>,
    pub(crate) max_layer: usize,
}

impl HnswIndex {
    /// Dispatch on `self.metric` once; all inner search logic is monomorphic.
    /// For NormalizedCosine, the query is normalized here before traversal so
    /// callers do not need to pre-normalize manually.
    ///
    /// When `flat_vecs_f16` is populated (set by `quantize_to_f16`) and the
    /// metric is `NormalizedCosine`, HNSW traversal uses F16 distances for
    /// cache efficiency, then re-scores the returned candidates with exact F32
    /// so the final ranking is correct despite F16 rounding. The re-score cost
    /// is O(top_k × dim) — negligible vs traversal over O(ef × dim) candidates.
    pub fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<(RowId, f32)> {
        match self.metric {
            VectorMetric::Cosine => self.search_typed::<CosineDist>(query, top_k, ef),
            VectorMetric::Euclidean => self.search_typed::<EuclideanDist>(query, top_k, ef),
            VectorMetric::DotProduct => self.search_typed::<DotProductDist>(query, top_k, ef),
            VectorMetric::NormalizedCosine => {
                let q_norm = normalize_l2(query);
                if self.flat_vecs_f16.is_some() {
                    // F16 error (~0.001) exceeds true 1-dot distances between similar unit
                    // vectors (~0.0002). The F16 top-k may not contain the true nearest
                    // neighbour at all. Fix: fetch a larger candidate pool (ef-sized or at
                    // least top_k*10), re-score every candidate with exact F32, then truncate.
                    let pool = ef.max(top_k * 10).max(top_k);
                    let mut candidates =
                        self.search_typed::<NormalizedCosineDist>(&q_norm, pool, ef);
                    let dim = self.dim as usize;
                    for (row_id, dist) in &mut candidates {
                        let idx = row_id.as_u64() as usize;
                        let v = &self.flat_vecs[idx * dim..(idx + 1) * dim];
                        *dist = NormalizedCosineDist::dist(&q_norm, v);
                    }
                    candidates.sort_unstable_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.truncate(top_k);
                    candidates
                } else {
                    self.search_typed::<NormalizedCosineDist>(&q_norm, top_k, ef)
                }
            }
        }
    }

    fn search_typed<M: DistFn>(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<(RowId, f32)> {
        if self.neighbors.is_empty() {
            return self.brute_force_typed::<M>(query, top_k);
        }
        let n = self.row_ids.len();
        VISITED.with(|cell| {
            let mut tracker = cell.borrow_mut();
            self.hnsw_search_typed::<M>(query, top_k, ef, &mut tracker, n)
        })
    }

    fn hnsw_search_typed<M: DistFn>(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        tracker: &mut VisitedTracker,
        n: usize,
    ) -> Vec<(RowId, f32)> {
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return vec![],
        };
        let dim = self.dim as usize;
        let mut eps = vec![ep];

        let f16s = self.flat_vecs_f16.as_deref();

        for lc in (1..=self.max_layer).rev() {
            tracker.prepare(n);
            let w = search_layer::<M>(
                query,
                &eps,
                1,
                lc,
                &self.flat_vecs,
                f16s,
                dim,
                &self.neighbors,
                &self.node_levels,
                tracker,
            );
            eps = vec![w[0].1];
        }

        tracker.prepare(n);
        let w = search_layer::<M>(
            query,
            &eps,
            ef.max(top_k),
            0,
            &self.flat_vecs,
            f16s,
            dim,
            &self.neighbors,
            &self.node_levels,
            tracker,
        );

        w.into_iter()
            .take(top_k)
            .map(|(d, idx)| (RowId::new(self.row_ids[idx]), d))
            .collect()
    }

    fn brute_force_typed<M: DistFn>(&self, query: &[f32], top_k: usize) -> Vec<(RowId, f32)> {
        let dim = self.dim as usize;
        let n = self.row_ids.len();
        let mut results: Vec<(RowId, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v = &self.flat_vecs[i * dim..(i + 1) * dim];
                (RowId::new(self.row_ids[i]), M::dist(query, v))
            })
            .collect();
        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Quantize flat_vecs → flat_vecs_f16.
    ///
    /// After calling this, HNSW traversal uses F16 distances (half the memory
    /// bandwidth per distance call vs F32). F32 vectors are retained for
    /// brute-force fallback and for exact re-scoring after traversal.
    ///
    /// For `NormalizedCosine`, F16 error (~0.001) can exceed the true `1-dot`
    /// distance between very similar unit vectors (~0.0002). The F16 cache
    /// benefit is still applied for the graph traversal phase; `search()` then
    /// re-scores the final `top_k` candidates with exact F32 to restore correct
    /// ranking. Pair with `rerank_factor` in `SearchConfig` for an additional
    /// Parquet-level exact re-score when maximum precision is required.
    pub fn quantize_to_f16(&mut self) {
        let f16_vecs: Vec<f16> = self.flat_vecs.iter().map(|&x| f16::from_f32(x)).collect();
        self.flat_vecs_f16 = Some(f16_vecs);
    }

    pub fn node_count(&self) -> u64 {
        self.row_ids.len() as u64
    }
    pub fn metric(&self) -> VectorMetric {
        self.metric
    }
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Insert one new node into the live HNSW graph.
    ///
    /// Uses the same insertion algorithm as the build pass (Algorithm 1, Malkov & Yashunin 2018):
    /// random level sampling, greedy descent to the insertion layer, bidirectional
    /// connections with SELECT-NEIGHBORS-HEURISTIC, and connection pruning.
    ///
    /// **Complexity**: O(log N) amortised per call.
    ///
    /// **F16 cache**: invalidated after insert. Call `quantize_to_f16()` once after
    /// a batch of insertions to restore the fast-traversal path.
    ///
    /// **Row ID contract**: the caller must supply a unique `row_id`. The value
    /// should match the row's position in the merged Parquet file so that search
    /// results can be correlated with tabular data without an extra lookup table.
    pub fn insert_node(&mut self, row_id: RowId, vector: Vec<f32>) {
        self.flat_vecs_f16 = None; // stale after new vectors added
        match self.metric {
            VectorMetric::Cosine => self.insert_node_typed::<CosineDist>(row_id, vector),
            VectorMetric::Euclidean => self.insert_node_typed::<EuclideanDist>(row_id, vector),
            VectorMetric::DotProduct => self.insert_node_typed::<DotProductDist>(row_id, vector),
            VectorMetric::NormalizedCosine => {
                let v = normalize_l2(&vector);
                self.insert_node_typed::<NormalizedCosineDist>(row_id, v);
            }
        }
    }

    fn insert_node_typed<M: DistFn>(&mut self, row_id: RowId, vector: Vec<f32>) {
        let dim = self.dim as usize;
        let i = self.row_ids.len(); // index of the new node (before push)

        // Append to flat storage. Clone the query vector before extending so
        // the borrow on flat_vecs is released before we need &mut self.neighbors.
        let q: Vec<f32> = vector.clone();
        self.flat_vecs.extend_from_slice(&vector);
        self.row_ids.push(row_id.as_u64());

        let m = self.config.m;
        let ef_c = self.config.ef_construction;
        let ml = 1.0_f64 / (m as f64).ln();
        let l = random_level(&mut rand::thread_rng(), ml);

        self.node_levels.push(l);
        self.neighbors.push(vec![Vec::new(); l + 1]);

        let n = i + 1; // total nodes after insertion

        let ep = match self.entry_point {
            None => {
                // First node: becomes the entry point.
                self.entry_point = Some(i);
                self.max_layer = l;
                return;
            }
            Some(ep) => ep,
        };

        let mut eps: Vec<usize> = vec![ep];
        let mut tracker = VisitedTracker::default();

        // Greedy descent above the insertion layer (ef=1): find the best entry
        // point for the layer where this node will be connected.
        for lc in (l + 1..=self.max_layer).rev() {
            tracker.prepare(n);
            let w = search_layer::<M>(
                &q,
                &eps,
                1,
                lc,
                &self.flat_vecs,
                None,
                dim,
                &self.neighbors,
                &self.node_levels,
                &mut tracker,
            );
            eps = vec![w[0].1];
        }

        // Connect the new node at each layer from min(l, max_layer) down to 0.
        for lc in (0..=l.min(self.max_layer)).rev() {
            let m_lc = if lc == 0 { 2 * m } else { m };
            tracker.prepare(n);
            let w = search_layer::<M>(
                &q,
                &eps,
                ef_c,
                lc,
                &self.flat_vecs,
                None,
                dim,
                &self.neighbors,
                &self.node_levels,
                &mut tracker,
            );

            let selected = select_neighbors_heuristic::<M>(&w, &self.flat_vecs, dim, m_lc, true);
            self.neighbors[i][lc] = selected.clone();

            // Add back-edges from each selected neighbor to the new node; prune if needed.
            for &nb in &selected {
                self.neighbors[nb][lc].push(i);
                let m_max = if lc == 0 { 2 * m } else { m };
                if self.neighbors[nb][lc].len() > m_max {
                    // Clone nb's vector to release the immutable borrow on flat_vecs
                    // before passing &mut self.neighbors[nb] to prune_connections.
                    let nb_start = nb * dim;
                    let nb_vec: Vec<f32> = self.flat_vecs[nb_start..nb_start + dim].to_vec();
                    prune_connections::<M>(
                        &mut self.neighbors[nb][lc],
                        &nb_vec,
                        &self.flat_vecs,
                        dim,
                        m_max,
                    );
                }
            }

            eps = w.iter().map(|&(_, idx)| idx).collect();
        }

        // Promote to entry point if this node spans more layers.
        if l > self.max_layer {
            self.entry_point = Some(i);
            self.max_layer = l;
        }
    }
}

// ── Heap types ────────────────────────────────────────────────────────────────

#[derive(PartialEq)]
struct MaxEntry {
    dist: f32,
    idx: usize,
}
impl Eq for MaxEntry {}
impl PartialOrd for MaxEntry {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MaxEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&o.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| o.idx.cmp(&self.idx))
    }
}

#[derive(PartialEq)]
struct MinEntry {
    neg_dist: f32,
    idx: usize,
}
impl Eq for MinEntry {}
impl PartialOrd for MinEntry {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.neg_dist
            .partial_cmp(&o.neg_dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| o.idx.cmp(&self.idx))
    }
}

// ── Algorithm 2: SEARCH-LAYER ─────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn search_layer<M: DistFn>(
    q: &[f32],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
    flat_vecs: &[f32],
    flat_vecs_f16: Option<&[f16]>,
    dim: usize,
    neighbors: &[Vec<Vec<usize>>],
    node_levels: &[usize],
    tracker: &mut VisitedTracker,
) -> Vec<(f32, usize)> {
    let mut cands: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(ef * 2);
    let mut w: BinaryHeap<MaxEntry> = BinaryHeap::with_capacity(ef + 1);

    macro_rules! distance {
        ($idx:expr) => {
            if let Some(f16s) = flat_vecs_f16 {
                M::dist_f16(q, &f16s[$idx * dim..($idx + 1) * dim])
            } else {
                M::dist(q, &flat_vecs[$idx * dim..($idx + 1) * dim])
            }
        };
    }

    for &ep in entry_points {
        if tracker.visit(ep) {
            let d = distance!(ep);
            cands.push(MinEntry {
                neg_dist: -d,
                idx: ep,
            });
            w.push(MaxEntry { dist: d, idx: ep });
        }
    }

    while let Some(c) = cands.pop() {
        let c_dist = -c.neg_dist;
        let f_dist = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
        if c_dist > f_dist {
            break;
        }

        if c.idx >= node_levels.len() || layer > node_levels[c.idx] {
            continue;
        }

        let nbs = &neighbors[c.idx][layer];
        for (i, &nb) in nbs.iter().enumerate() {
            // Prefetch next neighbor's vector while computing distance for this one.
            if let Some(&next_nb) = nbs.get(i + 1) {
                if let Some(f16s) = flat_vecs_f16 {
                    let offset = next_nb * dim;
                    if offset < f16s.len() {
                        prefetch_l1(f16s[offset..].as_ptr() as *const f32);
                    }
                } else {
                    let offset = next_nb * dim;
                    if offset < flat_vecs.len() {
                        prefetch_l1(flat_vecs[offset..].as_ptr());
                    }
                }
            }
            if tracker.visit(nb) {
                let d = distance!(nb);
                let f_dist = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
                if d < f_dist || w.len() < ef {
                    cands.push(MinEntry {
                        neg_dist: -d,
                        idx: nb,
                    });
                    w.push(MaxEntry { dist: d, idx: nb });
                    if w.len() > ef {
                        w.pop();
                    }
                }
            }
        }
    }

    let mut result: Vec<(f32, usize)> = w.into_iter().map(|e| (e.dist, e.idx)).collect();
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ── Parallel search layer (used during build_parallel) ───────────────────────

#[allow(clippy::too_many_arguments)]
fn search_layer_par<M: DistFn>(
    q: &[f32],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
    flat_vecs: &[f32],
    dim: usize,
    neighbors: &[Vec<RwLock<Vec<usize>>>],
    node_levels: &[usize],
) -> Vec<(f32, usize)> {
    let n = node_levels.len();
    VISITED.with(|cell| {
        let mut tracker = cell.borrow_mut();
        tracker.prepare(n);

        let mut cands: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(ef * 2);
        let mut w: BinaryHeap<MaxEntry> = BinaryHeap::with_capacity(ef + 1);

        for &ep in entry_points {
            if tracker.visit(ep) {
                let d = M::dist(q, &flat_vecs[ep * dim..(ep + 1) * dim]);
                cands.push(MinEntry {
                    neg_dist: -d,
                    idx: ep,
                });
                w.push(MaxEntry { dist: d, idx: ep });
            }
        }

        while let Some(c) = cands.pop() {
            let c_dist = -c.neg_dist;
            let f_dist = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
            if c_dist > f_dist {
                break;
            }
            if c.idx >= node_levels.len() || layer > node_levels[c.idx] {
                continue;
            }
            if c.idx >= neighbors.len() || layer >= neighbors[c.idx].len() {
                continue;
            }
            // Clone to release read lock before distance computations.
            let nbs: Vec<usize> = neighbors[c.idx][layer].read().unwrap().clone();
            for nb in nbs {
                if tracker.visit(nb) {
                    let d = M::dist(q, &flat_vecs[nb * dim..(nb + 1) * dim]);
                    let f_dist = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
                    if d < f_dist || w.len() < ef {
                        cands.push(MinEntry {
                            neg_dist: -d,
                            idx: nb,
                        });
                        w.push(MaxEntry { dist: d, idx: nb });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        let mut result: Vec<(f32, usize)> = w.into_iter().map(|e| (e.dist, e.idx)).collect();
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        result
    })
}

// ── Neighbor selection ────────────────────────────────────────────────────────

/// SELECT-NEIGHBORS-HEURISTIC — Algorithm 4 from the HNSW paper.
///
/// `candidates`: (dist_to_q, node_idx) sorted ascending — caller must pre-sort.
/// `q_vec`: vector of the base element q.
/// `keep_pruned`: fill remaining slots with discarded elements (maintains M connections
/// even in sparse graphs; recommended `true` for production).
///
/// Selects up to `m` neighbors such that each selected neighbor `e` is closer to `q`
/// than to any already-selected neighbor `r` — ensuring angular diversity and
/// longer-range graph edges for better recall vs. simple nearest-M selection.
fn select_neighbors_heuristic<M: DistFn>(
    candidates: &[(f32, usize)],
    flat_vecs: &[f32],
    dim: usize,
    m: usize,
    keep_pruned: bool,
) -> Vec<usize> {
    let mut result: Vec<(f32, usize)> = Vec::with_capacity(m);
    let mut discarded: Vec<(f32, usize)> = Vec::new();

    'cand: for &(d_q_e, e) in candidates {
        if result.len() >= m {
            break;
        }
        let e_vec = &flat_vecs[e * dim..(e + 1) * dim];
        // Discard e if any already-selected r is closer to e than q is.
        // This enforces diversity: selected neighbors span different directions from q.
        for &(_, r) in &result {
            let r_vec = &flat_vecs[r * dim..(r + 1) * dim];
            if M::dist(e_vec, r_vec) < d_q_e {
                if keep_pruned {
                    discarded.push((d_q_e, e));
                }
                continue 'cand;
            }
        }
        result.push((d_q_e, e));
    }

    if keep_pruned {
        for (d, e) in discarded {
            if result.len() >= m {
                break;
            }
            result.push((d, e));
        }
    }

    result.into_iter().map(|(_, nb)| nb).collect()
}

fn prune_connections<M: DistFn>(
    conn: &mut Vec<usize>,
    node_vec: &[f32],
    flat_vecs: &[f32],
    dim: usize,
    m_max: usize,
) {
    if conn.len() <= m_max {
        return;
    }
    let mut candidates: Vec<(f32, usize)> = conn
        .iter()
        .map(|&nb| (M::dist(node_vec, &flat_vecs[nb * dim..(nb + 1) * dim]), nb))
        .collect();
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    *conn = select_neighbors_heuristic::<M>(&candidates, flat_vecs, dim, m_max, true);
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
        let mut b = HnswBuilder::new(
            vecs[0].len() as u32,
            VectorMetric::Cosine,
            Default::default(),
        );
        for (i, v) in vecs.into_iter().enumerate() {
            b.insert(RowId::new(i as u64), v);
        }
        b.build()
    }

    #[test]
    fn top1_is_exact_match() {
        let idx = make_index(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let r = idx.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, RowId::new(0));
        assert!(r[0].1 < 1e-5);
    }

    #[test]
    fn top_k_returns_k() {
        let idx = make_index(vec![
            vec![1.0, 0.0],
            vec![0.8, 0.2],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ]);
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
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let mut gt: Vec<(f32, usize)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (cosine_distance(&query, v), i))
            .collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let gt_ids: std::collections::HashSet<usize> =
            gt.iter().take(10).map(|&(_, i)| i).collect();

        let mut b = HnswBuilder::new(
            dim as u32,
            VectorMetric::Cosine,
            HnswConfig {
                m: 16,
                ef_construction: 200,
                max_elements: 1000,
            },
        );
        for (i, v) in vecs.into_iter().enumerate() {
            b.insert(RowId::new(i as u64), v);
        }
        let idx = b.build();

        let results = idx.search(&query, 10, 50);
        let found: std::collections::HashSet<usize> =
            results.iter().map(|(id, _)| id.as_u64() as usize).collect();
        let recall = found.intersection(&gt_ids).count() as f64 / gt_ids.len() as f64;
        assert!(recall >= 0.8, "recall@10={recall:.2} < 0.8");
    }

    /// NormalizedCosine + F16 in-memory: traversal uses F16, final ranking uses exact F32.
    /// Verifies that nearest neighbor is correct despite F16 rounding errors.
    #[test]
    fn normedcosine_f16_quantize_correct_nearest() {
        // Two nearly-identical unit vectors whose 1-dot distance is ~0.0004 —
        // smaller than F16 quantization error (~0.001). Without the F32 re-score
        // step the nearest-neighbour would be wrong.
        let dim = 32usize;
        let mut v0: Vec<f32> = vec![0.0; dim];
        let mut v1: Vec<f32> = vec![0.0; dim];
        let mut v2: Vec<f32> = vec![0.0; dim];
        // v0 = unit vec along axis 0
        v0[0] = 1.0;
        // v1 = nearly same as v0 (1-dot ≈ 1e-4)
        v1[0] = (1.0f32 - 1e-4).sqrt();
        v1[1] = 1e-2_f32.sqrt();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v1 {
            *x /= norm1;
        }
        // v2 = moderately different (1-dot ≈ 0.1)
        v2[0] = 0.9f32.sqrt();
        v2[1] = 0.1f32.sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v2 {
            *x /= norm2;
        }

        let mut b = HnswBuilder::new(
            dim as u32,
            VectorMetric::NormalizedCosine,
            Default::default(),
        );
        b.insert(RowId::new(0), v0.clone());
        b.insert(RowId::new(1), v1.clone());
        b.insert(RowId::new(2), v2.clone());
        let mut idx = b.build();

        // Enable F16 in-memory quantization (now allowed for NormalizedCosine).
        idx.quantize_to_f16();
        assert!(
            idx.flat_vecs_f16.is_some(),
            "F16 should be populated for NormalizedCosine"
        );

        // Query is v0 itself — nearest must be row 0, then row 1.
        let results = idx.search(&v0, 2, 50);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].0,
            RowId::new(0),
            "nearest to v0 must be v0 (row 0)"
        );
        assert_eq!(
            results[1].0,
            RowId::new(1),
            "second nearest to v0 must be v1 (row 1)"
        );
        // Distances must be sorted ascending.
        assert!(results[0].1 <= results[1].1);
    }

    /// insert_node adds a new vector to an existing graph and the graph remains searchable.
    #[test]
    fn insert_node_extends_existing_graph() {
        let mut b = HnswBuilder::new(4, VectorMetric::Cosine, Default::default());
        b.insert(RowId::new(0), vec![1.0, 0.0, 0.0, 0.0]);
        b.insert(RowId::new(1), vec![0.0, 1.0, 0.0, 0.0]);
        b.insert(RowId::new(2), vec![0.0, 0.0, 1.0, 0.0]);
        let mut idx = b.build();

        assert_eq!(idx.node_count(), 3);

        // Insert a 4th node that is closest to the 3rd (row 2).
        idx.insert_node(RowId::new(3), vec![0.0, 0.0, 0.9, 0.1]);
        assert_eq!(idx.node_count(), 4);

        // Search for the newly inserted vector; it should be nearest to itself.
        let results = idx.search(&[0.0, 0.0, 0.9, 0.1], 1, 50);
        assert_eq!(
            results[0].0,
            RowId::new(3),
            "nearest to inserted vector must be itself"
        );

        // The original top-1 result for [1,0,0,0] must still be row 0.
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, RowId::new(0));
    }

    /// insert_node works for NormalizedCosine (vector is pre-normalised internally).
    #[test]
    fn insert_node_normalized_cosine() {
        let mut b = HnswBuilder::new(4, VectorMetric::NormalizedCosine, Default::default());
        b.insert(RowId::new(0), vec![1.0, 0.0, 0.0, 0.0]);
        b.insert(RowId::new(1), vec![0.0, 1.0, 0.0, 0.0]);
        let mut idx = b.build();

        // Un-normalised input: insert_node should normalise internally.
        idx.insert_node(RowId::new(2), vec![0.0, 0.0, 3.0, 0.0]);
        assert_eq!(idx.node_count(), 3);

        // After normalisation, [0,0,3,0] → [0,0,1,0], nearest to [0,0,1,0] is itself.
        let results = idx.search(&[0.0, 0.0, 1.0, 0.0], 1, 50);
        assert_eq!(results[0].0, RowId::new(2));
    }

    /// insert_node into a single-node graph (entry point with no neighbors yet).
    #[test]
    fn insert_node_into_single_node_graph() {
        let mut b = HnswBuilder::new(4, VectorMetric::Euclidean, Default::default());
        b.insert(RowId::new(0), vec![0.0, 0.0, 0.0, 0.0]);
        let mut idx = b.build();

        idx.insert_node(RowId::new(1), vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert_node(RowId::new(2), vec![0.0, 1.0, 0.0, 0.0]);

        assert_eq!(idx.node_count(), 3);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, RowId::new(1));
    }

    #[cfg(miri)]
    mod miri_tests {
        use super::*;

        /// Exercita o visited tracker (get_unchecked_mut) com 100 nós sob Miri.
        /// Miri detecta OOB writes mesmo com get_unchecked_mut se o idx for inválido.
        #[test]
        fn miri_hnsw_search_visited_tracker_bounds() {
            let dim = 4u32;
            let n = 100;
            let mut b = HnswBuilder::new(
                dim,
                VectorMetric::Cosine,
                HnswConfig {
                    m: 8,
                    ef_construction: 50,
                    max_elements: 200,
                },
            );
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            for i in 0..n {
                let v: Vec<f32> = (0..dim as usize).map(|_| rng.gen::<f32>()).collect();
                b.insert(RowId::new(i as u64), v);
            }
            let idx = b.build();

            let query: Vec<f32> = (0..dim as usize).map(|_| rng.gen::<f32>()).collect();
            let results = idx.search(&query, 10, 100);
            assert!(results.len() <= 10);
        }

        /// Serialização/deserialização sem mmap (Miri não suporta mmap).
        #[test]
        fn miri_hnsw_serialize_roundtrip() {
            use crate::serialize::HnswSerializer;
            let mut b = HnswBuilder::new(4, VectorMetric::Cosine, Default::default());
            b.insert(RowId::new(0), vec![1.0, 0.0, 0.0, 0.0]);
            b.insert(RowId::new(1), vec![0.0, 1.0, 0.0, 0.0]);
            let idx = b.build();
            let bytes = HnswSerializer::to_bytes(&idx).unwrap();
            let loaded = HnswSerializer::from_bytes(&bytes).unwrap();
            assert_eq!(loaded.node_count(), 2);
        }
    }

    /// quantize_to_f16 works for all metrics (including NormalizedCosine).
    #[test]
    fn quantize_to_f16_populates_for_all_metrics() {
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
            VectorMetric::DotProduct,
            VectorMetric::NormalizedCosine,
        ] {
            let mut b = HnswBuilder::new(4, metric, Default::default());
            b.insert(RowId::new(0), vec![1.0, 0.0, 0.0, 0.0]);
            b.insert(RowId::new(1), vec![0.0, 1.0, 0.0, 0.0]);
            let mut idx = b.build();
            idx.quantize_to_f16();
            assert!(
                idx.flat_vecs_f16.is_some(),
                "flat_vecs_f16 should be Some for metric {metric:?}"
            );
        }
    }
}
