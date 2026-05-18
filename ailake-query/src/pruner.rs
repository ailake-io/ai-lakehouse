// Phase 2: geometric pruning — skip files whose centroid is too far from the query.
// Phase 1: stub that always returns all files (no pruning).

use ailake_catalog::DataFileEntry;
use ailake_core::VectorMetric;

pub struct VectorPruner;

impl VectorPruner {
    /// Phase 1: returns all files unchanged.
    pub fn prune(
        files: Vec<DataFileEntry>,
        _query: &[f32],
        _metric: VectorMetric,
        _threshold: f32,
    ) -> Vec<DataFileEntry> {
        files
    }
}
