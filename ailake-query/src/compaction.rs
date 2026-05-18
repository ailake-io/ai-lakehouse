// Phase 2: CompactionPlanner — merge N small files into 1 large file, rebuild HNSW.
// Phase 1: stub.

pub struct CompactionConfig {
    pub min_files_to_compact: usize,
    pub target_file_size_bytes: u64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_files_to_compact: 4,
            target_file_size_bytes: 128 * 1024 * 1024, // 128 MB
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompactionMode {
    Full,
    Partial,
}

pub struct CompactionPlanner {
    config: CompactionConfig,
}

impl CompactionPlanner {
    pub fn new(config: CompactionConfig) -> Self {
        Self { config }
    }
}
