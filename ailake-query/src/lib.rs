// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-query — query planning and execution
//!
//! Integration layer. Depends on all data-plane crates.
//! Public surface: TableWriter, search(), ContextAssembler, CompactionPlanner, CompactionExecutor.

pub mod bloom;
pub mod bm25;
pub mod compaction;
pub mod schema_filler;
pub mod delete;
pub mod dv;
pub mod context_assembler;
pub mod mem_table;
pub mod memory_decay;
pub mod migration;
pub mod pruner;
pub mod scanner;
pub mod writer;

pub use ailake_index::IvfPqConfig;
pub use bm25::{BM25Scorer, HybridConfig, HybridFusion, IdfStats};
pub use compaction::{CompactionConfig, CompactionExecutor, CompactionMode, CompactionPlanner};
pub use context_assembler::{AssembledContext, Chunk, ContextAssembler, ContextAssemblerConfig};
pub use mem_table::{MemTableConfig, MemTableWriter, WorkingMemoryBuffer, WorkingMemoryEntry};
pub use memory_decay::MemoryDecayJob;
pub use migration::{EmbedFn, MigrationJob, MigrationProgress, MigrationStrategy, ProgressFn};
pub use bloom::BloomFilter;
pub use pruner::{BloomPruner, VectorPruner};
pub use schema_filler::SchemaFiller;
pub use scanner::{
    fetch_rows, search, search_multimodal, search_text, FusionMethod, ModalQuery, ScoreFn,
    SearchConfig, SearchResult, SearchSession,
};
pub use delete::{delete_rows, PuffinWriter};
pub use writer::{MultiVectorBatch, TableWriter};
