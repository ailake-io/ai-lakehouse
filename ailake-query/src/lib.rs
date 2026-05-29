// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-query — query planning and execution
//!
//! Integration layer. Depends on all data-plane crates.
//! Public surface: TableWriter, search(), ContextAssembler, CompactionPlanner, CompactionExecutor.

pub mod compaction;
pub mod context_assembler;
pub mod mem_table;
pub mod pruner;
pub mod scanner;
pub mod writer;

pub use ailake_index::IvfPqConfig;
pub use compaction::{CompactionConfig, CompactionExecutor, CompactionMode, CompactionPlanner};
pub use context_assembler::{AssembledContext, Chunk, ContextAssembler, ContextAssemblerConfig};
pub use mem_table::{MemTableConfig, MemTableWriter};
pub use pruner::VectorPruner;
pub use scanner::{search, SearchConfig, SearchResult, SearchSession};
pub use writer::{MultiVectorBatch, TableWriter};
