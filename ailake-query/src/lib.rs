//! ailake-query — query planning and execution
//!
//! Integration layer. Depends on all data-plane crates.
//! Public surface: TableWriter, search(), ContextAssembler, CompactionPlanner.

pub mod writer;
pub mod pruner;
pub mod scanner;
pub mod context_assembler;
pub mod compaction;

pub use writer::TableWriter;
pub use scanner::{search, SearchConfig, SearchResult};
pub use context_assembler::{ContextAssembler, ContextAssemblerConfig, AssembledContext};
pub use compaction::{CompactionPlanner, CompactionConfig, CompactionMode};
