//! ailake-query — query planning and execution
//!
//! Integration layer. Depends on all data-plane crates.
//! Public surface: TableWriter, search(), ContextAssembler, CompactionPlanner.

pub mod compaction;
pub mod context_assembler;
pub mod pruner;
pub mod scanner;
pub mod writer;

pub use compaction::{CompactionConfig, CompactionMode, CompactionPlanner};
pub use context_assembler::{AssembledContext, ContextAssembler, ContextAssemblerConfig};
pub use scanner::{search, SearchConfig, SearchResult};
pub use writer::TableWriter;
