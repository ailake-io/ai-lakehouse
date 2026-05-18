//! ailake-py — PyO3 Python bindings
//!
//! Thin layer. All logic in ailake-query and friends.
//! Exports: TableWriter, search(), assemble_context().

use pyo3::prelude::*;

#[pymodule]
fn ailake(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // TODO Phase 2: expose TableWriter, search, assemble_context
    let _ = m;
    Ok(())
}
