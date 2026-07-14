// SPDX-License-Identifier: MIT OR Apache-2.0
//! Column-level predicate for pushdown filtering during Parquet reads.
//!
//! Deliberately narrow: single-column comparison against a scalar. This is
//! the shape `ailake-parquet::ParquetVectorReader` can translate into a
//! Parquet row-group statistics skip (coarse) plus an Arrow `RowFilter`
//! (exact, evaluated only on surviving row groups) — see
//! `ailake-parquet/src/reader.rs::read_all_filtered`. Not a general
//! expression tree; combine filters at a higher layer if ever needed.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    I64(i64),
    F64(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnFilter {
    pub column: String,
    pub op: FilterOp,
    pub value: FilterValue,
}

impl ColumnFilter {
    pub fn new(column: impl Into<String>, op: FilterOp, value: FilterValue) -> Self {
        Self {
            column: column.into(),
            op,
            value,
        }
    }

    pub fn eq(column: impl Into<String>, value: FilterValue) -> Self {
        Self::new(column, FilterOp::Eq, value)
    }
}
