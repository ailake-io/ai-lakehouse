// SPDX-License-Identifier: MIT OR Apache-2.0
//! Schema evolution request types (Phase G).
//!
//! `SchemaEvolution` is a pure metadata operation — no data files are rewritten.
//! `initial-default` in each added field tells readers what value to return for
//! old files that predate the column. Corresponds to Iceberg V2/V3 spec §4.1.1.

/// Request to add a new column to the table schema.
///
/// # Iceberg types
///
/// Common type strings: `"int"`, `"long"`, `"float"`, `"double"`,
/// `"boolean"`, `"string"`, `"date"`, `"timestamp"`, `"timestamptz"`, `"binary"`.
///
/// Complex types (`"list<int>"`, `"map<string,long>"`, `"struct<…>"`) are stored
/// verbatim; `SchemaFiller` maps them to `Utf8` when injecting defaults.
pub struct AddColumnRequest {
    pub name: String,
    /// Iceberg type string — see struct-level doc for accepted values.
    pub iceberg_type: String,
    /// `false` (nullable) is the safe default for additions so old files never error.
    pub required: bool,
    /// Value returned when reading records from files written before this column existed.
    /// `None` → readers inject `null` for those rows.
    pub initial_default: Option<serde_json::Value>,
    /// Value written to new records when no explicit value is supplied.
    /// Defaults to `initial_default` when omitted.
    pub write_default: Option<serde_json::Value>,
    /// Human-readable field documentation stored in the schema JSON.
    pub doc: Option<String>,
}

/// Request to rename an existing column (field-id stays stable).
pub struct RenameColumnRequest {
    pub old_name: String,
    pub new_name: String,
}

/// Atomic schema evolution transaction applied in a single `metadata.json` rewrite.
///
/// Operations are applied in order: renames first (so you can rename a column
/// and also add a new column with the old name in one call), then additions.
#[derive(Default)]
pub struct SchemaEvolution {
    pub renames: Vec<RenameColumnRequest>,
    pub adds: Vec<AddColumnRequest>,
    /// Additional table-level properties to merge into `metadata.json` on commit.
    /// Used by `add_vector_column` to persist `ailake.dim-<col>`, `ailake.metric-<col>`, etc.
    pub extra_properties: std::collections::HashMap<String, String>,
}

impl SchemaEvolution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_column(mut self, req: AddColumnRequest) -> Self {
        self.adds.push(req);
        self
    }

    pub fn rename_column(
        mut self,
        old_name: impl Into<String>,
        new_name: impl Into<String>,
    ) -> Self {
        self.renames.push(RenameColumnRequest {
            old_name: old_name.into(),
            new_name: new_name.into(),
        });
        self
    }

    pub fn with_properties(mut self, props: std::collections::HashMap<String, String>) -> Self {
        self.extra_properties = props;
        self
    }
}
