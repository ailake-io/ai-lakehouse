// SPDX-License-Identifier: MIT OR Apache-2.0
//! Schema filler — inject missing columns at read time (Phase G).
//!
//! When the table schema has been evolved by adding new columns, old data files
//! do not contain those columns. `SchemaFiller::fill` detects absent columns and
//! appends them to the `RecordBatch`, filled with the field's `initial_default`
//! (or null when no default is set). This implements schema evolution without
//! rewriting data files, equivalent to Iceberg V2/V3 §4.1.1.

use std::sync::Arc;

use ailake_catalog::SchemaField;
use ailake_core::{AilakeError, AilakeResult};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array,
    StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};

pub struct SchemaFiller;

impl SchemaFiller {
    /// Inject columns present in `schema_fields` but absent from `batch`.
    ///
    /// Added columns appear after the existing columns, filled with `initial_default`
    /// (or null). Columns already in the batch are left untouched.
    ///
    /// Returns `batch` unchanged if `schema_fields` is empty or no columns are missing.
    pub fn fill(
        batch: arrow_array::RecordBatch,
        schema_fields: &[SchemaField],
    ) -> AilakeResult<arrow_array::RecordBatch> {
        if schema_fields.is_empty() {
            return Ok(batch);
        }

        let batch_schema = batch.schema();
        let existing: std::collections::HashSet<&str> = batch_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        let missing: Vec<&SchemaField> = schema_fields
            .iter()
            .filter(|sf| !existing.contains(sf.name.as_str()))
            .collect();

        if missing.is_empty() {
            return Ok(batch);
        }

        let n = batch.num_rows();
        let mut new_fields: Vec<arrow_schema::FieldRef> =
            batch.schema().fields().iter().cloned().collect();
        let mut new_cols: Vec<Arc<dyn Array>> = batch.columns().to_vec();

        for sf in missing {
            let dtype = iceberg_type_to_arrow(&sf.iceberg_type);
            let arr = make_default_array(&dtype, sf.initial_default.as_ref(), n)?;
            new_fields.push(Arc::new(Field::new(sf.name.clone(), dtype, !sf.required)));
            new_cols.push(arr);
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        arrow_array::RecordBatch::try_new(new_schema, new_cols)
            .map_err(|e| AilakeError::Arrow(e.to_string()))
    }
}

/// Map an Iceberg type string to an Arrow `DataType`.
///
/// Complex types (`list<…>`, `map<…>`, `struct<…>`) are mapped to `Utf8`
/// as a conservative fallback — the raw JSON is stored as a string.
pub fn iceberg_type_to_arrow(typ: &str) -> DataType {
    match typ.trim() {
        "boolean" => DataType::Boolean,
        "int" | "integer" => DataType::Int32,
        "long" => DataType::Int64,
        "float" => DataType::Float32,
        "double" => DataType::Float64,
        "date" => DataType::Date32,
        "time" => DataType::Time64(TimeUnit::Microsecond),
        "timestamp" => DataType::Timestamp(TimeUnit::Microsecond, None),
        "timestamptz" => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "string" | "uuid" => DataType::Utf8,
        "binary" | "fixed" => DataType::Binary,
        _ => DataType::Utf8,
    }
}

/// Build an array of `n` rows filled with `default_val` (or all-null if `None`).
fn make_default_array(
    dtype: &DataType,
    default_val: Option<&serde_json::Value>,
    n: usize,
) -> AilakeResult<ArrayRef> {
    use serde_json::Value;

    Ok(match dtype {
        DataType::Boolean => {
            let v = default_val.and_then(Value::as_bool);
            Arc::new(BooleanArray::from(vec![v; n]))
        }
        DataType::Int32 => {
            let v = default_val.and_then(Value::as_i64).map(|i| i as i32);
            Arc::new(Int32Array::from(vec![v; n]))
        }
        DataType::Int64 => {
            let v = default_val.and_then(Value::as_i64);
            Arc::new(Int64Array::from(vec![v; n]))
        }
        DataType::Float32 => {
            let v = default_val.and_then(Value::as_f64).map(|f| f as f32);
            Arc::new(Float32Array::from(vec![v; n]))
        }
        DataType::Float64 => {
            let v = default_val.and_then(Value::as_f64);
            Arc::new(Float64Array::from(vec![v; n]))
        }
        DataType::Date32 => {
            // Iceberg date default is an integer (days since epoch).
            let v = default_val.and_then(Value::as_i64).map(|d| d as i32);
            Arc::new(Date32Array::from(vec![v; n]))
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            // Iceberg timestamp default is µs since epoch (i64).
            let v = default_val.and_then(Value::as_i64);
            let arr = TimestampMicrosecondArray::from(vec![v; n]);
            Arc::new(if tz.is_some() {
                arr.with_timezone("UTC")
            } else {
                arr
            })
        }
        DataType::Utf8 => {
            let v: Option<&str> = default_val.and_then(Value::as_str);
            Arc::new(StringArray::from(vec![v; n]))
        }
        _ => {
            // Binary, complex types, unknowns — inject null Utf8.
            Arc::new(StringArray::from(vec![None::<&str>; n]))
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ailake_catalog::SchemaField;
    use arrow_array::{Array, Float32Array, Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::SchemaFiller;

    fn make_base_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn no_op_when_no_schema_fields() {
        let batch = make_base_batch();
        let filled = SchemaFiller::fill(batch.clone(), &[]).unwrap();
        assert_eq!(filled.num_columns(), batch.num_columns());
        assert_eq!(filled.num_rows(), batch.num_rows());
    }

    #[test]
    fn no_op_when_all_columns_present() {
        let batch = make_base_batch();
        let fields = vec![
            SchemaField {
                id: 1,
                name: "id".into(),
                required: true,
                iceberg_type: "int".into(),
                initial_default: None,
                write_default: None,
            },
            SchemaField {
                id: 2,
                name: "text".into(),
                required: false,
                iceberg_type: "string".into(),
                initial_default: None,
                write_default: None,
            },
        ];
        let filled = SchemaFiller::fill(batch.clone(), &fields).unwrap();
        assert_eq!(filled.num_columns(), 2);
    }

    #[test]
    fn injects_missing_column_with_null_default() {
        let batch = make_base_batch();
        let fields = vec![SchemaField {
            id: 3,
            name: "score".into(),
            required: false,
            iceberg_type: "float".into(),
            initial_default: None,
            write_default: None,
        }];
        let filled = SchemaFiller::fill(batch, &fields).unwrap();
        assert_eq!(filled.num_columns(), 3);
        assert_eq!(filled.num_rows(), 3);
        let score_col = filled
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        // All nulls because initial_default is None.
        assert!(!score_col.is_valid(0));
        assert!(!score_col.is_valid(1));
    }

    #[test]
    fn injects_missing_column_with_value_default() {
        let batch = make_base_batch();
        let fields = vec![SchemaField {
            id: 4,
            name: "score".into(),
            required: false,
            iceberg_type: "float".into(),
            initial_default: Some(serde_json::json!(0.5)),
            write_default: None,
        }];
        let filled = SchemaFiller::fill(batch, &fields).unwrap();
        let score_col = filled
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!((score_col.value(0) - 0.5).abs() < 1e-6);
        assert!((score_col.value(2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn injects_string_column_with_default() {
        let batch = make_base_batch();
        let fields = vec![SchemaField {
            id: 5,
            name: "category".into(),
            required: false,
            iceberg_type: "string".into(),
            initial_default: Some(serde_json::json!("uncategorized")),
            write_default: None,
        }];
        let filled = SchemaFiller::fill(batch, &fields).unwrap();
        let cat = filled
            .column_by_name("category")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(cat.value(0), "uncategorized");
        assert_eq!(cat.value(2), "uncategorized");
    }
}
