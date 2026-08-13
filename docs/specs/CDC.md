# CDC.md — Change Data Capture for AI-Lake

## Goal

AI-Lake tables already store every commit as an immutable Iceberg snapshot. CDC exposes those snapshots as a change stream without requiring a table-level flag: any table with multiple snapshots and reachable data files can produce CDC output.

The reader compares two snapshots and emits rows annotated with a standard change envelope:

- `_change_type` — `insert`, `delete`, `update_before`, `update_after`
- `_snapshot_id` — snapshot that produced the change
- `_sequence_number` — Iceberg sequence number of that snapshot
- `_commit_timestamp` — snapshot commit time (milliseconds since epoch)

---

## What can trigger CDC output

CDC is a **read-only** operation. It never rewrites data files and adds no extra metadata to the table.

| Write operation | CDC emission |
|---|---|
| `INSERT` / `write_batch` (new data file) | `insert` for every surviving row in the new file |
| `DELETE WHERE` (equality delete file) | `delete` for each predicate value, using the equality-delete Avro file |
| `DELETE ROWS` / V3 deletion vectors | `delete` for rows newly masked by a DV, or for rows in files that exist only in the start snapshot |
| `Compaction` (file replacement) | `delete` for rows in replaced files, `insert` for rows in the merged file |
| Same-PK `delete` + `insert` within one snapshot | `update_before` + `update_after` when `coalesce_updates=True` |

---

## API

### Rust

```rust
use ailake_query::{read_changes, ChangeReaderConfig, ChangeType};
use ailake_catalog::{HadoopCatalog, TableIdent};
use ailake_store::LocalStore;
use std::sync::Arc;

let store: Arc<dyn Store> = Arc::new(LocalStore::new("/warehouse"));
let catalog: Arc<dyn CatalogProvider> = Arc::new(HadoopCatalog::new(store.clone(), ""));

let batch = read_changes(
    catalog,
    store,
    &TableIdent::new("default", "docs"),
    ChangeReaderConfig {
        start_snapshot_id: Some(1234567890),
        end_snapshot_id: Some(1234567891),
        pk_columns: vec!["doc_id".into()],
        coalesce_updates: true,
    },
).await?;
```

### Python

```python
import ailake

tbl = ailake.read_changes(
    "/warehouse/default/docs",
    start_snapshot=snap1,
    end_snapshot=snap2,
    pk_columns=["doc_id"],
    coalesce_updates=True,
)
# Returns a pyarrow.Table with the CDC envelope columns.
```

### CLI

```bash
ailake read-changes default.docs \
  --start-snapshot 1234567890 \
  --end-snapshot   1234567891 \
  --pk-column doc_id \
  --coalesce-updates
```

Output formats: `json` (default), `text`, `parquet`, `arrow`.

---

## Semantics

### Snapshot resolution

- `start_snapshot_id`: inclusive lower bound. If omitted, the parent of `end_snapshot_id` is used.
- `end_snapshot_id`: inclusive upper bound. If omitted, the current snapshot is used.

If `end_snapshot_id` has no parent and no explicit start is given, the call fails with a clear error.

### Insert detection

A data file that appears only in the end snapshot contributes all its surviving (non-DV-masked) rows as `insert`.

### Delete detection

A row is emitted as `delete` when:

1. It lives in a file that exists only in the start snapshot (file removed/replaced).
2. Its row position is newly masked by a V3 deletion vector in the end snapshot.
3. It matches a newly committed equality delete predicate.

### Update coalescing

When `coalesce_updates=True` and `pk_columns` is non-empty, the reader looks for a `DELETE` + `INSERT` pair with the same primary key within the same end snapshot. If found, the pair is emitted as `UPDATE_BEFORE` + `UPDATE_AFTER`.

Equality-delete predicates are resolved against the raw data files present in both snapshots, so `DELETE` rows carry the full pre-image. This makes coalesced `UPDATE_BEFORE` records complete.

---

## Requirements and constraints

1. **Snapshots must be reachable**: the Iceberg `metadata.json` must still list both snapshots, and the data/delete/manifest files they reference must still exist in the store.
2. **No table flag needed**: CDC works on any AI-Lake table with snapshot history.
3. **Primary-key columns for updates**: `coalesce_updates` requires `pk_columns`. Without PKs, updates are emitted as separate `DELETE` + `INSERT` rows.
4. **Equality deletes**: fully supported by reading the committed Avro delete file. The old write-path `inline_values` hint is still used as a fallback for in-memory catalog entries.

---

## Future work

- Pre-image enrichment: for equality-delete `DELETE` rows, fetch the full old row from the start snapshot's data files so `UPDATE_BEFORE` is complete.
- Range/positional deletes: currently only equality deletes and deletion vectors are handled.
- Streaming CDC: expose a cursor-based API that advances from one snapshot to the next without re-reading the full range.
