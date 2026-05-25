// Minimal Avro Object Container File writer that preserves custom schema
// properties (like "field-id") needed by PyIceberg.
//
// apache-avro 0.16 strips unknown schema properties when serializing back to
// JSON, breaking PyIceberg's avro_schema_to_iceberg conversion. This writer
// embeds the schema JSON verbatim, then uses zigzag-encoded binary for records.

// ---------------------------------------------------------------------------
// Encoding primitives
// ---------------------------------------------------------------------------

pub fn encode_long(n: i64, buf: &mut Vec<u8>) {
    let mut v = ((n << 1) ^ (n >> 63)) as u64;
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}

pub fn encode_int(n: i32, buf: &mut Vec<u8>) {
    encode_long(n as i64, buf);
}

pub fn encode_string(s: &str, buf: &mut Vec<u8>) {
    encode_long(s.len() as i64, buf);
    buf.extend_from_slice(s.as_bytes());
}

pub fn encode_bytes_field(b: &[u8], buf: &mut Vec<u8>) {
    encode_long(b.len() as i64, buf);
    buf.extend_from_slice(b);
}

/// Encode a union as `null` (index 0).
pub fn encode_union_null(buf: &mut Vec<u8>) {
    encode_long(0, buf);
}

/// Encode a union as the variant at `index` with a long payload.
pub fn encode_union_long(index: i64, val: i64, buf: &mut Vec<u8>) {
    encode_long(index, buf);
    encode_long(val, buf);
}

/// Encode a union as the variant at `index` with a bytes payload.
pub fn encode_union_bytes(index: i64, val: &[u8], buf: &mut Vec<u8>) {
    encode_long(index, buf);
    encode_bytes_field(val, buf);
}

/// Encode an empty array.
pub fn encode_empty_array(buf: &mut Vec<u8>) {
    encode_long(0, buf);
}

// ---------------------------------------------------------------------------
// Avro Object Container File
// ---------------------------------------------------------------------------

const AVRO_MAGIC: &[u8] = &[0x4F, 0x62, 0x6A, 0x01]; // "Obj\x01"
const SYNC_MARKER: &[u8] = &[
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
];

/// Write an Avro Object Container File with the given schema JSON (verbatim)
/// and pre-encoded binary records.
///
/// `schema_json` is embedded in the `avro.schema` file metadata entry exactly
/// as given, so properties like `"field-id"` survive the round-trip.
///
/// `extra_meta` allows callers to inject additional Avro file metadata entries
/// (e.g. Iceberg manifest fields: schema, partition-spec, format-version).
pub fn write_avro_container(
    schema_json: &str,
    extra_meta: &[(&str, &[u8])],
    records: &[Vec<u8>],
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(AVRO_MAGIC);

    // File metadata (Avro map): 2 base entries + extra_meta in one block
    encode_long((2 + extra_meta.len()) as i64, &mut buf);

    // Entry 1: avro.schema → schema_json bytes
    encode_string("avro.schema", &mut buf);
    encode_bytes_field(schema_json.as_bytes(), &mut buf);

    // Entry 2: avro.codec → "null"
    encode_string("avro.codec", &mut buf);
    encode_bytes_field(b"null", &mut buf);

    // Extra entries (e.g. Iceberg manifest metadata)
    for (key, val) in extra_meta {
        encode_string(key, &mut buf);
        encode_bytes_field(val, &mut buf);
    }

    // End of map
    encode_long(0, &mut buf);

    // Sync marker
    buf.extend_from_slice(SYNC_MARKER);

    // Data block (one block for all records).
    // The file ends after the sync marker — apache-avro's Reader detects EOF when
    // trying to read the next block count, which it treats as clean end-of-stream.
    // Writing an explicit count=0 terminator causes the Reader to then try to read
    // a byte_count after the zero, hitting EOF and returning an error instead.
    if !records.is_empty() {
        let block: Vec<u8> = records.iter().flat_map(|r| r.iter().copied()).collect();
        encode_long(records.len() as i64, &mut buf); // object count
        encode_long(block.len() as i64, &mut buf); // byte count
        buf.extend_from_slice(&block);
        buf.extend_from_slice(SYNC_MARKER);
    }

    buf
}
