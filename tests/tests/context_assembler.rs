//! Integration tests for ContextAssembler dedup and document ordering.

use ailake_query::{Chunk, ContextAssembler, ContextAssemblerConfig};

fn make_chunk(doc: &str, idx: u32, text: &str, dist: f32) -> Chunk {
    Chunk {
        document_id: doc.to_string(),
        chunk_index: idx,
        chunk_text: text.to_string(),
        document_title: Some(format!("Doc {doc}")),
        section_path: Some("Intro".into()),
        source_uri: None,
        distance: dist,
        embedding: None,
    }
}

#[test]
fn dedup_removes_near_identical_chunks() {
    let cfg = ContextAssemblerConfig {
        dedup_threshold: 0.01,
        ..Default::default()
    };
    let ca = ContextAssembler::new(cfg);
    let emb = vec![1.0f32, 0.0, 0.0];

    let mut c1 = make_chunk("doc-1", 0, "Identical content A.", 0.1);
    c1.embedding = Some(emb.clone());
    let mut c2 = make_chunk("doc-1", 1, "Identical content B.", 0.2);
    c2.embedding = Some(emb.clone());
    let mut c3 = make_chunk("doc-1", 2, "Different content.", 0.3);
    c3.embedding = Some(vec![0.0, 1.0, 0.0]); // orthogonal

    let ctx = ca.assemble_chunks(vec![c1, c2, c3]);
    assert_eq!(ctx.chunk_count, 2, "one duplicate should be removed");
    assert!(ctx.text.contains("Different content."));
}

#[test]
fn grouping_restores_chunk_order() {
    let ca = ContextAssembler::new(ContextAssemblerConfig::default());
    // Arrive in reverse order by distance, but chunk_index must be sorted in output
    let chunks = vec![
        make_chunk("doc-1", 2, "Chunk C.", 0.3),
        make_chunk("doc-1", 0, "Chunk A.", 0.1),
        make_chunk("doc-1", 1, "Chunk B.", 0.2),
    ];
    let ctx = ca.assemble_chunks(chunks);
    let pos_a = ctx.text.find("Chunk A.").expect("Chunk A missing");
    let pos_b = ctx.text.find("Chunk B.").expect("Chunk B missing");
    let pos_c = ctx.text.find("Chunk C.").expect("Chunk C missing");
    assert!(pos_a < pos_b && pos_b < pos_c, "chunks must appear in index order");
}
