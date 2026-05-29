// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::HashMap;

use ailake_vec::cosine_distance;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub document_id: String,
    pub chunk_index: u32,
    pub chunk_text: String,
    pub document_title: Option<String>,
    pub section_path: Option<String>,
    pub source_uri: Option<String>,
    /// Distance from query (lower = more relevant)
    pub distance: f32,
    /// Optional embedding used for similarity-based dedup
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct ContextAssemblerConfig {
    /// Approximate token budget (4 chars ≈ 1 token)
    pub max_tokens: usize,
    /// Cosine distance below which two chunks are considered duplicates
    pub dedup_threshold: f32,
    pub group_by_document: bool,
    pub max_chunks_per_document: usize,
}

impl Default for ContextAssemblerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            dedup_threshold: 0.05,
            group_by_document: true,
            max_chunks_per_document: 10,
        }
    }
}

pub struct AssembledContext {
    /// XML-structured context ready for LLM input
    pub text: String,
    pub chunk_count: usize,
    pub token_estimate: usize,
}

pub struct ContextAssembler {
    config: ContextAssemblerConfig,
}

impl ContextAssembler {
    pub fn new(config: ContextAssemblerConfig) -> Self {
        Self { config }
    }

    /// Assemble chunks into structured XML context:
    /// 1. Sort by relevance (distance ascending)
    /// 2. Deduplicate similar chunks via embedding cosine distance
    /// 3. Group by document, sort each group by chunk_index
    /// 4. Apply token budget
    /// 5. Render XML ready for LLM consumption
    pub fn assemble_chunks(&self, mut chunks: Vec<Chunk>) -> AssembledContext {
        chunks.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected = self.dedup(chunks);
        let groups = self.group(selected);
        self.render(groups)
    }

    /// Assemble from plain text strings — no dedup, no XML grouping.
    /// Kept for simpler callers that don't have document metadata.
    pub fn assemble_texts(&self, chunks: &[String]) -> AssembledContext {
        let char_budget = self.config.max_tokens * 4;
        let mut text = String::new();
        let mut count = 0;
        for chunk in chunks {
            if text.len() + chunk.len() + 2 > char_budget {
                break;
            }
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(chunk);
            count += 1;
        }
        AssembledContext {
            token_estimate: text.len() / 4,
            chunk_count: count,
            text,
        }
    }

    fn dedup(&self, chunks: Vec<Chunk>) -> Vec<Chunk> {
        let mut selected: Vec<Chunk> = Vec::new();
        'next: for chunk in chunks {
            if let Some(emb) = &chunk.embedding {
                for sel in &selected {
                    if let Some(sel_emb) = &sel.embedding {
                        if cosine_distance(emb, sel_emb) < self.config.dedup_threshold {
                            continue 'next;
                        }
                    }
                }
            }
            selected.push(chunk);
        }
        selected
    }

    fn group(&self, chunks: Vec<Chunk>) -> Vec<(String, Vec<Chunk>)> {
        let mut map: HashMap<String, Vec<Chunk>> = HashMap::new();
        let mut doc_order: Vec<String> = Vec::new();
        for chunk in chunks {
            if !map.contains_key(&chunk.document_id) {
                doc_order.push(chunk.document_id.clone());
            }
            map.entry(chunk.document_id.clone())
                .or_default()
                .push(chunk);
        }
        if self.config.group_by_document {
            for group in map.values_mut() {
                group.sort_by_key(|c| c.chunk_index);
            }
        }
        doc_order
            .into_iter()
            .map(|id| (id.clone(), map.remove(&id).unwrap_or_default()))
            .collect()
    }

    fn render(&self, groups: Vec<(String, Vec<Chunk>)>) -> AssembledContext {
        let char_budget = self.config.max_tokens * 4;
        let mut xml = String::from("<context>\n");
        let mut chunk_count = 0usize;

        'outer: for (doc_id, doc_chunks) in &groups {
            let title = doc_chunks
                .first()
                .and_then(|c| c.document_title.as_deref())
                .unwrap_or("");
            let source = doc_chunks
                .first()
                .and_then(|c| c.source_uri.as_deref())
                .unwrap_or("");
            xml.push_str(&format!(
                "  <document id=\"{}\" title=\"{}\" source=\"{}\">\n",
                escape_xml(doc_id),
                escape_xml(title),
                escape_xml(source)
            ));

            for chunk in doc_chunks.iter().take(self.config.max_chunks_per_document) {
                if xml.len() >= char_budget {
                    break 'outer;
                }
                let section = chunk.section_path.as_deref().unwrap_or("");
                xml.push_str(&format!(
                    "    <chunk index=\"{}\" section=\"{}\">\n      <text>{}</text>\n    </chunk>\n",
                    chunk.chunk_index,
                    escape_xml(section),
                    escape_xml(&chunk.chunk_text)
                ));
                chunk_count += 1;
            }

            xml.push_str("  </document>\n");
        }

        xml.push_str("</context>");
        AssembledContext {
            token_estimate: xml.len() / 4,
            chunk_count,
            text: xml,
        }
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(doc: &str, idx: u32, text: &str, dist: f32) -> Chunk {
        Chunk {
            document_id: doc.to_string(),
            chunk_index: idx,
            chunk_text: text.to_string(),
            document_title: Some(format!("Doc {doc}")),
            section_path: Some("Introduction".into()),
            source_uri: Some(format!("s3://lake/{doc}.parquet")),
            distance: dist,
            embedding: None,
        }
    }

    #[test]
    fn produces_valid_xml() {
        let ca = ContextAssembler::new(ContextAssemblerConfig::default());
        let chunks = vec![
            make_chunk("doc-1", 0, "First chunk.", 0.1),
            make_chunk("doc-1", 1, "Second chunk.", 0.15),
            make_chunk("doc-2", 0, "Doc 2 chunk.", 0.2),
        ];
        let ctx = ca.assemble_chunks(chunks);
        assert!(ctx.text.starts_with("<context>"));
        assert!(ctx.text.ends_with("</context>"));
        assert!(ctx.text.contains("doc-1"));
        assert!(ctx.text.contains("doc-2"));
        assert_eq!(ctx.chunk_count, 3);
    }

    #[test]
    fn dedup_removes_near_identical_embeddings() {
        let cfg = ContextAssemblerConfig {
            dedup_threshold: 0.01,
            ..Default::default()
        };
        let ca = ContextAssembler::new(cfg);
        let emb = vec![1.0f32, 0.0, 0.0];
        let mut c1 = make_chunk("doc-1", 0, "Text A.", 0.1);
        c1.embedding = Some(emb.clone());
        let mut c2 = make_chunk("doc-1", 1, "Text B.", 0.2);
        c2.embedding = Some(emb.clone());
        let ctx = ca.assemble_chunks(vec![c1, c2]);
        assert_eq!(ctx.chunk_count, 1, "duplicate chunk should be deduplicated");
    }

    #[test]
    fn grouping_restores_chunk_order() {
        let ca = ContextAssembler::new(ContextAssemblerConfig::default());
        // Chunks arrive out-of-order (by distance), but XML should group by doc + sort by index
        let chunks = vec![
            make_chunk("doc-1", 2, "Third chunk.", 0.3),
            make_chunk("doc-1", 0, "First chunk.", 0.1),
            make_chunk("doc-1", 1, "Second chunk.", 0.2),
        ];
        let ctx = ca.assemble_chunks(chunks);
        let first_pos = ctx.text.find("First chunk.").unwrap();
        let second_pos = ctx.text.find("Second chunk.").unwrap();
        let third_pos = ctx.text.find("Third chunk.").unwrap();
        assert!(first_pos < second_pos, "chunk 0 before chunk 1");
        assert!(second_pos < third_pos, "chunk 1 before chunk 2");
    }

    #[test]
    fn token_budget_limits_output() {
        let cfg = ContextAssemblerConfig {
            max_tokens: 10, // ~40 chars
            ..Default::default()
        };
        let ca = ContextAssembler::new(cfg);
        let chunks: Vec<Chunk> = (0..20)
            .map(|i| make_chunk("doc-1", i, &"word ".repeat(20), i as f32 * 0.01))
            .collect();
        let ctx = ca.assemble_chunks(chunks);
        assert!(ctx.token_estimate <= 100, "should respect token budget");
    }

    #[test]
    fn xml_escaping_applied() {
        let ca = ContextAssembler::new(ContextAssemblerConfig::default());
        let mut chunk = make_chunk("doc-1", 0, "Text with <b>bold</b> & \"quotes\".", 0.1);
        chunk.document_id = "doc<1>".into();
        let ctx = ca.assemble_chunks(vec![chunk]);
        assert!(ctx.text.contains("&lt;b&gt;"), "< should be escaped");
        assert!(ctx.text.contains("&amp;"), "& should be escaped");
    }

    #[test]
    fn assemble_texts_joins_with_budget() {
        let ca = ContextAssembler::new(ContextAssemblerConfig::default());
        let texts = vec!["Alpha".into(), "Beta".into(), "Gamma".into()];
        let ctx = ca.assemble_texts(&texts);
        assert!(ctx.text.contains("Alpha"));
        assert_eq!(ctx.chunk_count, 3);
    }
}
