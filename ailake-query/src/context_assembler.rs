// Phase 2: ContextAssembler — dedup, group by document, token budget, XML output.
// Phase 1: stub that returns chunks as plain text.

pub struct ContextAssemblerConfig {
    pub max_tokens: usize,
    pub dedup_threshold: f32,
}

impl Default for ContextAssemblerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            dedup_threshold: 0.05,
        }
    }
}

pub struct AssembledContext {
    pub text: String,
    pub chunk_count: usize,
}

pub struct ContextAssembler {
    #[allow(dead_code)]
    config: ContextAssemblerConfig,
}

impl ContextAssembler {
    pub fn new(config: ContextAssemblerConfig) -> Self {
        Self { config }
    }

    /// Phase 1: join chunk texts directly.
    pub fn assemble(&self, chunks: &[String]) -> AssembledContext {
        let text = chunks.join("\n\n");
        AssembledContext {
            chunk_count: chunks.len(),
            text,
        }
    }
}
