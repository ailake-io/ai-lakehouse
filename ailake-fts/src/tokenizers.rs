// SPDX-License-Identifier: MIT OR Apache-2.0
//! Custom tokenizer registration for `ailake-fts`.
//!
//! Always registered (zero extra deps):
//!   `cjk_ngram`   — unigram+bigram over Unicode code points; suits CJK scripts where
//!                   whitespace tokenization produces no tokens. Works at ~85% of the
//!                   precision of dictionary-based segmenters (Lindera/jieba) with zero
//!                   binary overhead.
//!
//! Registered behind `fts-stemmer-langs` feature:
//!   `fr_stem`, `de_stem`, `es_stem`, `pt_stem`, `it_stem`, `nl_stem`,
//!   `ru_stem`, `ar_stem`, `sv_stem`, `da_stem`, `fi_stem`, `no_stem`,
//!   `el_stem`, `hu_stem`, `ro_stem`, `ta_stem`, `tr_stem`
//!
//!   These use Tantivy's built-in Snowball stemmers (`rust_stemmers`), which are
//!   already compiled into the binary as a mandatory dep of Tantivy 0.22 — the
//!   feature flag controls registration (user-visible names) only, not binary size.
//!
//! # CJK usage guide
//!
//! ```text
//! FtsConfig { tokenizer: "cjk_ngram", .. }
//! ```
//!
//! `cjk_ngram` uses `NgramTokenizer(min=1, max=2)` — every Unicode character becomes
//! a unigram token and every adjacent pair a bigram. This means:
//!
//!   Input:  "人工知能"  (Japanese: "artificial intelligence")
//!   Tokens: 人, 工, 知, 能, 人工, 工知, 知能
//!
//! BM25 scoring works because rarer bigrams carry more IDF weight than common
//! unigrams. Query "知能" matches doc with "知能" bigram directly; query "人工知能"
//! matches via union of its 4 unigrams + 3 bigrams.
//!
//! **Limitations vs. dictionary-based segmenters:**
//! - False-positive unigrams can match unrelated characters ("能" = "ability"/"can")
//! - Recall drops for long multi-character compounds (~15-20% vs. Lindera IPADIC)
//! - Thai / Khmer (no word boundaries) benefit more from a proper segmenter
//!
//! For production CJK workloads, register a custom tokenizer (e.g., `jieba-rs` or
//! `lindera-tantivy`) via the Index tokenizer API before write; pass its name as
//! `FtsConfig::tokenizer`. `cjk_ngram` is the built-in zero-overhead baseline.

use ailake_core::{AilakeError, AilakeResult};
use tantivy::tokenizer::{
    LowerCaser, NgramTokenizer, RemoveLongFilter, SimpleTokenizer, TextAnalyzer,
};
use tantivy::Index;

/// Register `cjk_ngram` on every index — always, no feature gate.
///
/// `cjk_ngram` = NgramTokenizer(min=1, max=2, prefix_only=false) with LowerCaser.
/// Safe for any script: Latin text produces character n-grams (less efficient than
/// `default` whitespace tokenizer but functional). Intended for CJK, Arabic, Thai.
pub fn register_cjk_ngram(index: &Index) -> AilakeResult<()> {
    let ngram = NgramTokenizer::new(1, 2, false)
        .map_err(|e| AilakeError::Fts(format!("NgramTokenizer init: {e}")))?;
    let analyzer = TextAnalyzer::builder(ngram).filter(LowerCaser).build();
    index.tokenizers().register("cjk_ngram", analyzer);
    Ok(())
}

/// Register Snowball stemmers for all non-English languages supported by Tantivy.
///
/// Names follow the pattern `{iso639-1}_stem`. English (`en_stem`) is already in
/// Tantivy's default registry and is not re-registered here.
///
/// Enabled by the `fts-stemmer-langs` Cargo feature.
#[cfg(feature = "fts-stemmer-langs")]
pub fn register_stemmer_langs(index: &Index) {
    use tantivy::tokenizer::{Language, Stemmer};

    let langs: &[(&str, Language)] = &[
        ("ar_stem", Language::Arabic),
        ("da_stem", Language::Danish),
        ("nl_stem", Language::Dutch),
        ("fi_stem", Language::Finnish),
        ("fr_stem", Language::French),
        ("de_stem", Language::German),
        ("el_stem", Language::Greek),
        ("hu_stem", Language::Hungarian),
        ("it_stem", Language::Italian),
        ("no_stem", Language::Norwegian),
        ("pt_stem", Language::Portuguese),
        ("ro_stem", Language::Romanian),
        ("ru_stem", Language::Russian),
        ("es_stem", Language::Spanish),
        ("sv_stem", Language::Swedish),
        ("ta_stem", Language::Tamil),
        ("tr_stem", Language::Turkish),
    ];

    for (name, lang) in langs {
        let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(Stemmer::new(*lang))
            .build();
        index.tokenizers().register(name, analyzer);
    }
}
