//! What gets indexed from an item's text, besides its content key: the keyword vector (BM25 as a Qdrant sparse vector —
//! term frequencies here, the IDF on Qdrant's side with `modifier: idf`) and the small-to-big children.
//!
//! Measured on a private history of real work: the keyword channel beside the dense one put a needed item in the top 5
//! for +5–8 points; ~96-word children beat 192-word ones; adding raw text to the key itself hurt.
//!
//! Every length and position here is in UTF-16 code units, as the reference counts them.

use std::sync::LazyLock;

use icu_segmenter::options::WordBreakInvariantOptions;
use icu_segmenter::WordSegmenter;
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::js;

const K1: f64 = 1.2;
const B: f64 = 0.75;
/// Longer than this is not a word a person would type (a hash, a base64 blob).
const MAX_TOKEN: usize = 60;

/// A sparse vector as Qdrant takes it: term ids and their weights.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SparseVector {
    /// Term ids (`term_id`), ascending.
    pub indices: Vec<u32>,
    /// The weight of each.
    pub values: Vec<f64>,
}

fn re(pattern: &str) -> Regex {
    // The patterns are constants of this crate, checked by its tests.
    Regex::new(pattern).unwrap_or_else(|e| panic!("memtrails: invalid built-in pattern {pattern}: {e}"))
}

/// Scripts written without spaces between words: there the ICU word segmenter finds the words.
static UNSPACED: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]",
    )
});
static MARKS_AFTER_LGC: LazyLock<Regex> =
    LazyLock::new(|| re(r"([\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}])\p{M}+"));
static LETTER_OR_NUMBER: LazyLock<Regex> = LazyLock::new(|| re(r"[\p{L}\p{N}]"));
static NOT_WORD: LazyLock<Regex> = LazyLock::new(|| re(r"[^\p{L}\p{N}\p{M}_.\-/]+"));
static PUNCT_RUN: LazyLock<Regex> = LazyLock::new(|| re(r"[_.\-/]+"));
static CONCEPT_ENDS: LazyLock<Regex> =
    LazyLock::new(|| re(&format!(r#"^[{s}.,:;"'`()\[\]]+|[{s}.,:;"'`()\[\]]+$"#, s = js::SPACE_CLASS)));

/// The ICU word segmenter's word-like segments of `text`, as `[start, end)` byte offsets (dictionary-based for
/// Chinese, Japanese, Thai, Lao, Khmer and Burmese, as ICU4C and so `Intl.Segmenter` are). ICU4X gives a word that
/// ends in a combining mark (Devanagari's vowel signs) no word type where ICU4C calls it a letter word, so a segment
/// with a letter or a number in it is word-like here too.
pub(crate) fn word_segments(text: &str) -> Vec<(usize, usize)> {
    let segmenter = WordSegmenter::new_dictionary(WordBreakInvariantOptions::default());
    let mut out = Vec::new();
    let mut breaks = segmenter.segment_str(text);
    let mut prev = match breaks.next() {
        Some(b) => b,
        None => return out,
    };
    while let Some(b) = breaks.next() {
        if breaks.is_word_like() || LETTER_OR_NUMBER.is_match(&text[prev..b]) {
            out.push((prev, b));
        }
        prev = b;
    }
    out
}

/// Lower case, compatibility forms unified, accents dropped from Latin, Greek and Cyrillic letters — "bază" and
/// "baza" are the same word to a person typing either. Marks that make the word elsewhere (Devanagari, Thai) stay.
pub fn fold(text: &str) -> String {
    let nfkd: String = text.nfkd().collect();
    let stripped = MARKS_AFTER_LGC.replace_all(&nfkd, "$1");
    stripped.nfkc().collect::<String>().to_lowercase()
}

/// Words for the keyword channel, in any language: folded (`fold`); identifiers kept whole AND split at `_ . - /`; in
/// scripts without spaces (Chinese, Japanese, Thai…) the segmenter's words.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let folded = fold(text);
    for raw in NOT_WORD.split(&folded) {
        let w = raw.trim_matches(|c| matches!(c, '.' | '_' | '-' | '/'));
        if w.is_empty() {
            continue;
        }
        if UNSPACED.is_match(w) {
            for (a, b) in word_segments(w) {
                out.push(w[a..b].to_string());
            }
            continue;
        }
        let len = js::len16(w);
        if len < 2 {
            continue;
        }
        // The length cap is per word: a path too long to be one keyword still gives its parts.
        if len <= MAX_TOKEN {
            out.push(w.to_string());
        }
        if w.contains(['_', '.', '-', '/']) {
            for p in PUNCT_RUN.split(w) {
                let n = js::len16(p);
                if (2..=MAX_TOKEN).contains(&n) && p != w {
                    out.push(p.to_string());
                }
            }
        }
    }
    out
}

/// Where each word of a text starts and ends, in UTF-16 units — the segmenter's words, so a text without spaces
/// counts right.
pub(crate) fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut unit = 0usize;
    let mut byte = 0usize;
    for (a, b) in word_segments(text) {
        unit += js::len16(&text[byte..a]);
        let start = unit;
        unit += js::len16(&text[a..b]);
        byte = b;
        out.push((start, unit));
    }
    out
}

fn flat(s: &str) -> String {
    js::trim(&js::collapse_spaces(s)).to_string()
}

/// A text's first `n` words, as written (whitespace flattened).
pub fn first_words(text: &str, n: usize) -> String {
    let at = word_spans(text);
    if at.len() > n && n > 0 {
        flat(js::head16(text, at[n - 1].1))
    } else if at.len() > n {
        // `n` 0: JavaScript reads `at[-1]`, undefined, and slices to nothing.
        String::new()
    } else {
        flat(text)
    }
}

/// 32-bit FNV-1a over UTF-16 code units: a term's index in the sparse vector.
pub fn term_id(term: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for unit in term.encode_utf16() {
        h ^= u32::from(unit);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// A document's BM25 term weights (Qdrant applies the IDF), against the typical length of what is indexed (220).
pub fn doc_sparse(text: &str) -> SparseVector {
    doc_sparse_with(text, 220.0)
}

/// [`doc_sparse`] with another typical length.
pub fn doc_sparse_with(text: &str, avg_len: f64) -> SparseVector {
    let ts = tokens(text);
    let mut tf: IndexMap<u32, f64> = IndexMap::new();
    for t in &ts {
        *tf.entry(term_id(t)).or_insert(0.0) += 1.0;
    }
    let norm = K1 * (1.0 - B + (B * ts.len() as f64) / avg_len);
    let mut indices: Vec<u32> = tf.keys().copied().collect();
    indices.sort_unstable();
    let values = indices
        .iter()
        .map(|i| {
            let f = tf[i];
            (f * (K1 + 1.0)) / (f + norm)
        })
        .collect();
    SparseVector { indices, values }
}

/// A query's terms, each once.
pub fn query_sparse<S: AsRef<str>>(texts: &[S]) -> SparseVector {
    let mut ids: Vec<u32> = texts.iter().flat_map(|t| tokens(t.as_ref())).map(|t| term_id(&t)).collect();
    ids.sort_unstable();
    ids.dedup();
    let values = vec![1.0; ids.len()];
    SparseVector { indices: ids, values }
}

/// Words in a child window.
pub const CHILD_WORDS: usize = 96;
/// Words between the starts of two child windows.
pub const CHILD_STRIDE: usize = 72;

/// ~96-word windows of a body, a quarter overlapping, all of it — searched, never shown: the item is what comes back.
/// Words as the segmenter finds them, so a body in a script without spaces is windowed too.
pub fn children(text: &str) -> Vec<String> {
    children_with(text, CHILD_WORDS, CHILD_STRIDE)
}

/// [`children`] with other window sizes.
pub fn children_with(text: &str, words: usize, stride: usize) -> Vec<String> {
    let at = word_spans(text);
    if at.is_empty() {
        let f = flat(text);
        return if f.is_empty() { vec![] } else { vec![f] };
    }
    if at.len() <= words {
        return vec![flat(text)];
    }
    let mut out = Vec::new();
    let mut end = 0usize;
    let mut i = 0usize;
    let stride = stride.max(1);
    while i + words / 2 < at.len() {
        end = (i + words).min(at.len());
        out.push(flat(js::slice16(text, at[i].0, at[end - 1].1)));
        i += stride;
    }
    // The last words, when the stride stopped short of them: a window ending at the text's end.
    if end < at.len() {
        out.push(flat(js::slice16(text, at[at.len().saturating_sub(words)].0, at[at.len() - 1].1)));
    }
    out
}

/// The first `n` UTF-16 units and an ellipsis — never ending on half a surrogate pair (an emoji cut in two is not
/// valid UTF-8, and Qdrant refuses the whole write over it).
pub fn cut(s: &str, n: usize) -> String {
    if js::len16(s) <= n {
        return s.to_string();
    }
    format!("{}…", js::head16(s, n))
}

/// One spelling per concept, the graph's key: folded like a word (case, accents, compatibility forms), inner spaces
/// collapsed, stray punctuation off the ends — "Proxy.conf " and "proxy.conf" are one node.
pub fn concept_key(name: &str) -> String {
    let collapsed = js::collapse_spaces(&fold(name));
    CONCEPT_ENDS.replace_all(&collapsed, "").into_owned()
}
