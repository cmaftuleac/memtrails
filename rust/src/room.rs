//! A small room filled with sentences, not memories. An agent reads a few thousand characters of what memory found; a
//! memory's first 500 of them are rarely where its answer is. Each found memory's text is cut into sentences (at any
//! script's end marks, no word list), each sentence scored word by word against the question, and the room filled
//! with the best sentences, a memory's place in the search counting too (docs/RESEARCH.md: held out, the answer's
//! words in 6,000 characters 66.0 against 57.2 for Mem0's room on raw memory).
//!
//! Sentences are kept packed beside the memories' own words (`WordStore`), by their text, so each is encoded once.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::js;
use crate::words::{max_sim, pack_words, WordEncoder, WordKind, WordStore};

/// Sentences shorter than this say too little to be worth a place.
const MIN_SENTENCE: usize = 8;
/// A sentence longer than this is cut into pieces of it.
const MAX_SENTENCE: usize = 400;
/// A memory's date and kind, once, before its sentences.
const HEAD_CHARS: usize = 20;

/// `[start, end)` of each sentence, in UTF-16 units: after `.`, `!`, `?`, their full-width forms, `。` or a line
/// break, then a space.
pub fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let cs: Vec<char> = text.chars().collect();
    // The UTF-16 offset of each character, and of the end.
    let mut unit = Vec::with_capacity(cs.len() + 1);
    let mut u = 0usize;
    for c in &cs {
        unit.push(u);
        u += c.len_utf16();
    }
    unit.push(u);
    // A space at a UTF-16 offset: the character that starts there (spaces are all one unit).
    let index: BTreeMap<usize, usize> = unit.iter().enumerate().map(|(i, u)| (*u, i)).collect();
    let space_at = |at: usize| index.get(&at).and_then(|&i| cs.get(i)).is_some_and(|c| js::is_space(*c));
    let mut out = Vec::new();
    let mut push = |mut a: usize, mut b: usize| {
        while a < b && space_at(a) {
            a += 1;
        }
        while b > a && space_at(b - 1) {
            b -= 1;
        }
        let mut s = a;
        while s < b {
            out.push((s, b.min(s + MAX_SENTENCE)));
            s += MAX_SENTENCE;
        }
    };
    let mut from = 0usize;
    for (i, c) in cs.iter().enumerate() {
        let ends = match c {
            '.' | '!' | '?' | '。' | '！' | '？' => cs.get(i + 1).is_none_or(|n| js::is_space(*n)),
            '\n' => true,
            _ => false,
        };
        if ends {
            push(from, unit[i + 1]);
            from = unit[i + 1];
        }
    }
    push(from, u);
    out.retain(|(a, b)| b - a >= MIN_SENTENCE);
    out
}

/// A found memory as the room reads it: its head (date and kind), its text, and when it happened.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomItem {
    /// Its id.
    pub id: String,
    /// Shown once before its sentences.
    pub head: String,
    /// Its text.
    pub text: String,
    /// When it happened, for the order.
    pub ts: f64,
}

/// A found memory in the room's list: its rank, the memory and its sentences.
type Listed<'a> = (usize, &'a RoomItem, Vec<(usize, usize)>);

/// A sentence to score: its memory, its span, its weight by the memory's rank, and its packed words.
type Sentence = (usize, usize, usize, f64, Option<std::sync::Arc<crate::words::PackedWords>>);

/// The room: the best sentences of the first `top` memories, fitted in `chars`, each memory's head once, oldest first.
pub async fn sentence_room(
    query: &str,
    items: &[RoomItem],
    encode: &WordEncoder,
    store: &WordStore,
    chars: usize,
    top: usize,
) -> Result<String> {
    let listed: Vec<Listed<'_>> =
        items.iter().take(top).enumerate().map(|(rank, it)| (rank, it, sentence_spans(&it.text))).collect();
    let mut texts: Vec<String> = Vec::new();
    for (_, it, spans) in &listed {
        for (a, b) in spans {
            let t = js::slice16(&it.text, *a, *b).to_string();
            if !texts.contains(&t) {
                texts.push(t);
            }
        }
    }
    let missing: Vec<String> = texts.into_iter().filter(|t| store.get(t).is_none()).collect();
    if !missing.is_empty() {
        let encoded = encode(missing.clone(), WordKind::Document).await?;
        store.put(missing.into_iter().zip(encoded.iter().map(pack_words)).collect()).await?;
    }
    let q = encode(vec![query.to_string()], WordKind::Query)
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Model("no vectors for the question".into()))?;
    let mut work: Vec<Sentence> = Vec::new();
    for (k, (rank, it, spans)) in listed.iter().enumerate() {
        for (a, b) in spans {
            work.push((k, *a, *b, 1.0 + 2.0 / (1.0 + *rank as f64 / 10.0), store.get(js::slice16(&it.text, *a, *b))));
        }
    }
    // Word by word over every sentence: CPU work, on a blocking thread.
    let mut candidates: Vec<(usize, usize, usize, f64)> = crate::store::blocking(move || {
        work.into_iter()
            .map(|(k, a, b, weight, packed)| (k, a, b, packed.map_or(f64::NEG_INFINITY, |p| max_sim(&q, &p)) * weight))
            .collect()
    })
    .await?;
    candidates.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap_or(std::cmp::Ordering::Equal));
    let mut picked: indexmap::IndexMap<usize, Vec<(usize, usize)>> = indexmap::IndexMap::new();
    let mut used = 0usize;
    for (k, a, b, _) in candidates {
        let it = listed[k].1;
        let extra = b - a + 3 + if picked.contains_key(&k) { 0 } else { HEAD_CHARS + js::len16(&it.head) };
        if used + extra > chars {
            continue;
        }
        picked.entry(k).or_default().push((a, b));
        used += extra;
    }
    let mut groups: Vec<(usize, Vec<(usize, usize)>)> = picked.into_iter().collect();
    groups.sort_by(|x, y| listed[x.0].1.ts.partial_cmp(&listed[y.0].1.ts).unwrap_or(std::cmp::Ordering::Equal));
    Ok(groups
        .into_iter()
        .map(|(k, mut spans)| {
            let it = listed[k].1;
            spans.sort_by_key(|s| s.0);
            let said = spans.iter().map(|(a, b)| js::slice16(&it.text, *a, *b)).collect::<Vec<_>>().join(" … ");
            format!("{} {said}", it.head)
        })
        .collect::<Vec<_>>()
        .join("\n"))
}
