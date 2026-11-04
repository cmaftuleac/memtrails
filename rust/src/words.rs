//! Word by word: the search's order with no server. A memory's one vector averages it, so one exact detail in a long
//! memory is drowned out. The embedder has a vector per word before it averages them; a memory scores the sum, over
//! the question's words, of its closest word (MaxSim, as ColBERT does). Over the search's best 30, fused by rank as the
//! reranker is (`reranker::reranked_order`). No training, no word list, any language.
//!
//! Measured (docs/BENCHMARKS.md): all 2,011 questions, answer shown at 5 50.3 → 53.1 (+80/−24), at 10 57.1 → 58.4
//! (+67/−40); other languages against English memories gain more. One bit a number keeps nearly all of it, and 128
//! words are enough where 64 are not.
//!
//! A memory's words are kept small, the question's whole: its cosine with a word stored as signs is the sum of its
//! numbers with those signs, over √dims. Kept by the text's hash, so an unchanged text is never encoded twice.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::store::append_private;
use crate::{BoxFuture, RerankFn};

/// Words kept of a memory (64 lost most of the gain).
pub const WORD_LIMIT: usize = 128;
const KEY_BYTES: usize = 8;

/// Unit vectors, one a word, flat: `words × dims` numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct WordVectors {
    /// How many words.
    pub words: usize,
    /// Numbers a word.
    pub dims: usize,
    /// The numbers, word after word.
    pub data: Vec<f32>,
}

/// Whether a text is a question or a document: they are prompted differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordKind {
    /// A question.
    Query,
    /// A memory.
    Document,
}

/// Encodes texts into word vectors (the local model's `last_hidden_state`).
pub type WordEncoder = Arc<dyn Fn(Vec<String>, WordKind) -> BoxFuture<'static, Result<Vec<WordVectors>>> + Send + Sync>;

/// A memory's words as signs, one bit a number, a word to a whole number of bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedWords {
    /// How many words.
    pub words: usize,
    /// Numbers a word.
    pub dims: usize,
    /// The signs, word after word, `ceil(dims / 8)` bytes a word, the lowest bit first.
    pub bits: Vec<u8>,
}

/// The first `WORD_LIMIT` words of `v`, one bit a number.
pub fn pack_words(v: &WordVectors) -> PackedWords {
    pack_words_limit(v, WORD_LIMIT)
}

/// The first `limit` words of `v`, one bit a number.
pub fn pack_words_limit(v: &WordVectors, limit: usize) -> PackedWords {
    let words = v.words.min(limit);
    let stride = v.dims.div_ceil(8);
    let mut bits = vec![0u8; words * stride];
    for w in 0..words {
        for d in 0..v.dims {
            if v.data.get(w * v.dims + d).copied().unwrap_or(0.0) >= 0.0 {
                bits[w * stride + (d >> 3)] |= 1 << (d & 7);
            }
        }
    }
    PackedWords { words, dims: v.dims, bits }
}

/// The sum, over the question's words, of the closest word of the memory (`-inf` when either has none, or their
/// sizes differ).
pub fn max_sim(query: &WordVectors, memory: &PackedWords) -> f64 {
    if query.words == 0 || memory.words == 0 || query.dims != memory.dims {
        return f64::NEG_INFINITY;
    }
    let dims = query.dims;
    let stride = dims.div_ceil(8);
    let norm = (dims as f64).sqrt();
    let mut total = 0.0f64;
    for q in 0..query.words {
        let at = q * dims;
        let mut best = f64::NEG_INFINITY;
        for w in 0..memory.words {
            let row = w * stride;
            let mut sum = 0.0f64;
            for d in 0..dims {
                // The query's numbers are float32, summed in double as JavaScript does.
                let x = f64::from(query.data[at + d]);
                sum += if memory.bits[row + (d >> 3)] & (1 << (d & 7)) != 0 { x } else { -x };
            }
            if sum > best {
                best = sum;
            }
        }
        total += best / norm;
    }
    total
}

fn key_of(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))[..KEY_BYTES * 2].to_string()
}

/// The packed words by text, in memory and appended to a file (`words.bin`): an 8-byte key, the word count and the
/// numbers a word (u16 little-endian each), then the bits. Derived, like the index: losing it only means encoding
/// again.
#[derive(Clone)]
pub struct WordStore {
    file: Option<PathBuf>,
    by_key: Arc<Mutex<HashMap<String, Arc<PackedWords>>>>,
}

impl WordStore {
    /// A store kept in `file`, or in memory only.
    pub fn new(file: Option<PathBuf>) -> Self {
        WordStore { file, by_key: Arc::new(Mutex::new(HashMap::new())) }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, Arc<PackedWords>>> {
        self.by_key.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// How many texts it holds.
    pub fn size(&self) -> usize {
        self.lock().len()
    }

    /// The packed words of a text, when held.
    pub fn get(&self, text: &str) -> Option<Arc<PackedWords>> {
        self.lock().get(&key_of(text)).cloned()
    }

    /// Read the file. An entry cut short by a crash goes: everything appended after it would be read as part of it.
    pub async fn load(&self) -> Result<()> {
        let me = self.clone();
        crate::store::blocking(move || me.load_now()).await?
    }

    fn load_now(&self) -> Result<()> {
        let Some(file) = &self.file else { return Ok(()) };
        let Ok(b) = fs::read(file) else { return Ok(()) };
        let mut at = 0usize;
        let mut map = self.lock();
        while at + KEY_BYTES + 4 <= b.len() {
            let key = hex::encode(&b[at..at + KEY_BYTES]);
            let words = usize::from(u16::from_le_bytes([b[at + KEY_BYTES], b[at + KEY_BYTES + 1]]));
            let dims = usize::from(u16::from_le_bytes([b[at + KEY_BYTES + 2], b[at + KEY_BYTES + 3]]));
            let size = words * dims.div_ceil(8);
            let start = at + KEY_BYTES + 4;
            if start + size > b.len() {
                break;
            }
            map.insert(key, Arc::new(PackedWords { words, dims, bits: b[start..start + size].to_vec() }));
            at = start + size;
        }
        drop(map);
        if at < b.len() {
            // Read-only: encoded again when needed.
            if let Ok(f) = fs::OpenOptions::new().write(true).open(file) {
                let _ = f.set_len(at as u64);
            }
        }
        Ok(())
    }

    /// Keep these texts' packed words.
    pub async fn put(&self, entries: Vec<(String, PackedWords)>) -> Result<()> {
        let me = self.clone();
        crate::store::blocking(move || me.put_now(entries)).await?
    }

    fn put_now(&self, entries: Vec<(String, PackedWords)>) -> Result<()> {
        let mut bytes = Vec::new();
        {
            let mut map = self.lock();
            for (text, p) in entries {
                let key = key_of(&text);
                bytes.extend_from_slice(&hex::decode(&key).unwrap_or_default());
                bytes.extend_from_slice(&(p.words as u16).to_le_bytes());
                bytes.extend_from_slice(&(p.dims as u16).to_le_bytes());
                bytes.extend_from_slice(&p.bits);
                map.insert(key, Arc::new(p));
            }
        }
        let Some(file) = &self.file else { return Ok(()) };
        if bytes.is_empty() {
            return Ok(());
        }
        if let Some(dir) = file.parent() {
            fs::create_dir_all(dir).map_err(|e| crate::Error::io(dir, e))?;
        }
        append_private(file, &bytes).map_err(|e| crate::Error::io(file, e))
    }
}

/// A reranker for search that needs no server: what is not stored yet is encoded here, once. `None` — the search keeps
/// its own order — when the encoder fails or takes longer than it was given.
pub fn word_reranker(encode: WordEncoder, store: WordStore) -> RerankFn {
    Arc::new(move |query: String, documents: Vec<String>, timeout_ms: u64| {
        let encode = encode.clone();
        let store = store.clone();
        Box::pin(async move {
            let work = async {
                let mut missing: Vec<String> = Vec::new();
                for t in &documents {
                    if store.get(t).is_none() && !missing.contains(t) {
                        missing.push(t.clone());
                    }
                }
                if !missing.is_empty() {
                    let encoded = encode(missing.clone(), WordKind::Document).await?;
                    store.put(missing.into_iter().zip(encoded.iter().map(pack_words)).collect()).await?;
                }
                let q = encode(vec![query], WordKind::Query).await?;
                let q = q.into_iter().next().ok_or_else(|| crate::Error::Model("no query vectors".into()))?;
                let packed: Vec<Option<Arc<PackedWords>>> = documents.iter().map(|t| store.get(t)).collect();
                crate::store::blocking(move || {
                    packed
                        .iter()
                        .map(|p| p.as_ref().map_or(f64::NEG_INFINITY, |p| max_sim(&q, p)))
                        .collect::<Vec<f64>>()
                })
                .await
            };
            match tokio::time::timeout(Duration::from_millis(timeout_ms), work).await {
                Ok(Ok(scores)) => Some(scores),
                _ => None,
            }
        }) as BoxFuture<'static, Option<Vec<f64>>>
    })
}
