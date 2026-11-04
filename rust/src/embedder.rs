//! Embedding over HTTP, for a memory that embeds on a server: any service that speaks the OpenAI embeddings API
//! (`POST {url}/embeddings`, `{ model, input }`) and serves EmbeddingGemma-300m — text-embeddings-inference, Infinity,
//! Ollama (`embeddinggemma`), vLLM, your own. Vectors may come back as numbers or as little-endian float32 base64.
//!
//! Memory's prompts, indexes and measured thresholds are EmbeddingGemma's: a different model needs its own index, and
//! its results are not the measured ones.

use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::js;
use crate::llm::http;
use crate::BoxFuture;

/// The embedding model's id, name and dimensions.
pub struct MemoryEmbedder {
    /// The index's tag for it.
    pub id: &'static str,
    /// Its name.
    pub model: &'static str,
    /// Numbers a vector.
    pub dims: usize,
}

/// EmbeddingGemma-300m, 768 dimensions.
pub const MEMORY_EMBEDDER: MemoryEmbedder =
    MemoryEmbedder { id: "embeddinggemma-300m-768", model: "embeddinggemma-300m", dims: 768 };

/// A document as EmbeddingGemma's task prompt for a document gives it.
pub fn doc_text(title: &str, text: &str) -> String {
    format!("title: {} | text: {text}", if title.is_empty() { "none" } else { title })
}

/// A question as EmbeddingGemma's task prompt for a search query gives it.
pub fn query_text(q: &str) -> String {
    format!("task: search result | query: {q}")
}

/// Already prompted texts → unit vectors, in order: `(texts, timeout_ms)`.
pub type EmbedFn = Arc<dyn Fn(Vec<String>, Option<u64>) -> BoxFuture<'static, Result<Vec<Vec<f32>>>> + Send + Sync>;

/// An embeddings server.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmbedRoute {
    /// The API's base, e.g. `http://localhost:8080/v1` (`/embeddings` is added).
    pub url: String,
    /// Sent as a bearer token.
    pub api_key: Option<String>,
    /// The model name the service knows it by. Default `embeddinggemma-300m`.
    pub model: Option<String>,
    /// Send bodies gzipped (a service that accepts `content-encoding: gzip`).
    pub gzip: bool,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// The route the environment names: `MEMTRAILS_EMBED_URL`, `MEMTRAILS_EMBED_KEY`, `MEMTRAILS_EMBED_MODEL`; `None`
/// without one.
pub fn resolve_embed_route() -> Option<EmbedRoute> {
    Some(EmbedRoute {
        url: env("MEMTRAILS_EMBED_URL")?,
        api_key: env("MEMTRAILS_EMBED_KEY"),
        model: env("MEMTRAILS_EMBED_MODEL"),
        gzip: false,
    })
}

const MAX_INPUTS: usize = 256;
/// JSON-escaped UTF-8 bytes in one request.
const MAX_BYTES: usize = 600_000;
const MAX_TEXT: usize = 24_000;
const IN_FLIGHT: usize = 4;

pub(crate) fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    // Writing to a Vec cannot fail.
    let _ = enc.write_all(bytes);
    enc.finish().unwrap_or_default()
}

/// Validate the complete response before any vector reaches the index. Never mix dimensions or vector spaces.
fn vectors(body: &Value, count: usize) -> Result<Vec<Vec<f32>>> {
    let bad = |m: &str| Error::Embedding(format!("embeddings: {m}"));
    let mut data: Vec<&Value> = body["data"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    data.sort_by(|a, b| a["index"].as_f64().unwrap_or(f64::NAN).total_cmp(&b["index"].as_f64().unwrap_or(f64::NAN)));
    if data.len() != count {
        return Err(Error::Embedding(format!("embeddings: {} vectors for {count} texts", data.len())));
    }
    let dims = MEMORY_EMBEDDER.dims;
    data.iter()
        .enumerate()
        .map(|(n, d)| {
            if d["index"].as_f64() != Some(n as f64) {
                return Err(bad("missing or duplicate response index"));
            }
            let v: Vec<f32> = match &d["embedding"] {
                Value::String(s) => {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(s)
                        .map_err(|_| bad("invalid base64 vector"))?;
                    if bytes.len() != dims * 4 || base64::engine::general_purpose::STANDARD.encode(&bytes) != *s {
                        return Err(bad("invalid base64 vector"));
                    }
                    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
                }
                Value::Array(a) if a.len() == dims && a.iter().all(Value::is_number) => {
                    a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()
                }
                _ => return Err(bad("wrong vector dimensions")),
            };
            let norm = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
            if !norm.is_finite() || norm <= 0.0 {
                return Err(bad("non-finite or zero vector"));
            }
            Ok(v.iter().map(|x| (f64::from(*x) / norm) as f32).collect())
        })
        .collect()
}

/// `embed(texts)` over a server: unit vectors in the order of `texts`, however many requests a batch needs, at most
/// four at once, in one operation budget (default 30 s) that includes compression, retries and all batches.
pub fn remote_embedder(route: EmbedRoute) -> EmbedFn {
    let route = Arc::new(route);
    Arc::new(move |texts: Vec<String>, timeout_ms: Option<u64>| {
        let route = route.clone();
        Box::pin(async move { embed_remote(&route, texts, timeout_ms.unwrap_or(30_000)).await })
            as BoxFuture<'static, Result<Vec<Vec<f32>>>>
    })
}

async fn embed_remote(r: &EmbedRoute, texts: Vec<String>, timeout_ms: u64) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let until = Instant::now() + Duration::from_millis(timeout_ms);
    let remaining = move || -> Result<Duration> {
        let left = until.saturating_duration_since(Instant::now());
        if left < Duration::from_millis(1) {
            return Err(Error::Embedding("embeddings: operation deadline exceeded".into()));
        }
        Ok(left)
    };
    let input: Vec<String> = texts
        .iter()
        .map(|t| {
            let t = js::head16(t, MAX_TEXT);
            if t.is_empty() {
                " ".to_string()
            } else {
                t.to_string()
            }
        })
        .collect();
    let mut parts: Vec<(usize, Vec<String>)> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut bytes = 0usize;
    for (i, t) in input.iter().enumerate() {
        let size = serde_json::to_string(t).map(|s| s.len()).unwrap_or(t.len()) + 1;
        if !cur.is_empty() && (cur.len() >= MAX_INPUTS || bytes + size > MAX_BYTES) {
            parts.push((i - cur.len(), std::mem::take(&mut cur)));
            bytes = 0;
        }
        cur.push(t.clone());
        bytes += size;
    }
    if !cur.is_empty() {
        parts.push((input.len() - cur.len(), cur));
    }
    let url = format!("{}/embeddings", r.url.trim_end_matches('/').trim_end_matches("/embeddings"));
    let model = r.model.clone().unwrap_or_else(|| MEMORY_EMBEDDER.model.to_string());
    let call = |part: (usize, Vec<String>)| {
        let url = url.clone();
        let model = model.clone();
        async move {
            remaining()?;
            let count = part.1.len();
            let json = js::stringify(&json!({ "model": model, "input": part.1 })).into_bytes();
            let body = if r.gzip { gzip(&json) } else { json };
            let mut attempt = 0u32;
            loop {
                let mut req = http()
                    .post(&url)
                    .timeout(remaining()?)
                    .header("content-type", "application/json")
                    .body(body.clone());
                if r.gzip {
                    req = req.header("content-encoding", "gzip");
                }
                if let Some(k) = &r.api_key {
                    req = req.header("authorization", format!("Bearer {k}"));
                }
                let res = req.send().await.map_err(|e| Error::Embedding(format!("embeddings: {e}")))?;
                let status = res.status().as_u16();
                if res.status().is_success() {
                    let payload: Value = res.json().await.map_err(|e| Error::Embedding(format!("embeddings: {e}")))?;
                    remaining()?;
                    return Ok::<_, Error>((part.0, vectors(&payload, count)?));
                }
                // Drained before a retry (a keep-alive socket).
                let message = res.text().await.unwrap_or_default();
                if (status == 429 || status >= 500) && attempt < 3 {
                    let delay = Duration::from_millis(500 << attempt);
                    if remaining()? <= delay {
                        return Err(Error::Embedding("embeddings: operation deadline exceeded".into()));
                    }
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(Error::Embedding(format!("embeddings {status}: {}", js::head16(&message, 200))));
            }
        }
    };
    let mut out: Vec<Option<Vec<f32>>> = vec![None; input.len()];
    let mut pending = parts.into_iter();
    let mut running = futures::stream::FuturesUnordered::new();
    use futures::StreamExt;
    for p in pending.by_ref().take(IN_FLIGHT) {
        running.push(call(p));
    }
    while let Some(done) = running.next().await {
        let (at, vs) = done?;
        for (k, v) in vs.into_iter().enumerate() {
            out[at + k] = Some(v);
        }
        if let Some(p) = pending.next() {
            running.push(call(p));
        }
    }
    out.into_iter().map(|v| v.ok_or_else(|| Error::Embedding("embeddings: a vector is missing".into()))).collect()
}
