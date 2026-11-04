//! An optional cross-encoder reranker on a server, for search: any service that speaks the common rerank API
//! (`POST {url}/rerank`, `{ model, query, documents }` → `results[{ index, relevance_score }]`, as
//! text-embeddings-inference, Infinity, Jina and Cohere do). Measured with BAAI/bge-reranker-v2-m3 over the best 30
//! (docs/BENCHMARKS.md): on little text it matched word by word (words.rs), which needs no server — so it is off
//! unless you give it one.
//!
//! Optional by design: no route, a refusal, an error or a late answer — and the search keeps its own order.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::embedder::gzip;
use crate::js;
use crate::llm::http;
use crate::{BoxFuture, RerankFn};

/// The reranker's model by default.
pub const RERANK_MODEL: &str = "bge-reranker-v2-m3";
/// How many of the best candidates it reads (20 lost 4.6 points of LoCoMo multi-hop against 30).
pub const RERANK_TOP: usize = 30;
/// What it reads of each: the title and summary, cut here.
pub const RERANK_CHARS: usize = 1500;

/// A rerank server.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RerankRoute {
    /// The API's base, e.g. `http://localhost:8081/v1` (`/rerank` is added).
    pub url: String,
    /// Sent as a bearer token.
    pub api_key: Option<String>,
    /// Default `bge-reranker-v2-m3`.
    pub model: Option<String>,
    /// Send bodies gzipped.
    pub gzip: bool,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// The route the environment names: `MEMTRAILS_RERANK_URL`, `MEMTRAILS_RERANK_KEY`, `MEMTRAILS_RERANK_MODEL`.
pub fn resolve_rerank_route() -> Option<RerankRoute> {
    Some(RerankRoute {
        url: env("MEMTRAILS_RERANK_URL")?,
        api_key: env("MEMTRAILS_RERANK_KEY"),
        model: env("MEMTRAILS_RERANK_MODEL"),
        gzip: false,
    })
}

/// Each document's relevance to the query, in the documents' order — or `None` (refused, failed, late).
pub fn remote_reranker(route: RerankRoute) -> RerankFn {
    let route = Arc::new(route);
    Arc::new(move |query: String, documents: Vec<String>, timeout_ms: u64| {
        let r = route.clone();
        Box::pin(async move {
            if documents.is_empty() || timeout_ms < 100 {
                return None;
            }
            let docs: Vec<String> = documents
                .iter()
                .map(|t| {
                    let t = js::head16(t, RERANK_CHARS);
                    if t.is_empty() {
                        " ".to_string()
                    } else {
                        t.to_string()
                    }
                })
                .collect();
            let model = r.model.clone().unwrap_or_else(|| RERANK_MODEL.into());
            let json = js::stringify(&json!({ "model": model, "query": query, "documents": docs })).into_bytes();
            let body = if r.gzip { gzip(&json) } else { json };
            let mut req = http()
                .post(format!("{}/rerank", r.url.trim_end_matches('/').trim_end_matches("/rerank")))
                .timeout(Duration::from_millis(timeout_ms))
                .header("content-type", "application/json")
                .body(body);
            if r.gzip {
                req = req.header("content-encoding", "gzip");
            }
            if let Some(k) = &r.api_key {
                req = req.header("authorization", format!("Bearer {k}"));
            }
            let res = req.send().await.ok()?;
            if !res.status().is_success() {
                let _ = res.text().await;
                return None;
            }
            let reply: Value = res.json().await.ok()?;
            let mut scores = vec![f64::NAN; documents.len()];
            for x in reply["results"].as_array().into_iter().flatten() {
                if let (Some(i), Some(s)) = (x["index"].as_f64(), x["relevance_score"].as_f64()) {
                    if i >= 0.0 && i.fract() == 0.0 && (i as usize) < documents.len() {
                        scores[i as usize] = s;
                    }
                }
            }
            if scores.iter().any(|s| s.is_nan()) {
                None
            } else {
                Some(scores)
            }
        }) as BoxFuture<'static, Option<Vec<f64>>>
    })
}

/// The candidates in the reranker's order, fused with their own (by rank, k = 10: 0.7 to the reranker, 0.3 to the
/// search's own order — the measured mix); no scores leave the order as it was.
pub fn reranked_order<T: Clone>(items: &[T], scores: Option<&[f64]>) -> Vec<T> {
    let Some(scores) = scores.filter(|s| s.len() == items.len()) else { return items.to_vec() };
    let cmp = |a: f64, b: f64| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
    let mut by_score: Vec<usize> = (0..items.len()).collect();
    by_score.sort_by(|&a, &b| cmp(scores[b], scores[a]).then(a.cmp(&b)));
    let mut fused = vec![0.0f64; items.len()];
    for (r, &i) in by_score.iter().enumerate() {
        fused[i] += 0.7 / (10.0 + r as f64 + 1.0);
    }
    for (i, f) in fused.iter_mut().enumerate() {
        *f += 0.3 / (10.0 + i as f64 + 1.0);
    }
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&a, &b| cmp(fused[b], fused[a]).then(a.cmp(&b)));
    order.into_iter().map(|i| items[i].clone()).collect()
}
