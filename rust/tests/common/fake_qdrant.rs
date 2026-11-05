//! Qdrant's REST API as far as memory uses it, in memory, on loopback: points by id, payload filters, cosine over the
//! dense vectors and a dot product over the keyword vectors, grouped by item — so the real client is what talks to it.
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use memtrails::text::{term_id, tokens};
use serde_json::{json, Map, Value};

use super::{serve, Response, Server};

pub const DIMS: usize = 768;

/// The words of a text as a unit vector (the embedding prompt's wrapper left out): "the same" and "nearly the same"
/// are under a test's control.
pub fn word_vector(t: &str) -> Vec<f32> {
    let mut t = t.to_string();
    for p in ["task: search result | query: ", "title: "] {
        if let Some(rest) = t.strip_prefix(p) {
            t = format!(" {rest}");
        }
    }
    let t = t.replace(" | text: ", " ");
    let mut v = vec![0.0f32; DIMS];
    let mut seen = std::collections::HashSet::new();
    for w in tokens(&t) {
        if seen.insert(w.clone()) {
            v[term_id(&w) as usize % DIMS] = 1.0;
        }
    }
    let n = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    let n = if n == 0.0 { 1.0 } else { n };
    v.iter().map(|x| (f64::from(*x) / n) as f32).collect()
}

/// `word_vector` as an embedder.
pub fn embed() -> memtrails::embedder::EmbedFn {
    Arc::new(|texts: Vec<String>, _| Box::pin(async move { Ok(texts.iter().map(|t| word_vector(t)).collect()) }))
}

#[derive(Clone)]
pub struct Point {
    pub dense: Vec<f64>,
    pub sparse: Option<(Vec<u64>, Vec<f64>)>,
    pub payload: Map<String, Value>,
}

#[derive(Default)]
pub struct State {
    pub exists: AtomicBool,
    pub points: Mutex<IndexMap<String, Point>>,
    pub drops: AtomicUsize,
    pub upserts: AtomicUsize,
    /// A pause before each upsert answers, in ms.
    pub delay: AtomicUsize,
    /// Points that tie: a chunk before its item's key (Qdrant may return either).
    pub chunk_first: AtomicBool,
}

pub struct Fake {
    pub server: Server,
    pub state: Arc<State>,
}

impl Fake {
    pub fn points(&self) -> Vec<Point> {
        self.state.points.lock().unwrap().values().cloned().collect()
    }

    pub fn of(&self, item: &str) -> Vec<Point> {
        self.points().into_iter().filter(|p| p.payload["item"] == item).collect()
    }
}

fn one(p: &Map<String, Value>, c: &Value) -> bool {
    if c.get("must").is_some() || c.get("should").is_some() || c.get("must_not").is_some() {
        return matches(p, c);
    }
    if let Some(e) = c.get("is_empty") {
        return p.get(e["key"].as_str().unwrap_or("")).is_none_or(Value::is_null);
    }
    let key = c["key"].as_str().unwrap_or("");
    let v = p.get(key).cloned().unwrap_or(Value::Null);
    if let Some(m) = c.get("match") {
        if let Some(want) = m.get("value") {
            return match &v {
                Value::Array(a) => a.contains(want),
                x => x == want,
            };
        }
        if let Some(any) = m.get("any").and_then(Value::as_array) {
            return match &v {
                Value::Array(a) => a.iter().any(|x| any.contains(x)),
                x => any.contains(x),
            };
        }
    }
    if let Some(r) = c.get("range") {
        let Some(x) = v.as_f64() else { return false };
        let ok = |k: &str, f: fn(f64, f64) -> bool| r.get(k).and_then(Value::as_f64).is_none_or(|b| f(x, b));
        return ok("lt", |x, b| x < b)
            && ok("lte", |x, b| x <= b)
            && ok("gte", |x, b| x >= b)
            && ok("gt", |x, b| x > b);
    }
    true
}

pub fn matches(p: &Map<String, Value>, f: &Value) -> bool {
    let list = |k: &str| f.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
    let should = list("should");
    list("must").iter().all(|c| one(p, c))
        && !list("must_not").iter().any(|c| one(p, c))
        && (should.is_empty() || should.iter().any(|c| one(p, c)))
}

fn score(p: &Point, query: &Value, using: &str) -> Option<f64> {
    if using == "bm25" {
        let (idx, vals) = p.sparse.as_ref()?;
        let q_idx: Vec<u64> = query["indices"].as_array()?.iter().filter_map(Value::as_u64).collect();
        let q_val: Vec<f64> = query["values"].as_array()?.iter().filter_map(Value::as_f64).collect();
        let s: f64 =
            q_idx.iter().zip(&q_val).filter_map(|(i, w)| idx.iter().position(|x| x == i).map(|k| vals[k] * w)).sum();
        return (s > 0.0).then_some(s);
    }
    let q: Vec<f64> = query.as_array()?.iter().filter_map(Value::as_f64).collect();
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in q.iter().zip(&p.dense) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    Some(if aa > 0.0 && bb > 0.0 { ab / (aa * bb).sqrt() } else { 0.0 })
}

fn ranked(state: &State, query: &Value, using: &str, filter: &Value) -> Vec<(String, f64, Map<String, Value>)> {
    let points = state.points.lock().unwrap();
    let mut out: Vec<(String, f64, Map<String, Value>)> = points
        .iter()
        .filter(|(_, p)| matches(&p.payload, filter))
        .filter_map(|(id, p)| score(p, query, using).map(|s| (id.clone(), s, p.payload.clone())))
        .collect();
    let chunk_first = state.chunk_first.load(Ordering::SeqCst);
    let key = |p: &Map<String, Value>| chunk_first && p["part"] == "key";
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| key(&a.2).cmp(&key(&b.2))));
    out
}

fn result(v: Value) -> Response {
    Response::json(200, &json!({ "result": v }))
}

fn point_json(id: &str, p: &Point, with_vector: bool) -> Value {
    let mut v = json!({ "id": id, "payload": p.payload });
    if with_vector {
        v["vector"] = json!({ "dense": p.dense });
    }
    v
}

pub async fn start() -> Fake {
    let state = Arc::new(State::default());
    let s = state.clone();
    let server = serve(move |r| {
        let path = r.path.split('?').next().unwrap_or("").to_string();
        let b: Value = if r.body.is_empty() { json!({}) } else { r.json() };
        let Some(rest) = path.strip_prefix("/collections/") else { return result(json!({ "collections": [] })) };
        let rest = rest.split_once('/').map(|x| format!("/{}", x.1)).unwrap_or_default();
        if rest.is_empty() {
            return match r.method.as_str() {
                "GET" if s.exists.load(Ordering::SeqCst) => result(
                    json!({ "config": { "optimizer_config": { "default_segment_number": 2 } }, "payload_schema": {} }),
                ),
                "GET" => Response::json(404, &json!({ "status": { "error": "Not found" } })),
                "PUT" => {
                    s.exists.store(true, Ordering::SeqCst);
                    result(json!(true))
                }
                "DELETE" => {
                    s.exists.store(false, Ordering::SeqCst);
                    s.points.lock().unwrap().clear();
                    s.drops.fetch_add(1, Ordering::SeqCst);
                    result(json!(true))
                }
                _ => result(json!(true)),
            };
        }
        if !s.exists.load(Ordering::SeqCst) {
            return Response::json(404, &json!({ "status": { "error": "Not found" } }));
        }
        let filter = b.get("filter").cloned().unwrap_or(json!({}));
        match (r.method.as_str(), rest.as_str()) {
            (_, "/index") => result(json!(true)),
            ("PUT", "/points") => {
                s.upserts.fetch_add(1, Ordering::SeqCst);
                let delay = s.delay.load(Ordering::SeqCst);
                if delay > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(delay as u64));
                }
                if !s.exists.load(Ordering::SeqCst) {
                    return Response::json(404, &json!({}));
                }
                let mut points = s.points.lock().unwrap();
                for p in b["points"].as_array().into_iter().flatten() {
                    let v = &p["vector"];
                    let sparse = v.get("bm25").map(|x| {
                        (
                            x["indices"].as_array().unwrap().iter().filter_map(Value::as_u64).collect(),
                            x["values"].as_array().unwrap().iter().filter_map(Value::as_f64).collect(),
                        )
                    });
                    points.insert(
                        p["id"].as_str().unwrap().to_string(),
                        Point {
                            dense: v["dense"].as_array().unwrap().iter().filter_map(Value::as_f64).collect(),
                            sparse,
                            payload: p["payload"].as_object().unwrap().clone(),
                        },
                    );
                }
                result(json!(true))
            }
            ("POST", "/points") => {
                let points = s.points.lock().unwrap();
                let with_vector = b["with_vector"] != false;
                let found: Vec<Value> = b["ids"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|id| points.get(id.as_str()?).map(|p| point_json(id.as_str().unwrap(), p, with_vector)))
                    .collect();
                result(json!(found))
            }
            (_, "/points/delete") => {
                s.points.lock().unwrap().retain(|_, p| !matches(&p.payload, &filter));
                result(json!(true))
            }
            (_, "/points/payload") => {
                for p in s.points.lock().unwrap().values_mut() {
                    if matches(&p.payload, &filter) {
                        for (k, v) in b["payload"].as_object().unwrap() {
                            p.payload.insert(k.clone(), v.clone());
                        }
                    }
                }
                result(json!(true))
            }
            (_, "/points/count") => result(json!({ "count": s.points.lock().unwrap().len() })),
            (_, "/points/scroll") => {
                let points = s.points.lock().unwrap();
                let limit = b["limit"].as_u64().unwrap_or(256) as usize;
                let with_vector = b["with_vector"] != false;
                let found: Vec<Value> = points
                    .iter()
                    .filter(|(_, p)| matches(&p.payload, &filter))
                    .take(limit)
                    .map(|(id, p)| point_json(id, p, with_vector))
                    .collect();
                result(json!({ "points": found }))
            }
            (_, "/points/query") => result(json!({ "points": query(&s, &b) })),
            (_, "/points/query/batch") => result(json!(b["searches"]
                .as_array()
                .unwrap()
                .iter()
                .map(|q| json!({ "points": query(&s, q) }))
                .collect::<Vec<_>>())),
            (_, "/points/query/groups") => {
                let mut groups: IndexMap<String, (String, f64)> = IndexMap::new();
                for (id, sc, p) in ranked(&s, &b["query"], b["using"].as_str().unwrap_or("dense"), &filter) {
                    let item = p["item"].as_str().unwrap_or("").to_string();
                    groups.entry(item).or_insert((id, sc));
                }
                let limit = b["limit"].as_u64().unwrap_or(10) as usize;
                let out: Vec<Value> = groups
                    .into_iter()
                    .take(limit)
                    .map(|(item, (id, sc))| json!({ "id": item, "hits": [{ "id": id, "score": sc }] }))
                    .collect();
                result(json!({ "groups": out }))
            }
            _ => result(json!(true)),
        }
    })
    .await;
    Fake { server, state }
}

fn query(s: &State, b: &Value) -> Vec<Value> {
    let limit = b["limit"].as_u64().unwrap_or(10) as usize;
    ranked(s, &b["query"], b["using"].as_str().unwrap_or("dense"), &b.get("filter").cloned().unwrap_or(json!({})))
        .into_iter()
        .take(limit)
        .map(|(id, sc, p)| json!({ "id": id, "score": sc, "payload": p }))
        .collect()
}

/// A client of the fake, its collection made.
pub async fn client(f: &Fake) -> memtrails::qdrant_client::QdrantClient {
    let c = memtrails::qdrant_client::QdrantClient::new(
        memtrails::qdrant_client::Endpoint::Fixed(memtrails::qdrant_process::QdrantEndpoint {
            url: f.server.url.clone(),
            api_key: String::new(),
        }),
        None,
    );
    c.ensure_collection().await.unwrap();
    c
}

/// An embeddings server that gives `word_vector`s, for a memory opened with `embedding.server`.
pub async fn embed_server() -> Server {
    serve(|r| {
        let input = r.json()["input"].as_array().unwrap().clone();
        let data: Vec<Value> = input
            .iter()
            .enumerate()
            .map(|(i, t)| json!({ "index": i, "embedding": word_vector(t.as_str().unwrap()) }))
            .collect();
        Response::json(200, &json!({ "data": data }))
    })
    .await
}
