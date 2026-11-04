//! The few Qdrant REST calls the memory index makes — plain HTTP, no client library, so the payloads are the
//! reference's own. One collection, `memory`, two tiers (payload `tier`):
//!
//!   - `stm` — short-term memory: one point per record (its start as the key) and its chunks;
//!   - `ltm` — long-term memory: per item a KEY point (`part: key` — its content title + ~150-word summary as the
//!     dense vector, a BM25 sparse vector over its title, summary, body and files, the item itself as payload) and
//!     CHILD points (`part: child`, ~96-word windows of its body, dense only) and CUE points pointing at it.

use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{json, Value};

use crate::embedder::MEMORY_EMBEDDER;
use crate::error::{Error, Result};
use crate::llm::http;
use crate::qdrant_process::{QdrantEndpoint, QdrantProcess};
use crate::text::SparseVector;
use sha1::{Digest, Sha1};

/// A point's id from its item and its place (0 = the key, n = the n-th child, -(k+1) = the k-th cue): the same item
/// writes the same ids. SHA-1 of `item#n`, laid out as a UUID version 5.
pub fn point_id(item: &str, n: i64) -> String {
    let h = hex::encode(Sha1::digest(format!("{item}#{n}").as_bytes()));
    let variant = (u8::from_str_radix(&h[16..17], 16).unwrap_or(0) & 0x3) | 0x8;
    format!("{}-{}-5{}-{:x}{}-{}", &h[0..8], &h[8..12], &h[13..16], variant, &h[17..20], &h[20..32])
}

/// The collection memory writes, unless a server of yours names another.
pub const COLLECTION: &str = "memory";
/// Numbers a dense vector.
pub const DIMS: usize = MEMORY_EMBEDDER.dims;
/// What the index is made of; a change empties it and it is written in again from the logs (rebuild.rs).
///  2: binary quantisation with rescoring, keyword terms folded and segmented for every language; 3: on-disk binary
///  copy, HNSW and payload indexes; 4: 768-dimension EmbeddingGemma vectors; 5: nothing cut — every chunk of a body,
///  short-term entries chunked too, cues numbered below the key, full payloads; 6: every captured request indexed for
///  good; 7: a record's kind (`engine`) on every point of it and indexed, so a kinds filter finds a record by its
///  chunks.
pub const INDEX_LAYOUT: u32 = 7;
/// Item-level fusion: each channel (a query vector, the keywords) is its own grouped query, its items' scores are
/// min-max normalised over its own list, and an item's score is the weighted sum (CombSUM) — the dense channels share
/// this much of the weight, keywords the rest. Measured against gold evidence (docs/RESEARCH.md): dense 0.7 was the
/// best share on both benchmarks, and CombSUM beat rank fusion at the top of both.
const DENSE_SHARE: f64 = 0.7;
/// Cues — the questions an item answers, in the asker's words — search as their own channel at this share of the
/// dense weight rather than among the item's points.
const CUE_SHARE: f64 = 0.1;

/// The memory-mapped 1-bit copy finds candidates; full vectors on disk re-score 3× as many as asked.
fn rescore() -> Value {
    json!({ "quantization": { "rescore": true, "oversampling": 3.0 } })
}

/// Two segments: Qdrant's default is one a core, which costs a laptop RAM and per-query work for nothing at memory's
/// size.
const SEGMENTS: u64 = 2;

/// A point as Qdrant takes it.
#[derive(Debug, Clone, Serialize)]
pub struct PointStruct {
    /// Its id (`point_id`).
    pub id: String,
    /// `dense` and, on a key point, `bm25`.
    pub vector: PointVectors,
    /// What it carries.
    pub payload: Value,
}

/// A point's named vectors.
#[derive(Debug, Clone, Serialize)]
pub struct PointVectors {
    /// The dense vector.
    pub dense: Vec<f32>,
    /// The keyword vector (key points only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25: Option<SparseVector>,
}

/// A point found, with its item, score and payload.
#[derive(Debug, Clone, PartialEq)]
pub struct Found {
    /// The item it belongs to.
    pub item: String,
    /// Its score.
    pub score: f64,
    /// What it carries.
    pub payload: Value,
}

/// A point read back.
#[derive(Debug, Clone, PartialEq)]
pub struct Stored {
    /// Its id.
    pub id: String,
    /// What it carries.
    pub payload: Value,
    /// Its dense vector, when asked for.
    pub vector: Option<Vec<f32>>,
}

/// An item a fused search ranked, with the point that matched best.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedHit {
    /// The item.
    pub item: String,
    /// Its fused score.
    pub score: f64,
    /// The point that matched (its key, a chunk, or a cue).
    pub point: Option<String>,
}

/// A fused search.
#[derive(Debug, Clone, Default)]
pub struct FusedQuery {
    /// One channel per query vector.
    pub dense: Vec<Vec<f32>>,
    /// The keywords.
    pub sparse: Option<SparseVector>,
    /// What may be found.
    pub filter: Option<Value>,
    /// Items each channel ranks (default: `limit`, at least 50).
    pub per: Option<usize>,
    /// Items returned (default 40).
    pub limit: Option<usize>,
}

/// Where the index answers.
#[derive(Clone)]
pub enum Endpoint {
    /// A server you run.
    Fixed(QdrantEndpoint),
    /// Memory's own process (none while it starts again).
    Process(QdrantProcess),
}

impl Endpoint {
    fn current(&self) -> Option<QdrantEndpoint> {
        match self {
            Endpoint::Fixed(e) => Some(e.clone()),
            Endpoint::Process(p) => p.current(),
        }
    }
}

/// `encodeURIComponent`: everything but `A–Z a–z 0–9 - _ . ! ~ * ' ( )` percent-encoded.
pub(crate) fn encode_uri_component(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The index's client: the REST calls memory makes, each retried when a kept-alive connection was closed under it.
#[derive(Clone)]
pub struct QdrantClient {
    endpoint: Endpoint,
    timeout_ms: u64,
    /// `/collections/<name>`, the name encoded: a name you give is a path segment, never more of the path.
    at: Arc<str>,
}

fn qerr(message: String, status: u16) -> Error {
    Error::Qdrant { message, status }
}

impl QdrantClient {
    /// A client of one collection (`COLLECTION` when not given).
    pub fn new(endpoint: Endpoint, collection: Option<&str>) -> Self {
        QdrantClient {
            endpoint,
            timeout_ms: 10_000,
            at: format!("/collections/{}", encode_uri_component(collection.unwrap_or(COLLECTION))).into(),
        }
    }

    async fn call(&self, method: &str, path: &str, body: Option<&Value>, timeout_ms: Option<u64>) -> Result<Value> {
        let e = self.endpoint.current().ok_or_else(|| qerr("memory index is not running".into(), 503))?;
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(self.timeout_ms));
        let payload = body.map(crate::js::stringify);
        let method_v = reqwest::Method::from_bytes(method.as_bytes()).map_err(|err| qerr(err.to_string(), 0))?;
        // Every call here can be sent again (upserts by id, deletes by filter, reads): a connection the server closed
        // while it sat idle fails before the request reaches it, so it is tried again — twice at most.
        let mut attempt = 0;
        let res = loop {
            let mut req = http()
                .request(method_v.clone(), format!("{}{path}", e.url))
                .timeout(timeout)
                .header("api-key", &e.api_key);
            if let Some(p) = &payload {
                req = req.header("content-type", "application/json").body(p.clone());
            }
            match req.send().await {
                Ok(r) => break r,
                Err(err) => {
                    if attempt < 2 && !err.is_timeout() {
                        attempt += 1;
                        tokio::time::sleep(Duration::from_millis(150 * attempt)).await;
                        continue;
                    }
                    return Err(qerr(format!("{method} {path}: {err}"), 0));
                }
            }
        };
        let status = res.status();
        let text = res.text().await.map_err(|err| qerr(format!("{method} {path}: {err}"), 0))?;
        if !status.is_success() {
            return Err(qerr(
                format!("{method} {path}: {} {}", status.as_u16(), crate::js::head16(&text, 300)),
                status.as_u16(),
            ));
        }
        if text.is_empty() {
            return Ok(Value::Null);
        }
        let json: Value = serde_json::from_str(&text).map_err(|err| qerr(format!("{method} {path}: {err}"), 0))?;
        Ok(json.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Drop the whole collection: the index is derived, written again from the logs.
    pub async fn drop_collection(&self) {
        let _ = self.call("DELETE", &self.at, None, None).await;
    }

    /// The collection and its payload indexes, made if missing — an index added since a collection was made is added
    /// to it in place. All of it on disk, memory-mapped: RAM holds what queries touch, and the system may take it back.
    pub async fn ensure_collection(&self) -> Result<()> {
        let info = match self.call("GET", &self.at, None, None).await {
            Ok(v) => Some(if v.is_null() { json!({}) } else { v }),
            Err(Error::Qdrant { status: 404, .. }) => None,
            Err(e) => return Err(e),
        };
        let have = match info {
            None => {
                let body = json!({
                    "vectors": { "dense": { "size": DIMS, "distance": "Cosine", "on_disk": true } },
                    "sparse_vectors": { "bm25": { "modifier": "idf", "index": { "on_disk": true } } },
                    "quantization_config": { "binary": { "always_ram": false } },
                    "hnsw_config": { "m": 16, "ef_construct": 100, "on_disk": true },
                    "on_disk_payload": true,
                    "optimizers_config": { "default_segment_number": SEGMENTS },
                });
                self.call("PUT", &self.at, Some(&body), None).await?;
                json!({})
            }
            Some(info) => {
                if info["config"]["optimizer_config"]["default_segment_number"].as_u64() != Some(SEGMENTS) {
                    // Made before: two segments from now on (the optimiser merges them in the background).
                    let body = json!({ "optimizers_config": { "default_segment_number": SEGMENTS } });
                    self.call("PATCH", &self.at, Some(&body), None).await?;
                }
                info["payload_schema"].clone()
            }
        };
        let fields = [
            ("item", "keyword"),
            ("part", "keyword"),
            ("tier", "keyword"),
            ("status", "keyword"),
            ("kind", "keyword"),
            ("engine", "keyword"),
            ("scope", "keyword"),
            ("project", "keyword"),
            ("session", "keyword"),
            ("ts", "integer"),
            ("synthetic", "bool"),
            ("concepts", "keyword"),
            ("eventFrom", "integer"),
            ("eventTo", "integer"),
        ];
        for (name, schema) in fields {
            if have.get(name).is_some() {
                continue;
            }
            let schema = if schema == "bool" { json!("bool") } else { json!({ "type": schema, "on_disk": true }) };
            let body = json!({ "field_name": name, "field_schema": schema });
            self.call("PUT", &format!("{}/index?wait=true", self.at), Some(&body), None).await?;
        }
        Ok(())
    }

    /// Write points (by id: the same item writes the same ids).
    pub async fn upsert(&self, points: &[PointStruct], wait: bool) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let body = json!({ "points": points });
        self.call("PUT", &format!("{}/points?wait={wait}", self.at), Some(&body), Some(60_000)).await.map(|_| ())
    }

    /// Every point of these items — before an item is written again (its children may be fewer now).
    pub async fn delete_items(&self, items: &[String]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let body = json!({ "filter": { "must": [{ "key": "item", "match": { "any": items } }] } });
        self.call("POST", &format!("{}/points/delete?wait=true", self.at), Some(&body), Some(60_000)).await.map(|_| ())
    }

    /// One kind of point of these items (`cue`), before that part is written again.
    pub async fn delete_parts(&self, items: &[String], part: &str) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let body = json!({ "filter": { "must": [{ "key": "item", "match": { "any": items } }, { "key": "part", "match": { "value": part } }] } });
        self.call("POST", &format!("{}/points/delete?wait=true", self.at), Some(&body), Some(60_000)).await.map(|_| ())
    }

    /// Change some payload fields of these items' points (status, a merged key's display fields).
    pub async fn set_payload(&self, items: &[String], payload: Value, key_only: bool) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let mut must = vec![json!({ "key": "item", "match": { "any": items } })];
        if key_only {
            must.push(json!({ "key": "part", "match": { "value": "key" } }));
        }
        let body = json!({ "payload": payload, "filter": { "must": must } });
        self.call("POST", &format!("{}/points/payload?wait=true", self.at), Some(&body), Some(60_000)).await.map(|_| ())
    }

    fn found(points: &Value) -> Vec<Found> {
        points
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| Found {
                item: value_string(&p["payload"]["item"]),
                score: p["score"].as_f64().unwrap_or(0.0),
                payload: if p["payload"].is_object() { p["payload"].clone() } else { json!({}) },
            })
            .collect()
    }

    /// The nearest key points to one vector (consolidation's "do I know this already?"), with their cosine.
    pub async fn neighbors(&self, vector: &[f32], filter: &Value, limit: usize) -> Result<Vec<Found>> {
        let mut f = filter.clone();
        push(&mut f, "must", json!({ "key": "part", "match": { "value": "key" } }));
        let body = json!({
            "query": vector, "using": "dense", "limit": limit, "params": rescore(), "filter": f,
            "with_payload": ["item", "title", "summary", "kind", "scope", "project", "status"],
        });
        let r = self.call("POST", &format!("{}/points/query", self.at), Some(&body), None).await?;
        Ok(Self::found(&r["points"]))
    }

    /// Each of these items' best cosine to one vector among its points of one part (key, child, cue). An item with
    /// no such point is not in the map.
    pub async fn best_per_item(
        &self,
        vector: &[f32],
        items: &[String],
        part: &str,
        timeout_ms: u64,
    ) -> Result<IndexMap<String, f64>> {
        if items.is_empty() {
            return Ok(IndexMap::new());
        }
        let body = json!({
            "query": vector, "using": "dense", "params": rescore(),
            "filter": { "must": [{ "key": "item", "match": { "any": items } }, { "key": "part", "match": { "value": part } }] },
            "group_by": "item", "group_size": 1, "limit": items.len(), "with_payload": false,
        });
        let r = self.call("POST", &format!("{}/points/query/groups", self.at), Some(&body), Some(timeout_ms)).await?;
        Ok(r["groups"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|g| (value_string(&g["id"]), g["hits"][0]["score"].as_f64().unwrap_or(0.0)))
            .collect())
    }

    /// The nearest points for several vectors in one call. `fields`: only these payload fields (and the item), for a
    /// caller that needs no more — a walk over many neighbours reads ids, not the bodies of long records.
    pub async fn nearest_many(
        &self,
        vectors: &[Vec<f32>],
        filter: &Value,
        limit: usize,
        timeout_ms: u64,
        fields: Option<&[&str]>,
    ) -> Result<Vec<Vec<Found>>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }
        let with_payload = match fields {
            Some(f) => {
                let mut v: Vec<&str> = Vec::new();
                for x in f.iter().copied().chain(["item"]) {
                    if !v.contains(&x) {
                        v.push(x);
                    }
                }
                json!(v)
            }
            None => json!(true),
        };
        let searches: Vec<Value> = vectors
            .iter()
            .map(|v| json!({ "query": v, "using": "dense", "limit": limit, "params": rescore(), "filter": filter, "with_payload": with_payload }))
            .collect();
        let body = json!({ "searches": searches });
        let r = self.call("POST", &format!("{}/points/query/batch", self.at), Some(&body), Some(timeout_ms)).await?;
        Ok(r.as_array().into_iter().flatten().map(|x| Self::found(&x["points"])).collect())
    }

    fn stored(points: &Value) -> Vec<Stored> {
        points
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| Stored {
                id: value_string(&p["id"]),
                payload: if p["payload"].is_object() { p["payload"].clone() } else { json!({}) },
                vector: p["vector"]["dense"]
                    .as_array()
                    .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()),
            })
            .collect()
    }

    /// Points within a filter, with their payload (and dense vector when asked) — at most `limit`.
    pub async fn scroll(
        &self,
        filter: &Value,
        limit: usize,
        with_vector: bool,
        timeout_ms: u64,
    ) -> Result<Vec<Stored>> {
        let body = json!({
            "filter": filter, "limit": limit, "with_payload": true,
            "with_vector": if with_vector { json!(["dense"]) } else { json!(false) },
        });
        let r = self.call("POST", &format!("{}/points/scroll", self.at), Some(&body), Some(timeout_ms)).await?;
        Ok(Self::stored(&r["points"]))
    }

    /// How many points (estimated), within a filter when given.
    pub async fn count(&self, filter: Option<&Value>) -> Result<u64> {
        let mut body = json!({ "exact": false });
        if let Some(f) = filter {
            body["filter"] = f.clone();
        }
        let r = self.call("POST", &format!("{}/points/count", self.at), Some(&body), None).await?;
        Ok(r["count"].as_u64().unwrap_or(0))
    }

    /// The points behind `ids`, with their payload — and their dense vector when asked.
    pub async fn retrieve(&self, ids: &[String], with_vector: bool) -> Result<Vec<Stored>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let body = json!({
            "ids": ids, "with_payload": true,
            "with_vector": if with_vector { json!(["dense"]) } else { json!(false) },
        });
        let r = self.call("POST", &format!("{}/points", self.at), Some(&body), None).await?;
        Ok(Self::stored(&r))
    }

    /// The best items for the query, fused — each with the point that matched best (its key, a chunk of its text, or a
    /// cue), so a caller can show the passage that matched. Each channel ranks up to `per` items; their scores are
    /// then fused (CombSUM, above).
    pub async fn fused_items(&self, q: &FusedQuery, timeout_ms: u64) -> Result<Vec<FusedHit>> {
        let limit = q.limit.unwrap_or(40);
        let per = q.per.unwrap_or(limit.max(50));
        let sparse = q.sparse.as_ref().filter(|s| !s.indices.is_empty());
        let dense_weight = (if sparse.is_some() { DENSE_SHARE } else { 1.0 }) / (q.dense.len().max(1) as f64);
        let cue = json!({ "key": "part", "match": { "value": "cue" } });
        let base = q.filter.clone().unwrap_or_else(|| json!({}));
        let mut cue_off = base.clone();
        push(&mut cue_off, "must_not", cue.clone());
        let mut cue_only = base;
        push(&mut cue_only, "must", cue);
        struct Channel {
            query: Value,
            using: &'static str,
            params: bool,
            weight: f64,
            filter: Option<Value>,
        }
        let mut channels: Vec<Channel> = Vec::new();
        for v in &q.dense {
            channels.push(Channel {
                query: json!(v),
                using: "dense",
                params: true,
                weight: dense_weight * (1.0 - CUE_SHARE),
                filter: Some(cue_off.clone()),
            });
            channels.push(Channel {
                query: json!(v),
                using: "dense",
                params: true,
                weight: dense_weight * CUE_SHARE,
                filter: Some(cue_only.clone()),
            });
        }
        if let Some(s) = sparse {
            channels.push(Channel {
                query: json!(s),
                using: "bm25",
                params: false,
                weight: 1.0 - DENSE_SHARE,
                filter: None,
            });
        }
        if channels.is_empty() {
            return Ok(vec![]);
        }
        let path = format!("{}/points/query/groups", self.at);
        let lists = futures::future::try_join_all(channels.iter().map(|c| {
            let mut body = serde_json::Map::new();
            body.insert("query".into(), c.query.clone());
            body.insert("using".into(), json!(c.using));
            if c.params {
                body.insert("params".into(), rescore());
            }
            if let Some(f) = c.filter.clone().or_else(|| q.filter.clone()) {
                body.insert("filter".into(), f);
            }
            body.insert("group_by".into(), json!("item"));
            body.insert("group_size".into(), json!(1));
            body.insert("limit".into(), json!(per));
            body.insert("with_payload".into(), json!(false));
            let body = Value::Object(body);
            let path = path.clone();
            async move { self.call("POST", &path, Some(&body), Some(timeout_ms)).await }
        }))
        .await?;
        // Per item: its fused score, its best rank in any channel, and the point it was found by there.
        let mut fused: IndexMap<String, Fusion> = IndexMap::new();
        for (c, r) in lists.iter().enumerate() {
            let groups: Vec<&Value> = r["groups"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
            let scores: Vec<f64> = groups.iter().map(|g| g["hits"][0]["score"].as_f64().unwrap_or(0.0)).collect();
            let lo = scores.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for (rank, g) in groups.iter().enumerate() {
                let item = value_string(&g["id"]);
                let raw = &g["hits"][0]["id"];
                let at = fused.entry(item).or_insert((0.0, f64::INFINITY, None));
                // A channel whose items all score alike (or that found one) counts each of them fully.
                at.0 += channels[c].weight * if hi > lo { (scores[rank] - lo) / (hi - lo) } else { 1.0 };
                // The matched passage: the point from the channel that ranked the item highest.
                let has_raw = !(raw.is_null() || raw.as_str() == Some("") || raw.as_i64() == Some(0));
                if (rank as f64) < at.1 && has_raw {
                    at.1 = rank as f64;
                    at.2 = Some(value_string(raw));
                }
            }
        }
        let mut out: Vec<(String, Fusion)> = fused.into_iter().collect();
        out.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out.into_iter().take(limit).map(|(item, (score, _, point))| FusedHit { item, score, point }).collect())
    }
}

/// An item's fused score, its best rank in any channel, and the point it was found by there.
type Fusion = (f64, f64, Option<String>);

/// `String(x)` for a payload value: a string as it is, anything else as JSON writes it (`""` for nothing).
pub(crate) fn value_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => crate::js::stringify(other),
    }
}

/// `{...f, key: [...f.key, x]}` on a filter.
pub(crate) fn push(filter: &mut Value, key: &str, x: Value) {
    if !filter.is_object() {
        *filter = json!({});
    }
    let list = filter[key].as_array().cloned().unwrap_or_default();
    let mut list = list;
    list.push(x);
    filter[key] = Value::Array(list);
}
