//! How memory reaches the index: points built, embedded in batches and written to Qdrant — both tiers.
//!
//!   - a record (a short-term entry): ONE key point, its start as the key, and its whole text in chunks: no model,
//!     searchable the moment it is added;
//!   - a long-term item: a key point (content title + ~150-word summary, BM25 over title, summary, body and files) and
//!     its ~96-word children (small-to-big: searched, never shown — the item is what comes back), its cues, and the
//!     keys of its concepts on the key point (the concept graph's links: graph.rs).

use std::path::Path;

use serde_json::{json, Map, Value};

use crate::embedder::{doc_text, EmbedFn};
use crate::episodes::episode_body;
use crate::error::{Error, Result};
use crate::js;
use crate::qdrant_client::{point_id, PointStruct, PointVectors, QdrantClient};
use crate::store::{MemoryItem, StmEntry};
use crate::text::{children, concept_key, cut, doc_sparse, first_words};

/// What writing to the index needs: the index and an embedder.
#[derive(Clone)]
pub struct WriterDeps {
    /// The index.
    pub client: QdrantClient,
    /// Already prompted texts → unit vectors.
    pub embed: EmbedFn,
}

/// Time for embedding this many texts: a 30 s floor, and more for a long batch (a long record is many chunks).
pub(crate) fn embed_budget(n: usize) -> u64 {
    (n as u64 * 150).max(30_000)
}

/// A short-term entry's item id in the index (its long-term episode, if consolidation keeps it, is `ep:<uid>`).
pub fn stm_item(uid: &str) -> String {
    format!("stm:{uid}")
}

/// The raw key of a captured request: what it asked and the start of what was done — ~150 words.
pub fn raw_key(e: &StmEntry) -> String {
    first_words(&format!("{}\n{}", e.request, e.reply), 150)
}

/// A captured request's own title, its first line: the conversation's title says where it was, not what it is (one
/// long conversation's requests would all read alike, and match alike).
pub fn stm_title(e: &StmEntry) -> String {
    let first = e.request.split('\n').map(js::trim).find(|l| !l.is_empty()).unwrap_or("");
    let t = cut(first, 100);
    if t.is_empty() {
        e.title.clone()
    } else {
        t
    }
}

/// A cue's point number: the k-th cue is -(k + 1) — below the key (0) and the children (1, 2, …), however many of
/// either there are, so the two never meet.
pub(crate) fn cue_point(k: usize) -> i64 {
    -(k as i64 + 1)
}

async fn embedded(d: &WriterDeps, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    let n = texts.len();
    let vecs = (d.embed)(texts, Some(embed_budget(n))).await?;
    if vecs.len() != n {
        return Err(Error::Embedding(format!("embeddings: {} vectors for {n} texts", vecs.len())));
    }
    Ok(vecs)
}

async fn upsert_all(client: &QdrantClient, points: &[PointStruct]) -> Result<()> {
    for part in points.chunks(256) {
        client.upsert(part, true).await?;
    }
    Ok(())
}

fn obj(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

/// Records into the index: the raw key (its start), and the whole body in chunks — every word of a record can be found.
pub async fn write_stm(d: &WriterDeps, entries: &[StmEntry]) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    let plan: Vec<(&StmEntry, String, Vec<String>)> = entries
        .iter()
        .map(|e| {
            let body = episode_body(e);
            let kids = children(&body);
            (e, body, kids)
        })
        .collect();
    let mut texts = Vec::new();
    for (e, _, kids) in &plan {
        let title = stm_title(e);
        texts.push(doc_text(&title, &raw_key(e)));
        texts.extend(kids.iter().map(|k| doc_text(&title, k)));
    }
    let vecs = embedded(d, texts).await?;
    let mut points = Vec::new();
    let mut at = 0;
    for (e, body, kids) in &plan {
        let item = stm_item(&e.uid);
        // The record's kind (`engine`) on every point: a kinds filter must find a record by any of its chunks.
        let common = json!({
            "item": item, "tier": "stm", "status": "active", "kind": "episode", "engine": e.engine, "scope": "project",
            "project": e.project, "session": e.session, "ts": js::ms_value(e.ts), "synthetic": e.synthetic,
        });
        let mut key = obj(common.clone());
        key.insert("part".into(), json!("key"));
        key.insert("cwd".into(), json!(e.cwd.clone().unwrap_or_default()));
        key.insert("where".into(), json!(e.title));
        key.insert("title".into(), json!(stm_title(e)));
        key.insert("summary".into(), json!(raw_key(e)));
        key.insert("body".into(), json!(body));
        key.insert("files".into(), json!(e.files));
        key.insert("importance".into(), json!(0));
        if let Some(r) = &e.record {
            key.insert("record".into(), serde_json::to_value(r).map_err(Error::json)?);
        }
        points.push(PointStruct {
            id: point_id(&item, 0),
            vector: PointVectors {
                dense: vecs[at].clone(),
                bm25: Some(doc_sparse(&format!("{}\n{body}", stm_title(e)))),
            },
            payload: Value::Object(key),
        });
        for k in 0..kids.len() {
            let mut child = obj(common.clone());
            child.insert("part".into(), json!("child"));
            points.push(PointStruct {
                id: point_id(&item, k as i64 + 1),
                vector: PointVectors { dense: vecs[at + 1 + k].clone(), bm25: None },
                payload: Value::Object(child),
            });
        }
        at += 1 + kids.len();
    }
    d.client.delete_items(&entries.iter().map(|e| stm_item(&e.uid)).collect::<Vec<_>>()).await?;
    upsert_all(&d.client, &points).await
}

/// What every point of a long-term item carries: what recall and the filters select on — for an observation, the
/// time it happened too (search's `from`/`to`): from its date, or from when what it tells began to when it ended.
fn ltm_payload(it: &MemoryItem) -> Map<String, Value> {
    let spans: Vec<DateSpan> =
        [&it.event_at, &it.start_at, &it.end_at].into_iter().filter_map(|d| d.as_deref().and_then(date_span)).collect();
    let mut m = obj(json!({
        "item": it.uid, "tier": "ltm", "status": it.status, "kind": it.kind, "scope": it.scope, "project": it.project,
        "session": it.session.clone().unwrap_or_default(), "ts": js::ms_value(it.ts), "synthetic": false,
    }));
    if !spans.is_empty() {
        let from = spans.iter().map(|s| s.from).fold(f64::INFINITY, f64::min);
        let to = spans.iter().map(|s| s.to).fold(f64::NEG_INFINITY, f64::max);
        m.insert("eventFrom".into(), js::ms_value(from));
        m.insert("eventTo".into(), js::ms_value(to));
    }
    m
}

fn part(common: &Map<String, Value>, part: &str) -> Value {
    let mut m = common.clone();
    m.insert("part".into(), json!(part));
    Value::Object(m)
}

fn nonempty(s: &Option<String>) -> Option<&String> {
    s.as_ref().filter(|x| !x.is_empty())
}

/// Only the cues of these items, written again (the key and the children stay as they are).
pub async fn write_cues(d: &WriterDeps, items: &[MemoryItem]) -> Result<()> {
    let live: Vec<&MemoryItem> = items.iter().filter(|it| it.status == "active").collect();
    let uids: Vec<String> = live.iter().map(|it| it.uid.clone()).collect();
    let texts: Vec<String> = live.iter().flat_map(|it| it.cue_list().iter().map(|c| doc_text(&it.title, c))).collect();
    if texts.is_empty() {
        return d.client.delete_parts(&uids, "cue").await;
    }
    let vecs = embedded(d, texts).await?;
    let mut points = Vec::new();
    let mut n = 0;
    for it in &live {
        let common = ltm_payload(it);
        for k in 0..it.cue_list().len() {
            points.push(PointStruct {
                id: point_id(&it.uid, cue_point(k)),
                vector: PointVectors { dense: vecs[n].clone(), bm25: None },
                payload: part(&common, "cue"),
            });
            n += 1;
        }
    }
    d.client.delete_parts(&uids, "cue").await?;
    upsert_all(&d.client, &points).await
}

fn basename(f: &str) -> String {
    Path::new(f).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| f.to_string())
}

/// Embed and write long-term items: their old points out, the key point, the children and the cues in. Cues are
/// embedded as documents titled by the item (how they were measured); the item, not the cue, is what comes back.
pub async fn write_ltm(d: &WriterDeps, items: &[MemoryItem]) -> Result<()> {
    let live: Vec<&MemoryItem> = items.iter().filter(|it| it.status == "active").collect();
    let all_uids: Vec<String> = items.iter().map(|it| it.uid.clone()).collect();
    if live.is_empty() {
        return d.client.delete_items(&all_uids).await;
    }
    let mut texts = Vec::new();
    let mut plan: Vec<(&MemoryItem, Vec<String>, usize)> = Vec::new();
    for it in &live {
        // Every chunk of the body — and of a summary too long for one key vector (the embedder reads its start).
        let mut kids = children(&it.body);
        let summary_kids = children(&it.summary);
        if summary_kids.len() >= LONG_SUMMARY_CHUNKS {
            kids.extend(summary_kids);
        }
        plan.push((it, kids.clone(), texts.len()));
        texts.push(doc_text(&it.title, &it.summary));
        texts.extend(kids.iter().map(|k| doc_text(&it.title, k)));
        texts.extend(it.cue_list().iter().map(|c| doc_text(&it.title, c)));
    }
    let vecs = embedded(d, texts).await?;
    let mut points = Vec::new();
    for (it, kids, at) in &plan {
        let common = ltm_payload(it);
        let mut key = common.clone();
        key.insert("part".into(), json!("key"));
        key.insert("engine".into(), json!(it.engine));
        key.insert("cwd".into(), json!(it.cwd.clone().unwrap_or_default()));
        key.insert("where".into(), json!(it.where_));
        key.insert("title".into(), json!(it.title));
        key.insert("summary".into(), json!(it.summary));
        key.insert("body".into(), json!(it.body));
        key.insert("files".into(), json!(it.files));
        key.insert("importance".into(), js::ms_value(it.importance));
        key.insert("appliesWhen".into(), json!(it.applies_when));
        key.insert("related".into(), json!(it.related_list()));
        key.insert("concepts".into(), json!(concept_keys(it.concept_list())));
        for (name, v) in [
            ("eventAt", &it.event_at),
            ("startAt", &it.start_at),
            ("endAt", &it.end_at),
            ("phase", &it.phase),
            ("about", &it.about),
        ] {
            if let Some(v) = nonempty(v) {
                key.insert(name.into(), json!(v));
            }
        }
        let files: Vec<String> = it.files.iter().map(|f| basename(f)).collect();
        points.push(PointStruct {
            id: point_id(&it.uid, 0),
            vector: PointVectors {
                dense: vecs[*at].clone(),
                bm25: Some(doc_sparse(&format!("{}\n{}\n{}\n{}", it.title, it.summary, it.body, files.join(" ")))),
            },
            payload: Value::Object(key),
        });
        for k in 0..kids.len() {
            points.push(PointStruct {
                id: point_id(&it.uid, k as i64 + 1),
                vector: PointVectors { dense: vecs[at + 1 + k].clone(), bm25: None },
                payload: part(&common, "child"),
            });
        }
        for k in 0..it.cue_list().len() {
            points.push(PointStruct {
                id: point_id(&it.uid, cue_point(k)),
                vector: PointVectors { dense: vecs[at + 1 + kids.len() + k].clone(), bm25: None },
                payload: part(&common, "cue"),
            });
        }
    }
    // Remote embedding can fail: the old searchable item stays until every replacement vector exists.
    d.client.delete_items(&all_uids).await?;
    upsert_all(&d.client, &points).await
}

/// Only the concepts of these items, written again (their keys on the key point).
pub async fn write_concepts(d: &WriterDeps, items: &[MemoryItem]) -> Result<()> {
    for it in items.iter().filter(|x| x.status == "active") {
        d.client
            .set_payload(std::slice::from_ref(&it.uid), json!({ "concepts": concept_keys(it.concept_list()) }), true)
            .await?;
    }
    Ok(())
}

/// The time a date stands for, as its first and last millisecond.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DateSpan {
    /// The first millisecond.
    pub from: f64,
    /// The last millisecond.
    pub to: f64,
}

/// The time a date stands for: a day, a month or a year (`YYYY-MM-DD`, `YYYY-MM` or `YYYY`) — `None` for anything
/// else, a month or day that does not exist too (2024-02-30 is not 1 March).
pub fn date_span(d: &str) -> Option<DateSpan> {
    let b = d.as_bytes();
    let num = |a: usize, n: usize| -> Option<i64> {
        b.get(a..a + n)
            .filter(|s| s.iter().all(u8::is_ascii_digit))?
            .iter()
            .try_fold(0i64, |v, c| Some(v * 10 + i64::from(c - b'0')))
    };
    let (y, mo, dd) = match b.len() {
        4 => (num(0, 4)?, None, None),
        7 if b[4] == b'-' => (num(0, 4)?, Some(num(5, 2)? - 1), None),
        10 if b[4] == b'-' && b[7] == b'-' => (num(0, 4)?, Some(num(5, 2)? - 1), Some(num(8, 2)?)),
        _ => return None,
    };
    if let Some(m) = mo {
        if !(0..=11).contains(&m) {
            return None;
        }
        if let Some(day) = dd {
            if day < 1 || js::utc_of(js::date_utc(y, m, day, 0, 0, 0, 0)).month - 1 != m {
                return None;
            }
        }
    }
    Some(match (mo, dd) {
        (None, _) => {
            DateSpan { from: js::date_utc(y, 0, 1, 0, 0, 0, 0), to: js::date_utc(y + 1, 0, 1, 0, 0, 0, 0) - 1.0 }
        }
        (Some(m), None) => {
            DateSpan { from: js::date_utc(y, m, 1, 0, 0, 0, 0), to: js::date_utc(y, m + 1, 1, 0, 0, 0, 0) - 1.0 }
        }
        (Some(m), Some(day)) => {
            DateSpan { from: js::date_utc(y, m, day, 0, 0, 0, 0), to: js::date_utc(y, m, day + 1, 0, 0, 0, 0) - 1.0 }
        }
    })
}

/// A summary this many chunks long (~380 words) is also chunked: its key vector reads only its start.
pub const LONG_SUMMARY_CHUNKS: usize = 5;

/// The keys of an item's concepts, one per concept.
pub fn concept_keys(names: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for n in names {
        let k = concept_key(n);
        if !k.is_empty() && !out.contains(&k) {
            out.push(k);
        }
    }
    out
}
