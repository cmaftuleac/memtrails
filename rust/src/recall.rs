//! Recall: what a search returns, an item opened whole, and exploring around a topic.
//!
//! `memory_search` embeds the query and searches dense vectors (each item's key, its chunks and its cues) and a keyword
//! vector at once, fused and grouped by item; with dates, what happened within them first. The best 30 can be
//! reordered (`rerank`: word by word on this machine, words.rs, or any cross-encoder) and fused with the search's own
//! order. `memory_open` reads one item whole. `memory_explore` walks the concept graph (graph.rs) from a topic or from
//! items to what is related but not found directly.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{json, Value};

pub use crate::recall_types::RecalledItem;

use crate::embedder::{query_text, EmbedFn};
use crate::error::{Error, Result};
use crate::graph::{explore, GraphQuery};
use crate::js;
use crate::qdrant_client::{point_id, push, value_string, FusedHit, FusedQuery, QdrantClient};
use crate::recall_log::RecallEvent;
use crate::records::collection_key;
use crate::reranker::{reranked_order, RERANK_TOP};
use crate::store::RecordRef;
use crate::text::{children, cut, flat, query_sparse};
use crate::writer::{date_span, LONG_SUMMARY_CHUNKS};
use crate::RerankFn;

/// Told what was shown (`false`) or opened (`true`): the sleep pass's "use".
pub type UsedFn = Arc<dyn Fn(&[String], bool) + Send + Sync>;
/// Told what each search showed, for what, and what was opened.
pub type RecordFn = Arc<dyn Fn(&RecallEvent) + Send + Sync>;

/// What recall needs.
#[derive(Clone)]
pub struct RecallDeps {
    /// The index.
    pub client: QdrantClient,
    /// Already prompted texts → unit vectors.
    pub embed: EmbedFn,
    /// What was shown / opened.
    pub used: Option<UsedFn>,
    /// What each search showed, for what, and what was opened (recall.jsonl).
    pub record: Option<RecordFn>,
    /// Each document's relevance to the query, or none — the search then keeps its own order.
    pub rerank: Option<RerankFn>,
}

/// A search: best first, in every collection or the ones named — and with dates, what happened within them first,
/// then the rest. Never only what lies within them: on 64 held-out questions that name a time, the answer reached the
/// reader whole for 54 searched without dates, 53 with what lies within them first, 44 with only that.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchQuery {
    /// What you need to know, in plain words.
    pub query: String,
    /// Only these collections (records' `collection`). Default: all.
    pub collections: Option<Vec<String>>,
    /// Only these kinds (records' `kind`, or what consolidation learned: `observation`, `fact`, …).
    pub kinds: Option<Vec<String>>,
    /// How many (default 8, at most 200; the list goes up to twice as deep, the best first).
    pub limit: Option<usize>,
    /// `YYYY-MM-DD`, `YYYY-MM` or `YYYY`: what happened from then first.
    pub from: Option<String>,
    /// `YYYY-MM-DD`, `YYYY-MM` or `YYYY`: what happened until then first.
    pub to: Option<String>,
}

impl SearchQuery {
    /// A search for these words, everything else by default.
    pub fn new(query: impl Into<String>) -> Self {
        SearchQuery { query: query.into(), ..Default::default() }
    }
}

/// The most a search returns.
pub const MAX_SEARCH: usize = 200;
/// A captured request's title shorter than this says nothing on its own.
const SHORT_TITLE: usize = 24;
/// How long a search waits for its reranker before keeping its own order.
const RERANK_WAIT_MS: u64 = 3000;
/// Up to this many asked for, the search ranks one fused list of twice as many; past it — a question that gathers —
/// the dated observations and everything else are two searches of that many each, interleaved, so neither crowds
/// the other out of the room.
const SINGLE_LIST_UP_TO: usize = 20;

/// What search may show: active items, never a throwaway run, never superseded or archived knowledge. `kinds`: a
/// record's kind is its `engine` (every point of a record has `kind: episode`), a learned item's its `kind`.
pub(crate) fn base_filter(kinds: Option<&[String]>) -> Value {
    let mut must = vec![
        json!({ "key": "synthetic", "match": { "value": false } }),
        json!({ "key": "status", "match": { "value": "active" } }),
    ];
    if let Some(k) = kinds.filter(|k| !k.is_empty()) {
        must.push(json!({ "should": [
            { "must": [{ "key": "tier", "match": { "value": "stm" } }, { "key": "engine", "match": { "any": k } }] },
            { "must": [{ "key": "tier", "match": { "value": "ltm" } }, { "key": "kind", "match": { "any": k } }] },
        ] }));
    }
    json!({ "must": must })
}

fn opt_string(p: &Value, key: &str) -> Option<String> {
    match &p[key] {
        Value::Null | Value::Bool(false) => None,
        Value::String(s) if s.is_empty() => None,
        v => Some(value_string(v)),
    }
}

/// The items behind these ids, from their key points, in the order asked (those not there left out).
pub(crate) async fn items(d: &RecallDeps, uids: &[String]) -> Result<Vec<RecalledItem>> {
    let got = d.client.retrieve(&uids.iter().map(|u| point_id(u, 0)).collect::<Vec<_>>(), false).await?;
    let by_item: IndexMap<String, Value> =
        got.into_iter().map(|p| (value_string(&p.payload["item"]), p.payload)).collect();
    Ok(uids
        .iter()
        .filter_map(|uid| {
            let p = by_item.get(uid)?;
            let s = |k: &str| value_string(&p[k]);
            let kind = if p["tier"] == "stm" {
                if p["engine"].is_null() {
                    "record".to_string()
                } else {
                    s("engine")
                }
            } else {
                s("kind")
            };
            Some(RecalledItem {
                id: uid.clone(),
                title: s("title"),
                summary: s("summary"),
                where_: s("where"),
                project: s("project"),
                engine: s("engine"),
                ts: p["ts"].as_f64().unwrap_or(0.0),
                kind,
                record: if p["record"].is_object() {
                    serde_json::from_value::<RecordRef>(p["record"].clone()).ok()
                } else {
                    None
                },
                event_at: opt_string(p, "eventAt"),
                start_at: opt_string(p, "startAt"),
                end_at: opt_string(p, "endAt"),
                match_: None,
                score: 0.0,
            })
        })
        .collect())
}

/// What happened within the dates (either end may be open): an observation by when it happened — its time
/// overlapping the range — anything else by when it was written. The caller reads the dates out of a question;
/// nothing here parses words.
fn within_dates(f: &Value, from: Option<&str>, to: Option<&str>) -> Value {
    let lo = from.and_then(date_span).map_or(0.0, |s| s.from);
    let hi = to.and_then(date_span).map_or(9_007_199_254_740_991.0, |s| s.to);
    let mut out = f.clone();
    push(
        &mut out,
        "must",
        json!({ "should": [
            { "must": [{ "key": "eventFrom", "range": { "lte": js::ms_value(hi) } }, { "key": "eventTo", "range": { "gte": js::ms_value(lo) } }] },
            { "must": [{ "is_empty": { "key": "eventFrom" } }, { "key": "ts", "range": { "gte": js::ms_value(lo), "lte": js::ms_value(hi) } }] },
        ] }),
    );
    out
}

/// The best `RERANK_TOP` in the reranker's order fused with their own; with dates, the hits within them stay first —
/// each block is reordered on its own.
async fn reranked(d: &RecallDeps, query: &str, list: Vec<RecalledItem>, dated: &HashSet<String>) -> Vec<RecalledItem> {
    let Some(rerank) = &d.rerank else { return list };
    if list.len() < 2 {
        return list;
    }
    let top: Vec<RecalledItem> = list.iter().take(RERANK_TOP).cloned().collect();
    let docs = top.iter().map(|it| format!("{}\n{}", it.title, it.summary)).collect();
    let Some(scores) = rerank(query.to_string(), docs, RERANK_WAIT_MS).await else { return list };
    if scores.len() != top.len() {
        return list;
    }
    let score: IndexMap<&str, f64> = top.iter().zip(&scores).map(|(it, s)| (it.id.as_str(), *s)).collect();
    let order = |xs: Vec<RecalledItem>| {
        let s: Vec<f64> = xs.iter().map(|it| score.get(it.id.as_str()).copied().unwrap_or(f64::NAN)).collect();
        reranked_order(&xs, Some(&s))
    };
    let (inside, outside): (Vec<RecalledItem>, Vec<RecalledItem>) =
        top.iter().cloned().partition(|it| dated.contains(&it.id));
    let mut out = order(inside);
    out.extend(order(outside));
    out.extend(list.into_iter().skip(RERANK_TOP));
    out
}

/// A search, best first.
pub async fn memory_search(d: &RecallDeps, q: &SearchQuery) -> Result<Vec<RecalledItem>> {
    // A date that is not one would leave its end open: the search would quietly ignore the range it was asked for.
    for (end, at) in [("from", &q.from), ("to", &q.to)] {
        if let Some(at) = at.as_deref().filter(|a| !a.is_empty()) {
            if date_span(at).is_none() {
                return Err(Error::InvalidInput(format!(
                    "memory search: {end} \"{at}\" is not a date (YYYY-MM-DD, YYYY-MM or YYYY)"
                )));
            }
        }
    }
    let projects: Option<Vec<String>> =
        q.collections.as_ref().filter(|c| !c.is_empty()).map(|c| c.iter().map(|x| collection_key(Some(x))).collect());
    // Up to 200: a question that gathers needs the facts of many threads at once, a dated observation a line.
    let limit = q.limit.unwrap_or(8).clamp(1, MAX_SEARCH);
    let vecs = (d.embed)(vec![query_text(&q.query)], None).await?;
    let mut scoped = base_filter(q.kinds.as_deref());
    if let Some(p) = &projects {
        push(&mut scoped, "must", json!({ "key": "project", "match": { "any": p } }));
    }
    let sparse = query_sparse(&[q.query.as_str()]);
    let observations = json!({ "key": "kind", "match": { "value": "observation" } });
    let ranked = |filter: Value| {
        let (vecs, sparse, observations) = (vecs.clone(), sparse.clone(), observations.clone());
        async move {
            let fq = |filter: Value, limit: usize| FusedQuery {
                dense: vecs.clone(),
                sparse: Some(sparse.clone()),
                filter: Some(filter),
                per: None,
                limit: Some(limit),
            };
            // A search for a few: one fused list over every kind, twice as deep as asked. Interleaving the kinds by
            // rank cost the top of the list; one list only as deep as asked lost the bottom.
            if limit <= SINGLE_LIST_UP_TO {
                return d.client.fused_items(&fq(filter, 2 * limit), 3000).await;
            }
            let mut obs_f = filter.clone();
            push(&mut obs_f, "must", observations.clone());
            let mut rest_f = filter;
            push(&mut rest_f, "must_not", observations);
            let (obs_q, rest_q) = (fq(obs_f, limit), fq(rest_f, limit));
            let (obs, rest) =
                tokio::try_join!(d.client.fused_items(&obs_q, 3000), d.client.fused_items(&rest_q, 3000))?;
            // Interleaved by rank, so each kind keeps its order and neither crowds out the other.
            let mut out = Vec::new();
            for i in 0..obs.len().max(rest.len()) {
                out.extend(rest.get(i).cloned());
                out.extend(obs.get(i).cloned());
            }
            Ok::<Vec<FusedHit>, Error>(out)
        }
    };
    let dated_search =
        q.from.as_deref().is_some_and(|x| !x.is_empty()) || q.to.as_deref().is_some_and(|x| !x.is_empty());
    let (in_dates, open) = tokio::try_join!(
        async {
            if dated_search {
                ranked(within_dates(&scoped, q.from.as_deref(), q.to.as_deref())).await
            } else {
                Ok(vec![])
            }
        },
        ranked(scoped.clone())
    )?;
    let dated: HashSet<String> = in_dates.iter().map(|h| h.item.clone()).collect();
    let cap = open.len().max(in_dates.len());
    let merged: Vec<FusedHit> =
        in_dates.into_iter().chain(open.into_iter().filter(|h| !dated.contains(&h.item))).take(cap).collect();
    // A kept episode and its captured request hold the same text: the one found first stands for both.
    let mut seen = HashSet::new();
    let hits: Vec<FusedHit> = merged
        .into_iter()
        .filter(|h| {
            let request = h.item.strip_prefix("ep:").or_else(|| h.item.strip_prefix("stm:"));
            match request {
                None => true,
                Some(r) => seen.insert(r.to_string()),
            }
        })
        .collect();
    let found = items(d, &hits.iter().map(|h| h.item.clone()).collect::<Vec<_>>()).await?;
    let list = reranked(d, &q.query, found, &dated).await;
    let score: IndexMap<&str, f64> = hits.iter().map(|h| (h.item.as_str(), h.score)).collect();
    let matched = passages(d, &hits).await?;
    // A record titled by its first line, when that line is as short as "continue" or "ok", says nothing of what was
    // found: it shows the start of its key instead, which is what matched.
    let out: Vec<RecalledItem> = list
        .into_iter()
        .map(|mut it| {
            if it.id.starts_with("stm:") && js::len16(&it.title) < SHORT_TITLE {
                it.title = cut(&js::collapse_spaces(&it.summary), 100);
            }
            it.score = score.get(it.id.as_str()).copied().unwrap_or(0.0);
            if let Some(m) = matched.get(&it.id) {
                it.match_ = Some(m.clone());
            }
            it
        })
        .collect();
    if let Some(record) = &d.record {
        record(&RecallEvent::Search {
            project: projects.map(|p| p.join(",")),
            query: q.query.clone(),
            shown: out.iter().map(|it| it.id.clone()).collect(),
        });
    }
    Ok(out)
}

/// For each hit whose best point was a chunk of its text, that chunk — found again from the text (a record's whole
/// body is chunked; an item's body, and a summary too long for one key vector) — unless the key starts with it.
async fn passages(d: &RecallDeps, hits: &[FusedHit]) -> Result<IndexMap<String, String>> {
    let chunked: Vec<&FusedHit> =
        hits.iter().filter(|h| h.point.as_ref().is_some_and(|p| *p != point_id(&h.item, 0))).collect();
    if chunked.is_empty() {
        return Ok(IndexMap::new());
    }
    let got = d.client.retrieve(&chunked.iter().map(|h| point_id(&h.item, 0)).collect::<Vec<_>>(), false).await?;
    let by_item: IndexMap<String, Value> =
        got.into_iter().map(|p| (value_string(&p.payload["item"]), p.payload)).collect();
    let mut out = IndexMap::new();
    for h in chunked {
        let Some(p) = by_item.get(&h.item) else { continue };
        let body = value_string(&p["body"]);
        let summary = value_string(&p["summary"]);
        let mut kids = children(&body);
        if p["tier"] != "stm" {
            let more = children(&summary);
            if more.len() >= LONG_SUMMARY_CHUNKS {
                kids.extend(more);
            }
        }
        // A chunk that is the key's own start is not a passage found elsewhere: a short record's only chunk is its
        // whole key, the two points tie, and either may come back first.
        if let Some(k) = (0..kids.len()).find(|i| Some(point_id(&h.item, *i as i64 + 1)) == h.point) {
            if !flat(&summary).starts_with(kids[k].as_str()) {
                out.insert(h.item.clone(), kids[k].clone());
            }
        }
    }
    Ok(out)
}

/// An agent exploring: from a topic or from items.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExploreQuery {
    /// What to explore, in plain words.
    pub topic: Option<String>,
    /// Items to explore from (memory ids).
    pub from: Option<Vec<String>>,
    /// Only these collections.
    pub collections: Option<Vec<String>>,
    /// How many related items (default 10, at most 20).
    pub limit: Option<usize>,
}

/// An item reached through the graph, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct RelatedItem {
    /// The item.
    pub item: RecalledItem,
    /// How it was reached: the concepts it shares, a link, or nearness.
    pub via: String,
}

/// What exploring found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Explored {
    /// What search found for the topic, best first.
    pub found: Vec<RecalledItem>,
    /// Reached through the graph and not found by search, best first, with the way it was reached.
    pub related: Vec<RelatedItem>,
    /// The concepts around it: where to look next.
    pub concepts: Vec<String>,
}

/// From a topic (search finds the first items; the topic's concepts start the walk too) or from items, what the
/// concept graph reaches that search did not, and the concepts around it.
pub async fn memory_explore(d: &RecallDeps, q: &ExploreQuery) -> Result<Explored> {
    let projects: Option<Vec<String>> =
        q.collections.as_ref().filter(|c| !c.is_empty()).map(|c| c.iter().map(|x| collection_key(Some(x))).collect());
    let mut filter = base_filter(None);
    if let Some(p) = &projects {
        push(&mut filter, "must", json!({ "key": "project", "match": { "any": p } }));
    }
    let r = explore(
        &d.client,
        &d.embed,
        &GraphQuery {
            topic: q.topic.clone(),
            from: q.from.clone(),
            filter,
            projects,
            limit: q.limit.unwrap_or(10).clamp(1, 20),
        },
    )
    .await?;
    let first: Vec<String> = r.found.iter().take(8).cloned().collect();
    let reached: Vec<String> = r.related.iter().map(|x| x.id.clone()).collect();
    let (found, related) = tokio::try_join!(items(d, &first), items(d, &reached))?;
    let how: IndexMap<&str, &crate::graph::Reached> = r.related.iter().map(|x| (x.id.as_str(), x)).collect();
    Ok(Explored {
        found,
        related: related
            .into_iter()
            .filter_map(|mut it| {
                let h = how.get(it.id.as_str())?;
                it.score = h.score;
                Some(RelatedItem { via: h.via.clone(), item: it })
            })
            .collect(),
        concepts: r.concepts.into_iter().map(|c| c.0).collect(),
    })
}

/// One item whole (its key point's payload), or `None`.
pub async fn memory_open(d: &RecallDeps, id: &str) -> Result<Option<Value>> {
    let got = d.client.retrieve(&[point_id(id, 0)], false).await?;
    let p = got.into_iter().next().map(|p| p.payload);
    if p.is_some() {
        if let Some(used) = &d.used {
            used(&[id.to_string()], true);
        }
        if let Some(record) = &d.record {
            record(&RecallEvent::Open { item: id.to_string() });
        }
    }
    Ok(p)
}
