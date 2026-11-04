//! The concept graph — memory's associative index (HippoRAG's hippocampal index), for an agent EXPLORING a topic or
//! what it found, rather than for a search's top few (as a ranker feature it added ~1 point; for reaching what search
//! does not, it is where the gain is).
//!
//! Nodes are long-term items and the concepts they are about (consolidation writes 3–12 per item), a concept being its
//! spelling key (`concept_key`). Edges: item ↔ concept, weighted by how specific the concept is (1 / log(2 + the items
//! that share it)); item ↔ item: `related` and each seed's nearest items by key vector, two hops out. Concepts are NOT
//! embedded: measured, embedding them found the same needed items for six embeddings an item more. It lives in Qdrant —
//! the concept keys on items' key points, indexed — so it grows with memory without a copy in RAM: `explore` reads the
//! neighbourhood of its seeds and walks it with Personalized PageRank (restart 0.5), within recall's walls.

use indexmap::{IndexMap, IndexSet};
use serde_json::{json, Value};

use crate::embedder::{query_text, EmbedFn};
use crate::error::Result;
use crate::qdrant_client::{point_id, push, value_string, FusedQuery, QdrantClient, Stored};
use crate::text::{concept_key, fold, query_sparse};

/// A seed's nearest items by key vector (from cos 0.5: a small memory's "nearest" can be anything), the walk's search
/// hits used as seeds, the items read through concepts.
const KNN: usize = 10;
const KNN_NEAR: f64 = 0.5;
const SEEDS: usize = 5;
const FOUND: usize = 20;
const POSTINGS: usize = 400;
/// A `related` link's weight, beside item ↔ concept's 1/log(2 + df).
const RELATED: f64 = 1.0;
const RESTART: f64 = 0.5;
const STEPS: usize = 20;

fn strings(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

/// Where a walk starts and what it may show.
#[derive(Debug, Clone)]
pub struct GraphQuery {
    /// What to explore, in plain words: search finds the first items, and the topic's concepts start the walk too.
    pub topic: Option<String>,
    /// Items to explore from (memory ids).
    pub from: Option<Vec<String>>,
    /// Recall's walls: what may be shown (active, real), and the collections when the search names some.
    pub filter: Value,
    /// The collections named, as keys.
    pub projects: Option<Vec<String>>,
    /// How many related items.
    pub limit: usize,
}

/// An item the walk reached.
#[derive(Debug, Clone, PartialEq)]
pub struct Reached {
    /// The item.
    pub id: String,
    /// Its share of the walk.
    pub score: f64,
    /// How it was reached.
    pub via: String,
}

/// What a walk found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Walked {
    /// What search found for the topic, best first.
    pub found: Vec<String>,
    /// Reached and not among `found` or the seeds, best first.
    pub related: Vec<Reached>,
    /// The concepts the walk spent most time on, with their share.
    pub concepts: Vec<(String, f64)>,
}

/// A small weighted undirected graph, walked by Personalized PageRank.
#[derive(Default)]
struct Walk {
    ids: IndexMap<String, usize>,
    adj: Vec<IndexMap<usize, f64>>,
}

impl Walk {
    fn node(&mut self, name: &str) -> usize {
        if let Some(i) = self.ids.get(name) {
            return *i;
        }
        let i = self.adj.len();
        self.ids.insert(name.to_string(), i);
        self.adj.push(IndexMap::new());
        i
    }

    fn link(&mut self, a: &str, b: &str, w: f64) {
        if a == b || w.is_nan() || w <= 0.0 {
            return;
        }
        let i = self.node(a);
        let j = self.node(b);
        let wij = self.adj[i].get(&j).copied().unwrap_or(0.0).max(w);
        self.adj[i].insert(j, wij);
        let wji = self.adj[j].get(&i).copied().unwrap_or(0.0).max(w);
        self.adj[j].insert(i, wji);
    }

    /// Each node's share of a walk that starts again at `seeds` (by weight) half the time.
    fn ppr(&self, seeds: &IndexMap<String, f64>) -> IndexMap<String, f64> {
        let n = self.adj.len();
        let mut e = vec![0.0f64; n];
        let mut total = 0.0;
        for (name, w) in seeds {
            if let Some(&i) = self.ids.get(name) {
                e[i] += w;
                total += w;
            }
        }
        if total == 0.0 {
            return IndexMap::new();
        }
        for x in &mut e {
            *x /= total;
        }
        let deg: Vec<f64> = self.adj.iter().map(|m| m.values().sum()).collect();
        let mut x = e.clone();
        for _ in 0..STEPS {
            let mut next = vec![0.0f64; n];
            for i in 0..n {
                if x[i] == 0.0 || deg[i] == 0.0 {
                    continue;
                }
                for (&j, &w) in &self.adj[i] {
                    next[j] += (1.0 - RESTART) * x[i] * (w / deg[i]);
                }
            }
            for i in 0..n {
                next[i] += RESTART * e[i];
            }
            x = next;
        }
        self.ids.iter().map(|(name, &i)| (name.clone(), x[i])).collect()
    }
}

fn desc(a: f64, b: f64) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

/// Walk the graph from a topic and/or items: what is related but not found directly, and the concepts around it.
pub async fn explore(client: &QdrantClient, embed: &EmbedFn, q: &GraphQuery) -> Result<Walked> {
    let key_only = |extra: Vec<Value>| -> Value {
        let mut f = q.filter.clone();
        push(&mut f, "must", json!({ "key": "tier", "match": { "value": "ltm" } }));
        push(&mut f, "must", json!({ "key": "part", "match": { "value": "key" } }));
        for x in extra {
            push(&mut f, "must", x);
        }
        f
    };
    let shown = |p: &Value| -> bool {
        p["tier"] == "ltm"
            && p["status"] == "active"
            && p["synthetic"] == false
            && q.projects.as_ref().is_none_or(|ps| ps.is_empty() || ps.contains(&value_string(&p["project"])))
    };
    let mut seed_items: IndexMap<String, f64> = IndexMap::new();
    let mut seed_concepts: IndexMap<String, f64> = IndexMap::new();
    let mut found: Vec<String> = Vec::new();
    if let Some(topic) = q.topic.as_deref().filter(|t| !crate::js::trim(t).is_empty()) {
        let qv = embed(vec![query_text(topic)], None).await?;
        let hits = client
            .fused_items(
                &FusedQuery {
                    dense: qv,
                    sparse: Some(query_sparse(&[topic])),
                    filter: Some(q.filter.clone()),
                    per: None,
                    limit: Some(FOUND),
                },
                3000,
            )
            .await?;
        found = hits.into_iter().map(|h| h.item).collect();
        for u in found.iter().take(SEEDS) {
            seed_items.insert(u.clone(), 1.0);
        }
        // The topic's own concepts: the topic named as one, and those of what search found that the topic names.
        let t = concept_key(topic);
        if !t.is_empty() {
            seed_concepts.insert(t, 1.0);
        }
        let said = fold(topic);
        let ids: Vec<String> = found.iter().take(SEEDS).map(|u| point_id(u, 0)).collect();
        for p in client.retrieve(&ids, false).await? {
            for k in strings(&p.payload["concepts"]) {
                if crate::js::len16(&k) > 2 && said.contains(&k) {
                    seed_concepts.insert(k, 1.0);
                }
            }
        }
    }
    for u in q.from.iter().flatten() {
        seed_items.insert(u.clone(), 1.0);
    }
    if seed_items.is_empty() && seed_concepts.is_empty() {
        return Ok(Walked { found, ..Default::default() });
    }

    let mut g = Walk::default();
    let mut items: IndexMap<String, Value> = IndexMap::new();
    // Hop 1: the seed items, their concepts and links, and their nearest items by key vector.
    let seeds: Vec<Stored> = client
        .retrieve(&seed_items.keys().map(|u| point_id(u, 0)).collect::<Vec<_>>(), true)
        .await?
        .into_iter()
        .filter(|p| p.vector.is_some())
        .collect();
    for p in &seeds {
        items.insert(value_string(&p.payload["item"]), p.payload.clone());
    }
    let vectors: Vec<Vec<f32>> = seeds.iter().filter_map(|p| p.vector.clone()).collect();
    let near = client.nearest_many(&vectors, &key_only(vec![]), KNN + 1, 5000, None).await?;
    let mut hop1: IndexMap<String, f64> = IndexMap::new();
    for (i, p) in seeds.iter().enumerate() {
        let me = value_string(&p.payload["item"]);
        for n in near.get(i).into_iter().flatten() {
            if n.item == me || n.score < KNN_NEAR {
                continue;
            }
            if !items.contains_key(&n.item) {
                items.insert(n.item.clone(), n.payload.clone());
            }
            if !seed_items.contains_key(&n.item) {
                let best = hop1.get(&n.item).copied().unwrap_or(0.0).max(n.score);
                hop1.insert(n.item.clone(), best);
            }
            g.link(&format!("i:{me}"), &format!("i:{}", n.item), n.score);
        }
    }
    // A second hop from the nearest of those (measured: the multi-hop item walk found more than one hop).
    let mut by_score: Vec<(String, f64)> = hop1.into_iter().collect();
    by_score.sort_by(|a, b| desc(a.1, b.1));
    let next_ids: Vec<String> = by_score.iter().take(KNN).map(|(u, _)| point_id(u, 0)).collect();
    let next: Vec<Stored> =
        client.retrieve(&next_ids, true).await?.into_iter().filter(|p| p.vector.is_some()).collect();
    let vectors: Vec<Vec<f32>> = next.iter().filter_map(|p| p.vector.clone()).collect();
    let near2 = client.nearest_many(&vectors, &key_only(vec![]), KNN + 1, 5000, None).await?;
    for (i, p) in next.iter().enumerate() {
        let me = value_string(&p.payload["item"]);
        for n in near2.get(i).into_iter().flatten() {
            if n.item == me || n.score < KNN_NEAR {
                continue;
            }
            if !items.contains_key(&n.item) {
                items.insert(n.item.clone(), n.payload.clone());
            }
            g.link(&format!("i:{me}"), &format!("i:{}", n.item), n.score);
        }
    }
    // The concepts of what was reached and of the topic.
    let mut concepts: IndexSet<String> = seed_concepts.keys().cloned().collect();
    for p in items.values() {
        concepts.extend(strings(&p["concepts"]));
    }
    // Hop 2: the items holding those concepts — a concept many of them share links each one less — and the items
    // linked to anything reached.
    let asked: Vec<String> = concepts.iter().take(300).cloned().collect();
    if !asked.is_empty() {
        let posted = client
            .scroll(&key_only(vec![json!({ "key": "concepts", "match": { "any": asked } })]), POSTINGS, false, 5000)
            .await?;
        for p in posted {
            let u = value_string(&p.payload["item"]);
            if !items.contains_key(&u) {
                items.insert(u, p.payload);
            }
        }
    }
    let mut linked_out: IndexSet<String> = IndexSet::new();
    for p in items.values() {
        for r in strings(&p["related"]) {
            if !items.contains_key(&r) {
                linked_out.insert(r);
            }
        }
    }
    let ids: Vec<String> = linked_out.iter().map(|u| point_id(u, 0)).collect();
    for p in client.retrieve(&ids, false).await? {
        items.insert(value_string(&p.payload["item"]), p.payload);
    }
    let mut df: IndexMap<String, f64> = IndexMap::new();
    for p in items.values() {
        for k in strings(&p["concepts"]) {
            if concepts.contains(&k) {
                *df.entry(k).or_insert(0.0) += 1.0;
            }
        }
    }
    for (u, p) in &items {
        if !shown(p) {
            continue;
        }
        for k in strings(&p["concepts"]) {
            if concepts.contains(&k) {
                g.link(&format!("i:{u}"), &format!("c:{k}"), 1.0 / (2.0 + df.get(&k).copied().unwrap_or(1.0)).ln());
            }
        }
        for r in strings(&p["related"]) {
            if items.get(&r).is_some_and(shown) {
                g.link(&format!("i:{u}"), &format!("i:{r}"), RELATED);
            }
        }
    }

    // The walk starts again half at the seed items and half at the topic's concepts (either alone when only one is).
    let mut start: IndexMap<String, f64> = IndexMap::new();
    let item_share = if seed_concepts.is_empty() { 1.0 } else { 0.5 };
    for u in seed_items.keys() {
        start.insert(format!("i:{u}"), item_share / seed_items.len() as f64);
    }
    let c_total: f64 = seed_concepts.values().sum();
    let concept_share = if seed_items.is_empty() { 1.0 } else { 0.5 };
    for (k, w) in &seed_concepts {
        start.insert(format!("c:{k}"), concept_share * (w / c_total));
    }
    let x = g.ppr(&start);
    let at = |name: &str| x.get(name).copied().unwrap_or(0.0);

    let skip: IndexSet<&String> = found.iter().chain(seed_items.keys()).collect();
    let seed_titles: IndexMap<&String, String> = seed_items
        .keys()
        .map(|u| {
            (
                u,
                items
                    .get(u)
                    .map(|p| p["title"].clone())
                    .filter(|t| !t.is_null())
                    .map_or_else(|| u.clone(), |t| value_string(&t)),
            )
        })
        .collect();
    let mut seed_keys: IndexSet<String> = seed_concepts.keys().cloned().collect();
    for u in seed_items.keys() {
        if let Some(p) = items.get(u) {
            seed_keys.extend(strings(&p["concepts"]));
        }
    }
    // How an item was reached: the concepts it shares with where the walk started, a link, or nearness to a seed.
    let via = |p: &Value| -> String {
        let cs = strings(&p["concepts"]);
        let shared: Vec<String> =
            cs.iter().filter(|k| seed_keys.contains(*k)).take(3).map(|k| format!("\"{k}\"")).collect();
        if !shared.is_empty() {
            return format!("shares {}", shared.join(", "));
        }
        if let Some(l) = strings(&p["related"]).into_iter().find(|r| seed_titles.contains_key(r)) {
            return format!("linked to \"{}\"", seed_titles[&l]);
        }
        if let Some(t) = cs.iter().find(|k| at(&format!("c:{k}")) > 0.0) {
            return format!("through \"{t}\"");
        }
        "near what was found".to_string()
    };
    let mut related: Vec<Reached> = items
        .iter()
        .filter(|(u, p)| !skip.contains(u) && shown(p) && at(&format!("i:{u}")) > 0.0)
        .map(|(u, p)| Reached { id: u.clone(), score: at(&format!("i:{u}")), via: via(p) })
        .collect();
    related.sort_by(|a, b| desc(a.score, b.score));
    related.truncate(q.limit);
    let mut concept_scores: Vec<(String, f64)> =
        x.iter().filter(|(n, s)| n.starts_with("c:") && **s > 0.0).map(|(n, s)| (n[2..].to_string(), *s)).collect();
    concept_scores.sort_by(|a, b| desc(a.1, b.1));
    concept_scores.truncate(12);
    Ok(Walked { found, related, concepts: concept_scores })
}
