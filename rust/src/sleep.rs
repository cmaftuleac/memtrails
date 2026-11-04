//! Sleep — the daily pass that keeps long-term memory organised:
//!
//!   1. RELATE what was learned since the last sleep to every near neighbour (cosine >= 0.80, any knowledge kind, the
//!      same scope) — not only the nearest one consolidation looked at: from 0.93 they merge; below, the fast model
//!      merges the two, says one replaces the other (superseded), links them or says they differ;
//!   2. PRIORITY for every active item = importance × recency (a half-life by kind: preferences, architecture,
//!      references and notes do not fade; facts and pitfalls fade in a year; episodes in three months) × use (how
//!      often recall showed it and an agent opened it);
//!   3. ARCHIVE what stays low and unused for half a year: out of the index, kept in the log. Nothing is forgotten
//!      outright;
//!   4. PROFILES (profiles.rs): for each subject enough items share, what they say gathered into one dated list.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use serde_json::{json, Map, Value};

use crate::consolidate::{
    learned_at, pair_size, relate_batches, ConsolidateDeps, Consolidator, Decision, Pair, RELATED, SAME,
};
use crate::embedder::doc_text;
use crate::error::Result;
use crate::js;
use crate::profiles::{write_profiles, PROFILES_PER_SLEEP};
use crate::qdrant_client::point_id;
use crate::store::{blocking, write_private, MemoryItem, UsageBook, Use};
use crate::writer::{write_ltm, WriterDeps};

const DAY: f64 = 24.0 * 60.0 * 60_000.0;
/// A fresh item's neighbours looked at.
const NEIGHBOURS: usize = 8;
/// The pairs a night asks about (a large first sleep goes on the next nights).
pub const PAIRS_PER_SLEEP: usize = 1600;
const RELATE_IN_FLIGHT: usize = 4;
/// Archived below this priority…
pub const ARCHIVE_BELOW: f64 = 0.02;
/// …when also unused this long.
pub const ARCHIVE_UNUSED_MS: f64 = 180.0 * DAY;

fn half_life(kind: &str) -> Option<f64> {
    Some(match kind {
        "episode" => 90.0 * DAY,
        "fact" | "pitfall" => 365.0 * DAY,
        "decision" | "procedure" | "snippet" | "template" => 540.0 * DAY,
        _ => return None,
    })
}

/// An item's priority: importance × recency (its kind's half-life) × use.
pub fn priority(kind: &str, importance: f64, ts: f64, used: &Use, now: f64) -> f64 {
    let recency = half_life(kind).map_or(1.0, |half| 0.5f64.powf((now - ts).max(0.0) / half));
    (importance / 5.0) * recency * (1.0 + (used.opened as f64).ln_1p() + 0.3 * (used.shown as f64).ln_1p())
}

/// What sleep reads and writes, besides what consolidation does.
#[derive(Clone)]
pub struct SleepDeps {
    /// What consolidation reads and writes.
    pub base: ConsolidateDeps,
    /// How often each item was shown and opened.
    pub usage: UsageBook,
    /// Where the last sleep is kept (`sleep.json`).
    pub state_path: PathBuf,
    /// Fresh items looked at at once (1).
    pub scan_at_once: usize,
    /// Relate asks in flight (4).
    pub relate_in_flight: usize,
    /// Answers applied at once (1, in order).
    pub apply_at_once: usize,
}

impl SleepDeps {
    /// Sleep deps with the defaults.
    pub fn new(base: ConsolidateDeps, usage: UsageBook, state_path: PathBuf) -> Self {
        SleepDeps { base, usage, state_path, scan_at_once: 1, relate_in_flight: RELATE_IN_FLIGHT, apply_at_once: 1 }
    }
}

/// What a night did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SleepReport {
    /// Fresh items looked at.
    pub looked: usize,
    /// Pairs merged.
    pub merged: usize,
    /// Items superseded.
    pub superseded: usize,
    /// Pairs linked.
    pub linked: usize,
    /// Items archived.
    pub archived: usize,
    /// Profiles written.
    pub profiled: usize,
    /// Profiles archived.
    pub unprofiled: usize,
    /// Set when the language model did not answer: why. What it was to read waits for the next night.
    pub paused: Option<String>,
}

struct Scored {
    pair: Pair,
    score: f64,
    /// When the fresh item of the pair was learned — where the next night starts when this pair is not asked.
    at: f64,
}

fn knowledge(status: &str, kind: &str) -> bool {
    // Observations are add-only: two events are never one, and an older one stays, dated. A profile is written again
    // from its items, never related to anything.
    status == "active" && !matches!(kind, "episode" | "note" | "observation" | "profile")
}

/// One offline pass.
pub async fn sleep(d: &SleepDeps) -> Result<SleepReport> {
    let b = &d.base;
    let now = (b.now)();
    let since = tokio::fs::read_to_string(&d.state_path)
        .await
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v["last"].as_f64())
        .unwrap_or(0.0);
    let mut r = SleepReport::default();
    let worker = Consolidator::new(b.clone());
    // 1. What was learned since the last sleep, against its near neighbours, oldest first — by when it was written,
    // not by its source's time: history imported after a sleep is new to memory.
    let mut fresh: Vec<(String, f64)> = b
        .ltm
        .entries()
        .into_iter()
        .filter(|(_, m)| knowledge(&m.status, &m.kind) && learned_at(m.ts, m.learned_at) >= since)
        .map(|(uid, m)| (uid, learned_at(m.ts, m.learned_at)))
        .collect();
    fresh.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut pairs: Vec<Scored> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut resume_at: Option<f64> = None;
    // Oldest first, a few at once: each group's neighbours become pairs in order, and a full night stops at the next
    // group, which the next night starts from.
    for group in fresh.chunks(d.scan_at_once.max(1)) {
        if (b.aborted)() {
            return Ok(r);
        }
        let group: Vec<MemoryItem> = b
            .ltm
            .get_many(group.iter().map(|g| g.0.clone()).collect())
            .await
            .into_iter()
            .flatten()
            .filter(|it| it.status == "active")
            .collect();
        if group.is_empty() {
            continue;
        }
        if pairs.len() >= PAIRS_PER_SLEEP {
            resume_at = Some(learned_at(group[0].ts, group[0].learned_at));
            break;
        }
        r.looked += group.len();
        let near = futures::future::try_join_all(group.iter().map(|it| look(b, it))).await?;
        for (it, near) in group.iter().zip(near) {
            for (item, score) in near {
                if item == it.uid || score < RELATED {
                    continue;
                }
                let mut key = [it.uid.clone(), item.clone()];
                key.sort();
                let key = key.join(" ");
                if seen.contains(&key) {
                    continue;
                }
                let Some(known) = b.ltm.get(&item).await.filter(|k| knowledge(&k.status, &k.kind)) else { continue };
                seen.insert(key);
                // The newer of the two is NEW, the older KNOWN: a contradiction retires the older.
                let at = learned_at(it.ts, it.learned_at);
                let pair = if it.ts >= known.ts {
                    Pair { it: it.clone(), known }
                } else {
                    Pair { it: known, known: it.clone() }
                };
                pairs.push(Scored { pair, score, at });
            }
        }
    }
    // The fast model reads the pairs below 0.93, several asks at once; the answers are applied in order, each to the
    // items as they are by then.
    let asked: Vec<usize> = (0..pairs.len()).filter(|&i| pairs[i].score < SAME).collect();
    let batches = relate_batches(asked, |&i| pair_size(&pairs[i].pair.it, &pairs[i].pair.known));
    let answers: Mutex<HashMap<usize, Decision>> = Mutex::new(HashMap::new());
    let next = AtomicUsize::new(0);
    let unanswered = AtomicBool::new(false);
    let paused: Mutex<(Option<String>, Option<f64>)> = Mutex::new((None, resume_at));
    let asker = || async {
        while !unanswered.load(Ordering::SeqCst) && !(b.aborted)() {
            let k = next.fetch_add(1, Ordering::SeqCst);
            let Some(batch) = batches.get(k) else { return };
            let ps: Vec<Pair> = batch.iter().map(|&i| pairs[i].pair.clone()).collect();
            match worker.relate(&ps).await {
                Ok(got) => {
                    let mut a = answers.lock().unwrap_or_else(|e| e.into_inner());
                    for (n, &i) in batch.iter().enumerate() {
                        if let Some(d) = got.get(&format!("p{}", n + 1)) {
                            a.insert(i, d.clone());
                        }
                    }
                }
                Err(why) => {
                    // One the model did not answer stops the asking; the next night starts from its pairs.
                    unanswered.store(true, Ordering::SeqCst);
                    let mut p = paused.lock().unwrap_or_else(|e| e.into_inner());
                    p.0.get_or_insert(why);
                    let first = batch.iter().map(|&i| pairs[i].at).fold(f64::INFINITY, f64::min);
                    p.1 = Some(p.1.unwrap_or(f64::INFINITY).min(first));
                    return;
                }
            }
        }
    };
    futures::future::join_all((0..d.relate_in_flight.max(1).min(batches.len())).map(|_| asker())).await;
    let (why, resume) = paused.into_inner().unwrap_or_else(|e| e.into_inner());
    r.paused = why;
    let resume_at = resume;
    let answers = answers.into_inner().unwrap_or_else(|e| e.into_inner());
    // The answers applied — in order, one at a time, unless a backfill applies several at once — each to the two items
    // as they are by then, under their lock (one an earlier answer retired is left alone).
    let counts = Mutex::new(SleepReport::default());
    let at = AtomicUsize::new(0);
    let apply = || async {
        while !(b.aborted)() {
            let i = at.fetch_add(1, Ordering::SeqCst);
            let Some(p) = pairs.get(i) else { return Ok::<(), crate::Error>(()) };
            let decision = if p.score >= SAME {
                Some(Decision {
                    relation: "same".into(),
                    title: String::new(),
                    summary: String::new(),
                    body: String::new(),
                })
            } else {
                answers.get(&i).cloned()
            };
            let w = worker.clone();
            let (iu, ku) = (p.pair.it.uid.clone(), p.pair.known.uid.clone());
            let counts = &counts;
            worker
                .locked(&[iu.clone(), ku.clone()], async move {
                    let (it, known) = (w.deps().ltm.get(&iu).await, w.deps().ltm.get(&ku).await);
                    let (Some(it), Some(known)) = (it, known) else { return Ok(()) };
                    if it.status != "active" || known.status != "active" {
                        return Ok(());
                    }
                    let Some(d) = decision.filter(|d| d.relation != "different") else { return Ok(()) };
                    match d.relation.as_str() {
                        "related" => {
                            w.link(it, known).await?;
                            counts.lock().unwrap_or_else(|e| e.into_inner()).linked += 1;
                        }
                        "contradicts" => {
                            let by = it.uid.clone();
                            w.retire(known, &by).await?;
                            counts.lock().unwrap_or_else(|e| e.into_inner()).superseded += 1;
                        }
                        _ => {
                            let body = if d.body.is_empty() { known.body.clone() } else { d.body.clone() };
                            let text = (!d.title.is_empty() && !d.summary.is_empty())
                                .then(|| (d.title.clone(), d.summary.clone(), body));
                            let into = known.uid.clone();
                            w.merge(known, it.clone(), text).await?;
                            w.retire(it, &into).await?;
                            counts.lock().unwrap_or_else(|e| e.into_inner()).merged += 1;
                        }
                    }
                    Ok(())
                })
                .await?;
        }
        Ok(())
    };
    for done in futures::future::join_all((0..d.apply_at_once.max(1)).map(|_| apply())).await {
        done?;
    }
    let c = counts.into_inner().unwrap_or_else(|e| e.into_inner());
    r.linked += c.linked;
    r.superseded += c.superseded;
    r.merged += c.merged;
    if (b.aborted)() {
        return Ok(r);
    }
    // 2–3. Priority, and the archive.
    let mut retire: Vec<String> = Vec::new();
    for (uid, m) in b.ltm.entries() {
        if m.status != "active" || m.kind == "note" || half_life(&m.kind).is_none() {
            continue;
        }
        let used = d.usage.of(&uid);
        // Unused for half a year since it was learned, not since its source was written: imported history is not
        // archived the night it arrives.
        if priority(&m.kind, m.importance, m.ts, &used, now) >= ARCHIVE_BELOW
            || now - m.ts.max(learned_at(m.ts, m.learned_at)).max(used.last_used) < ARCHIVE_UNUSED_MS
        {
            continue;
        }
        retire.push(uid);
    }
    let retire: Vec<MemoryItem> = b
        .ltm
        .get_many(retire)
        .await
        .into_iter()
        .flatten()
        .filter(|it| now - learned_at(it.ts, it.learned_at) >= ARCHIVE_UNUSED_MS)
        .map(|it| MemoryItem { status: "archived".into(), ..it })
        .collect();
    let w = WriterDeps { client: b.client.clone(), embed: b.embed.clone() };
    for part in retire.chunks(50) {
        if (b.aborted)() {
            break;
        }
        write_ltm(&w, part).await?;
        b.ltm.put(part.to_vec()).await?;
        r.archived += part.len();
    }
    // 4. Profiles of what is known now; one the fast model did not answer waits for the next night.
    if (b.aborted)() {
        return Ok(r);
    }
    let p = write_profiles(b, PROFILES_PER_SLEEP).await?;
    r.profiled = p.profiled;
    r.unprofiled = p.unprofiled;
    if let Some(why) = p.paused {
        (b.log)(&format!("memory sleep: profiles wait for the next night ({why})"));
        r.paused.get_or_insert(why);
    }
    b.ltm.compact().await?;
    // A night that stopped early (the pair cap, the fast model away) starts the next one where it stopped.
    let mut state = Map::new();
    state.insert("last".into(), js::ms_value(resume_at.unwrap_or(now)));
    for (k, v) in [
        ("looked", r.looked),
        ("merged", r.merged),
        ("superseded", r.superseded),
        ("linked", r.linked),
        ("archived", r.archived),
        ("profiled", r.profiled),
        ("unprofiled", r.unprofiled),
    ] {
        state.insert(k.into(), json!(v));
    }
    if let Some(p) = &r.paused {
        state.insert("paused".into(), json!(p));
    }
    let (path, text) = (d.state_path.clone(), js::stringify(&Value::Object(state)));
    // The next sleep looks at a little more when this cannot be written.
    let _ = blocking(move || {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        write_private(&path, text.as_bytes())
    })
    .await;
    Ok(r)
}

/// A fresh item's near neighbours among what may relate to it: the key vector the index already holds, embedded again
/// only when it is not there.
async fn look(b: &ConsolidateDeps, it: &MemoryItem) -> Result<Vec<(String, f64)>> {
    let held = b.client.retrieve(&[point_id(&it.uid, 0)], true).await?.into_iter().next().and_then(|p| p.vector);
    let v = match held {
        Some(v) => v,
        None => (b.embed)(vec![doc_text(&it.title, &it.summary)], None).await?.into_iter().next().unwrap_or_default(),
    };
    let mut must = vec![
        json!({ "key": "tier", "match": { "value": "ltm" } }),
        json!({ "key": "status", "match": { "value": "active" } }),
    ];
    must.push(if it.scope == "project" {
        json!({ "key": "project", "match": { "value": it.project } })
    } else {
        json!({ "key": "scope", "match": { "any": ["global", "person"] } })
    });
    let filter = json!({ "must": must, "must_not": [{ "key": "kind", "match": { "any": ["episode", "note", "observation", "profile"] } }] });
    Ok(b.client.neighbors(&v, &filter, NEIGHBOURS).await?.into_iter().map(|n| (n.item, n.score)).collect())
}
