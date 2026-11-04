//! Profiles — what memory knows about one subject, gathered in one place (sleep's last step).
//!
//! A question about a subject whose facts were told over many conversations needs the dated facts of all of them, and
//! a top 10 holds a few. For every concept enough long-term items share — at least `PROFILE_MIN_ITEMS` of a project's
//! items, and on at most `PROFILE_BROAD` of them (not the person themselves, not the project's own name) — the fast
//! model gathers what they say into one dated list of facts, word for word: a `profile` item, keyed and searched like
//! any other, whose provenance is every source it read.
//!
//!   - written again only when what it read changed (a hash of those items is its fingerprint), never from a profile;
//!   - a subject that no longer qualifies has its profile archived;
//!   - never related, merged or superseded by sleep or consolidation: it is written again from its items.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use indexmap::{IndexMap, IndexSet};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::consolidate::{new_item, ConsolidateDeps, Source};
use crate::error::Result;
use crate::js;
use crate::llm::{ask_or_pause, memory_request, LlmAnswer};
use crate::prompts::PROFILE_SYSTEM;
use crate::store::MemoryItem;
use crate::text::{concept_key, cut};
use crate::writer::{write_ltm, WriterDeps};

/// A concept on fewer items than this has no profile.
pub const PROFILE_MIN_ITEMS: usize = 3;
/// A concept on more than this share of a project's items is too broad for one.
pub const PROFILE_BROAD: f64 = 0.2;
/// The items a profile reads: the newest this many, oldest first, each line cut to `PROFILE_LINE`.
pub const PROFILE_READ: usize = 40;
const PROFILE_LINE: usize = 300;
/// Profiles a night writes (a large first sleep goes on the next nights), and asks in flight.
pub const PROFILES_PER_SLEEP: usize = 400;
const PROFILE_IN_FLIGHT: usize = 8;
/// What memory_search and gathering show of a profile: all of it, up to this (measured p90 1,082 characters).
pub const PROFILE_SHOWN: usize = 1500;

fn profile_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false, "required": ["title", "facts"],
        "properties": { "title": { "type": "string" }, "facts": { "type": "array", "items": { "type": "string" } } },
    })
}

/// A subject worth a profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileSubject {
    /// The profile's uid (`pf:…`).
    pub uid: String,
    /// The subject, named as most items spell it.
    pub concept: String,
    /// The scope most of its items have.
    pub scope: String,
    /// Its collection key.
    pub project: String,
    /// The items it reads, oldest first.
    pub read: Vec<MemoryItem>,
    /// A hash of what it reads.
    pub fingerprint: String,
}

/// What writing the profiles did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProfileReport {
    /// Profiles written.
    pub profiled: usize,
    /// Profiles archived.
    pub unprofiled: usize,
    /// Why the model did not answer, when it did not.
    pub paused: Option<String>,
}

fn when(it: &MemoryItem) -> String {
    it.event_at.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| js::long_day(it.ts))
}

fn hash(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

/// The subjects worth a profile, per project: each concept (as `concept_key` folds it, named as most items spell it)
/// on at least `PROFILE_MIN_ITEMS` items and on at most `PROFILE_BROAD` of the project's `sizes` (its active items).
pub fn profile_subjects(items: &[MemoryItem], sizes: &IndexMap<String, usize>) -> Vec<ProfileSubject> {
    struct On<'a> {
        project: String,
        key: String,
        names: IndexMap<String, usize>,
        items: Vec<&'a MemoryItem>,
    }
    let mut on: IndexMap<String, On> = IndexMap::new();
    for it in items {
        if it.status != "active" || it.kind == "profile" {
            continue;
        }
        let mut named: IndexMap<String, String> = IndexMap::new();
        for c in it.concept_list() {
            let key = concept_key(c);
            if !key.is_empty() && !named.contains_key(&key) {
                named.insert(key, js::trim(c).to_string());
            }
        }
        for (key, name) in named {
            let id = format!("{}\n{key}", it.project);
            let s = on.entry(id).or_insert_with(|| On {
                project: it.project.clone(),
                key,
                names: IndexMap::new(),
                items: vec![],
            });
            *s.names.entry(name).or_insert(0) += 1;
            s.items.push(it);
        }
    }
    let mut out = Vec::new();
    for (id, s) in on {
        if s.items.len() < PROFILE_MIN_ITEMS
            || s.items.len() as f64 > PROFILE_BROAD * sizes.get(&s.project).copied().unwrap_or(0) as f64
        {
            continue;
        }
        let mut read: Vec<&MemoryItem> = s.items.clone();
        read.sort_by(|a, b| {
            b.ts.partial_cmp(&a.ts)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| js::locale_compare(&a.uid, &b.uid))
        });
        read.truncate(PROFILE_READ);
        read.reverse();
        let mut scopes: IndexMap<&str, usize> = IndexMap::new();
        for it in &read {
            *scopes.entry(it.scope.as_str()).or_insert(0) += 1;
        }
        let mut names: Vec<(&String, &usize)> = s.names.iter().collect();
        names.sort_by(|a, b| b.1.cmp(a.1).then_with(|| js::locale_compare(a.0, b.0)));
        let mut scope_order: Vec<(&str, usize)> = scopes.into_iter().collect();
        scope_order.sort_by_key(|s| std::cmp::Reverse(s.1));
        let lines: Vec<String> = std::iter::once(s.key.clone())
            .chain(read.iter().map(|it| format!("{}\n{}\n{}\n{}", it.uid, when(it), it.title, it.summary)))
            .collect();
        out.push(ProfileSubject {
            uid: format!("pf:{}", &hash(&id)[..24]),
            concept: names.first().map(|n| n.0.clone()).unwrap_or_default(),
            scope: scope_order.first().map_or("project", |s| s.0).to_string(),
            project: s.project,
            fingerprint: hash(&lines.join("\n\n")),
            read: read.into_iter().cloned().collect(),
        });
    }
    out
}

/// The prompt a subject's profile is asked for with.
pub fn prompt_of(s: &ProfileSubject) -> String {
    let lines: Vec<String> = s
        .read
        .iter()
        .map(|it| format!("- ({}) {}: {}", when(it), it.title, cut(&js::collapse_spaces(&it.summary), PROFILE_LINE)))
        .collect();
    format!("SUBJECT: {}\n\nMEMORIES:\n{}", s.concept, lines.join("\n"))
}

fn profile_item(s: &ProfileSubject, title: &str, facts: &[String], now: f64) -> Option<MemoryItem> {
    let newest = s.read.last()?;
    let src = Source {
        project: &s.project,
        engine: &newest.engine,
        session: None,
        cwd: newest.cwd.as_deref(),
        where_: &s.concept,
        ts: now,
    };
    let title = js::trim(title);
    let mut it = new_item(
        s.uid.clone(),
        "profile",
        &s.scope,
        src,
        cut(if title.is_empty() { &s.concept } else { title }, 120),
        facts.join("\n"),
        String::new(),
        "curated",
        3.0,
        s.fingerprint.clone(),
    );
    it.ts = s.read.iter().map(|x| x.ts).fold(f64::NEG_INFINITY, f64::max);
    it.first_ts =
        s.read.iter().map(|x| if x.first_ts != 0.0 { x.first_ts } else { x.ts }).fold(f64::INFINITY, f64::min);
    it.provenance =
        s.read.iter().flat_map(|x| x.provenance.iter().cloned()).collect::<IndexSet<_>>().into_iter().collect();
    it.concepts = Some(vec![s.concept.clone()]);
    it.learned_at = Some(now);
    Some(it)
}

/// Profiles written for the subjects whose items changed (at most `max`, several asks at once), and archived for the
/// subjects that no longer qualify. An ask the model did not answer stops the night's profiles; the next night asks
/// again, as their fingerprints still differ.
pub async fn write_profiles(d: &ConsolidateDeps, max: usize) -> Result<ProfileReport> {
    let mut r = ProfileReport::default();
    let w = WriterDeps { client: d.client.clone(), embed: d.embed.clone() };
    let mut sizes: IndexMap<String, usize> = IndexMap::new();
    let mut wanted_items = Vec::new();
    for (uid, m) in d.ltm.entries() {
        if m.status != "active" || m.kind == "profile" {
            continue;
        }
        *sizes.entry(m.project.clone()).or_insert(0) += 1;
        if m.mapped {
            wanted_items.push(uid);
        }
    }
    let mapped: Vec<MemoryItem> = d.ltm.get_many(wanted_items).await.into_iter().flatten().collect();
    let subjects = profile_subjects(&mapped, &sizes);
    let wanted: IndexSet<&str> = subjects.iter().map(|s| s.uid.as_str()).collect();
    let gone_uids: Vec<String> = d
        .ltm
        .entries()
        .into_iter()
        .filter(|(uid, m)| m.kind == "profile" && m.status == "active" && !wanted.contains(uid.as_str()))
        .map(|(uid, _)| uid)
        .collect();
    let gone: Vec<MemoryItem> = d
        .ltm
        .get_many(gone_uids)
        .await
        .into_iter()
        .flatten()
        .map(|it| MemoryItem { status: "archived".into(), ..it })
        .collect();
    if !gone.is_empty() {
        write_ltm(&w, &gone).await?;
        r.unprofiled = gone.len();
        d.ltm.put(gone).await?;
    }
    let todo: Vec<&ProfileSubject> = subjects
        .iter()
        .filter(|s| d.ltm.meta(&s.uid).is_none_or(|m| m.status != "active" || m.fp != s.fingerprint))
        .take(max)
        .collect();
    let next = AtomicUsize::new(0);
    let state = Mutex::new((0usize, None::<String>));
    let worker = || async {
        loop {
            if state.lock().unwrap_or_else(|e| e.into_inner()).1.is_some() || (d.aborted)() {
                return Ok::<(), crate::Error>(());
            }
            let i = next.fetch_add(1, Ordering::SeqCst);
            let Some(s) = todo.get(i) else { return Ok(()) };
            let req = memory_request(
                PROFILE_SYSTEM,
                &prompt_of(s),
                profile_schema(),
                "memory_profile",
                3000,
                180_000,
                "so this subject's profile waits for the next night",
            );
            let answer = match ask_or_pause(d.ask.as_ref(), req).await {
                LlmAnswer::Answer { answer, .. } => answer,
                LlmAnswer::Unavailable(why) => {
                    state.lock().unwrap_or_else(|e| e.into_inner()).1.get_or_insert(why);
                    return Ok(());
                }
                LlmAnswer::Busy => {
                    state.lock().unwrap_or_else(|e| e.into_inner()).1.get_or_insert("busy".into());
                    return Ok(());
                }
            };
            let facts: Vec<String> = answer["facts"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|f| f.as_str())
                .filter(|f| !js::trim(f).is_empty())
                .map(|f| js::trim(f).to_string())
                .collect();
            if facts.is_empty() {
                continue;
            }
            let Some(it) = profile_item(s, answer["title"].as_str().unwrap_or(""), &facts, (d.now)()) else { continue };
            write_ltm(&w, std::slice::from_ref(&it)).await?;
            d.ltm.put(vec![it]).await?;
            state.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
        }
    };
    let workers: Vec<_> = (0..PROFILE_IN_FLIGHT.min(todo.len())).map(|_| worker()).collect();
    for done in futures::future::join_all(workers).await {
        done?;
    }
    let (profiled, paused) = state.into_inner().unwrap_or_else(|e| e.into_inner());
    r.profiled = profiled;
    r.paused = paused;
    Ok(r)
}
