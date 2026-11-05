//! Consolidation — LONG-TERM memory. Like the brain: short-term memory keeps everything as it came; long-term memory
//! filters, stores, reorganises and reprioritises.
//!
//! It reads the queue (the records log) a thread at a time — one that has settled (quiet for an hour, or a day old),
//! newest first — in windows of whole records (extract.rs: one language-model ask a window). What survives the gate is
//! RECONCILED with what long-term memory already holds before it is written, so nothing is stored twice:
//!
//!   - cosine >= 0.93 to a known item (0.97 for episodes) → the same memory: its provenance and `seen` grow, it is
//!     refreshed — no new item;
//!   - 0.80–0.93 → one fast-model decision a batch: same / updates / contradicts / related / different → merge,
//!     rewrite, supersede (the old one stays in the log, out of recall), add and link the two, or add;
//!   - below → added.
//!
//! Then the window's entries are marked done: what mattered of them is long-term now; the records stay searchable.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use indexmap::{IndexMap, IndexSet};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::embedder::{doc_text, EmbedFn};
use crate::error::{Error, Result};
use crate::extract::{
    clean_concepts, clean_cues, extract_window, observe_window, windows_of, Candidate, ExtractOptions, Extraction,
    KeptEpisode, Observation, Read, WindowMeta, WINDOW_CHARS,
};
use crate::js;
use crate::llm::{memory_request, Ask, LlmAnswer};
use crate::prompts::{CUE_SYSTEM, RELATE_SYSTEM};
use crate::qdrant_client::QdrantClient;
use crate::recall::{memory_search, RecallDeps, SearchQuery};
use crate::redact::redact_string;
use crate::store::{fingerprint, FpPart, LtmMeta, LtmStore, MemoryItem, StmEntry, StmStore};
use crate::text::{concept_key, cut};
use crate::writer::{embed_budget, write_concepts, write_cues, write_ltm, WriterDeps};
use crate::{BoxFuture, LogFn};

/// The same memory, from this cosine.
pub const SAME: f64 = 0.93;
/// The same episode, from this cosine.
pub const SAME_EPISODE: f64 = 0.97;
/// An observation is the same only when it says the same thing (this close) about the same date: two dinners on two
/// days are two observations, and a list question needs both.
pub const SAME_OBSERVATION: f64 = 0.97;
/// From here to `SAME`, the model decides.
pub const RELATED: f64 = 0.80;
/// A conversation is consolidated once it has been quiet this long — or its oldest waiting entry is a day old.
pub const SETTLED_MS: f64 = 60.0 * 60_000.0;
/// Extraction calls in flight: they are independent, and one at a time made a backfill take days.
pub const EXTRACT_CONCURRENCY: usize = 8;
const DAY_MS: f64 = 24.0 * 60.0 * 60_000.0;
/// Requests of a window searched for, and the passages of other conversations kept to name what it refers to.
const RESOLVE_QUERIES: usize = 4;
const RESOLVE_KEPT: usize = 8;
/// Items given their cues and concepts after a run's windows, and how many go in one ask.
pub const CUE_BACKLOG: usize = 40;
const CUE_BATCH: usize = 10;
/// Links an item keeps to others about the same subject (exploration walks them), newest last.
pub const MAX_LINKS: usize = 24;
/// Pairs asked about together: at most 8, and at most this many characters of their texts — a long pair on its own.
pub const RELATE_CHARS: usize = 60_000;
const RELATIONS: [&str; 5] = ["same", "updates", "contradicts", "related", "different"];

/// What a window's reading may keep, checked before anything is reconciled or written (e.g. facts only from the
/// person's own words, each with where it came from). An answer it cannot give leaves the window for the next run.
pub type GuardFn = Arc<dyn Fn(Vec<StmEntry>, Extraction) -> BoxFuture<'static, Read<Extraction>> + Send + Sync>;

/// What consolidation and sleep read and write.
#[derive(Clone)]
pub struct ConsolidateDeps {
    /// The index.
    pub client: QdrantClient,
    /// Already prompted texts → unit vectors.
    pub embed: EmbedFn,
    /// The records log.
    pub stm: StmStore,
    /// The knowledge log.
    pub ltm: LtmStore,
    /// Progress lines.
    pub log: LogFn,
    /// The memory is closing: stop at the next step.
    pub aborted: Arc<dyn Fn() -> bool + Send + Sync>,
    /// The language model.
    pub ask: Arc<dyn Ask>,
    /// The time, in ms.
    pub now: Arc<dyn Fn() -> f64 + Send + Sync>,
    /// Windows the model extracts at once (`EXTRACT_CONCURRENCY` by default).
    pub concurrency: usize,
    /// Windows reconciled at once (1, in order, by default).
    pub reconcile_concurrency: usize,
    /// A window's size in characters (`WINDOW_CHARS` by default).
    pub window_chars: usize,
    /// Observations read apart, in windows of this many characters; `None`: in the window's one ask.
    pub observe_chars: Option<usize>,
    /// Each window read with what other conversations say that it may refer to.
    pub resolve: bool,
    /// What a reading may keep.
    pub guard: Option<GuardFn>,
}

impl ConsolidateDeps {
    /// Deps with the defaults for everything but what must be given.
    pub fn new(client: QdrantClient, embed: EmbedFn, stm: StmStore, ltm: LtmStore, ask: Arc<dyn Ask>) -> Self {
        ConsolidateDeps {
            client,
            embed,
            stm,
            ltm,
            log: Arc::new(|_| {}),
            aborted: Arc::new(|| false),
            ask,
            now: Arc::new(js::now_ms),
            concurrency: EXTRACT_CONCURRENCY,
            reconcile_concurrency: 1,
            window_chars: WINDOW_CHARS,
            observe_chars: None,
            resolve: false,
            guard: None,
        }
    }

    fn now(&self) -> f64 {
        (self.now)()
    }
}

/// What a consolidation did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationReport {
    /// Windows read.
    pub windows: usize,
    /// Items added.
    pub added: usize,
    /// Items merged into known ones (or told again).
    pub merged: usize,
    /// Known items updated.
    pub updated: usize,
    /// Known items superseded.
    pub superseded: usize,
    /// Pairs linked.
    pub linked: usize,
    /// Items the gate dropped.
    pub dropped: usize,
    /// Requests kept as episodes.
    pub episodes: usize,
    /// Observations added.
    pub observed: usize,
    /// Items given cues and concepts.
    pub cued: usize,
    /// Why the model did not read, when it did not.
    pub paused: Option<String>,
}

impl ConsolidationReport {
    /// Add another report's counts to this one's.
    pub fn add(&mut self, o: &ConsolidationReport) {
        self.windows += o.windows;
        self.added += o.added;
        self.merged += o.merged;
        self.updated += o.updated;
        self.superseded += o.superseded;
        self.linked += o.linked;
        self.dropped += o.dropped;
        self.episodes += o.episodes;
        self.observed += o.observed;
        self.cued += o.cued;
    }
}

/// The model's decision on a pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// same, updates, contradicts, related or different.
    pub relation: String,
    /// The merged title (same, updates).
    pub title: String,
    /// The merged summary.
    pub summary: String,
    /// The merged body.
    pub body: String,
}

/// A new item and a known one, to be decided on.
#[derive(Debug, Clone)]
pub struct Pair {
    /// The newer.
    pub it: MemoryItem,
    /// The older.
    pub known: MemoryItem,
}

fn relate_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false, "required": ["pairs"],
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": false, "required": ["id", "relation", "title", "summary", "body"],
                    "properties": {
                        "id": { "type": "string" }, "relation": { "type": "string", "enum": RELATIONS },
                        "title": { "type": "string" }, "summary": { "type": "string" }, "body": { "type": "string" },
                    },
                },
            },
        },
    })
}

fn cue_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false, "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": false, "required": ["id", "cues", "concepts"],
                    "properties": { "id": { "type": "string" }, "cues": { "type": "array", "items": { "type": "string" } }, "concepts": { "type": "array", "items": { "type": "string" } } },
                },
            },
        },
    })
}

/// Pairs asked about together: at most 8, and at most `RELATE_CHARS` of their texts — a long pair on its own.
pub fn relate_batches<T>(pairs: Vec<T>, size_of: impl Fn(&T) -> usize) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let mut cur: Vec<T> = Vec::new();
    let mut chars = 0;
    for p in pairs {
        let size = size_of(&p);
        if !cur.is_empty() && (cur.len() >= 8 || chars + size > RELATE_CHARS) {
            out.push(std::mem::take(&mut cur));
            chars = 0;
        }
        cur.push(p);
        chars += size;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// What a pair's texts weigh.
pub(crate) fn pair_size(it: &MemoryItem, known: &MemoryItem) -> usize {
    js::len16(&it.summary) + js::len16(&it.body) + js::len16(&known.summary) + js::len16(&known.body)
}

/// When an item was written to long-term memory. `ts` is its source's own time — for history imported today, long ago
/// — so "learned since the last sleep" and "unused for half a year" read this.
pub fn learned_at(ts: f64, learned: Option<f64>) -> f64 {
    learned.unwrap_or(ts)
}

/// The number of a label the model wrote back: `p3`, or the line it labels quoted whole (`[p3] NEW …`) — 0 for none.
pub fn label_no(id: &str, letter: char) -> usize {
    let cs: Vec<char> = id.chars().collect();
    let word = |c: Option<&char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_');
    for i in 0..cs.len() {
        if cs[i] != letter || (i > 0 && word(cs.get(i - 1))) {
            continue;
        }
        let digits: String = cs[i + 1..].iter().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return digits.parse().unwrap_or(0);
        }
    }
    0
}

/// Where a long-term item comes from: its collection, kind of record, thread, folder, place and time.
pub(crate) struct Source<'a> {
    pub project: &'a str,
    pub engine: &'a str,
    pub session: Option<&'a str>,
    pub cwd: Option<&'a str>,
    pub where_: &'a str,
    pub ts: f64,
}

impl<'a> From<&'a StmEntry> for Source<'a> {
    fn from(e: &'a StmEntry) -> Self {
        Source {
            project: &e.project,
            engine: &e.engine,
            session: Some(&e.session),
            cwd: e.cwd.as_deref(),
            where_: &e.title,
            ts: e.ts,
        }
    }
}

/// A long-term item with every field given, the rest empty.
#[allow(clippy::too_many_arguments)]
pub(crate) fn new_item(
    uid: String,
    kind: &str,
    scope: &str,
    src: Source<'_>,
    title: String,
    summary: String,
    body: String,
    confidence: &str,
    importance: f64,
    fp: String,
) -> MemoryItem {
    MemoryItem {
        uid,
        kind: kind.into(),
        scope: scope.into(),
        project: src.project.to_string(),
        engine: src.engine.to_string(),
        session: src.session.map(str::to_string),
        cwd: src.cwd.map(str::to_string),
        where_: src.where_.to_string(),
        ts: src.ts,
        first_ts: src.ts,
        learned_at: None,
        title,
        summary,
        body,
        files: vec![],
        applies_when: String::new(),
        confidence: confidence.into(),
        importance,
        status: "active".into(),
        superseded_by: None,
        superseded_at: None,
        outcome: None,
        merged_into: None,
        absorbed: None,
        provenance: vec![],
        seen: 1.0,
        fingerprint: fp,
        cues: None,
        origin: None,
        related: None,
        concepts: None,
        event_at: None,
        start_at: None,
        end_at: None,
        phase: None,
        about: None,
        extra: IndexMap::new(),
    }
}

fn unique(xs: impl IntoIterator<Item = String>) -> Vec<String> {
    xs.into_iter().collect::<IndexSet<String>>().into_iter().collect()
}

/// One kept candidate as a long-term item (not yet reconciled).
fn item_of(c: &Candidate, src: &StmEntry, now: f64, sources: &[&StmEntry]) -> MemoryItem {
    let project = if c.scope == "project" { src.project.clone() } else { c.scope.clone() };
    let ts = if src.ts != 0.0 { src.ts } else { now };
    let mut it = new_item(
        format!(
            "mem:{}",
            fingerprint(&[
                c.kind.as_str().into(),
                c.scope.as_str().into(),
                project.as_str().into(),
                c.title.as_str().into(),
                c.summary.as_str().into()
            ])
        ),
        &c.kind,
        &c.scope,
        src.into(),
        c.title.clone(),
        c.summary.clone(),
        c.body.clone(),
        &c.confidence,
        c.importance,
        fingerprint(&[
            c.kind.as_str().into(),
            c.title.as_str().into(),
            c.summary.as_str().into(),
            c.body.as_str().into(),
        ]),
    );
    it.ts = ts;
    it.first_ts = ts;
    // The files its requests changed, from their records — by code, not the model: paths are what summaries lose.
    it.files = unique(sources.iter().flat_map(|e| e.files.iter().cloned()));
    it.applies_when = c.applies_when.clone();
    it.provenance = c.provenance.clone();
    it.cues = Some(c.cues.clone());
    it.concepts = Some(c.concepts.clone());
    it.origin = c.origin.clone();
    it.learned_at = Some(now);
    it
}

fn episode_item(e: &StmEntry, key: &KeptEpisode, now: f64) -> MemoryItem {
    let mut it = new_item(
        format!("ep:{}", e.uid),
        "episode",
        "project",
        e.into(),
        if key.title.is_empty() { e.title.clone() } else { key.title.clone() },
        key.summary.clone(),
        crate::episodes::episode_body(e),
        "observed",
        if e.files.is_empty() { 2.0 } else { 3.0 },
        fingerprint(&[e.request.as_str().into(), e.reply.as_str().into(), FpPart::List(&e.files)]),
    );
    it.files = e.files.clone();
    it.provenance = vec![e.uid.clone()];
    it.cues = Some(key.cues.clone());
    it.concepts = Some(key.concepts.clone());
    it.outcome = key.outcome.clone();
    it.origin = key.origin.clone();
    it.learned_at = Some(now);
    it
}

/// An observation as a long-term item: its text is its key, its date beside when it was said.
fn observation_item(o: &Observation, src: &StmEntry, now: f64) -> MemoryItem {
    let first = o.provenance.first().cloned().unwrap_or_else(|| src.uid.clone());
    let mut it = new_item(
        format!("obs:{}", fingerprint(&[o.text.as_str().into(), o.event_at.as_str().into(), first.as_str().into()])),
        "observation",
        "person",
        src.into(),
        cut(&o.text, 100),
        o.text.clone(),
        String::new(),
        "stated",
        2.0,
        fingerprint(&["observation".into(), o.text.as_str().into(), o.event_at.as_str().into()]),
    );
    it.provenance = o.provenance.clone();
    it.cues = Some(o.cues.clone());
    it.concepts = Some(o.concepts.clone());
    it.event_at = Some(o.event_at.clone()).filter(|d| !d.is_empty());
    it.start_at = o.start_at.clone();
    it.end_at = o.end_at.clone();
    it.phase = Some(o.phase.clone());
    it.about = Some(o.about.clone()).filter(|a| !a.is_empty());
    it.origin = o.origin.clone();
    it.learned_at = Some(now);
    it
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa != 0.0 && bb != 0.0 {
        ab / (aa * bb).sqrt()
    } else {
        0.0
    }
}

/// Where an item's duplicates can be: its project for project knowledge, anywhere for the global and the personal.
/// Observations are reconciled among themselves, and a profile is written again from its items: neither is a
/// candidate for a fact to merge into.
fn scope_filter(it: &MemoryItem) -> Value {
    let mut must = vec![
        json!({ "key": "tier", "match": { "value": "ltm" } }),
        json!({ "key": "status", "match": { "value": "active" } }),
    ];
    if it.scope == "project" {
        must.push(json!({ "key": "project", "match": { "value": it.project } }));
    } else {
        must.push(json!({ "key": "scope", "match": { "any": ["global", "person"] } }));
    }
    json!({ "must": must, "must_not": [{ "key": "kind", "match": { "any": ["observation", "profile"] } }] })
}

/// A thread with records waiting.
#[derive(Debug, Clone, PartialEq)]
pub struct Pending {
    /// The thread.
    pub session: String,
    /// Its records waiting.
    pub uids: Vec<String>,
    /// When the newest was written.
    pub last: f64,
}

/// What a window's settling says to the lanes.
enum Settled {
    Next,
    Stop,
}

/// The consolidation worker of one run (a memory's `consolidate` makes one, and runs it until the queue is read).
pub struct Consolidator {
    d: ConsolidateDeps,
    vecs: Mutex<HashMap<String, Vec<f32>>>,
    /// Windows whose extraction failed, by their first request: how many times in a row.
    failed: Mutex<HashMap<String, usize>>,
    /// Windows read in smaller parts from now on, by their first request: at most this many requests a part.
    narrowed: Mutex<HashMap<String, usize>>,
    /// An item changed by one window at a time: a lock by uid.
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

fn guard<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Consolidator {
    /// A worker over these deps.
    pub fn new(d: ConsolidateDeps) -> Arc<Self> {
        Arc::new(Consolidator {
            d,
            vecs: Mutex::new(HashMap::new()),
            failed: Mutex::new(HashMap::new()),
            narrowed: Mutex::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
        })
    }

    /// Runs with the items locked — taken in uid order, so two windows never wait on each other (sleep's answers too).
    pub async fn locked<T>(&self, uids: &[String], run: impl std::future::Future<Output = Result<T>>) -> Result<T> {
        let mut sorted: Vec<String> = uids.iter().cloned().collect::<HashSet<_>>().into_iter().collect();
        sorted.sort();
        let mut held = Vec::new();
        for uid in &sorted {
            let lock =
                guard(&self.locks).entry(uid.clone()).or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))).clone();
            held.push(lock.lock_owned().await);
        }
        let out = run.await;
        drop(held);
        let mut locks = guard(&self.locks);
        for uid in &sorted {
            if locks.get(uid).is_some_and(|l| Arc::strong_count(l) == 1) {
                locks.remove(uid);
            }
        }
        out
    }

    /// The item as it is now, or `None` when it stopped being active (merged or superseded by a window beside this).
    async fn latest(&self, uid: &str) -> Option<MemoryItem> {
        self.d.ltm.get(uid).await.filter(|it| it.status == "active")
    }

    /// Embed with a per-run cache: the key is embedded once for reconcile and reused when it is written.
    async fn embed(&self, texts: Vec<String>, timeout_ms: Option<u64>) -> Result<Vec<Vec<f32>>> {
        let missing: Vec<String> = {
            let v = guard(&self.vecs);
            unique(texts.iter().filter(|t| !v.contains_key(*t)).cloned())
        };
        if !missing.is_empty() {
            let got = (self.d.embed)(missing.clone(), timeout_ms).await?;
            let mut v = guard(&self.vecs);
            for (t, vec) in missing.into_iter().zip(got) {
                v.insert(t, vec);
            }
        }
        let v = guard(&self.vecs);
        texts
            .iter()
            .map(|t| v.get(t).cloned().ok_or_else(|| Error::Embedding("a text was not embedded".into())))
            .collect()
    }

    fn writer(self: &Arc<Self>) -> WriterDeps {
        let me = self.clone();
        WriterDeps {
            client: self.d.client.clone(),
            embed: Arc::new(move |texts, timeout| {
                let me = me.clone();
                Box::pin(async move { me.embed(texts, timeout).await })
            }),
        }
    }

    /// Threads with records waiting, settled, newest first.
    pub fn pending(&self, now: f64) -> Vec<Pending> {
        let mut by: IndexMap<String, (Vec<String>, f64, f64)> = IndexMap::new();
        for (uid, m) in self.d.stm.entries() {
            if m.done || m.synthetic {
                continue;
            }
            let s = by.entry(m.session.clone()).or_insert((vec![], f64::INFINITY, 0.0));
            s.0.push(uid);
            s.1 = s.1.min(m.ts);
            s.2 = s.2.max(m.ts);
        }
        // Quiet for an hour, or waiting a day since its oldest unread entry: a thread that never goes quiet is still
        // read.
        let mut out: Vec<Pending> = by
            .into_iter()
            .filter(|(_, s)| now - s.2 >= SETTLED_MS || now - s.1 >= DAY_MS)
            .map(|(session, (uids, _, last))| Pending { session, uids, last })
            .collect();
        out.sort_by(|a, b| b.last.partial_cmp(&a.last).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// A window as it is read now: in parts of at most the size its earlier failures narrowed it to, each part
    /// narrowed again by its own failures.
    fn narrow(&self, win: Vec<StmEntry>) -> Vec<Vec<StmEntry>> {
        let n = win.first().and_then(|h| guard(&self.narrowed).get(&h.uid).copied());
        match n {
            Some(n) if n < win.len() => win.chunks(n).flat_map(|c| self.narrow(c.to_vec())).collect(),
            _ => vec![win],
        }
    }

    /// Consolidate up to `max_windows` windows. Stops early when the model pauses or the memory closes. The model
    /// extracts several windows at once (they are independent); each is reconciled in order, newest conversation
    /// first, so a later item still sees what an earlier one wrote.
    pub async fn run(self: &Arc<Self>, max_windows: usize) -> Result<ConsolidationReport> {
        let mut wins: Vec<Vec<StmEntry>> = Vec::new();
        for conv in self.pending(self.d.now()) {
            let mut entries: Vec<StmEntry> = self.d.stm.get_many(conv.uids).await.into_iter().flatten().collect();
            entries.sort_by_key(|e| e.turn);
            for win in windows_of(&entries, self.d.window_chars).into_iter().flat_map(|w| self.narrow(w)) {
                if wins.len() < max_windows {
                    wins.push(win);
                }
            }
            if wins.len() >= max_windows {
                break;
            }
        }
        let report = Arc::new(Mutex::new(ConsolidationReport::default()));
        let limit = self.d.concurrency.max(1);
        let stopped = Arc::new(AtomicBool::new(false));
        // A pool: the next window starts when ANY one is read, not when the oldest is — one slow answer held every
        // other place idle while the windows after it waited their turn. They are still reconciled in order.
        let permits = Arc::new(Semaphore::new(limit));
        // The readings run in a set that aborts them when it is dropped: a run that fails, or is cancelled (its future
        // dropped, by close() or a caller's timeout), leaves no paid model call running behind it. Each window's
        // answer comes back on its own channel, so the windows are still reconciled in order.
        let mut pool = tokio::task::JoinSet::new();
        let mut reading = Vec::with_capacity(wins.len());
        for win in &wins {
            let (me, win, permits, stopped) = (self.clone(), win.clone(), permits.clone(), stopped.clone());
            let (tx, rx) = tokio::sync::oneshot::channel::<Result<Read<Extraction>>>();
            reading.push(Mutex::new(Some(rx)));
            pool.spawn(async move {
                let Ok(_permit) = permits.acquire_owned().await else { return };
                if stopped.load(Ordering::SeqCst) || (me.d.aborted)() {
                    return; // not read: its channel closes unanswered
                }
                let head = &win[0];
                let meta = WindowMeta { cwd: head.cwd.clone(), engine: head.engine.clone(), title: head.title.clone() };
                let read = std::panic::AssertUnwindSafe(me.read(&win, &meta))
                    .catch_unwind()
                    .await
                    .unwrap_or_else(|_| Err(Error::Other("a window's reading failed: it panicked".into())));
                // A failure starts no other window, as the reference's pool launches none after a rejected read.
                if read.is_err() {
                    stopped.store(true, Ordering::SeqCst);
                }
                let _ = tx.send(read);
            });
        }
        let together = self.d.reconcile_concurrency.clamp(1, limit);
        let at = AtomicUsize::new(0);
        let halt = AtomicBool::new(false);
        let lane = || async {
            while !halt.load(Ordering::SeqCst) && !(self.d.aborted)() {
                let i = at.fetch_add(1, Ordering::SeqCst);
                if i >= wins.len() {
                    break;
                }
                let Some(answer) = guard(&reading[i]).take() else { break };
                let Ok(read) = answer.await else {
                    halt.store(true, Ordering::SeqCst);
                    break;
                };
                let x = match read {
                    Ok(x) => x,
                    Err(e) => {
                        stopped.store(true, Ordering::SeqCst);
                        halt.store(true, Ordering::SeqCst);
                        return Err(e);
                    }
                };
                match self.settle(&wins[i], x, &report, together, &stopped).await {
                    Ok(Settled::Next) => {}
                    Ok(Settled::Stop) => {
                        halt.store(true, Ordering::SeqCst);
                        break;
                    }
                    Err(e) => {
                        halt.store(true, Ordering::SeqCst);
                        return Err(e);
                    }
                }
            }
            Ok::<(), Error>(())
        };
        let lanes: Vec<_> = (0..together).map(|_| lane()).collect();
        let done = futures::future::join_all(lanes).await;
        // Windows not reached are not read, and readings no lane waits for any more are stopped.
        stopped.store(true, Ordering::SeqCst);
        pool.abort_all();
        for d in done {
            d?;
        }
        let mut r = guard(&report).clone();
        if halt.load(Ordering::SeqCst) && r.paused.is_some() {
            return Ok(r);
        }
        guard(&self.vecs).clear();
        if !(self.d.aborted)() {
            let (cued, paused) = self.fill_cues(CUE_BACKLOG).await?;
            r.cued = cued;
            if paused.is_some() {
                r.paused = paused;
            }
        }
        self.d.ltm.compact().await?;
        self.d.stm.compact().await?;
        Ok(r)
    }

    /// Window `win`, once read: reconciled, observed, marked done — or `Stop` when the run pauses.
    async fn settle(
        self: &Arc<Self>,
        win: &[StmEntry],
        x: Read<Extraction>,
        report: &Mutex<ConsolidationReport>,
        together: usize,
        stopped: &AtomicBool,
    ) -> Result<Settled> {
        let head = &win[0];
        let x = match x {
            Read::Read(x) => x,
            Read::Unavailable(_) | Read::Busy => {
                // A window that failed before fails for itself (its answer too long to finish, say), not because the
                // model is away: it is read in halves from now on, and the windows after it go on. A first failure
                // before any window was read pauses the run; after others were read it is the window's own, and it
                // waits for the next run.
                let busy = matches!(x, Read::Busy);
                let times = if busy { 0 } else { guard(&self.failed).get(&head.uid).copied().unwrap_or(0) + 1 };
                if times < 2 {
                    if times > 0 {
                        guard(&self.failed).insert(head.uid.clone(), times);
                    }
                    let mut r = guard(report);
                    if r.windows > 0 {
                        return Ok(Settled::Next);
                    }
                    stopped.store(true, Ordering::SeqCst);
                    r.paused = Some(match x {
                        Read::Unavailable(why) => why,
                        _ => "the fast model is busy".into(),
                    });
                    return Ok(Settled::Stop);
                }
                guard(&self.failed).remove(&head.uid);
                if win.len() > 1 {
                    guard(&self.narrowed).insert(head.uid.clone(), win.len().div_ceil(2));
                }
                let why = match &x {
                    Read::Unavailable(w) => w.clone(),
                    _ => "busy".into(),
                };
                (self.d.log)(&format!(
                    "[memory] a window of {} requests failed again ({why}); {}",
                    win.len(),
                    if win.len() > 1 { "reading it in halves" } else { "it waits for the next run" }
                ));
                return Ok(Settled::Next);
            }
        };
        guard(&self.failed).remove(&head.uid);
        let now = self.d.now();
        let by_uid: IndexMap<&str, &StmEntry> = win.iter().map(|e| (e.uid.as_str(), e)).collect();
        // A request the window's typed memories already came from is not kept again as an episode: measured on a full
        // backfill, leaving those out took 4,631 records to 2,782 and raised hit@5.
        let learned: HashSet<&String> = x.candidates.iter().flat_map(|c| c.provenance.iter()).collect();
        let kept: Vec<&KeptEpisode> =
            x.episodes.iter().filter(|k| by_uid.contains_key(k.uid.as_str()) && !learned.contains(&k.uid)).collect();
        let mut fresh: Vec<MemoryItem> =
            kept.iter().filter_map(|k| by_uid.get(k.uid.as_str()).map(|e| episode_item(e, k, now))).collect();
        for c in &x.candidates {
            let src = c.provenance.last().and_then(|u| by_uid.get(u.as_str()).copied()).unwrap_or(head);
            let sources: Vec<&StmEntry> = c.provenance.iter().filter_map(|u| by_uid.get(u.as_str()).copied()).collect();
            fresh.push(item_of(c, src, now, &sources));
        }
        let mut r = ConsolidationReport::default();
        self.reconcile(fresh, &mut r).await?;
        let observed: Vec<MemoryItem> = x
            .observations
            .iter()
            .map(|o| {
                let src = o.provenance.last().and_then(|u| by_uid.get(u.as_str()).copied()).unwrap_or(head);
                observation_item(o, src, now)
            })
            .collect();
        self.observe(observed, &mut r).await?;
        r.dropped += x.dropped;
        r.episodes += kept.len();
        // Read: marked done in the queue. The records stay searchable — the raw record is the truth and the memories
        // learned from them are keys to it. Each as it is now, not as it was read before the model's long answer: one
        // removed meanwhile stays removed, one changed meanwhile is read again.
        // Checked and written as one step in the store, as the reference does it without an await in between: an add
        // landing between a check here and the write would be overwritten, or a version never read marked done.
        let read: HashMap<String, String> = win.iter().map(|e| (e.uid.clone(), e.fingerprint.clone())).collect();
        let uids: Vec<String> = win.iter().map(|e| e.uid.clone()).collect();
        self.d
            .stm
            .update_if(
                uids,
                move |uid, m| read.get(uid).is_some_and(|fp| *fp == m.fp),
                |mut e| {
                    e.done = true;
                    e
                },
            )
            .await?;
        r.windows += 1;
        guard(report).add(&r);
        // One at a time, each window's keys are dropped after it; windows at once share the cache until the run ends.
        if together == 1 {
            guard(&self.vecs).clear();
        }
        Ok(Settled::Next)
    }

    /// One window's reading, and what the guard (when there is one) lets it keep.
    async fn read(self: &Arc<Self>, win: &[StmEntry], meta: &WindowMeta) -> Result<Read<Extraction>> {
        let x = self.extract(win, meta).await?;
        match (&self.d.guard, x) {
            (Some(g), Read::Read(x)) => Ok(g(win.to_vec(), x).await),
            (_, x) => Ok(x),
        }
    }

    /// One window's extraction: the one ask — and, when observations are read apart, the window's small windows too,
    /// each with the one before it as context. Any part not answered leaves the whole window for the next run.
    async fn extract(self: &Arc<Self>, win: &[StmEntry], meta: &WindowMeta) -> Result<Read<Extraction>> {
        let known = if self.d.resolve { Some(self.known_for(win).await) } else { None };
        let ask = self.d.ask.as_ref();
        let Some(chars) = self.d.observe_chars.filter(|c| *c > 0) else {
            return extract_window(win, meta, &ExtractOptions { ask: Some(ask), observations: true, known }).await;
        };
        let subs = windows_of(win, chars);
        let empty: Vec<StmEntry> = vec![];
        let options = ExtractOptions { ask: Some(ask), observations: false, known: known.clone() };
        let main = extract_window(win, meta, &options);
        let seen = futures::future::join_all(subs.iter().enumerate().map(|(k, sub)| {
            observe_window(sub, if k > 0 { &subs[k - 1] } else { &empty }, meta, Some(ask), chars, known.as_deref())
        }));
        let (x, seen) = tokio::join!(main, seen);
        let mut x = match x? {
            Read::Read(x) => x,
            other => return Ok(other),
        };
        let mut observations = Vec::new();
        for o in seen {
            match o? {
                Read::Read(o) => observations.extend(o),
                Read::Unavailable(w) => return Ok(Read::Unavailable(w)),
                Read::Busy => return Ok(Read::Busy),
            }
        }
        x.observations = observations;
        Ok(Read::Read(x))
    }

    /// What other conversations say that this window may refer to: a few of its requests searched for, the best
    /// passages outside this conversation, dated. Nothing found, or a search that fails, reads the window without it.
    async fn known_for(&self, win: &[StmEntry]) -> Vec<String> {
        let step = win.len().div_ceil(RESOLVE_QUERIES).max(1);
        let asks: Vec<String> = win
            .iter()
            .enumerate()
            .filter(|(i, _)| i % step == 0)
            .take(RESOLVE_QUERIES)
            .map(|(_, e)| cut(&e.request, 600))
            .filter(|q| !q.is_empty())
            .collect();
        let session = &win[0].session;
        let deps = RecallDeps {
            client: self.d.client.clone(),
            embed: self.d.embed.clone(),
            used: None,
            record: None,
            rerank: None,
        };
        let searches = futures::future::try_join_all(asks.iter().map(|q| {
            let deps = deps.clone();
            let q = SearchQuery { limit: Some(6), ..SearchQuery::new(q.clone()) };
            async move { memory_search(&deps, &q).await }
        }))
        .await;
        let Ok(found) = searches else { return vec![] };
        let mut best: IndexMap<String, crate::recall::RecalledItem> = IndexMap::new();
        for h in found.into_iter().flatten() {
            let uid = h.id.strip_prefix("stm:").or_else(|| h.id.strip_prefix("ep:"));
            if uid.is_some_and(|u| self.d.stm.meta(u).is_some_and(|m| m.session == *session)) {
                continue;
            }
            if best.get(&h.id).map_or(-1.0, |b| b.score) < h.score {
                best.insert(h.id.clone(), h);
            }
        }
        let mut list: Vec<crate::recall::RecalledItem> = best.into_values().collect();
        list.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        list.into_iter()
            .take(RESOLVE_KEPT)
            .map(|h| {
                let when = h.event_at.clone().unwrap_or_else(|| js::iso_day(h.ts));
                format!("({when}) {}", cut(h.match_.as_deref().unwrap_or(&h.summary), 400))
            })
            .collect()
    }

    /// Encoding's backlog: active items without cues or concepts — saved notes, and what was learned before either
    /// existed — get them, newest first, a few an ask, several asks at once. Cues an item already has stay.
    pub async fn fill_cues(self: &Arc<Self>, max: usize) -> Result<(usize, Option<String>)> {
        // An observation is its own key, with the concepts it was extracted with; a profile has its subject as its
        // concept, and no cues (as measured).
        let mut todo: Vec<(String, LtmMeta)> = self
            .d
            .ltm
            .entries()
            .into_iter()
            .filter(|(_, m)| {
                m.status == "active" && m.kind != "observation" && m.kind != "profile" && (!m.cued || !m.mapped)
            })
            .collect();
        todo.sort_by(|a, b| b.1.ts.partial_cmp(&a.1.ts).unwrap_or(std::cmp::Ordering::Equal));
        todo.truncate(max);
        let items: Vec<MemoryItem> =
            self.d.ltm.get_many(todo.into_iter().map(|x| x.0).collect()).await.into_iter().flatten().collect();
        let batches: Vec<Vec<MemoryItem>> = items.chunks(CUE_BATCH).map(|c| c.to_vec()).collect();
        let limit = self.d.concurrency.max(1);
        let mut cued = 0;
        for group in batches.chunks(limit) {
            if (self.d.aborted)() {
                break;
            }
            let answers: Vec<std::result::Result<Vec<MemoryItem>, String>> =
                futures::future::join_all(group.iter().map(|b| self.cue_batch(b)))
                    .await
                    .into_iter()
                    .collect::<Result<_>>()?;
            let paused = answers.iter().find_map(|a| a.as_ref().err().cloned());
            let got: Vec<MemoryItem> = answers.into_iter().filter_map(|a| a.ok()).flatten().collect();
            if !got.is_empty() {
                let recued: Vec<MemoryItem> =
                    got.iter().filter(|it| !self.d.ltm.meta(&it.uid).is_some_and(|m| m.cued)).cloned().collect();
                let w = self.writer();
                if !recued.is_empty() {
                    write_cues(&w, &recued).await?;
                }
                write_concepts(&w, &got).await?;
                cued += got.len();
                self.d.ltm.put(got).await?;
            }
            if paused.is_some() {
                return Ok((cued, paused));
            }
        }
        Ok((cued, None))
    }

    /// One ask for a few items' cues and concepts; `Ok(Err(why))` when the model did not answer, an error when the ask
    /// failed.
    pub async fn cue_batch(&self, items: &[MemoryItem]) -> Result<std::result::Result<Vec<MemoryItem>, String>> {
        let prompt = items
            .iter()
            .enumerate()
            .map(|(n, it)| {
                let when = if it.applies_when.is_empty() {
                    String::new()
                } else {
                    format!("\nNeeded when: {}", it.applies_when)
                };
                format!("[i{}] ({}) {}\n{}{when}", n + 1, it.kind, it.title, it.summary)
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let req = memory_request(
            CUE_SYSTEM,
            &prompt,
            cue_schema(),
            "memory_cues",
            550 * items.len() as u64 + 200,
            120_000,
            "so these memories wait for their cues",
        );
        let answer = match self.d.ask.ask(req).await? {
            LlmAnswer::Answer { answer, .. } => answer,
            LlmAnswer::Unavailable(why) => return Ok(Err(why)),
            LlmAnswer::Busy => return Ok(Err("the fast model is busy".into())),
        };
        let mut out = Vec::new();
        for x in answer["items"].as_array().into_iter().flatten() {
            let Some(it) = x["id"].as_str().and_then(|id| label_no(id, 'i').checked_sub(1)).and_then(|k| items.get(k))
            else {
                continue;
            };
            let cues = if it.cue_list().is_empty() { clean_cues(&x["cues"]) } else { it.cue_list().to_vec() };
            let concepts =
                if it.concept_list().is_empty() { clean_concepts(&x["concepts"]) } else { it.concept_list().to_vec() };
            // What is missing is asked again next time.
            if !cues.is_empty() || !concepts.is_empty() {
                let mut it = it.clone();
                it.cues = Some(cues);
                it.concepts = Some(concepts);
                out.push(it);
            }
        }
        Ok(Ok(out))
    }

    /// Each item against what long-term memory holds — written as it goes, so a later item sees an earlier one.
    async fn reconcile(self: &Arc<Self>, items: Vec<MemoryItem>, r: &mut ConsolidationReport) -> Result<()> {
        let mut ask: Vec<Pair> = Vec::new();
        // Every key of the window in one embedding call (the cache serves each below and again when it is written).
        self.embed(items.iter().map(|it| doc_text(&it.title, &it.summary)).collect(), Some(embed_budget(items.len())))
            .await?;
        for it in items {
            let v = self.embed(vec![doc_text(&it.title, &it.summary)], None).await?;
            let near = self.d.client.neighbors(&v[0], &scope_filter(&it), 3).await?;
            let top = near.into_iter().find(|n| n.item != it.uid);
            let known = match &top {
                Some(t) => self.d.ltm.get(&t.item).await,
                None => None,
            };
            let score = top.as_ref().map_or(0.0, |t| t.score);
            let same = if it.kind == "episode" { SAME_EPISODE } else { SAME };
            if let Some(known) = known {
                if score >= same {
                    let me = self.clone();
                    let (k, i) = (known.uid.clone(), it.clone());
                    let merged = self
                        .locked(std::slice::from_ref(&known.uid), async move {
                            match me.latest(&k).await {
                                Some(now) => me.merge(now, i, None).await.map(|_| true),
                                None => Ok(false),
                            }
                        })
                        .await?;
                    if merged {
                        r.merged += 1;
                        continue;
                    }
                } else if score >= RELATED && it.kind != "episode" && known.kind != "episode" {
                    ask.push(Pair { it, known });
                    continue;
                }
            }
            self.add(it).await?;
            r.added += 1;
        }
        // The pairs are fixed before any is asked about: every batch asked at once, the answers applied in order.
        let batches = relate_batches(ask, |p| pair_size(&p.it, &p.known));
        let answers: Vec<std::result::Result<HashMap<String, Decision>, String>> =
            futures::future::join_all(batches.iter().map(|b| self.relate(b)))
                .await
                .into_iter()
                .collect::<Result<_>>()?;
        for (batch, got) in batches.into_iter().zip(answers) {
            for (n, Pair { it, known }) in batch.into_iter().enumerate() {
                let d = got.as_ref().ok().and_then(|m| m.get(&format!("p{}", n + 1)).cloned());
                let Some(d) = d.filter(|d| d.relation != "different") else {
                    self.add(it).await?;
                    r.added += 1;
                    continue;
                };
                let me = self.clone();
                let uids = [it.uid.clone(), known.uid.clone()];
                let counts = self
                    .locked(&uids, async move {
                        let mut c = ConsolidationReport::default();
                        // Changed by a window beside this one since it was asked about: the answer was about another
                        // text — kept beside it, linked, for the sleep pass to read again.
                        let now = me.latest(&known.uid).await;
                        let moved = now.as_ref().is_none_or(|n| n.fingerprint != known.fingerprint);
                        if d.relation == "related" || (moved && d.relation != "contradicts") {
                            me.add(it.clone()).await?;
                            c.added += 1;
                            if let Some(now) = now {
                                me.link(it, now).await?;
                                c.linked += 1;
                            }
                            return Ok(c);
                        }
                        if d.relation == "contradicts" {
                            let by = it.uid.clone();
                            me.add(it).await?;
                            c.added += 1;
                            if let Some(now) = now {
                                me.retire(now, &by).await?;
                                c.superseded += 1;
                            }
                            return Ok(c);
                        }
                        let Some(now) = now else { return Ok(c) };
                        let body = if d.body.is_empty() { now.body.clone() } else { d.body.clone() };
                        let text = (!d.title.is_empty() && !d.summary.is_empty())
                            .then(|| (d.title.clone(), d.summary.clone(), body));
                        me.merge(now, it, text).await?;
                        if d.relation == "updates" {
                            c.updated += 1;
                        } else {
                            c.merged += 1;
                        }
                        Ok(c)
                    })
                    .await?;
                r.add(&counts);
            }
        }
        Ok(())
    }

    /// Observations are only ever added: one is the same as a known one only when it says the same
    /// (`SAME_OBSERVATION`) about the same date — then the known one is told again (provenance, `seen`). Never asked
    /// about, never merged into another text, never superseded: an older one stays, dated, beside the newer.
    async fn observe(self: &Arc<Self>, items: Vec<MemoryItem>, r: &mut ConsolidationReport) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        // The window's observations looked up together and written together — and each also against the ones before
        // it in this window, as when each was written before the next looked.
        let vecs = self
            .embed(items.iter().map(|it| doc_text(&it.title, &it.summary)).collect(), Some(embed_budget(items.len())))
            .await?;
        let filter = json!({ "must": [
            { "key": "tier", "match": { "value": "ltm" } }, { "key": "status", "match": { "value": "active" } },
            { "key": "kind", "match": { "value": "observation" } }, { "key": "part", "match": { "value": "key" } },
        ] });
        let near = self.d.client.nearest_many(&vecs, &filter, 3, 30_000, None).await?;
        let mut fresh: Vec<(MemoryItem, Vec<f32>)> = Vec::new();
        // Told again in this window: how many times, by whom, the newest time.
        let mut told: IndexMap<String, (f64, f64, Vec<String>)> = IndexMap::new();
        for (n, it) in items.into_iter().enumerate() {
            let same_date = |e: &Option<String>| e.as_deref().unwrap_or("") == it.event_at.as_deref().unwrap_or("");
            if let Some(f) = fresh
                .iter_mut()
                .find(|(f, v)| f.uid != it.uid && same_date(&f.event_at) && cosine(v, &vecs[n]) >= SAME_OBSERVATION)
            {
                f.0.seen += 1.0;
                f.0.ts = f.0.ts.max(it.ts);
                f.0.provenance = unique(f.0.provenance.iter().cloned().chain(it.provenance.iter().cloned()));
                r.merged += 1;
                continue;
            }
            let hit = near.get(n).and_then(|hs| hs.iter().find(|h| h.item != it.uid));
            let known = match hit.filter(|h| h.score >= SAME_OBSERVATION) {
                Some(h) => self.d.ltm.get(&h.item).await,
                None => None,
            };
            if let Some(known) = known.filter(|k| same_date(&k.event_at)) {
                let t = told.entry(known.uid).or_insert((0.0, 0.0, vec![]));
                t.0 += 1.0;
                t.1 = t.1.max(it.ts);
                t.2 = unique(t.2.iter().cloned().chain(it.provenance.iter().cloned()));
                r.merged += 1;
                continue;
            }
            fresh.push((it, vecs[n].clone()));
            r.observed += 1;
        }
        // Told again: what this window adds put onto the item as it is by now — a window beside this one may have told
        // it too.
        for (uid, (times, ts, provenance)) in told {
            let me = self.clone();
            let u = uid.clone();
            self.locked(&[uid], async move {
                let Some(mut was) = me.d.ltm.get(&u).await else { return Ok(()) };
                was.ts = was.ts.max(ts);
                was.seen += times;
                was.provenance = unique(was.provenance.iter().cloned().chain(provenance));
                me.d.client.set_payload(std::slice::from_ref(&u), json!({ "ts": js::ms_value(was.ts) }), false).await?;
                me.d.ltm.put(vec![was]).await
            })
            .await?;
        }
        if !fresh.is_empty() {
            let items: Vec<MemoryItem> = fresh.into_iter().map(|f| f.0).collect();
            write_ltm(&self.writer(), &items).await?;
            self.d.ltm.put(items).await?;
        }
        Ok(())
    }

    async fn add(self: &Arc<Self>, it: MemoryItem) -> Result<()> {
        write_ltm(&self.writer(), std::slice::from_ref(&it)).await?;
        self.d.ltm.put(vec![it]).await
    }

    /// The known item absorbs the new one: provenance, `seen`, the newest time, the higher importance — and, when the
    /// model merged the texts, the merged key (embedded again). Nothing of either is lost: the new item's own text, and
    /// the known one's before a rewrite, stay in the log as `merged` records the item lists (`absorbed`).
    pub async fn merge(
        self: &Arc<Self>,
        known: MemoryItem,
        it: MemoryItem,
        text: Option<(String, String, String)>,
    ) -> Result<()> {
        let mut kept = vec![MemoryItem { status: "merged".into(), merged_into: Some(known.uid.clone()), ..it.clone() }];
        if text.is_some() {
            kept.push(MemoryItem {
                uid: format!("{}@{}", known.uid, known.fingerprint),
                status: "merged".into(),
                merged_into: Some(known.uid.clone()),
                ..known.clone()
            });
        }
        let mut merged = known.clone();
        if let Some((title, summary, body)) = &text {
            merged.title = redact_string(title);
            merged.summary = redact_string(summary);
            merged.body = redact_string(body);
        }
        merged.ts = known.ts.max(it.ts);
        merged.importance = known.importance.max(it.importance);
        merged.seen = known.seen + 1.0;
        merged.provenance = unique(known.provenance.iter().chain(&it.provenance).cloned());
        merged.files = unique(known.files.iter().chain(&it.files).cloned());
        merged.confidence = if known.confidence == "verified" || it.confidence == "verified" {
            "verified".into()
        } else {
            known.confidence.clone()
        };
        merged.cues = Some(unique(known.cue_list().iter().chain(it.cue_list()).cloned()));
        let related: Vec<String> = unique(known.related_list().iter().chain(it.related_list()).cloned())
            .into_iter()
            .filter(|u| *u != known.uid)
            .collect();
        merged.related = Some(related[related.len().saturating_sub(MAX_LINKS)..].to_vec());
        let mut concepts: IndexMap<String, String> = IndexMap::new();
        for c in known.concept_list().iter().chain(it.concept_list()) {
            concepts.insert(concept_key(c), c.clone());
        }
        merged.concepts = Some(concepts.into_values().collect());
        merged.absorbed =
            Some(unique(known.absorbed.iter().flatten().cloned().chain(kept.iter().map(|k| k.uid.clone()))));
        merged.fingerprint = fingerprint(&[
            merged.kind.as_str().into(),
            merged.title.as_str().into(),
            merged.summary.as_str().into(),
            merged.body.as_str().into(),
        ]);
        // Changed now: what sleep relates next.
        merged.learned_at = Some(learned_at(known.ts, known.learned_at).max(learned_at(it.ts, it.learned_at)));
        let w = self.writer();
        if text.is_some() {
            write_ltm(&w, std::slice::from_ref(&merged)).await?;
        } else {
            let uid = std::slice::from_ref(&known.uid);
            self.d
                .client
                .set_payload(
                    uid,
                    json!({ "ts": js::ms_value(merged.ts), "importance": js::ms_value(merged.importance) }),
                    false,
                )
                .await?;
            if merged.related_list().join("\n") != known.related_list().join("\n") {
                self.d.client.set_payload(uid, json!({ "related": merged.related_list() }), true).await?;
            }
            if merged.concept_list().len() > known.concept_list().len() {
                write_concepts(&w, std::slice::from_ref(&merged)).await?;
            }
            // Learned again in other words: its new cues are found from now on.
            if merged.cue_list().len() > known.cue_list().len() {
                write_cues(&w, std::slice::from_ref(&merged)).await?;
            }
        }
        kept.push(merged);
        self.d.ltm.put(kept).await
    }

    /// Two items about the same subject that say different things: each lists the other — the newest `MAX_LINKS`, so
    /// an item on a busy subject does not grow its list, and its payload, without end.
    pub async fn link(&self, a: MemoryItem, b: MemoryItem) -> Result<()> {
        let add = |x: &MemoryItem, y: &MemoryItem| -> MemoryItem {
            let mut list: Vec<String> = x.related_list().iter().filter(|u| **u != y.uid).cloned().collect();
            list.push(y.uid.clone());
            let start = list.len().saturating_sub(MAX_LINKS);
            MemoryItem { related: Some(list[start..].to_vec()), ..x.clone() }
        };
        let (one, two) = (add(&a, &b), add(&b, &a));
        self.d.ltm.put(vec![one.clone(), two.clone()]).await?;
        self.d
            .client
            .set_payload(std::slice::from_ref(&one.uid), json!({ "related": one.related_list() }), true)
            .await?;
        self.d.client.set_payload(std::slice::from_ref(&two.uid), json!({ "related": two.related_list() }), true).await
    }

    /// Superseded: out of the index and of recall, kept in the log with what replaced it.
    pub async fn retire(self: &Arc<Self>, known: MemoryItem, by: &str) -> Result<()> {
        let old = MemoryItem {
            status: "superseded".into(),
            superseded_by: Some(by.to_string()),
            superseded_at: Some(self.d.now()),
            ..known
        };
        write_ltm(&self.writer(), std::slice::from_ref(&old)).await?;
        self.d.ltm.put(vec![old]).await
    }

    /// The model's decision for each pair, by its label (`p1`, …) — or why it did not answer (`Ok(Err(why))`); an error
    /// when the ask failed.
    pub async fn relate(&self, batch: &[Pair]) -> Result<std::result::Result<HashMap<String, Decision>, String>> {
        // Whole texts: a merge the model writes from a shortened body loses the rest of it.
        let prompt = batch
            .iter()
            .enumerate()
            .map(|(n, p)| {
                format!(
                    "[p{}]\nNEW ({}): {}\n{}\n{}\n\nKNOWN ({}): {}\n{}\n{}",
                    n + 1,
                    p.it.kind,
                    p.it.title,
                    p.it.summary,
                    p.it.body,
                    p.known.kind,
                    p.known.title,
                    p.known.summary,
                    p.known.body
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        // Room for every merged text in full (≈ 2.5 characters a token, the two sides together), with the decisions.
        let room: u64 = batch.iter().map(|p| (pair_size(&p.it, &p.known) as f64 / 2.5).ceil() as u64).sum();
        let req = memory_request(
            RELATE_SYSTEM,
            &prompt,
            relate_schema(),
            "memory_reconcile",
            900 * batch.len() as u64 + 200 + room,
            300_000,
            "so these memories are kept apart for now",
        );
        let answer = match self.d.ask.ask(req).await? {
            LlmAnswer::Answer { answer, .. } => answer,
            LlmAnswer::Unavailable(why) => return Ok(Err(why)),
            LlmAnswer::Busy => return Ok(Err("the fast model is busy".into())),
        };
        let mut out = HashMap::new();
        for p in answer["pairs"].as_array().into_iter().flatten() {
            // A small model may quote the label's line rather than the label: the number is looked for.
            let n = p["id"].as_str().map_or(0, |id| label_no(id, 'p'));
            let relation = p["relation"].as_str().unwrap_or("");
            if n == 0 || !RELATIONS.contains(&relation) {
                continue;
            }
            let field = |k: &str| js::trim(p[k].as_str().unwrap_or("")).to_string();
            out.insert(
                format!("p{n}"),
                Decision {
                    relation: relation.into(),
                    title: field("title"),
                    summary: field("summary"),
                    body: field("body"),
                },
            );
        }
        Ok(Ok(out))
    }

    /// The deps it works with.
    pub fn deps(&self) -> &ConsolidateDeps {
        &self.d
    }
}
