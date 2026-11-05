//! What recall showed, for what, and what was opened after it — the record a model that learns from USE reads.
//!
//! The usage counts (`UsageBook`) say how often, never with what nor for what; this says both. One line per event in
//! `recall.jsonl` beside the logs — on this machine only, cleared with the index, never sent:
//!
//! ```text
//! {"t", "ev": "search", "project", "query", "shown": [ids]}    memory_search
//! {"t", "ev": "open",   "item"}                               memory_open
//! ```
//!
//! Appended in order, a line at a time; a write that fails is dropped (recall never waits on, or fails for, its
//! record). Kept: the last `RECALL_KEEP_MS`, at most `RECALL_KEEP_BYTES` of the newest (`trim`: when memory opens,
//! and every `RECALL_TRIM_EVERY` events).

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::js;
use crate::store::{append_private, blocking, write_private};

/// A query kept up to this: the start of a long message is what the search embedded first.
pub const RECALL_QUERY_CHARS: usize = 2000;
/// Events older than this are dropped.
pub const RECALL_KEEP_MS: f64 = 180.0 * 24.0 * 60.0 * 60_000.0;
/// At most this many bytes of the newest are kept.
pub const RECALL_KEEP_BYTES: usize = 32 * 1024 * 1024;
/// Events between trims of a book left open for long (a server): the cap holds without a restart.
pub const RECALL_TRIM_EVERY: u64 = 1000;

/// One event.
#[derive(Debug, Clone, PartialEq)]
pub enum RecallEvent {
    /// A search showed these items for this query.
    Search {
        /// The collections it was scoped to, joined by commas.
        project: Option<String>,
        /// What was asked.
        query: String,
        /// What was shown, best first.
        shown: Vec<String>,
    },
    /// An item was opened.
    Open {
        /// The item.
        item: String,
    },
}

enum Job {
    Line(String),
    Trim(tokio::sync::oneshot::Sender<usize>),
    Settled(tokio::sync::oneshot::Sender<()>),
}

/// The recall log of one memory. Events are written in order by one task of its own, the file work on a blocking
/// thread: recording never waits.
#[derive(Clone)]
pub struct RecallBook {
    now: Arc<dyn Fn() -> f64 + Send + Sync>,
    jobs: tokio::sync::mpsc::UnboundedSender<Job>,
    appended: Arc<AtomicU64>,
}

struct Files {
    path: PathBuf,
    now: Arc<dyn Fn() -> f64 + Send + Sync>,
    keep_ms: f64,
    keep_bytes: usize,
}

impl RecallBook {
    /// The log at `path`, kept as the constants say. Starts its writer on the current tokio runtime.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with(path, Arc::new(js::now_ms), RECALL_KEEP_MS, RECALL_KEEP_BYTES)
    }

    /// The log at `path` with its own clock and limits (tests).
    pub fn with(
        path: impl Into<PathBuf>,
        now: Arc<dyn Fn() -> f64 + Send + Sync>,
        keep_ms: f64,
        keep_bytes: usize,
    ) -> Self {
        let (jobs, mut rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let files = Arc::new(Files { path: path.into(), now: now.clone(), keep_ms, keep_bytes });
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let f = files.clone();
                match job {
                    // Recall never fails for its record.
                    Job::Line(line) => {
                        let _ = blocking(move || f.append(&line)).await;
                    }
                    Job::Trim(done) => {
                        let _ = done.send(blocking(move || f.trim()).await.unwrap_or(0));
                    }
                    Job::Settled(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        RecallBook { now, jobs, appended: Arc::new(AtomicU64::new(0)) }
    }

    /// Append one event (in order; a failed write is dropped).
    pub fn record(&self, e: &RecallEvent) {
        let mut m = Map::new();
        m.insert("t".into(), js::ms_value((self.now)()));
        match e {
            RecallEvent::Search { project, query, shown } => {
                m.insert("ev".into(), json!("search"));
                if let Some(p) = project {
                    m.insert("project".into(), json!(p));
                }
                m.insert("query".into(), json!(js::head16(query, RECALL_QUERY_CHARS)));
                m.insert("shown".into(), json!(shown));
            }
            RecallEvent::Open { item } => {
                m.insert("ev".into(), json!("open"));
                m.insert("item".into(), json!(item));
            }
        }
        let _ = self.jobs.send(Job::Line(format!("{}\n", js::stringify(&Value::Object(m)))));
        if (self.appended.fetch_add(1, Ordering::SeqCst) + 1).is_multiple_of(RECALL_TRIM_EVERY) {
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let _ = self.jobs.send(Job::Trim(tx));
        }
    }

    /// The events older than the time kept dropped, then the oldest until the rest fit the bytes kept — rewritten
    /// between appends (a new file renamed over the old). How many were dropped; 0 on any failure, the log as it was.
    /// Queued in order with the events at the call — what is recorded after it is the next trim's — and run between
    /// appends.
    pub fn trim(&self) -> impl std::future::Future<Output = usize> + Send + 'static {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let sent = self.jobs.send(Job::Trim(tx)).is_ok();
        async move {
            if sent {
                rx.await.unwrap_or(0)
            } else {
                0
            }
        }
    }

    /// Every event recorded so far written (tests, shutdown).
    pub async fn settled(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.jobs.send(Job::Settled(tx)).is_ok() {
            let _ = rx.await;
        }
    }
}

/// `Number(v)` for what a log line's time can be: a number, a numeric text, `true`/`false`/`null`; NaN otherwise.
fn js_number(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => {
            let t = crate::js::trim(s);
            match t {
                "" => 0.0,
                "Infinity" | "+Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                // Rust would also read `inf` and `nan`, which JavaScript does not.
                _ if t.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-')) => {
                    t.parse().unwrap_or(f64::NAN)
                }
                _ => f64::NAN,
            }
        }
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Null => 0.0,
        _ => f64::NAN,
    }
}

impl Files {
    fn append(&self, line: &str) {
        if let Some(dir) = self.path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = append_private(&self.path, line.as_bytes());
    }

    fn trim(&self) -> usize {
        // Decoded as the reference reads it (`readFile(path, 'utf8')`): a byte that is not UTF-8 is U+FFFD, not a
        // reason to leave the whole log untrimmed.
        let Ok(bytes) = fs::read(&self.path) else { return 0 };
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.split('\n').filter(|l| !l.is_empty()).collect();
        let since = (self.now)() - self.keep_ms;
        let recent: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|l| serde_json::from_str::<Value>(l).ok().is_some_and(|v| js_number(&v["t"]) >= since))
            .collect();
        let mut bytes: usize = recent.iter().map(|l| l.len() + 1).sum();
        let mut from = 0;
        while bytes > self.keep_bytes && from < recent.len() {
            bytes -= recent[from].len() + 1;
            from += 1;
        }
        let kept = &recent[from..];
        if kept.len() == lines.len() {
            return 0;
        }
        let next = PathBuf::from(format!("{}.trim", self.path.display()));
        let body: String = kept.iter().map(|l| format!("{l}\n")).collect();
        if write_private(&next, body.as_bytes()).is_err() || fs::rename(&next, &self.path).is_err() {
            let _ = fs::remove_file(&next);
            return 0;
        }
        lines.len() - kept.len()
    }
}
