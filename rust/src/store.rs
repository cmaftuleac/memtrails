//! What a memory holds, on disk, as two append-only logs in its home:
//!
//!   - `records.jsonl` — SHORT-TERM memory: every record as it was added (records.rs), kept for good; also the queue
//!     the consolidation worker reads (`done`);
//!   - `knowledge.jsonl` — LONG-TERM memory: what consolidation kept — typed items, dated observations, kept episodes,
//!     profiles — with their keys, status (active / superseded / archived / merged) and provenance.
//!
//! The logs are the truth; Qdrant is derived from them (rebuilt without asking a model, after a Qdrant upgrade or a
//! lost index). In memory: each record's place in its file and a small projection — never the text. A later line for
//! the same uid replaces the earlier one; `{"uid":…,"removed":true}` removes it; a log is rewritten without replaced
//! lines once it is twice what is live.
//!
//! `usage.json`: how often each long-term item was shown and opened (the sleep pass's "use").

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use indexmap::IndexMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha1::{Digest, Sha1};

use crate::error::{Error, Result};
use crate::js;
use crate::records::Metadata;

/// Milliseconds as JavaScript writes them: a whole number without a fraction.
pub(crate) mod ms {
    use super::*;

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> std::result::Result<S::Ok, S::Error> {
        crate::js::ms_value(*v).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(0.0))
    }

    /// The same for an optional time.
    pub mod opt {
        use super::*;

        pub fn serialize<S: Serializer>(v: &Option<f64>, s: S) -> std::result::Result<S::Ok, S::Error> {
            match v {
                Some(x) => crate::js::ms_value(*x).serialize(s),
                None => s.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<f64>, D::Error> {
            Option::<f64>::deserialize(d)
        }
    }
}

/// Where a record came from, when it is a record you added: its own id, its source, who wrote it, where to open it,
/// and your metadata.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RecordRef {
    /// The record's own id.
    pub id: String,
    /// Its source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Who wrote it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Where to open it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Your metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

/// A record as captured: short-term memory (a line of `records.jsonl`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StmEntry {
    /// The record's id in memory (`record_uid`).
    pub uid: String,
    /// What it holds, hashed: the same record added again is unchanged.
    pub fingerprint: String,
    /// The record's kind (`conversation` for an exchange of a conversation).
    pub engine: String,
    /// Its thread.
    pub session: String,
    /// Its place in the thread.
    pub turn: usize,
    /// A working folder (none for records).
    pub cwd: Option<String>,
    /// Its collection key.
    pub project: String,
    /// Its title.
    pub title: String,
    /// When it was written, in ms.
    #[serde(with = "ms")]
    pub ts: f64,
    /// A throwaway run, never shown.
    pub synthetic: bool,
    /// `human` or `unknown`.
    pub typed: String,
    /// What was asked (a record: its title).
    pub request: String,
    /// What was answered (a record: its author and text).
    pub reply: String,
    /// Everything the other side said, whole.
    pub said: String,
    /// Steps taken (none for records).
    #[serde(default)]
    pub steps: Vec<String>,
    /// Files it touched (a record: its url).
    #[serde(default)]
    pub files: Vec<String>,
    /// Read by the consolidation worker.
    pub done: bool,
    /// The record it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record: Option<RecordRef>,
    /// Fields another version wrote, kept as they are.
    #[serde(flatten)]
    pub extra: IndexMap<String, Value>,
}

/// What a kept thing was learned from, and the exact words that show it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryOrigin {
    /// `user` or `result`.
    pub by: String,
    /// The words that show it.
    pub evidence: String,
}

/// What long-term memory keeps (a line of `knowledge.jsonl`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryItem {
    /// `mem:…`, `ep:…`, `obs:…`, `pf:…`.
    pub uid: String,
    /// decision, procedure, snippet, template, pitfall, fact, architecture, preference, reference, episode, note,
    /// observation or profile.
    pub kind: String,
    /// project, global or person.
    pub scope: String,
    /// Its collection key.
    pub project: String,
    /// The kind of the record it came from.
    pub engine: String,
    /// The thread it came from.
    pub session: Option<String>,
    /// A working folder.
    pub cwd: Option<String>,
    /// Where it came from, for a person: the conversation's title, the note's name.
    #[serde(rename = "where")]
    pub where_: String,
    /// Last learned or refreshed (its source's time).
    #[serde(with = "ms")]
    pub ts: f64,
    /// First learned.
    #[serde(with = "ms")]
    pub first_ts: f64,
    /// When it was written to long-term memory.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "ms::opt")]
    pub learned_at: Option<f64>,
    /// A content title (part of the key).
    pub title: String,
    /// A ~150-word summary (the rest of the key).
    pub summary: String,
    /// Everything needed to reuse it.
    pub body: String,
    /// Files it names.
    #[serde(default)]
    pub files: Vec<String>,
    /// When it is needed.
    #[serde(default)]
    pub applies_when: String,
    /// verified, stated, curated or observed.
    pub confidence: String,
    /// 1–5.
    #[serde(with = "ms")]
    pub importance: f64,
    /// active, superseded, archived or merged.
    pub status: String,
    /// What replaced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
    /// When it was superseded.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "ms::opt")]
    pub superseded_at: Option<f64>,
    /// For an episode: success, partial, fail or uncertain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// For a `merged` record: the item that absorbed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_into: Option<String>,
    /// The `merged` records this item absorbed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absorbed: Option<Vec<String>>,
    /// The requests it was learned from.
    #[serde(default)]
    pub provenance: Vec<String>,
    /// How many times it was learned.
    #[serde(with = "ms")]
    pub seen: f64,
    /// What it says, hashed.
    pub fingerprint: String,
    /// Requests the person would type when it is needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cues: Option<Vec<String>>,
    /// Where it came from, when a guard checked it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<MemoryOrigin>,
    /// Items about the same subject that say different things.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related: Option<Vec<String>>,
    /// What it is about: the concept graph's links.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concepts: Option<Vec<String>>,
    /// For an observation: when it happened (YYYY-MM-DD, YYYY-MM or YYYY).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_at: Option<String>,
    /// For something that lasted: when it began.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_at: Option<String>,
    /// For something that lasted: when it ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_at: Option<String>,
    /// done, plan, ongoing or state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// Who or what it is about.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    /// Fields another version wrote, kept as they are.
    #[serde(flatten)]
    pub extra: IndexMap<String, Value>,
}

impl MemoryItem {
    /// Its cues, none when it has none.
    pub fn cue_list(&self) -> &[String] {
        self.cues.as_deref().unwrap_or(&[])
    }

    /// Its concepts, none when it has none.
    pub fn concept_list(&self) -> &[String] {
        self.concepts.as_deref().unwrap_or(&[])
    }

    /// Its links, none when it has none.
    pub fn related_list(&self) -> &[String] {
        self.related.as_deref().unwrap_or(&[])
    }
}

/// One part of a fingerprint: a text, a list (its items joined by U+0001), or nothing (as an empty text).
#[derive(Debug, Clone, Copy)]
pub enum FpPart<'a> {
    /// A text.
    Text(&'a str),
    /// A list.
    List(&'a [String]),
    /// Nothing.
    Null,
}

impl<'a> From<&'a str> for FpPart<'a> {
    fn from(s: &'a str) -> Self {
        FpPart::Text(s)
    }
}

impl<'a> From<&'a String> for FpPart<'a> {
    fn from(s: &'a String) -> Self {
        FpPart::Text(s)
    }
}

impl<'a> From<&'a [String]> for FpPart<'a> {
    fn from(s: &'a [String]) -> Self {
        FpPart::List(s)
    }
}

impl<'a> From<&'a Vec<String>> for FpPart<'a> {
    fn from(s: &'a Vec<String>) -> Self {
        FpPart::List(s)
    }
}

impl<'a> From<Option<&'a str>> for FpPart<'a> {
    fn from(s: Option<&'a str>) -> Self {
        s.map_or(FpPart::Null, FpPart::Text)
    }
}

/// SHA-1 over the parts, each followed by NUL (a list joined by U+0001), the first 16 hex.
pub fn fingerprint(parts: &[FpPart<'_>]) -> String {
    let mut h = Sha1::new();
    for p in parts {
        match p {
            FpPart::Text(s) => h.update(s.as_bytes()),
            FpPart::List(l) => h.update(l.join("\u{1}").as_bytes()),
            FpPart::Null => {}
        }
        h.update([0u8]);
    }
    hex::encode(h.finalize())[..16].to_string()
}

/// A record a log keeps: anything with a uid.
pub trait Keyed: Serialize + DeserializeOwned + Clone + Send + Sync + 'static {
    /// Its uid.
    fn uid(&self) -> &str;
}

impl Keyed for StmEntry {
    fn uid(&self) -> &str {
        &self.uid
    }
}

impl Keyed for MemoryItem {
    fn uid(&self) -> &str {
        &self.uid
    }
}

#[derive(Debug, Clone)]
struct Slot<M> {
    at: u64,
    len: u64,
    meta: M,
}

/// What a compaction holds before it writes.
const WRITE_BYTES: usize = 1 << 20;

struct Inner<M> {
    live: IndexMap<String, Slot<M>>,
    lines: usize,
    end: u64,
}

#[derive(Deserialize)]
struct Removal {
    uid: String,
    #[serde(default)]
    removed: bool,
}

/// An append-only JSONL log of records by uid, read back by byte offset: a later line for the same uid replaces the
/// earlier one. `meta` is what stays in memory for each record. Safe to share between tasks; every file access runs
/// on a blocking thread (`spawn_blocking`), never on the async runtime's own.
pub struct JsonlLog<T: Keyed, M: Clone + Send + 'static> {
    shared: Arc<Shared<T, M>>,
}

impl<T: Keyed, M: Clone + Send + 'static> Clone for JsonlLog<T, M> {
    fn clone(&self) -> Self {
        JsonlLog { shared: self.shared.clone() }
    }
}

struct Shared<T, M> {
    path: PathBuf,
    project: fn(&T) -> M,
    inner: Mutex<Inner<M>>,
}

#[cfg(unix)]
fn private(o: &mut OpenOptions) -> &mut OpenOptions {
    use std::os::unix::fs::OpenOptionsExt;
    o.mode(0o600)
}

#[cfg(not(unix))]
fn private(o: &mut OpenOptions) -> &mut OpenOptions {
    o
}

/// Append bytes to a file made private when it is made.
pub(crate) fn append_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = private(OpenOptions::new().create(true).append(true)).open(path)?;
    f.write_all(bytes)
}

/// Write a whole file, private when it is made.
pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = private(OpenOptions::new().create(true).write(true).truncate(true)).open(path)?;
    f.write_all(bytes)
}

/// Run file work on a blocking thread.
pub(crate) async fn blocking<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> Result<R> {
    tokio::task::spawn_blocking(f).await.map_err(|e| Error::Other(format!("a file task failed: {e}")))
}

impl<T: Keyed, M: Clone + Send + 'static> Shared<T, M> {
    fn lock(&self) -> MutexGuard<'_, Inner<M>> {
        // A panic while holding it left plain data; what it holds is still the log's index.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn load(&self) -> Result<()> {
        let mut g = self.lock();
        g.live.clear();
        g.lines = 0;
        g.end = 0;
        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(Error::io(&self.path, e)),
        };
        // By bytes, not decoded lines: a torn multi-byte character must not shift every offset after it.
        let mut reader = BufReader::new(file);
        let mut at = 0u64;
        let mut line = Vec::new();
        let take = |g: &mut Inner<M>, line: &[u8], at: u64, len: u64| -> bool {
            if line.is_empty() {
                return true;
            }
            let Ok(text) = std::str::from_utf8(line) else { return false };
            let Ok(v) = serde_json::from_str::<Value>(text) else { return false };
            if let Ok(r) = serde_json::from_value::<Removal>(v.clone()) {
                if r.removed {
                    g.live.shift_remove(&r.uid);
                    g.lines += 1;
                    return true;
                }
            }
            match serde_json::from_value::<T>(v) {
                Ok(rec) => {
                    let meta = (self.project)(&rec);
                    g.live.insert(rec.uid().to_string(), Slot { at, len, meta });
                    g.lines += 1;
                    true
                }
                Err(_) => false,
            }
        };
        loop {
            line.clear();
            let n = reader.read_until(b'\n', &mut line).map_err(|e| Error::io(&self.path, e))?;
            if n == 0 {
                break;
            }
            if line.last() == Some(&b'\n') {
                let body = &line[..line.len() - 1];
                take(&mut g, body, at, n as u64);
                at += n as u64;
            } else {
                // A crash cut the last line short of its '\n': whole but for it, it is ended; torn, it goes. Either
                // way what is added next starts on a line of its own, at the offset the index gives it.
                let tail = line.clone();
                if take(&mut g, &tail, at, tail.len() as u64) {
                    let _ = append_private(&self.path, b"\n");
                } else if let Ok(f) = OpenOptions::new().write(true).open(&self.path) {
                    let _ = f.set_len(at);
                }
                break;
            }
        }
        g.end = fs::metadata(&self.path).map_err(|e| Error::io(&self.path, e))?.len();
        Ok(())
    }

    fn read_slot(file: &mut File, s: &Slot<M>) -> Option<T> {
        file.seek(SeekFrom::Start(s.at)).ok()?;
        let mut buf = vec![0u8; s.len as usize];
        file.read_exact(&mut buf).ok()?;
        let text = std::str::from_utf8(&buf).ok()?;
        serde_json::from_str(js::trim_end(text)).ok()
    }

    fn get_many(&self, uids: &[String]) -> Vec<Option<T>> {
        let slots: Vec<Option<Slot<M>>> = {
            let g = self.lock();
            uids.iter().map(|u| g.live.get(u.as_str()).cloned()).collect()
        };
        let mut file = File::open(&self.path).ok();
        slots.into_iter().map(|s| Self::read_slot(file.as_mut()?, s.as_ref()?)).collect()
    }

    fn put(&self, recs: &[T]) -> Result<()> {
        let mut g = self.lock();
        self.put_locked(&mut g, recs)
    }

    /// The records whose projection `keep` still accepts, changed by `change` and appended — under the lock, so no
    /// other put lands between the check, the read and the write.
    fn update_if(
        &self,
        uids: &[String],
        keep: &(dyn Fn(&str, &M) -> bool + Send + Sync),
        change: &(dyn Fn(T) -> T + Send + Sync),
    ) -> Result<usize> {
        let mut g = self.lock();
        let slots: Vec<Slot<M>> =
            uids.iter().filter_map(|u| g.live.get(u.as_str()).filter(|s| keep(u, &s.meta)).cloned()).collect();
        if slots.is_empty() {
            return Ok(0);
        }
        let mut file = File::open(&self.path).map_err(|e| Error::io(&self.path, e))?;
        let recs: Vec<T> = slots.iter().filter_map(|s| Self::read_slot(&mut file, s)).map(change).collect();
        self.put_locked(&mut g, &recs)?;
        Ok(recs.len())
    }

    fn put_locked(&self, g: &mut Inner<M>, recs: &[T]) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        }
        if g.end == 0 {
            if let Ok(m) = fs::metadata(&self.path) {
                g.end = m.len();
            }
        }
        let mut bytes = Vec::new();
        let mut at = g.end;
        let mut slots = Vec::new();
        for rec in recs {
            let mut line = serde_json::to_vec(rec).map_err(Error::json)?;
            line.push(b'\n');
            slots.push((rec.uid().to_string(), Slot { at, len: line.len() as u64, meta: (self.project)(rec) }));
            at += line.len() as u64;
            bytes.extend_from_slice(&line);
        }
        append_private(&self.path, &bytes).map_err(|e| Error::io(&self.path, e))?;
        for (uid, slot) in slots {
            g.live.insert(uid, slot);
        }
        g.end = at;
        g.lines += recs.len();
        Ok(())
    }

    fn remove(&self, uids: &[String]) -> Result<()> {
        let mut g = self.lock();
        let gone: Vec<String> = uids.iter().filter(|u| g.live.contains_key(u.as_str())).cloned().collect();
        if gone.is_empty() {
            return Ok(());
        }
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        }
        let mut bytes = Vec::new();
        for uid in &gone {
            bytes.extend_from_slice(js::stringify(&serde_json::json!({ "uid": uid, "removed": true })).as_bytes());
            bytes.push(b'\n');
        }
        append_private(&self.path, &bytes).map_err(|e| Error::io(&self.path, e))?;
        if g.end == 0 {
            g.end = fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0).saturating_sub(bytes.len() as u64);
        }
        g.end += bytes.len() as u64;
        g.lines += gone.len();
        for uid in gone {
            g.live.shift_remove(&uid);
        }
        Ok(())
    }

    fn compact(&self) -> Result<()> {
        let mut g = self.lock();
        if g.lines < 2 * g.live.len() + 1000 {
            return Ok(());
        }
        let mut order: Vec<(String, Slot<M>)> = g.live.iter().map(|(k, s)| (k.clone(), s.clone())).collect();
        order.sort_by_key(|x| x.1.at);
        let tmp = PathBuf::from(format!("{}.compact", self.path.display()));
        let result = (|| -> Result<IndexMap<String, Slot<M>>> {
            let mut src = File::open(&self.path).map_err(|e| Error::io(&self.path, e))?;
            let mut out = private(OpenOptions::new().create(true).write(true).truncate(true))
                .open(&tmp)
                .map_err(|e| Error::io(&tmp, e))?;
            let mut next = IndexMap::new();
            let mut at = 0u64;
            let mut parts: Vec<u8> = Vec::new();
            for (uid, s) in order {
                let rec = match Self::read_slot(&mut src, &s) {
                    Some(r) if r.uid() == uid => r,
                    _ => {
                        return Err(Error::Store(format!(
                            "{}: record {uid} cannot be read at {}; not compacted",
                            self.path.display(),
                            s.at
                        )))
                    }
                };
                let mut line = serde_json::to_vec(&rec).map_err(Error::json)?;
                line.push(b'\n');
                next.insert(uid, Slot { at, len: line.len() as u64, meta: s.meta });
                at += line.len() as u64;
                parts.extend_from_slice(&line);
                if parts.len() >= WRITE_BYTES {
                    out.write_all(&parts).map_err(|e| Error::io(&tmp, e))?;
                    parts.clear();
                }
            }
            out.write_all(&parts).map_err(|e| Error::io(&tmp, e))?;
            out.sync_all().map_err(|e| Error::io(&tmp, e))?;
            drop(out);
            fs::rename(&tmp, &self.path).map_err(|e| Error::io(&self.path, e))?;
            Ok(next)
        })();
        match result {
            Ok(next) => {
                g.end = next.values().map(|s| s.at + s.len).max().unwrap_or(0);
                g.lines = next.len();
                g.live = next;
                Ok(())
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                Err(e)
            }
        }
    }
}

impl<T: Keyed, M: Clone + Send + 'static> JsonlLog<T, M> {
    /// A log at `path`, empty until `load`.
    pub fn new(path: impl Into<PathBuf>, project: fn(&T) -> M) -> Self {
        JsonlLog {
            shared: Arc::new(Shared {
                path: path.into(),
                project,
                inner: Mutex::new(Inner { live: IndexMap::new(), lines: 0, end: 0 }),
            }),
        }
    }

    /// Where it is kept.
    pub fn path(&self) -> &Path {
        &self.shared.path
    }

    /// How many live records.
    pub fn size(&self) -> usize {
        self.shared.lock().live.len()
    }

    /// A record's projection.
    pub fn meta(&self, uid: &str) -> Option<M> {
        self.shared.lock().live.get(uid).map(|s| s.meta.clone())
    }

    /// Every live record's uid and projection, in the order they were first written.
    pub fn entries(&self) -> Vec<(String, M)> {
        self.shared.lock().live.iter().map(|(k, s)| (k.clone(), s.meta.clone())).collect()
    }

    /// Every live record's uid, in file order (oldest first).
    pub fn order(&self) -> Vec<String> {
        let mut order: Vec<(String, u64)> = self.shared.lock().live.iter().map(|(k, s)| (k.clone(), s.at)).collect();
        order.sort_by_key(|x| x.1);
        order.into_iter().map(|x| x.0).collect()
    }

    /// Read the file: every live record's place and projection. A crash's cut-off last line is ended when it is
    /// whole but for its `\n`, and cut off when it is torn, so what is added next starts on a line of its own.
    pub async fn load(&self) -> Result<()> {
        let s = self.shared.clone();
        blocking(move || s.load()).await?
    }

    /// A live record, read from its place in the file; `None` when it is not live or cannot be read.
    pub async fn get(&self, uid: &str) -> Option<T> {
        self.get_many(vec![uid.to_string()]).await.into_iter().next().flatten()
    }

    /// Several records at once (one read of the file), in the order asked.
    pub async fn get_many(&self, uids: Vec<String>) -> Vec<Option<T>> {
        if uids.is_empty() {
            return vec![];
        }
        let n = uids.len();
        let s = self.shared.clone();
        blocking(move || s.get_many(&uids)).await.unwrap_or_else(|_| (0..n).map(|_| None).collect())
    }

    /// Append records (kept when this returns).
    pub async fn put(&self, recs: Vec<T>) -> Result<()> {
        if recs.is_empty() {
            return Ok(());
        }
        let s = self.shared.clone();
        blocking(move || s.put(&recs)).await?
    }

    /// These records changed by `change` and appended, each only while its projection is one `keep` accepts — the
    /// check, the read and the write as one step, so a put made meanwhile is neither overwritten nor changed unread.
    /// How many were changed.
    pub async fn update_if(
        &self,
        uids: Vec<String>,
        keep: impl Fn(&str, &M) -> bool + Send + Sync + 'static,
        change: impl Fn(T) -> T + Send + Sync + 'static,
    ) -> Result<usize> {
        if uids.is_empty() {
            return Ok(0);
        }
        let s = self.shared.clone();
        blocking(move || s.update_if(&uids, &keep, &change)).await?
    }

    /// Forget these records: gone from memory, and from the file when it is next compacted.
    pub async fn remove(&self, uids: Vec<String>) -> Result<()> {
        let s = self.shared.clone();
        blocking(move || s.remove(&uids)).await?
    }

    /// Every live record, in file order (oldest first).
    pub async fn all(&self) -> Vec<T> {
        let order = self.order();
        self.get_many(order).await.into_iter().flatten().collect()
    }

    /// Rewrite the log without replaced lines once it is twice what is live. Record by record into a file beside it,
    /// made durable and then renamed over it: a log of any size, and a crash leaves the old one whole. A record that
    /// cannot be read is never dropped by a rewrite: it fails, and the log stays as it is.
    pub async fn compact(&self) -> Result<()> {
        let s = self.shared.clone();
        blocking(move || s.compact()).await?
    }
}

/// What stays in memory of a short-term entry.
#[derive(Debug, Clone, PartialEq)]
pub struct StmMeta {
    /// Its fingerprint.
    pub fp: String,
    /// Read by consolidation.
    pub done: bool,
    /// When it was written.
    pub ts: f64,
    /// Its thread.
    pub session: String,
    /// Its collection key.
    pub project: String,
    /// A throwaway run.
    pub synthetic: bool,
}

/// What stays in memory of a long-term item.
#[derive(Debug, Clone, PartialEq)]
pub struct LtmMeta {
    /// Its fingerprint.
    pub fp: String,
    /// Its status.
    pub status: String,
    /// Its kind.
    pub kind: String,
    /// Its scope.
    pub scope: String,
    /// Its collection key.
    pub project: String,
    /// Its source's time.
    pub ts: f64,
    /// 1–5.
    pub importance: f64,
    /// It has cues.
    pub cued: bool,
    /// It has concepts.
    pub mapped: bool,
    /// When it was written to long-term memory.
    pub learned_at: Option<f64>,
}

/// The projection of a short-term entry.
pub fn stm_meta_of(r: &StmEntry) -> StmMeta {
    StmMeta {
        fp: r.fingerprint.clone(),
        done: r.done,
        ts: r.ts,
        session: r.session.clone(),
        project: r.project.clone(),
        synthetic: r.synthetic,
    }
}

/// The projection of a long-term item.
pub fn ltm_meta_of(r: &MemoryItem) -> LtmMeta {
    LtmMeta {
        fp: r.fingerprint.clone(),
        status: r.status.clone(),
        kind: r.kind.clone(),
        scope: r.scope.clone(),
        project: r.project.clone(),
        ts: r.ts,
        importance: r.importance,
        cued: !r.cue_list().is_empty(),
        mapped: !r.concept_list().is_empty(),
        learned_at: r.learned_at,
    }
}

/// The records log.
pub type StmStore = JsonlLog<StmEntry, StmMeta>;
/// The knowledge log.
pub type LtmStore = JsonlLog<MemoryItem, LtmMeta>;

/// The records log at `path`.
pub fn stm_log(path: impl Into<PathBuf>) -> StmStore {
    JsonlLog::new(path, stm_meta_of)
}

/// The knowledge log at `path`.
pub fn ltm_log(path: impl Into<PathBuf>) -> LtmStore {
    JsonlLog::new(path, ltm_meta_of)
}

/// How often one item was shown and opened, and when last.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Use {
    /// Shown by a search.
    pub shown: u64,
    /// Opened by an agent.
    pub opened: u64,
    /// When last.
    #[serde(with = "ms")]
    pub last_used: f64,
}

struct Book {
    data: IndexMap<String, Use>,
    dirty: bool,
    armed: bool,
}

/// How often each long-term item was shown with a search and opened by an agent — what the sleep pass reads as "use".
/// Saved at most once a minute, and on `flush`.
#[derive(Clone)]
pub struct UsageBook {
    path: PathBuf,
    book: Arc<Mutex<Book>>,
}

impl UsageBook {
    /// The book at `path` (empty when there is none, or it cannot be read).
    pub async fn open(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let data =
            tokio::fs::read_to_string(&path).await.ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        UsageBook { path, book: Arc::new(Mutex::new(Book { data, dirty: false, armed: false })) }
    }

    fn lock(&self) -> MutexGuard<'_, Book> {
        self.book.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One item's use.
    pub fn of(&self, uid: &str) -> Use {
        self.lock().data.get(uid).copied().unwrap_or_default()
    }

    /// These items were shown (or opened) at `now`.
    pub fn note(&self, uids: &[String], opened: bool, now: f64) {
        let arm = {
            let mut b = self.lock();
            for uid in uids {
                let u = b.data.entry(uid.clone()).or_default();
                if opened {
                    u.opened += 1;
                } else {
                    u.shown += 1;
                }
                u.last_used = now;
            }
            b.dirty = true;
            !std::mem::replace(&mut b.armed, true)
        };
        if arm {
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                let me = self.clone();
                rt.spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    me.flush().await;
                });
            }
        }
    }

    /// Write it now, when anything changed (a read-only home keeps it in memory).
    pub async fn flush(&self) {
        let me = self.clone();
        let _ = blocking(move || {
            let mut b = me.lock();
            b.armed = false;
            if !b.dirty {
                return;
            }
            if let Some(dir) = me.path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            let text = serde_json::to_string(&b.data).unwrap_or_else(|_| "{}".into());
            if write_private(&me.path, text.as_bytes()).is_ok() {
                b.dirty = false;
            }
        })
        .await;
    }
}
