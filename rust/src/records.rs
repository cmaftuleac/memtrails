//! What memtrails stores: records. Anything with text is a record — an e-mail, a chat message, a note, a memo, a news
//! article, a commit message, a ticket, a page of a document. Records that belong together (an e-mail thread, a chat
//! thread, a meeting's notes) share a `thread`; a conversation between a person and an assistant is a thread of turns.
//!
//! Every record is searchable the moment it is added, with no model asked. Consolidation (optional, with a language
//! model) reads threads in windows and keeps typed knowledge from them; the records themselves are never lost.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::js;
use crate::store::{RecordRef, StmEntry};

/// When something was written: milliseconds since the epoch, or a date as text (ISO 8601 or RFC 2822).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum At {
    /// Milliseconds since the epoch.
    Ms(f64),
    /// A date as text, read the way JavaScript's `Date.parse` reads it.
    Text(String),
}

impl From<f64> for At {
    fn from(ms: f64) -> Self {
        At::Ms(ms)
    }
}

impl From<i64> for At {
    fn from(ms: i64) -> Self {
        At::Ms(ms as f64)
    }
}

impl From<&str> for At {
    fn from(s: &str) -> Self {
        At::Text(s.to_string())
    }
}

impl From<String> for At {
    fn from(s: String) -> Self {
        At::Text(s)
    }
}

impl From<std::time::SystemTime> for At {
    fn from(t: std::time::SystemTime) -> Self {
        let ms = match t.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_millis() as f64,
            Err(e) => -(e.duration().as_millis() as f64),
        };
        At::Ms(ms)
    }
}

impl At {
    /// As it reads in a message: the text, or the number as JavaScript writes it.
    fn shown(&self) -> String {
        match self {
            At::Ms(ms) => js::number(*ms),
            At::Text(s) => s.clone(),
        }
    }
}

/// Metadata kept with a record and returned with it (strings, numbers, booleans or null). Not searched.
pub type Metadata = IndexMap<String, Value>;

/// One piece of information to remember. Only `id` and `text` are required.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Stable and unique within its `source`: the same id added again replaces the record (unchanged: nothing is done).
    pub id: String,
    /// The content.
    pub text: String,
    /// A subject, headline or name; when absent, the first line of the text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// What it is, in one word of your choosing: `email`, `message`, `note`, `memo`, `article`, `commit`, `ticket`, …
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Where it came from: a mailbox, a workspace, a feed, a repository. Part of the record's identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// What it belongs to, for searches scoped to one collection: a project, a team, a customer, a person.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    /// Records that belong together share a thread (an e-mail thread, a chat thread, a repository's history).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<String>,
    /// Who wrote it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// When it was written. Default: when it was added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<At>,
    /// Where to open it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Anything else, kept with the record and returned with it. Not searched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl MemoryRecord {
    /// A record with an id and a text, the rest unset.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        MemoryRecord { id: id.into(), text: text.into(), ..Default::default() }
    }
}

/// Who speaks in a conversation's turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The person whose memory it is.
    User,
    /// The other side.
    Assistant,
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// Who speaks.
    pub role: Role,
    /// What was said.
    pub text: String,
    /// Who it was, by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// When it was said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<At>,
}

/// A conversation: turns between a person and an assistant (or any two parties), in order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    /// Stable: the same id added again is read again, not doubled.
    pub id: String,
    /// For searches scoped to one collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    /// Part of each turn record's identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// When it began (each turn may carry its own `at`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<At>,
    /// The turns, in order.
    pub turns: Vec<Turn>,
}

/// The collection records without one belong to.
pub const DEFAULT_COLLECTION: &str = "default";

/// A time as milliseconds; `None` for nothing or something unreadable.
pub fn time_of(at: Option<&At>) -> Option<f64> {
    match at? {
        At::Ms(ms) => ms.is_finite().then_some(*ms),
        At::Text(s) if s.is_empty() => None,
        At::Text(s) => js::date_parse(s),
    }
}

/// A collection as the index keys it: one path segment, any script kept.
pub fn collection_key(collection: Option<&str>) -> String {
    let c = js::trim(collection.unwrap_or(""));
    if c.is_empty() {
        return DEFAULT_COLLECTION.to_string();
    }
    let mut out = String::new();
    let mut run = false;
    for ch in c.chars() {
        if js::is_space(ch) || matches!(ch, '/' | '\\' | ':') {
            if !run {
                out.push('-');
            }
            run = true;
        } else {
            out.push(ch);
            run = false;
        }
    }
    out
}

/// SHA-256 of the parts joined by NUL, as hex.
pub(crate) fn hash_parts(parts: &[&str]) -> String {
    hex::encode(Sha256::digest(parts.join("\u{0}").as_bytes()))
}

/// A record's id in memory: its source and id, hashed (any characters may be in either).
pub fn record_uid(id: &str, source: Option<&str>) -> String {
    hash_parts(&[source.unwrap_or(""), id])[..32].to_string()
}

fn present(s: &Option<String>) -> Option<&str> {
    s.as_deref().filter(|x| !x.is_empty())
}

fn trimmed(s: &Option<String>) -> Option<&str> {
    s.as_deref().map(js::trim).filter(|x| !x.is_empty())
}

/// Checks a record and says what is wrong with it, or `None`.
pub fn invalid_record(r: &MemoryRecord) -> Option<String> {
    if js::trim(&r.id).is_empty() {
        return Some("a record needs an id".into());
    }
    if js::trim(&r.text).is_empty() && trimmed(&r.title).is_none() {
        return Some(format!("record {}: it has no text and no title", r.id));
    }
    if let Some(at) = &r.at {
        if time_of(Some(at)).is_none() {
            return Some(format!("record {}: its time cannot be read ({})", r.id, at.shown()));
        }
    }
    None
}

/// A record as memory keeps it: the title is what it is about (the first line when there is none), the text after the
/// author is the content. Searchable as written: its start is its key, and every word of it is in a chunk.
pub fn entry_of(r: &MemoryRecord, now: f64, turn: usize) -> StmEntry {
    let text = js::trim(&r.text);
    let first_line = text.split('\n').map(js::trim).find(|l| !l.is_empty()).unwrap_or("");
    let title = trimmed(&r.title).unwrap_or(first_line).to_string();
    // Without a title its first line is the title, and not said twice.
    let body = if trimmed(&r.title).is_some() {
        text.to_string()
    } else {
        let at = text.find(first_line).unwrap_or(0) + first_line.len();
        js::trim(&text[at..]).to_string()
    };
    let by = trimmed(&r.author).map(|a| format!("{a}: ")).unwrap_or_default();
    let uid = record_uid(&r.id, r.source.as_deref());
    let reply = format!("{by}{body}");
    let ts = time_of(r.at.as_ref());
    let metadata = r.metadata.as_ref().map(|m| js::stringify(&Value::Object(m.clone().into_iter().collect())));
    let fingerprint = hash_parts(&[
        &title,
        &reply,
        r.url.as_deref().unwrap_or(""),
        r.kind.as_deref().unwrap_or(""),
        r.collection.as_deref().unwrap_or(""),
        r.thread.as_deref().unwrap_or(""),
        &ts.map(js::number).unwrap_or_default(),
        metadata.as_deref().unwrap_or("{}"),
    ]);
    StmEntry {
        uid: uid.clone(),
        fingerprint,
        engine: trimmed(&r.kind).unwrap_or("record").to_string(),
        session: trimmed(&r.thread).map(str::to_string).unwrap_or_else(|| uid.clone()),
        turn,
        cwd: None,
        project: collection_key(r.collection.as_deref()),
        title: title.clone(),
        ts: ts.unwrap_or(now),
        synthetic: false,
        typed: "human".into(),
        request: title,
        reply: reply.clone(),
        said: reply,
        steps: vec![],
        files: present(&r.url).map(|u| vec![u.to_string()]).unwrap_or_default(),
        done: false,
        record: Some(RecordRef {
            id: r.id.clone(),
            source: present(&r.source).map(str::to_string),
            author: present(&r.author).map(str::to_string),
            url: present(&r.url).map(str::to_string),
            metadata: r.metadata.as_ref().map(js::in_js_order),
        }),
        extra: IndexMap::new(),
    }
}

fn first_nonblank(lines: &[String]) -> Option<String> {
    lines.join("\n").split('\n').find(|l| !js::trim(l).is_empty()).map(|l| js::trim(l).to_string())
}

/// A conversation's turns as records: each time the person speaks, what they said is the request and what the other
/// side answered until they speak again is its reply — one record a request, so a question finds the exchange it was
/// in. Turns before the first request (an assistant that opened) are the first record's own.
pub fn conversation_entries(c: &Conversation, now: f64) -> Vec<StmEntry> {
    let began = time_of(c.at.as_ref());
    let mut out: Vec<StmEntry> = Vec::new();
    let mut request: Vec<String> = Vec::new();
    let mut reply: Vec<String> = Vec::new();
    let mut at: Option<f64> = None;
    let flush = |request: &mut Vec<String>, reply: &mut Vec<String>, at: &mut Option<f64>, out: &mut Vec<StmEntry>| {
        if request.is_empty() && reply.is_empty() {
            return;
        }
        let n = out.len();
        // A time only the caller gave is part of the fingerprint: one defaulted from when it is added would make the
        // same conversation added again a changed one — read, embedded and consolidated again.
        let given = at.or_else(|| began.map(|b| b + n as f64 * 60_000.0));
        let title = first_nonblank(request).or_else(|| first_nonblank(reply)).unwrap_or_else(|| c.id.clone());
        let record = MemoryRecord {
            id: format!("{}#{n}", c.id),
            source: c.source.clone(),
            collection: c.collection.clone(),
            thread: Some(c.id.clone()),
            kind: Some("conversation".into()),
            title: Some(title),
            text: request.join("\n"),
            at: given.map(At::Ms),
            ..Default::default()
        };
        let mut e = entry_of(&record, now, n);
        e.ts = given.unwrap_or(now + n as f64 * 60_000.0);
        let req = request.join("\n");
        e.request = if req.is_empty() { e.title.clone() } else { req };
        e.reply = reply.join("\n");
        e.said = e.reply.clone();
        out.push(e);
        request.clear();
        reply.clear();
        *at = None;
    };
    for t in &c.turns {
        let text = js::trim(&t.text);
        if text.is_empty() {
            continue;
        }
        let said = match trimmed(&t.author) {
            Some(a) => format!("{a}: {text}"),
            None => text.to_string(),
        };
        if t.role == Role::User && !reply.is_empty() {
            flush(&mut request, &mut reply, &mut at, &mut out);
        }
        if t.role == Role::User {
            request.push(said);
        } else {
            reply.push(said);
        }
        if at.is_none() {
            at = time_of(t.at.as_ref());
        }
    }
    flush(&mut request, &mut reply, &mut at, &mut out);
    for e in &mut out {
        e.fingerprint = hash_parts(&[&e.fingerprint, &e.request, &e.reply]);
    }
    out
}
