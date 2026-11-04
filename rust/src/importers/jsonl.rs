//! JSON Lines: one JSON object per line, read a line at a time. With no `map`, each line is a record as it is
//! (`{ "id": "…", "text": "…", "title": …, "at": … }`); with one, `map(obj, line_number)` turns each object into a
//! record, or into `None` to leave it out. Lines that are not JSON or not a record are skipped, counted and reported
//! through `on_skip`.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::text::{js_string, Draft};
use super::Records;
use crate::error::{Error, Result};
use crate::js;
use crate::records::{invalid_record, At, MemoryRecord};

/// Turns one parsed line into a record, `None` to leave it out, or an error (the line is skipped with it).
pub type JsonlMap = Arc<dyn Fn(&Value, usize) -> std::result::Result<Option<MemoryRecord>, String> + Send + Sync>;

/// Called for each line left out: the line, why, its number.
pub type JsonlSkip = Arc<dyn Fn(&str, &str, usize) + Send + Sync>;

/// What reading a file came to, filled in as it is read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JsonlStats {
    /// Lines read (blank ones not counted).
    pub lines: usize,
    /// Records given.
    pub records: usize,
    /// Lines skipped.
    pub skipped: usize,
}

/// How a JSON Lines file is read.
#[derive(Clone, Default)]
pub struct JsonlOptions {
    /// Each parsed line into a record.
    pub map: Option<JsonlMap>,
    /// Called for each line left out, with why.
    pub on_skip: Option<JsonlSkip>,
    /// Filled in as the file is read.
    pub stats: Option<Arc<Mutex<JsonlStats>>>,
    /// For records that have none.
    pub source: Option<String>,
    /// For records that have none.
    pub collection: Option<String>,
    /// For records that have none.
    pub kind: Option<String>,
}

/// Lines as Node's `readline` gives them: ended by `\n`, `\r\n` or a lone `\r`, without their ends.
pub(crate) struct Lines<R: BufRead> {
    reader: R,
    queue: VecDeque<Vec<u8>>,
    done: bool,
}

impl<R: BufRead> Lines<R> {
    pub(crate) fn new(reader: R) -> Self {
        Lines { reader, queue: VecDeque::new(), done: false }
    }
}

impl<R: BufRead> Iterator for Lines<R> {
    type Item = std::io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(l) = self.queue.pop_front() {
                return Some(Ok(l));
            }
            if self.done {
                return None;
            }
            let mut buf = Vec::new();
            match self.reader.read_until(b'\n', &mut buf) {
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {}
            }
            let content: &[u8] = if buf.ends_with(b"\r\n") {
                &buf[..buf.len() - 2]
            } else if buf.ends_with(b"\n") || buf.ends_with(b"\r") {
                &buf[..buf.len() - 1]
            } else {
                &buf
            };
            for piece in content.split(|b| *b == b'\r') {
                self.queue.push_back(piece.to_vec());
            }
        }
    }
}

/// A record read from a JSON object the way the reference reads one: the keys a record has, an `id` given as a number
/// written as text, `""` fields left out; what cannot be a record says why.
pub(crate) fn record_of_object(obj: &serde_json::Map<String, Value>) -> std::result::Result<MemoryRecord, String> {
    let id = match obj.get("id") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => js::number(n.as_f64().unwrap_or(f64::NAN)),
        _ => return Err("a record needs an id".into()),
    };
    let text = match obj.get("text") {
        Some(Value::String(s)) => s.clone(),
        _ => return Err(format!("record {id}: text must be a string")),
    };
    let field = |name: &str| -> std::result::Result<Option<String>, String> {
        match obj.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(format!("record {id}: {name} must be a string")),
        }
    };
    let at = match obj.get("at") {
        None => None,
        Some(Value::Number(n)) => Some(At::Ms(n.as_f64().unwrap_or(f64::NAN))),
        Some(Value::String(s)) => Some(At::Text(s.clone())),
        Some(other) => return Err(format!("record {id}: its time cannot be read ({})", js_string(other))),
    };
    let metadata = match obj.get("metadata") {
        None | Some(Value::Null) => vec![],
        Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), Some(v.clone()))).collect(),
        Some(_) => return Err(format!("record {id}: metadata must be an object")),
    };
    let (title, kind, source) = (field("title")?, field("kind")?, field("source")?);
    let (collection, thread, author, url) = (field("collection")?, field("thread")?, field("author")?, field("url")?);
    Ok(Draft { id, text, title, kind, source, collection, thread, author, at, url, metadata }.finish())
}

struct Reader {
    lines: Lines<BufReader<File>>,
    n: usize,
    o: JsonlOptions,
    stats: Arc<Mutex<JsonlStats>>,
}

impl Reader {
    fn tally(&self, f: impl FnOnce(&mut JsonlStats)) {
        f(&mut self.stats.lock().unwrap_or_else(|e| e.into_inner()));
    }

    fn left_out(&self, line: &str, reason: &str) {
        self.tally(|s| s.skipped += 1);
        if let Some(cb) = &self.o.on_skip {
            cb(line, reason, self.n);
        }
    }

    /// One line's record, or why it was left out (`None` for a blank line).
    fn read(&mut self, raw: &[u8]) -> Option<std::result::Result<MemoryRecord, String>> {
        let decoded = String::from_utf8_lossy(raw);
        let line = if self.n == 1 { decoded.strip_prefix('\u{feff}').unwrap_or(&decoded) } else { &decoded };
        if js::trim(line).is_empty() {
            return None;
        }
        self.tally(|s| s.lines += 1);
        let obj: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return Some(Err(format!("not JSON: {e}"))),
        };
        let r = if let Some(map) = &self.o.map {
            match map(&obj, self.n) {
                Err(e) => return Some(Err(format!("map failed: {e}"))),
                Ok(None) => return Some(Err("left out by map".into())),
                Ok(Some(r)) => r,
            }
        } else {
            match &obj {
                Value::Object(m) => match record_of_object(m) {
                    Ok(r) => r,
                    Err(e) => return Some(Err(e)),
                },
                _ => return Some(Err("not an object".into())),
            }
        };
        let filled = Draft {
            source: r.source.clone().or_else(|| self.o.source.clone()),
            collection: r.collection.clone().or_else(|| self.o.collection.clone()),
            kind: r.kind.clone().or_else(|| self.o.kind.clone()),
            id: r.id,
            text: r.text,
            title: r.title,
            thread: r.thread,
            author: r.author,
            at: r.at,
            url: r.url,
            metadata: r.metadata.unwrap_or_default().into_iter().map(|(k, v)| (k, Some(v))).collect(),
        }
        .finish();
        if let Some(wrong) = invalid_record(&filled) {
            return Some(Err(wrong));
        }
        Some(Ok(filled))
    }
}

impl Iterator for Reader {
    type Item = Result<MemoryRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let raw = match self.lines.next()? {
                Ok(l) => l,
                Err(e) => return Some(Err(Error::Import(format!("cannot read the JSON Lines file: {e}")))),
            };
            self.n += 1;
            match self.read(&raw) {
                None => continue,
                Some(Ok(r)) => {
                    self.tally(|s| s.records += 1);
                    return Some(Ok(r));
                }
                Some(Err(reason)) => {
                    let decoded = String::from_utf8_lossy(&raw).into_owned();
                    self.left_out(&decoded, &reason);
                }
            }
        }
    }
}

/// Records from a JSON Lines file.
pub fn from_jsonl(path: impl AsRef<Path>, opts: JsonlOptions) -> Result<Records> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| Error::io(path, e))?;
    let stats = opts.stats.clone().unwrap_or_default();
    Ok(Box::new(Reader { lines: Lines::new(BufReader::new(file)), n: 0, o: opts, stats }))
}
