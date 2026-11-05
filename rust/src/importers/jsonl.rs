//! JSON Lines: one JSON object per line, read a line at a time. With no `map`, each line is a record as it is
//! (`{ "id": "…", "text": "…", "title": …, "at": … }`); with one, `map(obj, line_number)` turns each object into a
//! record, or into `None` to leave it out. Lines that are not JSON or not a record are skipped, counted and reported
//! through `on_skip`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::text::{js_string, Draft};
use super::{blocking_stream, RecordStream};
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

/// The longest line read: one past it is skipped with a reason. The reference's `readline` has no bound but V8's
/// longest string, and fails there; unbounded, one hostile line held the whole file in memory.
pub(crate) const MAX_LINE_BYTES: usize = 64 * 1024 * 1024;

/// What `Lines` gives for a line longer than its bound (the line itself is passed over, and reading goes on).
pub(crate) fn too_long(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::InvalidData && e.to_string().starts_with("a line longer than")
}

/// Lines as Node's `readline` gives them: ended by `\n`, `\r\n` or a lone `\r`, without their ends. One longer
/// than its bound comes as an error `too_long` reads, and the lines after it follow.
pub(crate) struct Lines<R: BufRead> {
    reader: R,
    max: usize,
    /// The last line ended with `\r`: a `\n` right after it is the same end.
    after_cr: bool,
    done: bool,
}

impl<R: BufRead> Lines<R> {
    pub(crate) fn new(reader: R) -> Self {
        Self::with_max(reader, MAX_LINE_BYTES)
    }

    pub(crate) fn with_max(reader: R, max: usize) -> Self {
        Lines { reader, max, after_cr: false, done: false }
    }
}

impl<R: BufRead> Iterator for Lines<R> {
    type Item = std::io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut line = Vec::new();
        let (mut read, mut long) = (0usize, false);
        loop {
            let avail = match self.reader.fill_buf() {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            if avail.is_empty() {
                self.done = true;
                if read == 0 {
                    return None;
                }
                break;
            }
            if std::mem::take(&mut self.after_cr) && avail[0] == b'\n' {
                self.reader.consume(1);
                continue;
            }
            let end = avail.iter().position(|b| *b == b'\n' || *b == b'\r');
            let piece = &avail[..end.unwrap_or(avail.len())];
            if !long {
                if line.len() + piece.len() > self.max {
                    long = true;
                    line = Vec::new();
                } else {
                    line.extend_from_slice(piece);
                }
            }
            read += piece.len();
            match end {
                Some(i) => {
                    self.after_cr = avail[i] == b'\r';
                    self.reader.consume(i + 1);
                    break;
                }
                None => {
                    let n = avail.len();
                    self.reader.consume(n);
                }
            }
        }
        if long {
            return Some(Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("a line longer than {} bytes ({read})", self.max),
            )));
        }
        Some(Ok(line))
    }
}

/// A record read from a JSON object the way the reference reads one: the keys a record has, an `id` given as a number
/// written as text; what cannot be a record says why. A `""` field stays `Some("")` here — it is given, so no default
/// fills it — and is left out when the record is finished.
pub(crate) fn record_of_object(obj: &serde_json::Map<String, Value>) -> std::result::Result<Draft, String> {
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
    Ok(Draft { id, text, title, kind, source, collection, thread, author, at, url, metadata })
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
        let d = if let Some(map) = &self.o.map {
            match map(&obj, self.n) {
                Err(e) => return Some(Err(format!("map failed: {e}"))),
                Ok(None) => return Some(Err("left out by map".into())),
                Ok(Some(r)) => Draft {
                    id: r.id,
                    text: r.text,
                    title: r.title,
                    kind: r.kind,
                    source: r.source,
                    collection: r.collection,
                    thread: r.thread,
                    author: r.author,
                    at: r.at,
                    url: r.url,
                    metadata: r.metadata.unwrap_or_default().into_iter().map(|(k, v)| (k, Some(v))).collect(),
                },
            }
        } else {
            match &obj {
                Value::Object(m) => match record_of_object(m) {
                    Ok(d) => d,
                    Err(e) => return Some(Err(e)),
                },
                _ => return Some(Err("not an object".into())),
            }
        };
        // Filled only where absent (`r.source ?? o.source`): a `""` given stays, and `finish` leaves it out.
        let filled = Draft {
            source: d.source.clone().or_else(|| self.o.source.clone()),
            collection: d.collection.clone().or_else(|| self.o.collection.clone()),
            kind: d.kind.clone().or_else(|| self.o.kind.clone()),
            ..d
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
                Err(e) if too_long(&e) => {
                    self.n += 1;
                    self.tally(|s| s.lines += 1);
                    self.left_out("", &e.to_string());
                    continue;
                }
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

/// Records from a JSON Lines file, read a line at a time.
pub async fn from_jsonl(path: impl AsRef<Path>, opts: JsonlOptions) -> Result<RecordStream> {
    let path = path.as_ref().to_path_buf();
    blocking_stream(move || {
        let file = File::open(&path).map_err(|e| Error::io(&path, e))?;
        let stats = opts.stats.clone().unwrap_or_default();
        Ok(Reader { lines: Lines::new(BufReader::new(file)), n: 0, o: opts, stats })
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &[u8], max: usize) -> Vec<std::result::Result<String, String>> {
        Lines::with_max(std::io::Cursor::new(text.to_vec()), max)
            .map(|l| l.map(|b| String::from_utf8_lossy(&b).into_owned()).map_err(|e| e.to_string()))
            .collect()
    }

    #[test]
    fn lines_end_as_readline_ends_them_and_one_past_the_bound_is_passed_over() {
        let ok = |s: &str| Ok(s.to_string());
        assert_eq!(lines(b"a\r\nb\rc\n\nd", 100), [ok("a"), ok("b"), ok("c"), ok(""), ok("d")]);
        assert_eq!(lines(b"a\r\r\n", 100), [ok("a"), ok("")]);
        assert_eq!(lines(b"", 100), Vec::<std::result::Result<String, String>>::new());
        let got = lines(b"short\nfar too long a line\nafter\n", 8);
        assert_eq!(got[0], ok("short"));
        assert!(got[1].as_ref().is_err_and(|e| e.starts_with("a line longer than 8 bytes")), "{got:?}");
        assert_eq!(got[2..], [ok("after")]);
    }

    #[test]
    fn a_json_line_past_the_bound_is_skipped_with_why_and_the_next_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.jsonl");
        let long = format!("{{\"id\":\"big\",\"text\":\"{}\"}}", "x".repeat(200));
        std::fs::write(&path, format!("{long}\n{{\"id\":\"ok\",\"text\":\"fine\"}}\n")).unwrap();
        let why = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = why.clone();
        let o = JsonlOptions {
            on_skip: Some(Arc::new(move |_l: &str, r: &str, n: usize| seen.lock().unwrap().push(format!("{n}: {r}")))),
            ..Default::default()
        };
        let reader = Reader {
            lines: Lines::with_max(BufReader::new(File::open(&path).unwrap()), 100),
            n: 0,
            o,
            stats: Arc::default(),
        };
        let got: Vec<MemoryRecord> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(got.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["ok"]);
        let why = why.lock().unwrap();
        assert!(why.len() == 1 && why[0].starts_with("1: a line longer than 100 bytes"), "{why:?}");
    }
}
