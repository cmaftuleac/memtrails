//! A folder of text files: notes, documents, saved pages. Every file whose extension is in `include` becomes a record,
//! read one at a time; dot folders, dot files and `node_modules` are passed by.
//!
//! id: the path inside the folder (with `/`). title: the Markdown front matter's `title`, else the first `# heading`
//! (an Org `#+TITLE:`, an RST underlined title, an HTML `<title>` or `<h1>`), else the file's name. text: the file,
//! front matter taken off, HTML reduced to text. at: the front matter's `date`, else when the file was last changed.
//! The other front-matter fields go into metadata (`author` also to the record's author). A file longer than
//! `max_chars` is cut into parts with ids `path#1`, `path#2`, … sharing the thread `path`.

use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use encoding_rs::{UTF_16BE, UTF_16LE};
use indexmap::IndexMap;
use regex::Regex;
use serde_json::Value;

use super::text::{decode_bytes, file_url, html_title, html_to_text, re, split_text, Draft};
use super::{blocking_stream, RecordStream};
use crate::error::{Error, Result};
use crate::js;
use crate::records::{time_of, At, MemoryRecord};

/// The extensions read by default: text formats only.
pub const TEXT_EXTENSIONS: &[&str] = &[
    ".md",
    ".markdown",
    ".mdx",
    ".txt",
    ".text",
    ".html",
    ".htm",
    ".xhtml",
    ".rst",
    ".org",
    ".adoc",
    ".asciidoc",
    ".textile",
    ".wiki",
    ".tex",
];

/// Called for each file left out: its path inside the folder and why.
pub type FilesSkip = Arc<dyn Fn(&str, &str) + Send + Sync>;

/// Leaves out a file by its path inside the folder.
pub type FilesExclude = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// How a folder is read.
#[derive(Clone, Default)]
pub struct FilesOptions {
    /// Extensions to read, with or without the dot. Default: `TEXT_EXTENSIONS`.
    pub include: Option<Vec<String>>,
    /// Folder names never entered (dot folders never are). Default: `node_modules`.
    pub skip_folders: Option<Vec<String>>,
    /// Leave out a file by its path inside the folder.
    pub exclude: Option<FilesExclude>,
    /// Longer files are cut into parts of at most this many characters (UTF-16 units). Default 20,000.
    pub max_chars: Option<usize>,
    /// Larger files are not read (reported through `on_skip`). Default 64 MiB.
    pub max_bytes: Option<u64>,
    /// The records' collection.
    pub collection: Option<String>,
    /// Default: the folder's name.
    pub source: Option<String>,
    /// Default `note`.
    pub kind: Option<String>,
    /// Called for each file left out, with why.
    pub on_skip: Option<FilesSkip>,
}

/// A front-matter value: a string, a number, a boolean or null.
fn scalar(raw: &str) -> Value {
    let v = js::trim(raw);
    let b = v.as_bytes();
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        return Value::String(v[1..v.len() - 1].replace("\\\"", "\"").replace("''", "'"));
    }
    match v.to_ascii_lowercase().as_str() {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" | "~" | "" => return Value::Null,
        _ => {}
    }
    static NUMBER: LazyLock<Regex> = LazyLock::new(|| re(r"^-?(0|[1-9][0-9]{0,14})(\.[0-9]+)?$"));
    if NUMBER.is_match(v) {
        if let Ok(x) = v.parse::<f64>() {
            return super::text::number_value(x);
        }
    }
    Value::String(v.to_string())
}

fn list_value(items: &[&str]) -> Value {
    Value::String(
        items
            .iter()
            .map(|i| scalar(i))
            .filter(|i| !i.is_null() && i.as_str() != Some(""))
            .map(|i| super::text::js_string(&i))
            .collect::<Vec<_>>()
            .join(", "),
    )
}

fn starts_with_space(s: &str) -> bool {
    s.chars().next().is_some_and(js::is_space)
}

/// One front-matter line as key and value: TOML `key = value`, YAML `key: value` (a `:` then a space or the end).
/// Split at the first `=` or `:` by hand: `^([^=]*?)\s*=` tried a run of spaces at every position, quadratic on a
/// long line.
fn key_value(line: &str, toml: bool) -> Option<(String, String)> {
    let at = line.find(if toml { '=' } else { ':' })?;
    let first = line.chars().next()?;
    if at == 0
        || js::is_space(first)
        || if toml { matches!(first, '=' | '#' | '[') } else { matches!(first, ':' | '#' | '-') }
    {
        return None;
    }
    let value = &line[at + 1..];
    if !toml && !value.is_empty() && !starts_with_space(value) {
        return None;
    }
    Some((js::trim_end(&line[..at]).to_string(), value.to_string()))
}

static FRONT: LazyLock<Regex> =
    LazyLock::new(|| re(r"^(---|\+\+\+)[ \t]*\r?\n((?s:.)*?)\r?\n(?:---|\+\+\+|\.\.\.)[ \t]*(?:\r?\n|$)"));
static LINE_BREAK: LazyLock<Regex> = LazyLock::new(|| re(r"\r?\n"));
static BLOCK_MARK: LazyLock<Regex> = LazyLock::new(|| re(r"^[|>][-+]?$"));

/// YAML (`---`) or TOML (`+++`) front matter, flat fields only: scalars, lists (joined with `, `) and block text.
/// Returns the fields, in their order, and the text after them.
pub fn front_matter(text: &str) -> (IndexMap<String, Value>, String) {
    let Some(c) = FRONT.captures(text) else { return (IndexMap::new(), text.to_string()) };
    let (Some(whole), Some(kind), Some(inner)) = (c.get(0), c.get(1), c.get(2)) else {
        return (IndexMap::new(), text.to_string());
    };
    let toml = kind.as_str() == "+++";
    let mut fields: IndexMap<String, Value> = IndexMap::new();
    let lines: Vec<&str> = LINE_BREAK.split(inner.as_str()).collect();
    let mut i = 0;
    while i < lines.len() {
        let Some((k, v)) = key_value(lines[i], toml) else {
            i += 1;
            continue;
        };
        let key = js::trim(&k);
        let key = key.strip_prefix(['"', '\'']).unwrap_or(key);
        let key = key.strip_suffix(['"', '\'']).unwrap_or(key).to_string();
        let value = js::trim(&v);
        if value.starts_with('[') && value.ends_with(']') && !value.is_empty() {
            let inner = if value.len() >= 2 { &value[1..value.len() - 1] } else { "" };
            fields.insert(key, list_value(&inner.split(',').collect::<Vec<_>>()));
        } else if !toml && BLOCK_MARK.is_match(value) {
            let mut block: Vec<&str> = Vec::new();
            while i + 1 < lines.len() && (starts_with_space(lines[i + 1]) || js::trim(lines[i + 1]).is_empty()) {
                i += 1;
                block.push(js::trim(lines[i]));
            }
            let joined = block.join(if value.starts_with('|') { "\n" } else { " " });
            fields.insert(key, Value::String(js::trim(&joined).to_string()));
        } else if !toml && value.is_empty() {
            let mut items: Vec<String> = Vec::new();
            while i + 1 < lines.len() && list_item(lines[i + 1]).is_some() {
                i += 1;
                items.push(list_item(lines[i]).unwrap_or_default().to_string());
            }
            // A nested map: not kept.
            while i + 1 < lines.len()
                && starts_with_space(lines[i + 1])
                && !super::text::trim_start(lines[i + 1]).is_empty()
            {
                i += 1;
            }
            if !items.is_empty() {
                fields.insert(key, list_value(&items.iter().map(String::as_str).collect::<Vec<_>>()));
            }
        } else {
            // Not /\s+#.*$/: quadratic on a long run of spaces for a backtracking engine.
            let comment =
                value.char_indices().find(|(at, c)| js::is_space(*c) && value[at + c.len_utf8()..].starts_with('#'));
            fields.insert(key, scalar(comment.map_or(value, |(at, _)| &value[..at])));
        }
        i += 1;
    }
    (fields, text[whole.end()..].to_string())
}

/// `^\s*-\s+(rest)`: a YAML list item's text.
fn list_item(line: &str) -> Option<&str> {
    let t = super::text::trim_start(line);
    let rest = t.strip_prefix('-')?;
    let after = super::text::trim_start(rest);
    (after.len() < rest.len()).then_some(after)
}

/// Title rules see at most this much of a file, a line at a time: per-line regexes over a whole hostile file were
/// slow.
const TITLE_SCAN: usize = 64 * 1024;

/// An RST underline (or overline): one punctuation character, three or more times, then spaces at most.
fn rst_rule(line: Option<&str>) -> bool {
    let t = js::trim_end(line.unwrap_or("")); // not /[ \t]+$/: quadratic on a long run of spaces
    let Some(first) = t.chars().next() else { return false };
    js::len16(t) >= 3 && "=-~^\"'`#*+".contains(first) && t.chars().all(|c| c == first)
}

static LINES: LazyLock<Regex> = LazyLock::new(|| re(r"\r\n?|\n"));
static ORG_TITLE: LazyLock<Regex> = LazyLock::new(|| re(r"^(?i-u:#\+title:)"));
static SETEXT_RULE: LazyLock<Regex> = LazyLock::new(|| re(r"^=+[ \t]*$"));

fn title_of(ext: &str, body: &str, raw: &str) -> Option<String> {
    if matches!(ext, ".html" | ".htm" | ".xhtml") {
        return html_title(raw);
    }
    let scanned = js::head16(body, TITLE_SCAN);
    let lines: Vec<&str> = LINES.split(scanned).collect();
    let first = |f: &dyn Fn(&str, usize) -> Option<String>| -> Option<String> {
        lines.iter().enumerate().find_map(|(i, l)| f(l, i))
    };
    let tab_or_space = |c: Option<char>| matches!(c, Some(' ' | '\t'));
    if ext == ".org" {
        let titled = first(&|l, _| (ORG_TITLE.is_match(l) && js::len16(l) > 8).then(|| js::trim(&l[8..]).to_string()));
        return match titled {
            Some(t) if !t.is_empty() => Some(t),
            _ => first(&|l, _| {
                let rest = l.trim_start_matches('*');
                (rest.len() < l.len() && tab_or_space(rest.chars().next()) && !js::trim(rest).is_empty())
                    .then(|| js::trim(rest).to_string())
            }),
        };
    }
    if ext == ".rst" {
        return first(&|l, i| {
            (!js::trim(l).is_empty() && rst_rule(lines.get(i + 1).copied())).then(|| js::trim(l).to_string())
        });
    }
    if ext == ".adoc" || ext == ".asciidoc" {
        return first(&|l, _| {
            let rest = l.strip_prefix('=')?;
            (tab_or_space(rest.chars().next()) && !js::trim(rest).is_empty()).then(|| js::trim(rest).to_string())
        });
    }
    // `# Title`, a closing run of `#` after a space dropped (by hand: `[ \t]+#+$` is quadratic on a run of spaces).
    let atx = first(&|l, _| {
        let rest = l.strip_prefix('#')?;
        if !tab_or_space(rest.chars().next()) {
            return None;
        }
        let t = js::trim(rest);
        let j = t.trim_end_matches('#').len();
        let title =
            if j > 0 && j < t.len() && matches!(t.as_bytes()[j - 1], b' ' | b'\t') { js::trim_end(&t[..j]) } else { t };
        (!title.is_empty()).then(|| title.to_string())
    });
    if atx.is_some() {
        return atx;
    }
    // A Setext `Title\n=====`.
    first(&|l, i| {
        (l.chars().next().is_some_and(|c| !js::is_space(c))
            && SETEXT_RULE.is_match(lines.get(i + 1).copied().unwrap_or("")))
        .then(|| js::trim_end(l).to_string())
    })
}

/// A file's extension as Node's `extname` gives it: from the last `.` that does not start the name.
pub(crate) fn extname(name: &str) -> &str {
    match name.rfind('.') {
        Some(0) | None => "",
        Some(i) => &name[i..],
    }
}

fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

fn decode_file(bytes: &[u8]) -> Option<String> {
    if bytes.starts_with(&[0xff, 0xfe]) {
        return Some(UTF_16LE.decode_with_bom_removal(bytes).0.into_owned());
    }
    if bytes.starts_with(&[0xfe, 0xff]) {
        return Some(UTF_16BE.decode_with_bom_removal(bytes).0.into_owned());
    }
    if bytes[..bytes.len().min(8192)].contains(&0) {
        return None; // binary
    }
    Some(decode_bytes(bytes, None))
}

struct Walk {
    root: PathBuf,
    include: HashSet<String>,
    skip_folders: HashSet<String>,
    /// Folders still to list, and the files of the one being read, in order.
    pending: Vec<VecDeque<(String, bool)>>,
    max_chars: usize,
    max_bytes: u64,
    source: String,
    o: FilesOptions,
    out: VecDeque<MemoryRecord>,
}

impl Walk {
    /// A folder's entries, sorted as JavaScript sorts names: files to read, folders to enter.
    fn list(&self, rel: &str) -> Result<VecDeque<(String, bool)>> {
        let dir = if rel.is_empty() { self.root.clone() } else { self.root.join(rel) };
        let mut entries: Vec<(String, bool)> = Vec::new();
        for e in fs::read_dir(&dir).map_err(|e| Error::io(&dir, e))? {
            let e = e.map_err(|e| Error::io(&dir, e))?;
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let ft = e.file_type().map_err(|err| Error::io(e.path(), err))?;
            let path = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
            if ft.is_dir() {
                if !self.skip_folders.contains(&name) {
                    entries.push((path, true));
                }
            } else if ft.is_file() && self.include.contains(&extname(&name).to_lowercase()) {
                entries.push((path, false));
            }
        }
        entries.sort_by(|a, b| utf16_cmp(a.0.rsplit('/').next().unwrap_or(""), b.0.rsplit('/').next().unwrap_or("")));
        Ok(entries.into())
    }

    fn left_out(&self, path: &str, why: &str) {
        if let Some(cb) = &self.o.on_skip {
            cb(path, why);
        }
    }

    /// The records of one file (several when it is cut into parts).
    fn read(&mut self, path: &str) -> Result<()> {
        if self.o.exclude.as_ref().is_some_and(|ex| ex(path)) {
            return Ok(());
        }
        let abs = self.root.join(path);
        let info = fs::metadata(&abs).map_err(|e| Error::io(&abs, e))?;
        if info.len() > self.max_bytes {
            self.left_out(path, &format!("larger than {} bytes", self.max_bytes));
            return Ok(());
        }
        let bytes = fs::read(&abs).map_err(|e| Error::io(&abs, e))?;
        let Some(raw) = decode_file(&bytes) else {
            self.left_out(path, "not text");
            return Ok(());
        };
        let name = path.rsplit('/').next().unwrap_or(path);
        let ext = extname(name).to_lowercase();
        let is_html = matches!(ext.as_str(), ".html" | ".htm" | ".xhtml");
        let (fields, body) = if is_html { (IndexMap::new(), raw.clone()) } else { front_matter(&raw) };
        let fm_title = match fields.get("title") {
            Some(v @ (Value::String(_) | Value::Number(_))) => js::trim(&super::text::js_string(v)).to_string(),
            _ => String::new(),
        };
        let stem = &name[..name.len() - extname(name).len()];
        let title = if !fm_title.is_empty() {
            fm_title
        } else {
            title_of(&ext, &body, &raw).filter(|t| !t.is_empty()).unwrap_or_else(|| stem.to_string())
        };
        let text =
            if is_html { html_to_text(&raw, false) } else { js::trim(&super::text::crlf_to_lf(&body)).to_string() };
        let dated = match fields.get("date") {
            Some(v @ (Value::String(_) | Value::Number(_))) => time_of(Some(&At::Text(super::text::js_string(v)))),
            _ => None,
        };
        let mtime = info
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0.0, |d| d.as_secs_f64() * 1000.0);
        let at = dated.unwrap_or(mtime);
        let author = match fields.get("author") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        let metadata: Vec<(String, Option<Value>)> =
            fields.iter().filter(|(k, _)| k.as_str() != "title").map(|(k, v)| (k.clone(), Some(v.clone()))).collect();
        let base = Draft {
            id: path.to_string(),
            text: text.clone(),
            title: Some(title.clone()),
            kind: Some(self.o.kind.clone().unwrap_or_else(|| "note".into())),
            source: Some(self.source.clone()),
            collection: self.o.collection.clone(),
            author,
            at: Some(At::Ms((at + 0.5).floor())),
            url: Some(file_url(&abs)),
            ..Default::default()
        };
        if text.is_empty() && title.is_empty() {
            return Ok(());
        }
        let parts = split_text(&text, self.max_chars);
        if parts.len() == 1 {
            self.out.push_back(Draft { metadata, ..base }.finish());
            return Ok(());
        }
        let n = parts.len();
        for (i, part) in parts.into_iter().enumerate() {
            let mut meta = metadata.clone();
            set(&mut meta, "part", Value::from(i + 1));
            set(&mut meta, "parts", Value::from(n));
            self.out.push_back(
                Draft {
                    id: format!("{path}#{}", i + 1),
                    thread: Some(path.to_string()),
                    text: part,
                    title: Some(format!("{title} ({}/{n})", i + 1)),
                    metadata: meta,
                    ..base.clone()
                }
                .finish(),
            );
        }
        Ok(())
    }
}

/// A key set in an object's order: an existing one keeps its place.
fn set(meta: &mut Vec<(String, Option<Value>)>, key: &str, v: Value) {
    match meta.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = Some(v),
        None => meta.push((key.to_string(), Some(v))),
    }
}

impl Iterator for Walk {
    type Item = Result<MemoryRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(r) = self.out.pop_front() {
                return Some(Ok(r));
            }
            let top = self.pending.last_mut()?;
            let Some((path, is_dir)) = top.pop_front() else {
                self.pending.pop();
                continue;
            };
            if is_dir {
                match self.list(&path) {
                    Ok(entries) => self.pending.push(entries),
                    Err(e) => return Some(Err(e)),
                }
                continue;
            }
            if let Err(e) = self.read(&path) {
                return Some(Err(e));
            }
        }
    }
}

/// The walk over a folder, its top level listed.
fn walk(dir: &Path, opts: FilesOptions) -> Result<Walk> {
    let root = super::text::resolve_path(dir).map_err(|e| Error::io(dir, e))?;
    let include = opts
        .include
        .clone()
        .unwrap_or_else(|| TEXT_EXTENSIONS.iter().map(|s| s.to_string()).collect())
        .into_iter()
        .map(|e| if e.starts_with('.') { e } else { format!(".{e}") }.to_lowercase())
        .collect();
    let skip_folders = opts.skip_folders.clone().unwrap_or_else(|| vec!["node_modules".into()]).into_iter().collect();
    let source = opts
        .source
        .clone()
        .unwrap_or_else(|| root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    let mut walk = Walk {
        root,
        include,
        skip_folders,
        pending: Vec::new(),
        max_chars: opts.max_chars.unwrap_or(20_000),
        max_bytes: opts.max_bytes.unwrap_or(64 * 1024 * 1024),
        source,
        o: opts,
        out: VecDeque::new(),
    };
    let first = walk.list("")?;
    walk.pending.push(first);
    Ok(walk)
}

/// Records from the text files in a folder and its subfolders.
pub async fn from_files(dir: impl AsRef<Path>, opts: FilesOptions) -> Result<RecordStream> {
    let dir = dir.as_ref().to_path_buf();
    blocking_stream(move || walk(&dir, opts)).await
}
