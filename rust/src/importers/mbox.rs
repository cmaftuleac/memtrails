//! E-mail: an mbox file (RFC 4155: each message opens with a `From ` line; `>From ` in a body is unescaped), read a
//! message at a time, or one `.eml` file. Headers are unfolded and their RFC 2047 encoded-words (Q and B, any charset)
//! decoded; MIME parts are taken apart (text/plain preferred, else text/html reduced to text), base64 and
//! quoted-printable decoded, and every charset the Encoding Standard knows read as such.
//!
//! id: the Message-ID (else a hash of the message). title: the Subject. text: the body, with the names of attachments
//! after it; lines quoted from earlier messages (`>`) and the signature (after `-- `) are left out unless
//! `keep_quoted` / `keep_signature`, since the earlier message is a record of its own. author: the sender's name, else
//! address. at: the Date. kind `email`. thread: the first id in References, else In-Reply-To, else the message's own
//! id. metadata { from, to, cc, inReplyTo }.
//!
//! Lines are read as Latin-1 — every byte one character — and each part is decoded later in its own charset.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use indexmap::IndexMap;
use regex::{Captures, Regex};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::jsonl::Lines;
use super::text::{base64_lenient, crlf_to_lf, decode_bytes, html_to_text, latin1, latin1_bytes, re, Draft};
use super::{blocking_stream, RecordStream};
use crate::error::{Error, Result};
use crate::js;
use crate::records::{time_of, At, MemoryRecord};

/// Called for each message left out: why, and its place in the file.
pub type MboxSkip = Arc<dyn Fn(&str, usize) + Send + Sync>;

/// How mail is read.
#[derive(Clone, Default)]
pub struct MboxOptions {
    /// Keep lines quoted from earlier messages (`> …`).
    pub keep_quoted: bool,
    /// Keep the signature (what follows a `-- ` line).
    pub keep_signature: bool,
    /// Default: the file's name without its extension.
    pub source: Option<String>,
    /// The records' collection.
    pub collection: Option<String>,
    /// A message's body past this many bytes is not read (its headers are). Default 32 MiB.
    pub max_message_bytes: Option<usize>,
    /// Called for each message left out (no subject and no text), with its place in the file.
    pub on_skip: Option<MboxSkip>,
}

/// One MIME entity: headers (the first of each name, raw bytes as Latin-1 characters, unfolded) and body lines.
struct Entity {
    headers: HashMap<String, String>,
    body: Vec<String>,
}

struct Content {
    text: String,
    html: bool,
    attachments: Vec<String>,
}

fn entity_of(lines: &[String]) -> Entity {
    let mut headers: HashMap<String, String> = HashMap::new();
    let mut name = String::new();
    let mut i = 0;
    while i < lines.len() {
        let line = &lines[i];
        if line.is_empty() || line == "\r" {
            break;
        }
        if line.starts_with([' ', '\t']) {
            if !name.is_empty() {
                let folded = format!(" {}", line.trim_start_matches([' ', '\t']));
                if let Some(v) = headers.get_mut(&name) {
                    v.push_str(&folded);
                }
            }
            i += 1;
            continue;
        }
        let Some(colon) = line.find(':').filter(|c| *c > 0) else {
            i += 1;
            continue;
        };
        let key = js::trim(&line[..colon]).to_lowercase();
        if headers.contains_key(&key) {
            name.clear(); // a later copy of a header: the first is kept
            i += 1;
            continue;
        }
        name = key.clone();
        headers.insert(key, js::trim(&line[colon + 1..]).to_string());
        i += 1;
    }
    Entity { headers, body: lines.get(i + 1..).map(<[String]>::to_vec).unwrap_or_default() }
}

fn hex_byte(units: &[u16]) -> Option<u8> {
    if units.len() != 2 {
        return None;
    }
    let s = String::from_utf16(units).ok()?;
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u8::from_str_radix(&s, 16).ok()
}

fn q_bytes(s: &str) -> Vec<u8> {
    let u: Vec<u16> = s.encode_utf16().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < u.len() {
        if u[i] == u16::from(b'_') {
            out.push(0x20);
        } else if u[i] == u16::from(b'=') && u.get(i + 1..i + 3).and_then(hex_byte).is_some() {
            out.push(u.get(i + 1..i + 3).and_then(hex_byte).unwrap_or(0));
            i += 2;
        } else {
            out.push(u[i] as u8);
        }
        i += 1;
    }
    out
}

static ENCODED_WORD: LazyLock<Regex> =
    LazyLock::new(|| re(&format!(r"=\?([^?{s}]+)\?([bBqQ])\?([^?{s}]*)\?=", s = js::SPACE_CLASS)));

/// RFC 2047 encoded-words decoded; words side by side join with no space, and their bytes are read together.
pub fn decode_encoded_words(s: &str) -> String {
    let mut out = String::new();
    let mut last = 0usize;
    let mut pending: Option<(String, Vec<u8>)> = None;
    let flush = |pending: &mut Option<(String, Vec<u8>)>, out: &mut String| {
        if let Some((charset, bytes)) = pending.take() {
            out.push_str(&decode_bytes(&bytes, Some(&charset)));
        }
    };
    for c in ENCODED_WORD.captures_iter(s) {
        let Some(m) = c.get(0) else { continue };
        let gap = &s[last..m.start()];
        if pending.is_none() || !gap.chars().all(js::is_space) {
            flush(&mut pending, &mut out);
            out.push_str(gap);
        }
        let charset = c[1].split('*').next().unwrap_or("").to_lowercase();
        let bytes = if c[2].eq_ignore_ascii_case("b") { base64_lenient(&c[3]) } else { q_bytes(&c[3]) };
        if pending.as_ref().is_some_and(|p| p.0 != charset) {
            flush(&mut pending, &mut out);
        }
        match &mut pending {
            Some((_, b)) => b.extend_from_slice(&bytes),
            None => pending = Some((charset, bytes)),
        }
        last = m.end();
    }
    flush(&mut pending, &mut out);
    out.push_str(&s[last..]);
    out
}

/// A raw header as text: bytes sent unencoded read as UTF-8 (else Windows-1252), then encoded-words decoded.
fn header_text(raw: Option<&String>) -> String {
    let Some(raw) = raw.filter(|r| !r.is_empty()) else { return String::new() };
    js::trim(&js::collapse_spaces(&decode_encoded_words(&decode_bytes(&latin1_bytes(raw), None)))).to_string()
}

struct Params {
    value: String,
    params: HashMap<String, String>,
}

static EXTENDED: LazyLock<Regex> = LazyLock::new(|| re(r"^([^\n\r\u{2028}\u{2029}]+?)\*([0-9]+)?(\*)?$"));
static CHARSET_LANG: LazyLock<Regex> = LazyLock::new(|| re(r"^([^']*)'[^']*'([^\n\r\u{2028}\u{2029}]*)$"));
static ESCAPED: LazyLock<Regex> = LazyLock::new(|| re(r"\\([^\n\r\u{2028}\u{2029}])"));

/// `%XX` as its byte, everything else as its Latin-1 byte.
fn percent_bytes(v: &str) -> Vec<u8> {
    let u: Vec<u16> = v.encode_utf16().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < u.len() {
        if u[i] == u16::from(b'%') {
            if let Some(b) = u.get(i + 1..i + 3).and_then(hex_byte) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(u[i] as u8);
        i += 1;
    }
    out
}

/// `type/sub; a=1; b="x y"` as its value and parameters (RFC 2231 continued and encoded ones joined and decoded).
fn parse_params(raw: Option<&String>) -> Params {
    let mut pieces: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for ch in raw.map(String::as_str).unwrap_or("").chars() {
        if ch == '"' {
            quoted = !quoted;
        }
        if ch == ';' && !quoted {
            pieces.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    pieces.push(cur);
    let value = js::trim(&pieces.remove(0)).to_lowercase();
    let mut params: HashMap<String, String> = HashMap::new();
    let mut sections: IndexMap<String, Vec<(u64, bool, String)>> = IndexMap::new();
    for p in &pieces {
        let Some(eq) = p.find('=').filter(|e| *e > 0) else { continue };
        let key = js::trim(&p[..eq]).to_lowercase();
        let mut v = js::trim(&p[eq + 1..]).to_string();
        if v.len() > 1 && v.starts_with('"') && v.ends_with('"') {
            v = ESCAPED.replace_all(&v[1..v.len() - 1], "$1").into_owned();
        }
        if let Some(ext) = EXTENDED.captures(&key) {
            let n = ext.get(2).map_or(0, |d| d.as_str().parse().unwrap_or(u64::MAX));
            let encoded = ext.get(2).is_none() || ext.get(3).is_some();
            sections.entry(ext[1].to_string()).or_default().push((n, encoded, v));
        } else {
            let decoded = if key == "name" || key == "filename" { header_text(Some(&v)) } else { v };
            params.insert(key, decoded);
        }
    }
    for (key, mut list) in sections {
        list.sort_by_key(|s| s.0);
        let mut charset = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        for (i, (_, encoded, v)) in list.iter().enumerate() {
            let mut v = v.clone();
            if *encoded && i == 0 {
                if let Some(m) = CHARSET_LANG.captures(&v) {
                    charset = m[1].to_string();
                    v = m[2].to_string();
                }
            }
            if *encoded {
                bytes.extend(percent_bytes(&v));
            } else {
                bytes.extend(latin1_bytes(&v));
            }
        }
        params.insert(key, decode_bytes(&bytes, Some(&charset)));
    }
    Params { value, params }
}

static QP_ESCAPE: LazyLock<Regex> = LazyLock::new(|| re(r"=([0-9A-Fa-f]{2})"));

fn quoted_printable(lines: &[String]) -> Vec<u8> {
    let mut s = String::new();
    for (i, raw) in lines.iter().enumerate() {
        // By hand: /[ \t\r]+$/ is quadratic on a run of spaces for a backtracking engine.
        let line = raw.trim_end_matches([' ', '\t', '\r']);
        if let Some(soft) = line.strip_suffix('=') {
            s.push_str(soft);
        } else {
            s.push_str(line);
            if i < lines.len() - 1 {
                s.push('\n');
            }
        }
    }
    let decoded =
        QP_ESCAPE.replace_all(&s, |c: &Captures| char::from(u8::from_str_radix(&c[1], 16).unwrap_or(0)).to_string());
    latin1_bytes(&decoded)
}

fn body_bytes(e: &Entity) -> Vec<u8> {
    let cte = e.headers.get("content-transfer-encoding").map(|v| js::trim(v).to_lowercase()).unwrap_or_default();
    match cte.as_str() {
        "base64" => base64_lenient(&e.body.concat()),
        "quoted-printable" => quoted_printable(&e.body),
        _ => latin1_bytes(&e.body.join("\n")),
    }
}

/// `format=flowed` (RFC 3676) text with its soft line breaks joined.
fn unflow(text: &str, delsp: bool) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut cur: Option<String> = None;
    for raw in text.split('\n') {
        let quote_len = raw.len() - raw.trim_start_matches('>').len();
        let quote = &raw[..quote_len];
        let mut line = &raw[quote_len..];
        if let Some(rest) = line.strip_prefix(' ') {
            line = rest;
        }
        let soft = line.ends_with(' ') && line != "-- ";
        if soft && delsp {
            line = &line[..line.len() - 1];
        }
        cur = Some(match cur {
            None if !quote.is_empty() => format!("{quote} {line}"),
            None => line.to_string(),
            Some(c) => c + line,
        });
        if !soft {
            out.extend(cur.take());
        }
    }
    out.extend(cur);
    out.join("\n")
}

static MANY_BLANKS: LazyLock<Regex> = LazyLock::new(|| re(r"\n{3,}"));

fn clean(text: &str, html: bool, opts: &MboxOptions) -> String {
    let mut lines: Vec<&str> = text.split('\n').collect();
    if !opts.keep_signature {
        if let Some(sig) = lines.iter().position(|l| *l == "-- " || (html && *l == "--")) {
            lines.truncate(sig);
        }
    }
    if !opts.keep_quoted {
        lines.retain(|l| !(l.starts_with('>') || l.starts_with(" >")));
    }
    let joined = lines.iter().map(|l| if *l == "-- " { *l } else { js::trim_end(l) }).collect::<Vec<_>>().join("\n");
    js::trim(&MANY_BLANKS.replace_all(&joined, "\n\n")).to_string()
}

/// Each name once, first place kept — through a set: a list searched for each name was quadratic in a message with
/// thousands of parts.
fn dedupe(v: Vec<String>) -> Vec<String> {
    v.into_iter().collect::<indexmap::IndexSet<String>>().into_iter().collect()
}

fn content_of(e: &Entity, opts: &MboxOptions, depth: usize) -> Content {
    let default_type = "text/plain".to_string();
    let ty = parse_params(Some(e.headers.get("content-type").unwrap_or(&default_type)));
    let disposition = parse_params(e.headers.get("content-disposition"));
    let name = disposition.params.get("filename").or_else(|| ty.params.get("name")).cloned();
    let attached = disposition.value == "attachment";
    let mime = if ty.value.is_empty() { "text/plain".to_string() } else { ty.value.clone() };
    if mime.starts_with("multipart/") && depth < 30 {
        if let Some(boundary) = ty.params.get("boundary").filter(|b| !b.is_empty()) {
            let kids: Vec<Content> =
                split_parts(&e.body, boundary).iter().map(|k| content_of(k, opts, depth + 1)).collect();
            let attachments: Vec<String> = kids.iter().flat_map(|k| k.attachments.clone()).collect();
            if mime == "multipart/alternative" {
                let pick = kids
                    .iter()
                    .find(|k| !k.text.is_empty() && !k.html)
                    .or_else(|| kids.iter().find(|k| !k.text.is_empty()));
                return Content {
                    text: pick.map(|p| p.text.clone()).unwrap_or_default(),
                    html: pick.is_some_and(|p| p.html),
                    attachments: dedupe(attachments),
                };
            }
            let said: Vec<&Content> = kids.iter().filter(|k| !k.text.is_empty()).collect();
            return Content {
                text: said.iter().map(|k| k.text.as_str()).collect::<Vec<_>>().join("\n\n"),
                html: !said.is_empty() && said.iter().all(|k| k.html),
                attachments,
            };
        }
    }
    if mime == "message/rfc822" && depth < 30 {
        let lines: Vec<String> = split_crlf(&latin1(&body_bytes(e)));
        let inner = entity_of(&lines);
        let c = content_of(&inner, opts, depth + 1);
        let mut head: Vec<String> = [("From", "from"), ("Date", "date"), ("Subject", "subject")]
            .iter()
            .filter_map(|(label, key)| inner.headers.get(*key).map(|v| format!("{label}: {}", header_text(Some(v)))))
            .collect();
        head.push(String::new());
        head.push(c.text);
        return Content { text: js::trim(&head.join("\n")).to_string(), html: false, attachments: c.attachments };
    }
    if !attached && matches!(mime.as_str(), "text/plain" | "text/html" | "text/markdown") {
        let raw = crlf_to_lf(&decode_bytes(&body_bytes(e), ty.params.get("charset").map(String::as_str)));
        if mime == "text/html" {
            return Content {
                text: clean(&html_to_text(&raw, !opts.keep_quoted), true, opts),
                html: true,
                attachments: vec![],
            };
        }
        let flowed = ty.params.get("format").is_some_and(|f| f.to_lowercase() == "flowed");
        let text =
            if flowed { unflow(&raw, ty.params.get("delsp").is_some_and(|d| d.to_lowercase() == "yes")) } else { raw };
        return Content { text: clean(&text, false, opts), html: false, attachments: vec![] };
    }
    Content { text: String::new(), html: false, attachments: name.into_iter().filter(|n| !n.is_empty()).collect() }
}

fn split_parts(body: &[String], boundary: &str) -> Vec<Entity> {
    let open = format!("--{boundary}");
    let close = format!("--{boundary}--");
    let mut parts: Vec<Vec<String>> = Vec::new();
    let mut cur: Option<Vec<String>> = None;
    for line in body {
        let t = js::trim_end(line);
        if t == close {
            break;
        }
        if t == open {
            parts.extend(cur.take());
            cur = Some(Vec::new());
        } else if let Some(c) = &mut cur {
            c.push(line.clone());
        }
    }
    parts.extend(cur);
    parts
        .into_iter()
        .map(|p| {
            if p.first().is_some_and(String::is_empty) {
                Entity { headers: HashMap::new(), body: p[1..].to_vec() }
            } else {
                entity_of(&p)
            }
        })
        .collect()
}

static BRACKETED: LazyLock<Regex> = LazyLock::new(|| re(&format!(r"<([^<>{}]+)>", js::SPACE_CLASS)));

fn ids(raw: Option<&String>) -> Vec<String> {
    let v = raw.map(String::as_str).unwrap_or("");
    let bracketed: Vec<String> = BRACKETED.captures_iter(v).map(|c| c[1].to_string()).collect();
    if !bracketed.is_empty() {
        return bracketed;
    }
    v.split(js::is_space).filter(|x| x.contains('@')).map(str::to_string).collect()
}

/// `Name <address>` → Name, `address (Name)` → Name. By hand: `^\s*(\S+@\S+)\s*\(` is quadratic on `a@a@a@…` for a
/// backtracking engine.
fn person_of(from: &str) -> String {
    let t = js::trim(from);
    if let Some(lt) = t.rfind('<') {
        if t.ends_with('>') && lt < t.len() - 1 && !t[lt + 1..t.len() - 1].contains('>') {
            let name = js::trim(&t[..lt]);
            let name = match name.strip_prefix('"').and_then(|n| n.strip_suffix('"')) {
                Some(inner) if name.len() >= 2 => inner,
                _ => name,
            };
            let name = js::trim(name);
            return if name.is_empty() { js::trim(&t[lt + 1..t.len() - 1]).to_string() } else { name.to_string() };
        }
    }
    let open = t.find('(');
    let address = match open {
        Some(o) if o > 0 => js::trim(&t[..o]),
        _ => "",
    };
    if let Some(o) = open {
        if t.ends_with(')')
            && js::len16(&t[o..]) > 2
            && !address.is_empty()
            && !address.chars().any(js::is_space)
            && address.find('@').is_some_and(|a| a > 0)
            && !address.ends_with('@')
        {
            return js::trim(&t[o + 1..t.len() - 1]).to_string();
        }
    }
    t.to_string()
}

static COMMENTS: LazyLock<Regex> = LazyLock::new(|| re(r"\([^)]*\)"));

/// `(comments)` taken out of a header, up to its last `)`: past it, each `(` would scan to the end and fail.
fn drop_comments(v: &str) -> String {
    let end = v.rfind(')').map_or(0, |i| i + 1);
    format!("{}{}", COMMENTS.replace_all(&v[..end], " "), &v[end..])
}

static FROM_LINE: LazyLock<Regex> = LazyLock::new(|| re(&format!(r"^From [^{s}]+[{s}]+", s = js::SPACE_CLASS)));

/// One message's lines (raw bytes as Latin-1 characters, no line endings) as a record, or `None` when it says nothing.
fn message_record(mut lines: Vec<String>, from_line: &str, source: &str, opts: &MboxOptions) -> Option<MemoryRecord> {
    while lines.last().is_some_and(|l| js::trim(l).is_empty()) {
        lines.pop();
    }
    let e = entity_of(&lines);
    let h = |key: &str| header_text(e.headers.get(key));
    let content = content_of(&e, opts, 0);
    let attachments = dedupe(content.attachments);
    let listed =
        if attachments.is_empty() { String::new() } else { format!("Attachments: {}", attachments.join(", ")) };
    let text = [content.text, listed].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    let subject = h("subject");
    if text.is_empty() && subject.is_empty() {
        return None;
    }
    let from = h("from");
    let date = js::trim(&drop_comments(&h("date"))).to_string();
    let after_from = FROM_LINE.replace(from_line, "");
    let at =
        time_of(Some(&At::Text(date.clone()))).or_else(|| time_of(Some(&At::Text(js::trim(&after_from).to_string()))));
    let own = ids(e.headers.get("message-id")).into_iter().next();
    let id = own.unwrap_or_else(|| {
        let joined = [from.as_str(), date.as_str(), subject.as_str(), js::head16(&text, 4000)].join("\u{0}");
        hex::encode(Sha256::digest(joined.as_bytes()))[..32].to_string()
    });
    let in_reply_to = ids(e.headers.get("in-reply-to")).into_iter().next();
    let thread = ids(e.headers.get("references"))
        .into_iter()
        .next()
        .or_else(|| in_reply_to.clone())
        .unwrap_or_else(|| id.clone());
    Some(
        Draft {
            id,
            title: Some(subject),
            text,
            kind: Some("email".into()),
            source: Some(source.to_string()),
            collection: opts.collection.clone(),
            author: (!from.is_empty()).then(|| person_of(&from)),
            at: at.map(At::Ms),
            thread: Some(thread),
            metadata: vec![
                ("from".into(), Some(Value::String(from))),
                ("to".into(), Some(Value::String(h("to")))),
                ("cc".into(), Some(Value::String(h("cc")))),
                ("inReplyTo".into(), in_reply_to.map(Value::String)),
            ],
            ..Default::default()
        }
        .finish(),
    )
}

/// `s.split(/\r?\n/)`.
fn split_crlf(s: &str) -> Vec<String> {
    s.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l).to_string()).collect()
}

fn stem(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = super::files::extname(&name).len();
    name[..name.len() - ext].to_string()
}

static TIME_IN_LINE: LazyLock<Regex> = LazyLock::new(|| re(r"[0-9]:[0-9][0-9]"));

struct Mbox {
    lines: Lines<BufReader<File>>,
    msg: Option<Vec<String>>,
    from_line: String,
    size: usize,
    previous_blank: bool,
    index: usize,
    max: usize,
    source: String,
    o: MboxOptions,
    done: bool,
}

impl Mbox {
    fn finish(&mut self) -> Option<MemoryRecord> {
        let msg = self.msg.take()?;
        let r = message_record(msg, &self.from_line, &self.source, &self.o);
        if r.is_none() {
            if let Some(cb) = &self.o.on_skip {
                cb("no subject and no text", self.index);
            }
        }
        self.index += 1;
        r
    }
}

impl Iterator for Mbox {
    type Item = Result<MemoryRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.done {
            let line = match self.lines.next() {
                None => {
                    self.done = true;
                    return self.finish().map(Ok);
                }
                Some(Err(e)) => {
                    self.done = true;
                    return Some(Err(Error::Import(format!("cannot read the mailbox: {e}"))));
                }
                Some(Ok(bytes)) => latin1(&bytes),
            };
            // A new message: a `From ` line at the start or after an empty line, carrying a time
            // (`From - Mon Jan 1 00:00:00 2024`).
            if line.starts_with("From ") && self.previous_blank && (self.msg.is_none() || TIME_IN_LINE.is_match(&line))
            {
                let r = self.finish();
                self.msg = Some(Vec::new());
                self.from_line = line;
                self.size = 0;
                self.previous_blank = false;
                if let Some(r) = r {
                    return Some(Ok(r));
                }
                continue;
            }
            self.previous_blank = line.is_empty() || line == "\r";
            if self.msg.is_none() {
                if js::trim(&line).is_empty() {
                    continue;
                }
                self.msg = Some(Vec::new()); // a file that does not open with a `From ` line: one message from its first line
            }
            self.size += line.chars().count() + 1;
            if self.size > self.max {
                continue;
            }
            let unescaped = if line.starts_with('>') && line.trim_start_matches('>').starts_with("From ") {
                line[1..].to_string()
            } else {
                line
            };
            if let Some(m) = &mut self.msg {
                m.push(unescaped);
            }
        }
        None
    }
}

/// Records from an mbox file, a message at a time.
pub async fn from_mbox(path: impl AsRef<Path>, opts: MboxOptions) -> Result<RecordStream> {
    let path = path.as_ref().to_path_buf();
    blocking_stream(move || {
        let file = File::open(&path).map_err(|e| Error::io(&path, e))?;
        Ok(Mbox {
            lines: Lines::new(BufReader::new(file)),
            msg: None,
            from_line: String::new(),
            size: 0,
            previous_blank: true,
            index: 0,
            max: opts.max_message_bytes.unwrap_or(32 * 1024 * 1024),
            source: opts.source.clone().unwrap_or_else(|| stem(&path)),
            o: opts,
            done: false,
        })
    })
    .await
}

/// The record of one `.eml` file (an RFC 5322 message).
pub async fn from_eml(path: impl AsRef<Path>, opts: MboxOptions) -> Result<RecordStream> {
    let path = path.as_ref().to_path_buf();
    blocking_stream(move || {
        let bytes = fs::read(&path).map_err(|e| Error::io(&path, e))?;
        let mut lines = split_crlf(&latin1(&bytes));
        let from_line =
            if lines.first().is_some_and(|l| l.starts_with("From ")) { lines.remove(0) } else { String::new() };
        let source = opts.source.clone().unwrap_or_else(|| stem(&path));
        let r = message_record(lines, &from_line, &source, &opts);
        if r.is_none() {
            if let Some(cb) = &opts.on_skip {
                cb("no subject and no text", 0);
            }
        }
        Ok(r.into_iter().map(Ok))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn many_attachment_names_are_deduplicated_in_linear_time() {
        let names: Vec<String> = (0..60_000).map(|i| format!("file-{}.pdf", i % 40_000)).collect();
        let t = std::time::Instant::now();
        let out = dedupe(names);
        assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
        assert_eq!(out.len(), 40_000);
        assert_eq!((out[0].as_str(), out[39_999].as_str()), ("file-0.pdf", "file-39999.pdf"));
    }
}
