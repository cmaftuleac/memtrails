//! Small text tools the importers share: HTML reduced to text, character entities, bytes in any charset, long texts
//! cut into parts, and the record cleanup every importer ends with. Any language and script is kept as written.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use encoding_rs::{Encoding, REPLACEMENT, UTF_8, WINDOWS_1252};
use regex::{Captures, Regex};
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::js;
use crate::records::{At, MemoryRecord, Metadata};

pub(crate) fn re(pattern: &str) -> Regex {
    // The patterns are constants of this crate, checked by its tests.
    Regex::new(pattern).unwrap_or_else(|e| panic!("memtrails: invalid built-in pattern {pattern}: {e}"))
}

static NAMED: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
    let pairs: &[(&str, &str)] = &[
        ("amp", "&"),
        ("lt", "<"),
        ("gt", ">"),
        ("quot", "\""),
        ("apos", "'"),
        ("nbsp", "\u{a0}"),
        ("ensp", "\u{2002}"),
        ("emsp", "\u{2003}"),
        ("thinsp", "\u{2009}"),
        ("shy", ""),
        ("zwnj", "\u{200c}"),
        ("zwj", "\u{200d}"),
        ("lrm", "\u{200e}"),
        ("rlm", "\u{200f}"),
        ("copy", "©"),
        ("reg", "®"),
        ("trade", "™"),
        ("hellip", "…"),
        ("mdash", "—"),
        ("ndash", "–"),
        ("minus", "−"),
        ("lsquo", "‘"),
        ("rsquo", "’"),
        ("sbquo", "‚"),
        ("ldquo", "“"),
        ("rdquo", "”"),
        ("bdquo", "„"),
        ("laquo", "«"),
        ("raquo", "»"),
        ("lsaquo", "‹"),
        ("rsaquo", "›"),
        ("euro", "€"),
        ("pound", "£"),
        ("yen", "¥"),
        ("cent", "¢"),
        ("curren", "¤"),
        ("deg", "°"),
        ("middot", "·"),
        ("bull", "•"),
        ("times", "×"),
        ("divide", "÷"),
        ("sect", "§"),
        ("para", "¶"),
        ("iexcl", "¡"),
        ("iquest", "¿"),
        ("szlig", "ß"),
        ("aelig", "æ"),
        ("AElig", "Æ"),
        ("oelig", "œ"),
        ("OElig", "Œ"),
        ("oslash", "ø"),
        ("Oslash", "Ø"),
        ("eth", "ð"),
        ("ETH", "Ð"),
        ("thorn", "þ"),
        ("THORN", "Þ"),
        ("ordf", "ª"),
        ("ordm", "º"),
        ("micro", "µ"),
        ("plusmn", "±"),
        ("frac12", "½"),
        ("frac14", "¼"),
        ("frac34", "¾"),
        ("sup1", "¹"),
        ("sup2", "²"),
        ("sup3", "³"),
        ("dagger", "†"),
        ("Dagger", "‡"),
        ("permil", "‰"),
        ("prime", "′"),
        ("Prime", "″"),
        ("le", "≤"),
        ("ge", "≥"),
        ("ne", "≠"),
        ("infin", "∞"),
        ("asymp", "≈"),
        ("larr", "←"),
        ("rarr", "→"),
        ("uarr", "↑"),
        ("darr", "↓"),
        ("harr", "↔"),
        ("hearts", "♥"),
        ("not", "¬"),
        ("brvbar", "¦"),
        ("uml", "¨"),
        ("acute", "´"),
        ("cedil", "¸"),
        ("macr", "¯"),
        ("circ", "ˆ"),
        ("tilde", "˜"),
    ];
    let mut m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    // Greek letters by name (Alpha … Omega, alpha … omega, sigmaf).
    const GREEK: [&str; 24] = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu",
        "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    ];
    for (i, name) in GREEK.iter().enumerate() {
        let offset = i as u32 + if i >= 17 { 1 } else { 0 };
        let upper = char::from_u32(0x391 + offset).map(String::from).unwrap_or_default();
        let lower = char::from_u32(0x3b1 + offset).map(String::from).unwrap_or_default();
        m.insert(name.to_string(), upper);
        m.insert(name.to_lowercase(), lower);
    }
    m.insert("sigmaf".into(), "ς".into());
    m
});

/// A Latin letter with a mark: `&eacute;` `&Uuml;` `&ccedil;` `&scaron;` … composed from the letter and a combining
/// mark.
fn mark(name: &str) -> Option<char> {
    Some(match name {
        "acute" => '\u{301}',
        "grave" => '\u{300}',
        "circ" => '\u{302}',
        "tilde" => '\u{303}',
        "uml" => '\u{308}',
        "ring" => '\u{30a}',
        "cedil" => '\u{327}',
        "caron" => '\u{30c}',
        "macr" => '\u{304}',
        "breve" => '\u{306}',
        "ogon" => '\u{328}',
        "dot" => '\u{307}',
        "dblac" => '\u{30b}',
        _ => return None,
    })
}

fn named_entity(name: &str) -> Option<String> {
    if let Some(v) = NAMED.get(name) {
        return Some(v.clone());
    }
    let mut cs = name.chars();
    let letter = cs.next().filter(char::is_ascii_alphabetic)?;
    let m = mark(cs.as_str())?;
    let composed: String = [letter, m].iter().collect::<String>().nfc().collect();
    (composed.chars().count() == 1).then_some(composed)
}

fn code_point(n: Option<u32>) -> String {
    match n {
        // What browsers do with `&#150;` and the like.
        Some(n @ 0x80..=0x9f) => WINDOWS_1252.decode_without_bom_handling(&[n as u8]).0.into_owned(),
        Some(n) if n > 0 && n <= 0x10ffff && !(0xd800..=0xdfff).contains(&n) => {
            char::from_u32(n).map(String::from).unwrap_or_else(|| "\u{fffd}".into())
        }
        _ => "\u{fffd}".into(),
    }
}

static ENTITY: LazyLock<Regex> =
    LazyLock::new(|| re(r"&(#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7}|[a-zA-Z][a-zA-Z0-9]{1,31});"));

/// `&amp;`, `&#233;`, `&#x1F600;`, `&eacute;` and the other common names, decoded; unknown names are left as written.
pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    ENTITY
        .replace_all(s, |c: &Captures| {
            let r = &c[1];
            if let Some(num) = r.strip_prefix('#') {
                let n = match num.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok(),
                    None => num.parse::<u32>().ok(),
                };
                return code_point(n);
            }
            named_entity(r).unwrap_or_else(|| c[0].to_string())
        })
        .into_owned()
}

const BLOCK: &str = "address|article|aside|blockquote|caption|center|dd|details|dialog|div|dl|dt|fieldset|figcaption|figure|footer|form|h[1-6]|header|hr|main|nav|ol|p|pre|section|summary|table|tbody|thead|tfoot|tr|ul";

/// The end of `s` up to its last `>` (a byte offset).
fn last_gt(s: &str) -> usize {
    s.rfind('>').map_or(0, |i| i + 1)
}

/// `f` applied to `s` up to its last `>`, the rest kept. Every tag pattern here ends in `[^>]*>`: past the last `>`
/// each `<` is a start that scans to the end and fails, so a run of unclosed `<a<a<a…` was quadratic for a
/// backtracking engine; before it, each scan stops at the next `>` and matches.
fn before_last_gt(s: &str, f: impl Fn(&str) -> String) -> String {
    let end = last_gt(s);
    if end == s.len() {
        f(s)
    } else {
        format!("{}{}", f(&s[..end]), &s[end..])
    }
}

/// Removes every `<tag …>…</tag>` (nested ones too), for `blockquote`: the quoted earlier message in an HTML reply.
fn drop_elements(html: &str, tag: &str) -> String {
    let pattern = re(&format!(r"<(/?)(?i-u:{tag})(?-u:\b)[^>]*>"));
    let head = &html[..last_gt(html)]; // see before_last_gt
    let mut out = String::new();
    let mut depth = 0usize;
    let mut from = 0usize;
    for c in pattern.captures_iter(head) {
        let (Some(m), closing) = (c.get(0), c.get(1).is_some_and(|x| !x.as_str().is_empty())) else { continue };
        if !closing {
            if depth == 0 {
                out.push_str(&html[from..m.start()]);
            }
            depth += 1;
        } else if depth > 0 {
            depth -= 1;
            if depth == 0 {
                from = m.end();
            }
        }
    }
    if depth == 0 {
        out.push_str(&html[from..]);
    }
    out
}

/// Each `<tag …>…</tag>` found by `open` (the tag's name in group 1) replaced by `f(name, element)`, scanning once:
/// `<(script|…)\b[\s\S]*?<\/\1>` rescanned to the end for every unclosed opening tag, quadratic on
/// `<script<script…`. An unclosed one has `element` `None`; `f` then gives what replaces it and everything after it,
/// or `None` to leave it as written. A tag with no closing tag after it has none after a later opening tag either, so
/// it is looked for once. The text between elements goes through `outside`.
fn replace_elements(
    s: &str,
    open: &Regex,
    f: impl Fn(&str, Option<&str>) -> Option<String>,
    outside: impl Fn(&str) -> String,
) -> String {
    let mut closes: HashMap<String, Regex> = HashMap::new();
    let mut unclosed: HashSet<String> = HashSet::new();
    let mut out = String::new();
    let mut from = 0usize;
    let mut at = 0usize;
    while let Some(c) = open.captures_at(s, at) {
        let (Some(m), Some(name)) = (c.get(0), c.get(1)) else { break };
        let name = name.as_str().to_lowercase();
        let mut end: Option<usize> = None;
        if !unclosed.contains(&name) {
            let close =
                closes.entry(name.clone()).or_insert_with(|| re(&format!(r"(?i-u:</{name})[{}]*>", js::SPACE_CLASS)));
            match close.find_at(s, m.end()) {
                Some(x) => end = Some(x.end()),
                None => {
                    unclosed.insert(name.clone());
                }
            }
        }
        let replaced = f(&name, end.map(|e| &s[m.start()..e]));
        let Some(replaced) = replaced else {
            // Left as written: look on after the opening tag.
            at = m.end().max(m.start() + 1);
            continue;
        };
        out.push_str(&outside(&s[from..m.start()]));
        out.push_str(&replaced);
        match end {
            None => return out,
            Some(e) => {
                from = e;
                at = e;
            }
        }
    }
    out.push_str(&outside(&s[from..]));
    out
}

/// `<![CDATA[…]]>` as its text; an unclosed one, and everything after it, left as written.
fn unwrap_cdata(s: &str) -> String {
    let mut out = String::new();
    let mut from = 0usize;
    while let Some(i) = s[from..].find("<![CDATA[").map(|i| i + from) {
        let Some(j) = s[i + 9..].find("]]>").map(|j| j + i + 9) else { break };
        out.push_str(&s[from..i]);
        out.push_str(&s[i + 9..j]);
        from = j + 3;
    }
    out.push_str(&s[from..]);
    out
}

static COMMENT: LazyLock<Regex> = LazyLock::new(|| re(r"<!--(?s:.)*?(?:-->|$)"));
static HIDDEN: LazyLock<Regex> =
    LazyLock::new(|| re(r"<((?i-u:script|style|head|noscript|template|svg|math|iframe|object))(?-u:\b)"));
static PRE: LazyLock<Regex> = LazyLock::new(|| re(r"<((?i-u:pre))(?-u:\b)"));
static BR: LazyLock<Regex> = LazyLock::new(|| re(r"<(?i-u:br)(?-u:\b)[^>]*>"));
static LI: LazyLock<Regex> = LazyLock::new(|| re(r"<(?i-u:li)(?-u:\b)[^>]*>"));
static BLOCKS: LazyLock<Regex> = LazyLock::new(|| re(&format!(r"</?(?i-u:{BLOCK})(?-u:\b)[^>]*>")));
static CELL_END: LazyLock<Regex> = LazyLock::new(|| re(&format!(r"</(?i-u:t[dh])[{}]*>", js::SPACE_CLASS)));
static TAG: LazyLock<Regex> = LazyLock::new(|| re(r"</?[a-zA-Z][^>]*>"));
static DECL: LazyLock<Regex> = LazyLock::new(|| re(r"<[!?][^>]*>"));
static ANY_TAG: LazyLock<Regex> = LazyLock::new(|| re(r"<[^>]*>"));
static CRLF: LazyLock<Regex> = LazyLock::new(|| re(r"\r\n?"));

/// Title rules see at most this much of a page (UTF-16 units): a title is near the top, and a hostile page is not read
/// to its end.
const TITLE_SCAN: usize = 64 * 1024;

/// The `<title>` of an HTML page, else its first `<h1>`, as text; `None` when it has neither.
pub fn html_title(html: &str) -> Option<String> {
    let top = js::head16(html, TITLE_SCAN);
    let head = &top[..last_gt(top)]; // see before_last_gt
    for tag in ["title", "h1"] {
        let open = re(&format!(r"<(?i-u:{tag})(?-u:\b)[^>]*>"));
        let Some(o) = open.find(head) else { continue };
        let close = re(&format!(r"(?i-u:</{tag})[{}]*>", js::SPACE_CLASS));
        let Some(end) = close.find_at(head, o.end()) else { continue };
        let inner = before_last_gt(&head[o.end()..end.start()], |h| ANY_TAG.replace_all(h, " ").into_owned());
        let t = js::trim(&js::collapse_spaces(&decode_entities(&inner))).to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    None
}

/// HTML as plain text: scripts, styles, comments and the head dropped, blocks and line breaks kept as lines, list
/// items as `- `, tags removed, entities decoded, `<pre>` kept as written. `drop_quotes` also drops `<blockquote>`s.
pub fn html_to_text(html: &str, drop_quotes: bool) -> String {
    let mut s = unwrap_cdata(&COMMENT.replace_all(html, " "));
    // Hidden elements dropped; a script or style left open runs to the end, any other unclosed one is left to the
    // tag rules.
    s = replace_elements(
        &s,
        &HIDDEN,
        |name, element| (element.is_some() || name == "script" || name == "style").then(|| " ".to_string()),
        str::to_string,
    );
    if drop_quotes {
        s = drop_elements(&s, "blockquote");
    }
    // Whitespace in the source is one space, except inside <pre>.
    s = replace_elements(
        &s,
        &PRE,
        |_name, element| element.map(|e| CRLF.replace_all(e, "\n").into_owned()),
        js::collapse_spaces,
    );
    s = before_last_gt(&s, |h| {
        let h = BR.replace_all(h, "\n");
        let h = LI.replace_all(&h, "\n- ");
        let h = BLOCKS.replace_all(&h, "\n");
        let h = CELL_END.replace_all(&h, "\t");
        let h = TAG.replace_all(&h, "");
        DECL.replace_all(&h, "").into_owned()
    });
    tidy(&decode_entities(&s))
}

static INLINE_SPACE: LazyLock<Regex> = LazyLock::new(|| re("[ \t\u{a0}]+"));
static BLANK_LINES: LazyLock<Regex> = LazyLock::new(|| re(r"\n{3,}"));

/// Lines trimmed at the end, spaces and tabs inside a line kept to one, at most one empty line in a row.
pub(crate) fn tidy(s: &str) -> String {
    let unified = CRLF.replace_all(s, "\n");
    let lines: Vec<String> = unified
        .split('\n')
        .map(|l| {
            let one = INLINE_SPACE.replace_all(l, " ");
            // A single leading space before text goes (`/^ (?=\S)/`).
            let one = match one.strip_prefix(' ') {
                Some(rest) if rest.chars().next().is_some_and(|c| !js::is_space(c)) => rest.to_string(),
                _ => one.into_owned(),
            };
            js::trim_end(&one).to_string()
        })
        .collect();
    js::trim(&BLANK_LINES.replace_all(&lines.join("\n"), "\n\n")).to_string()
}

/// Bytes as text the way a WHATWG `TextDecoder` reads them: the encoding's own byte order mark dropped, errors as
/// U+FFFD.
fn decode_with(encoding: &'static Encoding, bytes: &[u8]) -> String {
    encoding.decode_with_bom_removal(bytes).0.into_owned()
}

static PLAIN_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    re(r"^(us-?ascii|ascii|ansi_x3\.4-1968|utf-?8|unicode-1-1-utf-8|x-unknown|unknown(-8bit)?|default)$")
});

/// Bytes as text in the charset named (any the WHATWG Encoding Standard knows); unnamed, ASCII or unknown: UTF-8,
/// else Windows-1252.
pub fn decode_bytes(bytes: &[u8], charset: Option<&str>) -> String {
    let label = js::trim(charset.unwrap_or(""));
    let label = label.strip_prefix(['"', '\'']).unwrap_or(label);
    let label = label.strip_suffix(['"', '\'']).unwrap_or(label);
    let label = label.split('*').next().unwrap_or("").to_lowercase();
    if !label.is_empty() && !PLAIN_LABEL.is_match(&label) {
        // A charset the Encoding Standard does not know (or maps to "replacement", which a TextDecoder refuses): read
        // it as below.
        if let Some(enc) = Encoding::for_label(label.as_bytes()).filter(|e| *e != REPLACEMENT) {
            return decode_with(enc, bytes);
        }
    }
    match UTF_8.decode_without_bom_handling_and_without_replacement(bytes) {
        Some(_) => decode_with(UTF_8, bytes),
        None => decode_with(WINDOWS_1252, bytes),
    }
}

static SENTENCE_END: LazyLock<Regex> = LazyLock::new(|| re(r"\p{Sentence_Terminal}+"));

fn last_index_of(hay: &[u16], needle: &[u16]) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).rev().find(|&i| &hay[i..i + needle.len()] == needle)
}

fn is_space_unit(u: u16) -> bool {
    char::from_u32(u32::from(u)).is_some_and(js::is_space)
}

/// A long text in parts of at most `max_chars` UTF-16 units (100 at least), each cut at a paragraph, else a line, else
/// a sentence's end (in any script), else a space; never inside a character.
pub fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    let max = max_chars.max(100);
    let units: Vec<u16> = text.encode_utf16().collect();
    if units.len() <= max {
        return vec![text.to_string()];
    }
    let mut parts = Vec::new();
    let mut at = 0usize;
    let push = |parts: &mut Vec<String>, piece: &[u16]| {
        let s = String::from_utf16_lossy(piece);
        let t = js::trim(&s);
        if !t.is_empty() {
            parts.push(t.to_string());
        }
    };
    while units.len() - at > max {
        let window = &units[at..at + max];
        let half = max / 2;
        let as_i = |x: Option<usize>| x.map_or(-1, |v| v as i64);
        let mut cut = as_i(last_index_of(window, &[10, 10]));
        if cut < half as i64 {
            cut = as_i(window.iter().rposition(|u| *u == 10));
        }
        if cut < half as i64 {
            // A lone surrogate at `half` reads as U+FFFD: one unit either way, so offsets stay right.
            let tail = String::from_utf16_lossy(&window[half..]);
            cut = match SENTENCE_END.find_iter(&tail).last() {
                Some(m) => (half + js::len16(&tail[..m.end()])) as i64,
                None => -1,
            };
        }
        if cut < half as i64 {
            cut = as_i(window.iter().rposition(|u| is_space_unit(*u)));
        }
        if cut < half as i64 {
            cut = max as i64;
        }
        let mut cut = cut as usize;
        if (0xd800..=0xdbff).contains(&window[cut - 1]) {
            cut -= 1; // keep a surrogate pair together
        }
        push(&mut parts, &units[at..at + cut]);
        at += cut;
    }
    push(&mut parts, &units[at..]);
    parts
}

/// A record being put together by an importer: fields left `None` or empty are dropped, as are metadata entries left
/// `None` or `""` (the reference's `record()`).
#[derive(Debug, Clone, Default)]
pub(crate) struct Draft {
    pub id: String,
    pub text: String,
    pub title: Option<String>,
    pub kind: Option<String>,
    pub source: Option<String>,
    pub collection: Option<String>,
    pub thread: Option<String>,
    pub author: Option<String>,
    pub at: Option<At>,
    pub url: Option<String>,
    pub metadata: Vec<(String, Option<Value>)>,
}

fn kept(s: Option<String>) -> Option<String> {
    s.filter(|x| !x.is_empty())
}

impl Draft {
    /// The record, without its empty fields.
    pub fn finish(self) -> MemoryRecord {
        let metadata: Metadata = self
            .metadata
            .into_iter()
            .filter_map(|(k, v)| match v {
                None => None,
                Some(Value::String(s)) if s.is_empty() => None,
                Some(v) => Some((k, v)),
            })
            .collect();
        MemoryRecord {
            id: self.id,
            text: self.text,
            title: kept(self.title),
            kind: kept(self.kind),
            source: kept(self.source),
            collection: kept(self.collection),
            thread: kept(self.thread),
            author: kept(self.author),
            at: self.at.filter(|a| !matches!(a, At::Text(s) if s.is_empty())),
            url: kept(self.url),
            metadata: (!metadata.is_empty()).then_some(metadata),
        }
    }
}

/// `Buffer.from(s, 'base64')` as Node reads it: both alphabets, anything else skipped, ending at the first `=`.
pub(crate) fn base64_lenient(s: &str) -> Vec<u8> {
    let mut sextets: Vec<u8> = Vec::with_capacity(s.len());
    for c in s.chars() {
        let v = match c {
            'A'..='Z' => c as u8 - b'A',
            'a'..='z' => c as u8 - b'a' + 26,
            '0'..='9' => c as u8 - b'0' + 52,
            '+' | '-' => 62,
            '/' | '_' => 63,
            '=' => break,
            _ => continue,
        };
        sextets.push(v);
    }
    let mut out = Vec::with_capacity(sextets.len() * 3 / 4);
    for chunk in sextets.chunks(4) {
        let n = chunk.iter().enumerate().fold(0u32, |acc, (i, v)| acc | (u32::from(*v) << (18 - 6 * i)));
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        let take = match chunk.len() {
            4 => 3,
            3 => 2,
            2 => 1,
            _ => 0,
        };
        out.extend_from_slice(&bytes[..take]);
    }
    out
}

/// Bytes as Latin-1 characters: every byte one character, U+0000–U+00FF.
pub(crate) fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|b| char::from(*b)).collect()
}

/// Latin-1 characters back to their bytes (`Buffer.from(s, 'latin1')`: a character past U+00FF keeps its low byte).
pub(crate) fn latin1_bytes(s: &str) -> Vec<u8> {
    s.encode_utf16().map(|u| u as u8).collect()
}

/// `String(v)` for a JSON value.
pub(crate) fn js_string(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => js::number(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => s.clone(),
        Value::Array(a) => {
            a.iter().map(|x| if x.is_null() { String::new() } else { js_string(x) }).collect::<Vec<_>>().join(",")
        }
        Value::Object(_) => "[object Object]".into(),
    }
}

/// A JavaScript truthy value.
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0 && !x.is_nan()),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

/// A number as JavaScript's `Number(s)` reads a string: decimal, `0x`/`0o`/`0b`, `Infinity`; NaN for anything else.
pub(crate) fn js_number_of(s: &str) -> f64 {
    let t = js::trim(s);
    if t.is_empty() {
        return 0.0;
    }
    let radix = |p: &str, r: u32| u64::from_str_radix(p, r).map(|v| v as f64).unwrap_or(f64::NAN);
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return radix(h, 16);
    }
    if let Some(o) = t.strip_prefix("0o").or_else(|| t.strip_prefix("0O")) {
        return radix(o, 8);
    }
    if let Some(b) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        return radix(b, 2);
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    static DECIMAL: LazyLock<Regex> = LazyLock::new(|| re(r"^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$"));
    if DECIMAL.is_match(t) {
        t.parse().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

/// A file's path as a `file:` URL, percent-encoded as Node's `pathToFileURL` does.
pub(crate) fn file_url(path: &std::path::Path) -> String {
    // On Windows the separators become slashes; elsewhere a backslash is part of a name, and is encoded.
    let p = if cfg!(windows) { path.to_string_lossy().replace('\\', "/") } else { path.to_string_lossy().into_owned() };
    let mut out = String::from("file://");
    if !p.starts_with('/') {
        out.push('/');
    }
    for b in p.bytes() {
        let keep = b.is_ascii_alphanumeric() || b"-._/!$&'()*+,;=:@".contains(&b);
        if keep {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// A number as a JSON value, a whole one without a fraction (as JavaScript keeps it).
pub(crate) fn number_value(x: f64) -> Value {
    if x.fract() == 0.0 && x.abs() < 9.0e15 {
        Value::from(x as i64)
    } else {
        serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
    }
}

/// `String.prototype.trimStart`.
pub(crate) fn trim_start(s: &str) -> &str {
    s.trim_start_matches(js::is_space)
}

/// `s.replace(/\r\n?/g, '\n')`.
pub(crate) fn crlf_to_lf(s: &str) -> String {
    CRLF.replace_all(s, "\n").into_owned()
}

/// A path made absolute and normalised by its text, as Node's `path.resolve` does (`..` taken back, `.` dropped),
/// without following links.
pub(crate) fn resolve_path(p: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    use std::path::Component;
    let abs = std::path::absolute(p)?;
    let mut out = std::path::PathBuf::new();
    for c in abs.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    Ok(out)
}
