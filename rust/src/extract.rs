//! Consolidation's reading of a thread — one ask of the language model per window of whole records: what is worth
//! keeping, and in what form.
//!
//! The prompt is the measured one, extended with what long-term memory needs to keep only the useful: an importance,
//! when the item applies, whether it is contested or ambiguous — and which requests are worth keeping as episodes,
//! each with its content key (title + ~150-word summary). Each kept episode and item also gets its retrieval CUES
//! (requests the person would type later when it is needed) and its CONCEPTS (the graph's nodes).
//!
//! The GATE is code, not a hope: an item the model inferred without proof, flagged contested or ambiguous, or rated
//! below importance 2 is dropped. Everything kept passes `redact_string`.

use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::episodes::{calendar_of, render_for_extraction};
use crate::js;
use crate::llm::{ask_or_pause, memory_request, Ask, LlmAnswer};
use crate::prompts::{EXTRACT_SYSTEM, EXTRACT_SYSTEM_WITHOUT_OBSERVATIONS, OBSERVE_SYSTEM};
use crate::redact::redact_string;
use crate::store::{MemoryOrigin, StmEntry};
use crate::text::concept_key;

/// A window's size in characters.
pub const WINDOW_CHARS: usize = 60_000;
/// The answer's room. A window's kept episodes, items, cues and concepts measured on a whole real history: 7,056
/// output tokens at the median, 9,223 at p90 — and 3% of windows ran past 12,000, cut off mid-JSON. A cut answer is no
/// answer, and the worker asks the same window again.
pub const EXTRACT_MAX_TOKENS: u64 = 32_000;

const KINDS: [&str; 9] =
    ["decision", "procedure", "snippet", "template", "pitfall", "fact", "architecture", "preference", "reference"];
const OUTCOMES: [&str; 4] = ["success", "partial", "fail", "uncertain"];
/// Kinds that say "this works": kept only from a request that was seen working, at least in part.
const NEEDS_WORKING: [&str; 4] = ["procedure", "snippet", "template", "pitfall"];
const PHASES: [&str; 4] = ["done", "plan", "ongoing", "state"];
const CONSEQUENCE: &str = "so this part of the conversation waits to be consolidated";

fn observations_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object", "additionalProperties": false, "required": ["text", "about", "event_date", "start_date", "end_date", "phase", "requests", "concepts", "cues"],
            "properties": {
                "text": { "type": "string" }, "about": { "type": "string" }, "event_date": { "type": "string" },
                "start_date": { "type": "string" }, "end_date": { "type": "string" },
                "phase": { "type": "string", "enum": PHASES },
                "requests": { "type": "array", "items": { "type": "integer" } }, "concepts": { "type": "array", "items": { "type": "string" } },
                "cues": { "type": "array", "items": { "type": "string" } },
            },
        },
    })
}

/// The extraction's JSON Schema, with or without observations.
pub(crate) fn extract_schema(with_observations: bool) -> Value {
    let mut properties = json!({
        "segment_summary": { "type": "string" },
        "episodes": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": false, "required": ["request", "outcome", "keep", "title", "summary", "cues", "concepts"],
                "properties": {
                    "request": { "type": "integer" }, "outcome": { "type": "string", "enum": OUTCOMES }, "keep": { "type": "boolean" }, "title": { "type": "string" }, "summary": { "type": "string" },
                    "cues": { "type": "array", "items": { "type": "string" } }, "concepts": { "type": "array", "items": { "type": "string" } },
                },
            },
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": false,
                "required": ["kind", "title", "summary", "body", "applies_when", "scope", "requests", "confidence", "importance", "contested", "ambiguous", "cues", "concepts"],
                "properties": {
                    "kind": { "type": "string", "enum": KINDS },
                    "title": { "type": "string" }, "summary": { "type": "string" }, "body": { "type": "string" }, "applies_when": { "type": "string" },
                    "scope": { "type": "string", "enum": ["project", "global", "person"] },
                    "requests": { "type": "array", "items": { "type": "integer" } },
                    "confidence": { "type": "string", "enum": ["verified", "stated", "inferred"] },
                    "importance": { "type": "integer" },
                    "contested": { "type": "boolean" }, "ambiguous": { "type": "boolean" },
                    "cues": { "type": "array", "items": { "type": "string" } }, "concepts": { "type": "array", "items": { "type": "string" } },
                },
            },
        },
    });
    let mut required = vec!["segment_summary", "episodes", "items"];
    if with_observations {
        properties["observations"] = observations_schema();
        required.push("observations");
    }
    json!({ "type": "object", "additionalProperties": false, "required": required, "properties": properties })
}

fn observe_schema() -> Value {
    json!({ "type": "object", "additionalProperties": false, "required": ["observations"], "properties": { "observations": observations_schema() } })
}

/// Something worth keeping, as the gate let it through.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Its kind.
    pub kind: String,
    /// Its title.
    pub title: String,
    /// Its ~150-word summary.
    pub summary: String,
    /// Everything needed to reuse it.
    pub body: String,
    /// When it is needed.
    pub applies_when: String,
    /// project, global or person.
    pub scope: String,
    /// The request uids it was learned from.
    pub provenance: Vec<String>,
    /// verified or stated.
    pub confidence: String,
    /// 1–5.
    pub importance: f64,
    /// Requests that would need it.
    pub cues: Vec<String>,
    /// What it is about.
    pub concepts: Vec<String>,
    /// Where it came from, when a guard checked it.
    pub origin: Option<MemoryOrigin>,
}

/// One dated thing the conversation told — kept as said, never merged with a different one.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// The event or fact itself.
    pub text: String,
    /// Who or what it is about.
    pub about: String,
    /// When it happened (YYYY-MM-DD, YYYY-MM or YYYY), or empty.
    pub event_at: String,
    /// For what lasts: when it began.
    pub start_at: Option<String>,
    /// For what lasts: when it ended.
    pub end_at: Option<String>,
    /// done, plan, ongoing or state.
    pub phase: String,
    /// The request uids it was told in.
    pub provenance: Vec<String>,
    /// What it names.
    pub concepts: Vec<String>,
    /// Questions it answers.
    pub cues: Vec<String>,
    /// Where it came from, when a guard checked it.
    pub origin: Option<MemoryOrigin>,
}

/// A request kept as an episode, with its key.
#[derive(Debug, Clone, PartialEq)]
pub struct KeptEpisode {
    /// The request's uid.
    pub uid: String,
    /// 4–10 words.
    pub title: String,
    /// ~150 words.
    pub summary: String,
    /// Requests that would need it.
    pub cues: Vec<String>,
    /// What it is about.
    pub concepts: Vec<String>,
    /// success, partial, fail or uncertain.
    pub outcome: Option<String>,
    /// Where it came from, when a guard checked it.
    pub origin: Option<MemoryOrigin>,
}

/// What a window's reading keeps.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Extraction {
    /// What the part worked on.
    pub summary: String,
    /// Typed knowledge.
    pub candidates: Vec<Candidate>,
    /// Requests kept as episodes.
    pub episodes: Vec<KeptEpisode>,
    /// Dated observations.
    pub observations: Vec<Observation>,
    /// Items the gate dropped.
    pub dropped: usize,
}

/// A reading, or why there is none.
#[derive(Debug, Clone, PartialEq)]
pub enum Read<T> {
    /// What was read.
    Read(T),
    /// The model could not be asked, or did not answer: why.
    Unavailable(String),
    /// One for this purpose is already being asked.
    Busy,
}

fn str_of(v: &Value) -> Option<&str> {
    v.as_str()
}

/// A cue list as memory keeps it: non-empty, scrubbed, no repeats — every one the model wrote, whole.
pub fn clean_cues(raw: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in raw.as_array().into_iter().flatten() {
        let Some(c) = c.as_str() else { continue };
        if js::trim(c).is_empty() {
            continue;
        }
        let v = redact_string(js::trim(c));
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// A concept list as memory keeps it: named as written (scrubbed), one per concept key — every one, whole.
pub fn clean_concepts(raw: &Value) -> Vec<String> {
    let mut out: IndexMap<String, String> = IndexMap::new();
    for c in raw.as_array().into_iter().flatten() {
        let Some(c) = c.as_str() else { continue };
        let name = redact_string(js::trim(c));
        let key = concept_key(&name);
        if !key.is_empty() && !out.contains_key(&key) {
            out.insert(key, name);
        }
    }
    out.into_values().collect()
}

fn is_date(d: &str) -> bool {
    let b = d.as_bytes();
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
    match b.len() {
        4 => digits(0..4),
        7 => digits(0..4) && b[4] == b'-' && digits(5..7),
        10 => digits(0..4) && b[4] == b'-' && digits(5..7) && b[7] == b'-' && digits(8..10),
        _ => false,
    }
}

fn date_of(d: &Value) -> String {
    d.as_str().map(js::trim).filter(|t| is_date(t)).unwrap_or("").to_string()
}

/// The uid of request `n` (1-based), when `n` is a whole number naming one.
fn uid_at(uids: &[String], n: &Value) -> Option<String> {
    let n = n.as_f64()?;
    if n.fract() != 0.0 || n < 1.0 {
        return None;
    }
    uids.get(n as usize - 1).cloned()
}

fn requests(x: &Value) -> Vec<&Value> {
    x["requests"].as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

/// The observations as memory keeps them: with text, from a request of this part, scrubbed; a date only when it is
/// one.
pub fn clean_observations(raw: &[Value], uids: &[String]) -> Vec<Observation> {
    let mut out = Vec::new();
    for x in raw {
        let text = str_of(&x["text"]).map(|t| redact_string(js::trim(t))).unwrap_or_default();
        let provenance: Vec<String> = requests(x).into_iter().filter_map(|n| uid_at(uids, n)).collect();
        if text.is_empty() || provenance.is_empty() {
            continue;
        }
        let phase = x["phase"].as_str().filter(|p| PHASES.contains(p)).unwrap_or("done");
        let mut cues = clean_cues(&x["cues"]);
        cues.truncate(4);
        out.push(Observation {
            text,
            about: redact_string(js::trim(str_of(&x["about"]).unwrap_or(""))),
            event_at: date_of(&x["event_date"]),
            start_at: Some(date_of(&x["start_date"])).filter(|d| !d.is_empty()),
            end_at: Some(date_of(&x["end_date"])).filter(|d| !d.is_empty()),
            phase: phase.to_string(),
            provenance,
            concepts: clean_concepts(&x["concepts"]),
            cues,
            origin: None,
        });
    }
    out
}

/// The windows a conversation's pending requests are read in: whole requests, at most `max` characters — a request
/// longer than that alone in its window, read there in parts.
pub fn windows_of(eps: &[StmEntry], max: usize) -> Vec<Vec<StmEntry>> {
    let mut out = Vec::new();
    let mut cur: Vec<StmEntry> = Vec::new();
    let mut size = 0;
    for e in eps {
        let n = js::len16(&render_for_extraction(e, cur.len() + 1)).min(max);
        if !cur.is_empty() && size + n > max {
            out.push(std::mem::take(&mut cur));
            size = 0;
        }
        cur.push(e.clone());
        size += n;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn outcome_in(x: Option<&Value>) -> Option<String> {
    x?["outcome"].as_str().filter(|o| OUTCOMES.contains(o)).map(str::to_string)
}

/// What survives the gate. Pure: the rules, in one place. `outcome_of` is how each request ended: a kind that says
/// "this works" is dropped when every request it came from failed or is unknown.
pub fn gate(raw: &[Value], uids: &[String], outcome_of: impl Fn(&Value) -> Option<String>) -> (Vec<Candidate>, usize) {
    let mut kept = Vec::new();
    let mut dropped = 0;
    for x in raw {
        let kind = x["kind"].as_str();
        let nonblank = |k: &str| x[k].as_str().is_some_and(|t| !js::trim(t).is_empty());
        let importance = x["importance"].as_f64();
        let ok = kind.is_some_and(|k| KINDS.contains(&k))
            && nonblank("title")
            && nonblank("summary")
            && matches!(x["confidence"].as_str(), Some("verified" | "stated"))
            && x["contested"] != true
            && x["ambiguous"] != true
            && importance.is_some_and(|i| i >= 2.0);
        let ends: Vec<String> = requests(x).into_iter().filter_map(&outcome_of).collect();
        let unproven = kind.is_some_and(|k| NEEDS_WORKING.contains(&k))
            && !ends.is_empty()
            && ends.iter().all(|o| o == "fail" || o == "uncertain");
        if !ok || unproven {
            dropped += 1;
            continue;
        }
        let text = |k: &str| redact_string(js::trim(x[k].as_str().unwrap_or("")));
        let scope = x["scope"].as_str().filter(|s| matches!(*s, "global" | "person")).unwrap_or("project");
        kept.push(Candidate {
            kind: kind.unwrap_or("fact").to_string(),
            title: text("title"),
            summary: text("summary"),
            body: text("body"),
            applies_when: text("applies_when"),
            scope: scope.to_string(),
            provenance: requests(x).into_iter().filter_map(|n| uid_at(uids, n)).collect(),
            confidence: x["confidence"].as_str().unwrap_or("stated").to_string(),
            importance: importance.unwrap_or(2.0).round().clamp(1.0, 5.0),
            cues: clean_cues(&x["cues"]),
            concepts: clean_concepts(&x["concepts"]),
            origin: None,
        });
    }
    (kept, dropped)
}

/// A request's text in parts of at most `max` UTF-16 units, split at line ends (a longer line at `max`) — every
/// character in exactly one part.
pub fn parts_of(text: &str, max: usize) -> Vec<String> {
    if js::len16(text) <= max {
        return vec![text.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in text.split('\n') {
        let mut rest = line.to_string();
        while js::len16(&rest) > max {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            let mut head = js::head16(&rest, max).to_string();
            if head.is_empty() {
                // A character wider than the part (max 1, an emoji): it goes whole.
                head = rest.chars().next().map(String::from).unwrap_or_default();
            }
            rest = rest[head.len()..].to_string();
            out.push(head);
        }
        if !cur.is_empty() && js::len16(&cur) + 1 + js::len16(&rest) > max {
            out.push(std::mem::replace(&mut cur, rest));
        } else if cur.is_empty() {
            cur = rest;
        } else {
            cur = format!("{cur}\n{rest}");
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Where a window comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowMeta {
    /// A working folder.
    pub cwd: Option<String>,
    /// The records' kind.
    pub engine: String,
    /// The thread's title.
    pub title: String,
}

/// What the model reads before the part: where it comes from, the calendar of its days, and what other conversations
/// say that it may refer to.
fn header_of(win: &[StmEntry], meta: &WindowMeta, known: Option<&[String]>) -> String {
    let folder = meta.cwd.as_deref().filter(|c| !c.is_empty()).map(|c| format!(" · Folder: {c}")).unwrap_or_default();
    let mut lines = vec![format!("Source: {} · Thread: {}{folder}", meta.engine, meta.title)];
    let cal = calendar_of(&win.iter().map(|e| e.ts).collect::<Vec<_>>());
    if !cal.is_empty() {
        lines.push(format!("Calendar (the days the requests were made; weeks start on Monday):\n{cal}"));
    }
    if let Some(k) = known.filter(|k| !k.is_empty()) {
        lines.push(format!(
            "KNOWN FROM OTHER CONVERSATIONS (only to name what this part refers to; never a fact of its own):\n{}",
            k.iter().map(|x| format!("- {x}")).collect::<Vec<_>>().join("\n")
        ));
    }
    lines.join("\n\n")
}

/// How a window is read.
#[derive(Clone, Default)]
pub struct ExtractOptions<'a> {
    /// The model.
    pub ask: Option<&'a dyn Ask>,
    /// `false`: observations are read apart (observe_window).
    pub observations: bool,
    /// Passages of other conversations related to this part: what a description refers to is named.
    pub known: Option<Vec<String>>,
}

async fn ask_extraction(ask: &dyn Ask, prompt: &str, with_observations: bool) -> Read<Value> {
    let system = if with_observations { EXTRACT_SYSTEM.as_str() } else { EXTRACT_SYSTEM_WITHOUT_OBSERVATIONS.as_str() };
    let req = memory_request(
        system,
        prompt,
        extract_schema(with_observations),
        "memory_consolidation",
        EXTRACT_MAX_TOKENS,
        300_000,
        CONSEQUENCE,
    );
    match ask_or_pause(ask, req).await {
        LlmAnswer::Answer { answer, .. } => Read::Read(answer),
        LlmAnswer::Unavailable(why) => Read::Unavailable(why),
        LlmAnswer::Busy => Read::Busy,
    }
}

fn values(v: &Value) -> Vec<Value> {
    v.as_array().cloned().unwrap_or_default()
}

/// One request too long for a window, read in parts IN ORDER, each folded into what the parts before it showed — the
/// running summary of the request is handed to the next part, which returns it updated. Every part's items are kept;
/// the episode is the last part's, with every part's cues and concepts. A part not answered leaves the whole request
/// for the next run. The request's whole text stays as the episode's body.
async fn extract_long(e: &StmEntry, header: &str, ask: &dyn Ask, observations: bool) -> Read<Extraction> {
    let parts = parts_of(&render_for_extraction(e, 1), WINDOW_CHARS - 2000);
    let uids = vec![e.uid.clone()];
    let mut x = Extraction::default();
    let mut summaries: Vec<String> = Vec::new();
    let (mut cues, mut concepts) = (Vec::new(), Vec::new());
    let (mut title, mut summary) = (String::new(), String::new());
    let mut outcome: Option<String> = None;
    let mut keep = false;
    for (k, text) in parts.iter().enumerate() {
        let before = if summary.is_empty() {
            String::new()
        } else {
            format!("\nWhat the earlier parts showed of Request 1 — fold this part into it, keeping every fact it holds:\nTitle: {title}\nSummary: {summary}\n")
        };
        let prompt = format!(
            "{header}\nThis is part {} of {} of ONE long request (Request 1), read in order.{before}For Request 1, write the episode of the request as read so far (every part up to this one); items and observations: what THIS part adds.\n\n{text}",
            k + 1,
            parts.len()
        );
        let a = match ask_extraction(ask, &prompt, observations).await {
            Read::Read(a) => a,
            Read::Unavailable(w) => return Read::Unavailable(w),
            Read::Busy => return Read::Busy,
        };
        let episodes = values(&a["episodes"]);
        let mine = episodes.iter().find(|x| x["request"] == 1);
        let mine_outcome = outcome_in(mine);
        let (kept, dropped) = gate(&values(&a["items"]), &uids, |n| if *n == 1 { mine_outcome.clone() } else { None });
        x.candidates.extend(kept);
        x.observations.extend(clean_observations(&values(&a["observations"]), &uids));
        x.dropped += dropped;
        if let Some(s) = a["segment_summary"].as_str().filter(|s| !s.is_empty()) {
            summaries.push(redact_string(s));
        }
        let ep = episodes
            .iter()
            .find(|x| x["request"] == 1 && x["summary"].as_str().is_some_and(|s| !js::trim(s).is_empty()));
        if let Some(ep) = ep {
            let t = redact_string(js::trim(ep["title"].as_str().unwrap_or("")));
            if !t.is_empty() {
                title = t;
            }
            summary = redact_string(js::trim(ep["summary"].as_str().unwrap_or("")));
            outcome = outcome_in(Some(ep)).or(outcome);
            keep = keep || ep["keep"] == true;
            cues.extend(clean_cues(&ep["cues"]));
            concepts.extend(clean_concepts(&ep["concepts"]));
        }
    }
    x.summary = summaries.join("\n");
    if keep && !summary.is_empty() {
        x.episodes.push(KeptEpisode {
            uid: e.uid.clone(),
            title,
            summary,
            cues: clean_cues(&json!(cues)),
            concepts: clean_concepts(&json!(concepts)),
            outcome,
            origin: None,
        });
    }
    Read::Read(x)
}

/// Read one window. `Unavailable`/`Busy` come back as they are, so the worker can pause.
pub async fn extract_window(win: &[StmEntry], meta: &WindowMeta, o: &ExtractOptions<'_>) -> Read<Extraction> {
    let default = crate::llm::default_ask();
    let ask = o.ask.unwrap_or(default.as_ref());
    let header = header_of(win, meta, o.known.as_deref());
    let rendered: Vec<String> = win.iter().enumerate().map(|(i, e)| render_for_extraction(e, i + 1)).collect();
    if win.len() == 1 && js::len16(&rendered[0]) > WINDOW_CHARS {
        return extract_long(&win[0], &header, ask, o.observations).await;
    }
    let a = match ask_extraction(ask, &format!("{header}\n\n{}", rendered.join("\n\n")), o.observations).await {
        Read::Read(a) => a,
        Read::Unavailable(w) => return Read::Unavailable(w),
        Read::Busy => return Read::Busy,
    };
    let uids: Vec<String> = win.iter().map(|e| e.uid.clone()).collect();
    let mut ends: IndexMap<i64, String> = IndexMap::new();
    for x in values(&a["episodes"]) {
        if let (Some(n), Some(o)) = (x["request"].as_f64(), outcome_in(Some(&x))) {
            ends.insert(n as i64, o);
        }
    }
    let (candidates, dropped) = gate(&values(&a["items"]), &uids, |n| {
        n.as_f64().filter(|f| f.fract() == 0.0).and_then(|f| ends.get(&(f as i64)).cloned())
    });
    let mut episodes = Vec::new();
    for x in values(&a["episodes"]) {
        let Some(uid) = uid_at(&uids, &x["request"]) else { continue };
        let Some(summary) = x["summary"].as_str().filter(|s| !js::trim(s).is_empty()) else { continue };
        if x["keep"] != true {
            continue;
        }
        episodes.push(KeptEpisode {
            uid,
            title: redact_string(js::trim(x["title"].as_str().unwrap_or(""))),
            summary: redact_string(js::trim(summary)),
            cues: clean_cues(&x["cues"]),
            concepts: clean_concepts(&x["concepts"]),
            outcome: outcome_in(Some(&x)),
            origin: None,
        });
    }
    Read::Read(Extraction {
        summary: redact_string(a["segment_summary"].as_str().unwrap_or("")),
        candidates,
        episodes,
        observations: clean_observations(&values(&a["observations"]), &uids),
        dropped,
    })
}

/// Observations read apart: a small window of requests in one focused ask, with what came just before it as context —
/// what a large window's one ask for everything let slip. A request longer than the window is read in parts, in
/// order, each part with the one before as its context.
pub async fn observe_window(
    win: &[StmEntry],
    before: &[StmEntry],
    meta: &WindowMeta,
    ask: Option<&dyn Ask>,
    chars: usize,
    known: Option<&[String]>,
) -> Read<Vec<Observation>> {
    let default = crate::llm::default_ask();
    let ask = ask.unwrap_or(default.as_ref());
    let header = header_of(win, meta, known);
    let context = before
        .iter()
        .map(|e| {
            let r = render_for_extraction(e, 0);
            for (from, to) in [("## Request 0", "## Earlier Request"), ("## Record 0", "## Earlier Record")] {
                if let Some(rest) = r.strip_prefix(from) {
                    return format!("{to}{rest}");
                }
            }
            r
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let context = js::tail16(&context, chars / 2).to_string();
    let rendered =
        win.iter().enumerate().map(|(i, e)| render_for_extraction(e, i + 1)).collect::<Vec<_>>().join("\n\n");
    let parts = parts_of(&rendered, chars);
    let uids: Vec<String> = win.iter().map(|e| e.uid.clone()).collect();
    let mut out = Vec::new();
    for (k, text) in parts.iter().enumerate() {
        let earlier = if k == 0 { context.clone() } else { js::tail16(&parts[k - 1], chars / 2).to_string() };
        let just_before =
            if earlier.is_empty() { String::new() } else { format!("JUST BEFORE (context only):\n{earlier}\n\n") };
        let which = if parts.len() > 1 { format!(" (part {} of {})", k + 1, parts.len()) } else { String::new() };
        let prompt = format!("{header}\n\n{just_before}TO READ{which}:\n{text}");
        let req = memory_request(
            &OBSERVE_SYSTEM,
            &prompt,
            observe_schema(),
            "memory_observations",
            16_000,
            300_000,
            CONSEQUENCE,
        );
        match ask_or_pause(ask, req).await {
            LlmAnswer::Answer { answer, .. } => out.extend(clean_observations(&values(&answer["observations"]), &uids)),
            LlmAnswer::Unavailable(w) => return Read::Unavailable(w),
            LlmAnswer::Busy => return Read::Busy,
        }
    }
    Read::Read(out)
}
