//! Gathering: a SET of memories for a question, not a top-k.
//!
//! A question that counts, lists, totals or compares needs every instance, and one ranked below the cut is a wrong
//! answer however good the rest are. Here the fast model reads them:
//!   1. a wide search (memory_search at its limit: up to 200 observations and 200 other memories);
//!   2. the fast model reads every candidate, in parts at once, and keeps each one the answer needs — each instance of
//!      what is counted, what changed or cancelled one, the dates — and names what it did NOT see that could also
//!      count (other kinds, other words);
//!   3. those searches run, and what they find that is new is read the same way.
//!
//! The kept set comes first, in rank order; the rest follows by rank. Nothing is dropped: a part the model did not
//! answer keeps all of its candidates.

use indexmap::IndexSet;
use serde_json::{json, Value};

use crate::consolidate::label_no;
use crate::error::Result;
use crate::js;
use crate::llm::{memory_request, Ask, LlmAnswer};
use crate::profiles::PROFILE_SHOWN;
use crate::prompts::{GATHER_SYSTEM, NOTES_RULE};
use crate::recall::{memory_search, RecallDeps, RecalledItem, SearchQuery, MAX_SEARCH};
use crate::redact::redact_string;
use crate::text::cut;

/// Candidates the model reads in one ask.
const PART: usize = 40;
/// With notes, all of them in one ask, so a question's notes are written once.
const NOTES_PART: usize = 80;
/// Searches it may ask for.
pub(crate) const MORE: usize = 6;

fn schema(notes: bool) -> Value {
    let mut properties = json!({
        "keep": { "type": "array", "items": { "type": "string" } },
        "searches": { "type": "array", "items": { "type": "string" } },
    });
    let mut required = vec!["keep", "searches"];
    if notes {
        properties["notes"] = json!({ "type": "array", "items": { "type": "string" } });
        required.push("notes");
    }
    json!({ "type": "object", "additionalProperties": false, "required": required, "properties": properties })
}

fn day(ts: f64) -> String {
    crate::tools::day(ts)
}

/// One candidate as the model reads it: when it happened (from when to when, for what lasted) and when it was told,
/// its text, and the passage that matched.
fn line(h: &RecalledItem, n: usize) -> String {
    let some = |s: &Option<String>| s.as_deref().filter(|x| !x.is_empty()).map(str::to_string);
    let happened =
        [some(&h.event_at), some(&h.start_at).map(|s| format!("from {s}")), some(&h.end_at).map(|s| format!("to {s}"))]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" ");
    let when = if happened.is_empty() { day(h.ts) } else { format!("{happened}; told {}", day(h.ts)) };
    let text = if h.kind == "observation" {
        h.summary.clone()
    } else {
        format!("{} — {}", h.title, cut(&h.summary, if h.kind == "profile" { PROFILE_SHOWN } else { 600 }))
    };
    let matched =
        h.match_.as_deref().filter(|m| !m.is_empty()).map(|m| format!("\n    … {}", cut(m, 600))).unwrap_or_default();
    format!("[m{n}] ({when}) {text}{matched}")
}

#[derive(Default)]
struct Picked {
    keep: IndexSet<String>,
    searches: Vec<String>,
    notes: Vec<String>,
    paused: Option<String>,
}

async fn pick(ask: &dyn Ask, question: &str, list: &[RecalledItem], notes: bool) -> Result<Picked> {
    let size = if notes { NOTES_PART } else { PART };
    let system = if notes { format!("{}{NOTES_RULE}", *GATHER_SYSTEM) } else { GATHER_SYSTEM.clone() };
    let answers = futures::future::try_join_all(list.chunks(size).map(|part| {
        let system = system.clone();
        async move {
            let lines = part.iter().enumerate().map(|(n, h)| line(h, n + 1)).collect::<Vec<_>>().join("\n");
            let prompt = format!("Question: {question}\n\nMemories:\n{lines}");
            let req = memory_request(
                &system,
                &prompt,
                schema(notes),
                "memory_gather",
                40 * part.len() as u64 + if notes { 1400 } else { 600 },
                120_000,
                "so these memories are kept unread",
            );
            let got = match ask.ask(req).await? {
                LlmAnswer::Answer { answer, .. } => answer,
                LlmAnswer::Unavailable(why) => {
                    return Ok(Picked {
                        keep: part.iter().map(|h| h.id.clone()).collect(),
                        paused: Some(why),
                        ..Default::default()
                    })
                }
                LlmAnswer::Busy => {
                    return Ok(Picked {
                        keep: part.iter().map(|h| h.id.clone()).collect(),
                        paused: Some("the fast model is busy".into()),
                        ..Default::default()
                    })
                }
            };
            let strings = |k: &str| -> Vec<String> {
                got[k].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect()
            };
            // A small model can quote the line it means ("[m3] (2023-06-03) …"), so the label is looked for, not
            // assumed to be all of it.
            let keep = strings("keep")
                .iter()
                .filter_map(|x| label_no(x, 'm').checked_sub(1).and_then(|k| part.get(k)).map(|h| h.id.clone()))
                .collect();
            let searches = strings("searches").into_iter().filter(|s| !js::trim(s).is_empty()).collect();
            let said = if notes {
                strings("notes")
                    .into_iter()
                    .filter(|s| !js::trim(s).is_empty())
                    .map(|s| redact_string(js::trim(&s)))
                    .collect()
            } else {
                vec![]
            };
            Ok::<Picked, crate::Error>(Picked { keep, searches, notes: said, paused: None })
        }
    }))
    .await?; // the parts asked at once; one that fails drops the others (the reference's allOrAbort)
    let mut out = Picked::default();
    let mut searches: IndexSet<String> = IndexSet::new();
    for a in answers {
        out.keep.extend(a.keep);
        searches.extend(a.searches.iter().map(|s| js::trim(s).to_string()));
        out.notes.extend(a.notes);
        if out.paused.is_none() {
            out.paused = a.paused;
        }
    }
    out.searches = searches.into_iter().collect();
    Ok(out)
}

/// A gathered set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gathered {
    /// What the answer needs, best first, then the rest of what was found, by rank.
    pub found: Vec<RecalledItem>,
    /// The ids kept.
    pub kept: Vec<String>,
    /// The searches the model asked for.
    pub searches: Vec<String>,
    /// With notes: what the kept memories say that answers the question, a few short lines (empty otherwise). Show
    /// them before the memories, as notes on them.
    pub notes: Vec<String>,
    /// Set when the model did not read some candidates (none configured, unreachable, busy): why. They are all kept,
    /// by rank — the result is then a wide search, not a gathered set.
    pub paused: Option<String>,
}

/// How a gathering runs.
#[derive(Clone, Copy, Default)]
pub struct GatherOptions<'a> {
    /// The model.
    pub ask: Option<&'a dyn Ask>,
    /// How many each search finds (default `MAX_SEARCH`).
    pub limit: Option<usize>,
    /// Write notes on what was kept.
    pub notes: bool,
}

/// A set for a question. `from`/`to`: the time the question asks about. What happened then is searched for besides
/// everything else and read first — not instead: a question also names dates it only counts from, and a date filter
/// alone would hide the event it counts.
pub async fn memory_gather(d: &RecallDeps, q: &SearchQuery, o: GatherOptions<'_>) -> Result<Gathered> {
    let default = crate::llm::default_ask();
    let ask = o.ask.unwrap_or(default.as_ref());
    let limit = o.limit.unwrap_or(MAX_SEARCH);
    let open = SearchQuery { from: None, to: None, ..q.clone() };
    let dated = q.from.as_deref().is_some_and(|x| !x.is_empty()) || q.to.as_deref().is_some_and(|x| !x.is_empty());
    let (in_dates, wide) = tokio::try_join!(
        async {
            if dated {
                memory_search(d, &SearchQuery { limit: Some(limit), ..q.clone() }).await
            } else {
                Ok(vec![])
            }
        },
        async {
            let wide = SearchQuery { limit: Some(limit), ..open.clone() };
            memory_search(d, &wide).await
        }
    )?;
    let in_ids: IndexSet<String> = in_dates.iter().map(|h| h.id.clone()).collect();
    let first: Vec<RecalledItem> =
        in_dates.into_iter().chain(wide.into_iter().filter(|h| !in_ids.contains(&h.id))).collect();
    let one = pick(ask, &q.query, &first, o.notes).await?;
    let searches: Vec<String> = one.searches.iter().take(MORE).cloned().collect();
    let mut have: IndexSet<String> = first.iter().map(|h| h.id.clone()).collect();
    let found_more = futures::future::try_join_all(searches.iter().map(|s| {
        let q = SearchQuery { query: s.clone(), limit: Some(30), ..open.clone() };
        async move { memory_search(d, &q).await }
    }))
    .await?;
    let mut more: Vec<RecalledItem> = Vec::new();
    for h in found_more.into_iter().flatten() {
        if have.insert(h.id.clone()) {
            more.push(h);
        }
    }
    let two = if more.is_empty() { Picked::default() } else { pick(ask, &q.query, &more, o.notes).await? };
    let all: Vec<RecalledItem> = first.into_iter().chain(more).collect();
    let (kept, rest): (Vec<RecalledItem>, Vec<RecalledItem>) =
        all.into_iter().partition(|h| one.keep.contains(&h.id) || two.keep.contains(&h.id));
    Ok(Gathered {
        kept: kept.iter().map(|h| h.id.clone()).collect(),
        found: kept.into_iter().chain(rest).collect(),
        searches,
        notes: one.notes.into_iter().chain(two.notes).collect(),
        paused: one.paused.or(two.paused),
    })
}
