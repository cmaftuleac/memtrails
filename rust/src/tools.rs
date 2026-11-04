//! An agent's tools over a memory: `memory_search` (find), `memory_open` (read one whole, with the records it was
//! learned from) and `memory_explore` (from a topic or items to what is related but not found by search), and the
//! text a reader is shown of a search.

use crate::js;
use crate::profiles::PROFILE_SHOWN;
use crate::recall_types::RecalledItem;
use crate::text::cut;
use crate::writer::date_span;

pub(crate) fn day(ts: f64) -> String {
    if ts != 0.0 && ts.is_finite() {
        js::iso_day(ts)
    } else {
        String::new()
    }
}

pub(crate) fn hit_line(h: &RecalledItem, n: usize) -> String {
    let kind = if h.kind == "episode" { "earlier work" } else { h.kind.as_str() };
    let author = h.record.as_ref().and_then(|r| r.author.as_deref()).filter(|a| !a.is_empty());
    format!(
        "{}. {} ({kind} · {}{} · id {})",
        n + 1,
        h.title,
        day(h.ts),
        author.map(|a| format!(" · {a}")).unwrap_or_default(),
        h.id
    )
}

/// A date as a reader reckons with it — the weekday too ("the Friday before …"): 2023-07-14 → "Fri 14 Jul 2023".
/// Not a date (2023-13, 2023-02-30): shown as written.
pub fn readable_date(d: &str) -> String {
    if date_span(d).is_none() {
        return d.to_string();
    }
    let parts: Vec<i64> = d.split('-').map(|p| p.parse().unwrap_or(0)).collect();
    let (y, m, dd) = (parts[0], parts.get(1).copied().unwrap_or(0), parts.get(2).copied().unwrap_or(0));
    if y == 0 {
        return String::new();
    }
    if m == 0 {
        return y.to_string();
    }
    let month = js::MONTHS[(m - 1) as usize];
    if dd == 0 {
        return format!("{month} {y}");
    }
    let wd = js::WEEKDAYS[js::utc_of(js::date_utc(y, m - 1, dd, 0, 0, 0, 0)).weekday as usize];
    format!("{wd} {dd} {month} {y}")
}

/// How long something lasted, as a reader reckons with it: "since Feb 2022", "Nov 2021 – May 2022".
pub fn lasted_text(start_at: Option<&str>, end_at: Option<&str>) -> String {
    let show = |d: Option<&str>| d.filter(|d| !d.is_empty()).map(readable_date).unwrap_or_default();
    let (from, to) = (show(start_at), show(end_at));
    if !from.is_empty() && !to.is_empty() {
        format!("{from} – {to}")
    } else if !from.is_empty() {
        format!("since {from}")
    } else if !to.is_empty() {
        format!("until {to}")
    } else {
        String::new()
    }
}

/// The order `search_text` lists hits in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Order {
    /// Best first.
    #[default]
    Relevance,
    /// Oldest first, by when it happened.
    Time,
}

/// When a hit happened, for ordering by time: an observation's own date, else when it was learned.
fn when(h: &RecalledItem) -> String {
    h.event_at.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| day(h.ts))
}

/// What memory_search shows: an observation as one dated line (when it happened, and when it was told when that
/// differs), a profile whole (up to `PROFILE_SHOWN`), anything else as its title line and summary — shorter when many
/// are asked for. Best first, or by time.
pub fn search_text(hits: &[RecalledItem], order: Order) -> String {
    let mut list: Vec<&RecalledItem> = hits.iter().collect();
    if order == Order::Time {
        list.sort_by_key(|h| when(h));
    }
    // Many hits: one line each — the date, what it is, its summary — so a list question sees every one clearly.
    let compact = list.len() > 20;
    let lines: Vec<String> = list
        .iter()
        .enumerate()
        .map(|(n, h)| {
            if h.kind == "profile" {
                return format!("{}\n   {}", hit_line(h, n), cut(&h.summary, PROFILE_SHOWN).replace('\n', "\n   "));
            }
            if h.kind != "observation" {
                // The passage that matched, when it was not the start: a fact said deep in a long request is what
                // was found.
                let matched = h.match_.as_deref().filter(|m| !m.is_empty());
                let shown = matched.map(|m| format!(" … {}", js::collapse_spaces(m))).unwrap_or_default();
                if !compact {
                    let n_chars = if matched.is_some() { 200 } else { 500 };
                    return format!("{}\n   {}{shown}", hit_line(h, n), js::head16(&h.summary, n_chars));
                }
                let what = if h.kind == "episode" { "earlier work" } else { h.kind.as_str() };
                let n_chars = if matched.is_some() { 150 } else { 300 };
                return format!(
                    "{}. ({} · {what}) {}{shown} (id {})",
                    n + 1,
                    readable_date(&day(h.ts)),
                    js::head16(&js::collapse_spaces(&h.summary), n_chars),
                    h.id
                );
            }
            let told = day(h.ts);
            let lasted = lasted_text(h.start_at.as_deref(), h.end_at.as_deref());
            let event = h.event_at.as_deref().filter(|e| !e.is_empty());
            let happened = [event.map(readable_date).unwrap_or_default(), lasted.clone()]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            let at = if happened.is_empty() {
                format!("told {}", readable_date(&told))
            } else if h.event_at.as_deref() != Some(told.as_str()) || !lasted.is_empty() {
                format!("{happened}; told {}", readable_date(&told))
            } else {
                happened
            };
            format!("{}. ({at}) {} (id {})", n + 1, h.summary, h.id)
        })
        .collect();
    let head = if order == Order::Time {
        "Oldest first — the date is when it happened (and when it was told, when that differs).\n"
    } else {
        ""
    };
    format!("{head}{}", lines.join("\n"))
}
