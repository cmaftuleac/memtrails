//! How a record reads: to the index (its body, every word of which is searched in chunks) and to the consolidation
//! model.
//!
//! A conversation's exchange keeps the format consolidation was measured on: what was asked, the steps, what was said.
//! Any other record reads as itself: its title, then its text.

use crate::js;
use crate::store::StmEntry;

/// A record's kind says it is an exchange of a conversation.
pub(crate) fn is_conversation(engine: &str) -> bool {
    engine == "conversation"
}

/// A record's body: what its key summarises, its chunks are cut from, and opening it shows — whole, never shortened.
pub fn episode_body(e: &StmEntry) -> String {
    if !is_conversation(&e.engine) {
        return if js::trim(&e.reply).is_empty() { e.request.clone() } else { format!("{}\n{}", e.request, e.reply) };
    }
    let done = if js::trim(&e.said).is_empty() { &e.reply } else { &e.said };
    let mut parts = vec![format!("Asked: {}", e.request), format!("Done: {done}")];
    if !e.files.is_empty() {
        parts.push(format!("Files: {}", e.files.join(", ")));
    }
    parts.join("\n")
}

const DAY: f64 = 86_400_000.0;

fn wd(t: f64) -> &'static str {
    js::WEEKDAYS[js::utc_of(t).weekday as usize]
}

/// For each day a request was made, the days a time said relatively that day means — yesterday, the week before, the
/// last weekend, each of the six days before by name, this week, last month, last year (UTC as the requests' times) —
/// so the model reads "last week" or "on Friday" off a line instead of counting. "Last week" is the seven days before,
/// as people mean it, not the Monday-to-Sunday week before this one: measured on LoCoMo, the calendar week put events
/// said on a Sunday thirteen days back, a week early.
pub fn calendar_of(tss: &[f64]) -> String {
    let mut days: Vec<f64> = Vec::new();
    for t in tss.iter().filter(|t| **t > 0.0) {
        let d = (t / DAY).floor() * DAY;
        if !days.contains(&d) {
            days.push(d);
        }
    }
    days.sort_by(|a, b| a.total_cmp(b));
    let iso = js::iso_day;
    days.iter()
        .map(|&d| {
            let u = js::utc_of(d);
            let monday = d - ((u.weekday + 6) % 7) as f64 * DAY;
            let sunday = d - (if u.weekday == 0 { 7 } else { u.weekday }) as f64 * DAY;
            let last_month = js::iso_head(js::date_utc(u.year, u.month - 2, 1, 0, 0, 0, 0), 7);
            let before = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
                .iter()
                .map(|n| format!("{} {}", wd(d - n * DAY), iso(d - n * DAY)))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "- {} {}: yesterday {} {} · the days before {before} · last week (the week before) {} to {} · last weekend {} to {} · this week {} to {} · last month {last_month} · last year {}",
                wd(d), iso(d), wd(d - DAY), iso(d - DAY), iso(d - 7.0 * DAY), iso(d - DAY), iso(sunday - DAY), iso(sunday),
                iso(monday), iso(monday + 6.0 * DAY), u.year - 1
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A record as the consolidation model reads it, numbered `n` in its window.
pub fn render_for_extraction(e: &StmEntry, n: usize) -> String {
    let when = if e.ts != 0.0 && !e.ts.is_nan() { js::iso_head(e.ts, 16).replacen('T', " ", 1) } else { String::new() };
    // Whole: a record too long for one window is read in parts and its parts joined, never shortened.
    if !is_conversation(&e.engine) {
        return [format!("## Record {n} · {when} · {}", e.engine), e.request.clone(), e.reply.clone()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
    }
    let mut parts = vec![format!("## Request {n} · {when} · typed by the person"), e.request.clone()];
    if !e.said.is_empty() {
        parts.push(format!("Agent said:\n{}", e.said));
    }
    if !e.files.is_empty() {
        parts.push(format!("Files changed: {}", e.files.join(", ")));
    }
    parts.join("\n")
}
