//! How memory reaches the index: points built, embedded in batches and written to Qdrant — both tiers.
//!
//!   - a record (a short-term entry): ONE key point, its start as the key, and its whole text in chunks: no model,
//!     searchable the moment it is added;
//!   - a long-term item: a key point (content title + ~150-word summary, BM25 over title, summary, body and files) and
//!     its ~96-word children (small-to-big: searched, never shown — the item is what comes back), its cues, and the
//!     keys of its concepts on the key point (the concept graph's links: graph.rs).

use crate::js;
use crate::store::StmEntry;
use crate::text::{concept_key, cut, first_words};

/// A short-term entry's item id in the index (its long-term episode, if consolidation keeps it, is `ep:<uid>`).
pub fn stm_item(uid: &str) -> String {
    format!("stm:{uid}")
}

/// The raw key of a captured request: what it asked and the start of what was done — ~150 words.
pub fn raw_key(e: &StmEntry) -> String {
    first_words(&format!("{}\n{}", e.request, e.reply), 150)
}

/// A captured request's own title, its first line: the conversation's title says where it was, not what it is (one
/// long conversation's requests would all read alike, and match alike).
pub fn stm_title(e: &StmEntry) -> String {
    let first = e.request.split('\n').map(js::trim).find(|l| !l.is_empty()).unwrap_or("");
    let t = cut(first, 100);
    if t.is_empty() {
        e.title.clone()
    } else {
        t
    }
}

/// The time a date stands for, as its first and last millisecond.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DateSpan {
    /// The first millisecond.
    pub from: f64,
    /// The last millisecond.
    pub to: f64,
}

/// The time a date stands for: a day, a month or a year (`YYYY-MM-DD`, `YYYY-MM` or `YYYY`) — `None` for anything
/// else, a month or day that does not exist too (2024-02-30 is not 1 March).
pub fn date_span(d: &str) -> Option<DateSpan> {
    let b = d.as_bytes();
    let num = |a: usize, n: usize| -> Option<i64> {
        b.get(a..a + n)
            .filter(|s| s.iter().all(u8::is_ascii_digit))?
            .iter()
            .try_fold(0i64, |v, c| Some(v * 10 + i64::from(c - b'0')))
    };
    let (y, mo, dd) = match b.len() {
        4 => (num(0, 4)?, None, None),
        7 if b[4] == b'-' => (num(0, 4)?, Some(num(5, 2)? - 1), None),
        10 if b[4] == b'-' && b[7] == b'-' => (num(0, 4)?, Some(num(5, 2)? - 1), Some(num(8, 2)?)),
        _ => return None,
    };
    if let Some(m) = mo {
        if !(0..=11).contains(&m) {
            return None;
        }
        if let Some(day) = dd {
            if day < 1 || js::utc_of(js::date_utc(y, m, day, 0, 0, 0, 0)).month - 1 != m {
                return None;
            }
        }
    }
    Some(match (mo, dd) {
        (None, _) => {
            DateSpan { from: js::date_utc(y, 0, 1, 0, 0, 0, 0), to: js::date_utc(y + 1, 0, 1, 0, 0, 0, 0) - 1.0 }
        }
        (Some(m), None) => {
            DateSpan { from: js::date_utc(y, m, 1, 0, 0, 0, 0), to: js::date_utc(y, m + 1, 1, 0, 0, 0, 0) - 1.0 }
        }
        (Some(m), Some(day)) => {
            DateSpan { from: js::date_utc(y, m, day, 0, 0, 0, 0), to: js::date_utc(y, m, day + 1, 0, 0, 0, 0) - 1.0 }
        }
    })
}

/// A summary this many chunks long (~380 words) is also chunked: its key vector reads only its start.
pub const LONG_SUMMARY_CHUNKS: usize = 5;

/// The keys of an item's concepts, one per concept.
pub fn concept_keys(names: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for n in names {
        let k = concept_key(n);
        if !k.is_empty() && !out.contains(&k) {
            out.push(k);
        }
    }
    out
}
