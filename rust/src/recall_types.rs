//! What a search returns.

use serde::{Deserialize, Serialize};

use crate::store::RecordRef;

/// One item a search found: a record you added, or something consolidation learned.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecalledItem {
    /// Pass it to `open`.
    pub id: String,
    /// Its title.
    pub title: String,
    /// Its key text.
    pub summary: String,
    /// Where it came from: the thread's title, the note's name.
    #[serde(default, rename = "where")]
    pub where_: String,
    /// Its collection key.
    #[serde(default)]
    pub project: String,
    /// The kind of the record it is or came from.
    #[serde(default)]
    pub engine: String,
    /// When it was written or learned, in ms.
    #[serde(with = "crate::store::ms")]
    pub ts: f64,
    /// For a record you added: its kind (`email`, `message`, …, `record` when none was given). For what consolidation
    /// learned: decision, procedure, pitfall, fact, preference, …, observation, profile, episode.
    pub kind: String,
    /// The record it is, for a record you added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record: Option<RecordRef>,
    /// For an observation: when it happened (YYYY-MM-DD, YYYY-MM or YYYY) — `ts` is when it was said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_at: Option<String>,
    /// For something that lasted: when it began.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_at: Option<String>,
    /// For something that lasted: when it ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_at: Option<String>,
    /// The passage of its text that matched the search, when that was not its key.
    #[serde(default, rename = "match", skip_serializing_if = "Option::is_none")]
    pub match_: Option<String>,
    /// The search's fused score.
    #[serde(default)]
    pub score: f64,
}
