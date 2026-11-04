//! Importers: records from what people already have — JSON Lines, a folder of text files, a Git history, a Slack
//! export, an mbox or `.eml` mailbox. Each one streams: records come one at a time as the source is read, never the
//! whole source in memory, so `batches(from_mbox(path, opts)?, 200)` works on a mailbox of any size.

use crate::error::Result;
use crate::records::MemoryRecord;

mod files;
mod git;
mod jsonl;
mod mbox;
mod slack;
mod text;

pub use files::{from_files, front_matter, FilesExclude, FilesOptions, FilesSkip, TEXT_EXTENSIONS};
pub use git::{from_git, GitOptions, GitTime, GitUrl};
pub use jsonl::{from_jsonl, JsonlMap, JsonlOptions, JsonlSkip, JsonlStats};
pub use mbox::{decode_encoded_words, from_eml, from_mbox, MboxOptions, MboxSkip};
pub use slack::{from_slack_export, slack_text, SlackOptions, SLACK_SKIPPED_SUBTYPES};
pub use text::{decode_bytes, decode_entities, html_title, html_to_text, split_text};

/// What an importer gives: records one at a time, or why one could not be read.
pub type Records = Box<dyn Iterator<Item = Result<MemoryRecord>> + Send>;

/// Items in groups of `size` (the last may be shorter), as they come.
pub fn batches<I: IntoIterator>(items: I, size: usize) -> Batches<I::IntoIter> {
    Batches { items: items.into_iter(), size: size.max(1) }
}

/// The groups `batches` gives.
pub struct Batches<I> {
    items: I,
    size: usize,
}

impl<I: Iterator> Iterator for Batches<I> {
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch: Vec<I::Item> = self.items.by_ref().take(self.size).collect();
        (!batch.is_empty()).then_some(batch)
    }
}
