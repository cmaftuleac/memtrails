//! Importers: records from what people already have — JSON Lines, a folder of text files, a Git history, a Slack
//! export, an mbox or `.eml` mailbox. Each one streams: records come one at a time as the source is read, never the
//! whole source in memory, so `batches(from_mbox(path, opts).await?, 200)` works on a mailbox of any size.
//!
//! Files are read on tokio's blocking threads and handed over a small channel; Git runs as an async child process.
//! No importer blocks an async worker thread.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use tokio::sync::{mpsc, oneshot};

use crate::error::{Error, Result};
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
pub type RecordStream = Pin<Box<dyn Stream<Item = Result<MemoryRecord>> + Send>>;

/// Records read ahead of the consumer: enough to keep the reader busy, few enough to stay small.
const READ_AHEAD: usize = 64;

/// A blocking reader as a stream: `open` and the iterator it gives run on a blocking thread, each record sent over a
/// bounded channel. An error from `open` is returned here, before any record; the reader stops when the stream is
/// dropped.
pub(crate) async fn blocking_stream<I, F>(open: F) -> Result<RecordStream>
where
    F: FnOnce() -> Result<I> + Send + 'static,
    I: Iterator<Item = Result<MemoryRecord>>,
{
    let (tx, rx) = mpsc::channel(READ_AHEAD);
    let (opened, ready) = oneshot::channel();
    tokio::task::spawn_blocking(move || {
        let records = match open() {
            Ok(it) => {
                let _ = opened.send(Ok(()));
                it
            }
            Err(e) => {
                let _ = opened.send(Err(e));
                return;
            }
        };
        for r in records {
            if tx.blocking_send(r).is_err() {
                break; // nobody reads any more
            }
        }
    });
    ready.await.map_err(|_| Error::Import("the reader stopped before it opened its source".into()))??;
    Ok(Box::pin(futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|r| (r, rx)) })))
}

/// Items in groups of `size` (the last may be shorter), as they come.
pub fn batches<T, S: Stream<Item = T> + Send>(s: S, size: usize) -> impl Stream<Item = Vec<T>> {
    s.chunks(size.max(1))
}
