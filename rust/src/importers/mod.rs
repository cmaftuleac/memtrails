//! Importers: records from what people already have — JSON Lines, a folder of text files, a Git history, a Slack
//! export, an mbox or `.eml` mailbox. Each one streams: records come one at a time as the source is read, never the
//! whole source in memory.

use crate::error::Result;
use crate::records::MemoryRecord;

/// What an importer gives: records one at a time, or why one could not be read.
pub type Records = Box<dyn Iterator<Item = Result<MemoryRecord>> + Send>;
