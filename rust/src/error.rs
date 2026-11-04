//! What can go wrong, as one error type.

use std::path::Path;

/// A memtrails error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The file or folder.
        path: String,
        /// What the system said.
        #[source]
        source: std::io::Error,
    },
    /// JSON that could not be read or written.
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A record (or conversation) that cannot be added, and why.
    #[error("{0}")]
    InvalidRecord(String),
    /// An argument that cannot be used, and why (a date that is not one, say).
    #[error("{0}")]
    InvalidInput(String),
    /// A log that cannot be rewritten without losing a record.
    #[error("{0}")]
    Store(String),
    /// The index answered with an error (`status` 0 when it could not be reached).
    #[error("{message}")]
    Qdrant {
        /// What happened.
        message: String,
        /// The HTTP status.
        status: u16,
    },
    /// Embedding failed.
    #[error("{0}")]
    Embedding(String),
    /// There is no way to embed: neither a model directory nor a server was given.
    #[error("{0}")]
    EmbeddingUnavailable(String),
    /// A download failed, or what came was not what is pinned.
    #[error("{0}")]
    Download(String),
    /// The Qdrant process could not be started or proven ours.
    #[error("{0}")]
    Process(String),
    /// The local model could not be loaded or run.
    #[error("{0}")]
    Model(String),
    /// The memory's home is held by another process.
    #[error("{0}")]
    Locked(String),
    /// The memory was closed.
    #[error("memory is closed")]
    Closed,
    /// An importer could not read its source.
    #[error("{0}")]
    Import(String),
    /// Anything else, said in words.
    #[error("{0}")]
    Other(String),
}

impl Error {
    pub(crate) fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Error::Io { path: path.as_ref().display().to_string(), source }
    }

    pub(crate) fn json(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

/// A memtrails result.
pub type Result<T, E = Error> = std::result::Result<T, E>;
