//! memtrails — retrieval and exploration for any text: e-mail, chat, notes, documents, news, commits,
//! conversations. A native port of the TypeScript library of the same name: the same algorithms, prompts, files and
//! index layout, so a memory written by one opens and searches in the other.
#![deny(missing_docs)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub mod episodes;
pub mod error;
pub mod importers;
pub mod js;
pub mod profiles;
pub mod qdrant_client;
pub mod recall_types;
pub mod records;
pub mod redact;
pub mod room;
pub mod store;
pub mod text;
pub mod tools;
pub mod words;
pub mod writer;

pub use error::{Error, Result};

/// A boxed future, as the pluggable functions (embedders, rerankers, language models) return.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Each document's relevance to the query, or `None` (the search keeps its own order): `(query, documents,
/// timeout_ms)`.
pub type RerankFn = Arc<dyn Fn(String, Vec<String>, u64) -> BoxFuture<'static, Option<Vec<f64>>> + Send + Sync>;
