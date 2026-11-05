#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub mod consolidate;
pub mod download;
pub mod embedder;
pub mod episodes;
pub mod error;
pub mod extract;
pub mod gather;
pub mod graph;
pub mod importers;
pub mod js;
pub mod llm;
pub mod local_embedder;
pub mod mcp;
pub mod memory;
pub mod model;
pub mod profiles;
mod prompts;
pub mod qdrant_binary;
pub mod qdrant_client;
pub mod qdrant_process;
pub mod rebuild;
pub mod recall;
pub mod recall_log;
pub mod recall_types;
pub mod records;
pub mod redact;
pub mod reranker;
pub mod room;
pub mod runtime;
pub mod sleep;
pub mod store;
mod sys;
pub mod text;
pub mod tools;
pub mod words;
pub mod writer;

pub use consolidate::ConsolidationReport;
pub use error::{Error, Result};
pub use gather::Gathered;
pub use llm::{default_ask, openai_ask, Ask, LlmAnswer, LlmRequest, LlmUsage, OpenAiAskOptions, ReasoningEffort};
pub use memory::{
    Added, EmbeddingOptions, Llm, Memory, MemoryOptions, QdrantOption, RecordId, Rerank, SearchOptions, Stats,
};
pub use recall::{ExploreQuery, Explored, RecalledItem, RelatedItem, SearchQuery};
pub use records::{At, Conversation, MemoryRecord, Role, Turn, DEFAULT_COLLECTION};
pub use sleep::SleepReport;
pub use tools::{AgentTool, Order, ToolResult};

/// A boxed future, as the pluggable functions (embedders, rerankers, language models) return.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Each document's relevance to the query, or `None` (the search keeps its own order): `(query, documents,
/// timeout_ms)`.
pub type RerankFn = Arc<dyn Fn(String, Vec<String>, u64) -> BoxFuture<'static, Option<Vec<f64>>> + Send + Sync>;

/// Where progress lines go.
pub type LogFn = Arc<dyn Fn(&str) + Send + Sync>;

#[cfg(test)]
mod tests {
    /// A memory, its options and its errors cross tasks: an async app holds them anywhere.
    #[test]
    fn memory_options_and_errors_are_send_and_sync() {
        fn both<T: Send + Sync>() {}
        both::<crate::Memory>();
        both::<crate::MemoryOptions>();
        both::<crate::Error>();
        both::<crate::AgentTool>();
    }
}
