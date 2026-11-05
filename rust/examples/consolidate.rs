//! With a language model: consolidate what was added into typed knowledge and dated events, sleep, then gather every
//! instance for a listing question. Any OpenAI-compatible endpoint works (`base_url`).
//!
//!     OPENAI_API_KEY=… cargo run --example consolidate

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memtrails::{Llm, LlmUsage, Memory, MemoryOptions, OpenAiAskOptions, ReasoningEffort, SearchOptions, SearchQuery};

#[tokio::main]
async fn main() -> memtrails::Result<()> {
    let tokens = Arc::new(AtomicU64::new(0));
    let counted = tokens.clone();
    let llm = Llm::OpenAi(OpenAiAskOptions {
        model: Some(std::env::var("MEMTRAILS_MODEL").unwrap_or_else(|_| "gpt-6-luna".into())),
        reasoning_effort: Some(ReasoningEffort::Low),
        on_usage: Some(Arc::new(move |u: LlmUsage| {
            counted.fetch_add(u.input_tokens + u.output_tokens, Ordering::Relaxed);
        })),
        ..Default::default()
    });
    let memory = Memory::open(MemoryOptions { llm, ..MemoryOptions::new("./example-memory") }).await?;

    println!("{:?}", memory.consolidate().await?);
    println!("{:?}", memory.sleep().await?);
    let set = memory.gather(&SearchQuery::new("every decision we made about the proxy"), false).await?;
    println!("{} kept of {}", set.kept.len(), set.found.len());
    let kinds = ["decision", "observation", "profile"].map(String::from).to_vec();
    let q = SearchOptions {
        query: SearchQuery { kinds: Some(kinds), limit: Some(10), ..SearchQuery::new("proxy") },
        ..Default::default()
    };
    println!("{}", memory.search_text(&q).await?);
    println!("tokens used: {}", tokens.load(Ordering::Relaxed));
    memory.close().await
}
