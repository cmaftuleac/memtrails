//! Memory as an agent's tools: an MCP server over stdio for Claude Desktop, Claude Code, Cursor and others (the same as
//! `memtrails mcp`), or JSON Schemas for any function-calling API.
//!
//!     cargo run --example agent_tools               # serve MCP on stdin/stdout
//!     cargo run --example agent_tools -- --schemas  # print the tools for function calling

use std::sync::Arc;

use memtrails::mcp::{serve_mcp_stdio, McpServerOptions};
use memtrails::{BoxFuture, Memory, MemoryOptions};
use serde_json::json;

#[tokio::main]
async fn main() -> memtrails::Result<()> {
    if std::env::args().any(|a| a == "--schemas") {
        // For function calling: name, description and JSON Schema; call `(tool.handler)(args).await` with the model's
        // arguments.
        let memory = Memory::open(MemoryOptions::new("./example-memory")).await?;
        let tools: Vec<_> = memory
            .tools()
            .into_iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "parameters": t.json_schema }))
            .collect();
        println!("{}", serde_json::to_string_pretty(&tools).expect("JSON"));
        return memory.close().await;
    }
    // The memory opens at the first tool call and closes after five idle minutes.
    let open = Arc::new(|| {
        Box::pin(Memory::open(MemoryOptions::new("./example-memory"))) as BoxFuture<'static, memtrails::Result<Memory>>
    });
    serve_mcp_stdio(McpServerOptions::new(open, env!("CARGO_PKG_VERSION"))).await
}
