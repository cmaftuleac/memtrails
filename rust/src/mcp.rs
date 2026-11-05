//! A memory as an MCP server over stdio (newline-delimited JSON-RPC), for Claude Desktop, Claude Code, Cursor and any
//! other MCP client: the agent tools (memory_search, memory_open, memory_explore) and, unless read-only, memory_save
//! (mcp.ts, ported; its answers are held to `conformance/fixtures/mcp.json`).
//!
//! The memory is opened at the first tool call and closed again after `idle` without one, so the home is free in
//! between: for `memtrails add`, or another client on the same memory. A memory another process holds is a tool error
//! that says which process, not a server that will not start.
//!
//! ```no_run
//! # async fn run() -> memtrails::Result<()> {
//! use std::sync::Arc;
//! use memtrails::mcp::{serve_mcp_stdio, McpServerOptions};
//! use memtrails::{Memory, MemoryOptions};
//!
//! let open = Arc::new(|| Box::pin(Memory::open(MemoryOptions::new("./memory"))) as memtrails::BoxFuture<'static, memtrails::Result<Memory>>);
//! serve_mcp_stdio(McpServerOptions::new(open, env!("CARGO_PKG_VERSION"))).await
//! # }
//! ```

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rand::RngCore;
use serde_json::{json, Value};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinSet;

use crate::memory::{Added, Memory};
use crate::records::{At, MemoryRecord};
use crate::tools::{build_agent_tools, AgentTool, ToolProvider, ToolResult};
use crate::{BoxFuture, ExploreQuery, Explored, RecalledItem, Result, SearchQuery};

/// The protocol versions this server speaks, newest first; a client asking for another gets the newest.
pub const MCP_PROTOCOL_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "Long-term memory: what this person or team worked on, wrote, decided and said, and what was learned from it. Search it before answering about past work, people, decisions or dates, and open a result to read it whole. Save what is worth remembering later: a decision and why, a preference, a fact about a person or a system.";

const SAVE_DESCRIPTION: &str = "Save something to memory, to be found later by memory_search: a decision and why, a preference, a fact about a person, a project or a system, what was learned. Write it to be understood on its own, later, by someone who was not here: name who and what, and the date when it matters.";

const TITLE_MAX: usize = 200;

/// What the server needs of a memory: its tools, adding a note, closing. [`Memory`] is one.
#[async_trait::async_trait]
pub trait McpMemory: Send + Sync + 'static {
    /// memory_search, memory_open, memory_explore over it.
    fn tools(&self) -> Vec<AgentTool>;
    /// Add records (memory_save).
    async fn add(&self, records: Vec<MemoryRecord>) -> Result<Added>;
    /// Release it.
    async fn close(&self) -> Result<()>;
}

#[async_trait::async_trait]
impl McpMemory for Memory {
    fn tools(&self) -> Vec<AgentTool> {
        Memory::tools(self)
    }
    async fn add(&self, records: Vec<MemoryRecord>) -> Result<Added> {
        Memory::add(self, records).await
    }
    async fn close(&self) -> Result<()> {
        Memory::close(self).await
    }
}

/// Opens the memory: at the first tool call, and again after an idle close.
pub type McpOpen<M> = Arc<dyn Fn() -> BoxFuture<'static, Result<M>> + Send + Sync>;

/// How the server runs.
pub struct McpServerOptions<M> {
    /// Opens the memory.
    pub open: McpOpen<M>,
    /// The server's version, as clients show it.
    pub version: String,
    /// Without memory_save.
    pub read_only: bool,
    /// Close the memory after this long without a tool call (default 5 minutes; zero keeps it open).
    pub idle: Duration,
    /// Where its own failures are told (default: stderr).
    pub log: Option<crate::LogFn>,
}

impl<M> McpServerOptions<M> {
    /// The defaults: memory_save offered, closed after five idle minutes.
    pub fn new(open: McpOpen<M>, version: impl Into<String>) -> Self {
        McpServerOptions { open, version: version.into(), read_only: false, idle: Duration::from_secs(300), log: None }
    }
}

fn text(t: impl Into<String>, is_error: bool) -> Value {
    let mut r = json!({ "content": [{ "type": "text", "text": t.into() }] });
    if is_error {
        r["isError"] = json!(true);
    }
    r
}

fn result_value(r: ToolResult) -> Value {
    serde_json::to_value(r).unwrap_or_else(|e| text(e.to_string(), true))
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn save_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "text": { "type": "string", "minLength": 1, "description": "What to remember, written to be understood on its own." },
            "title": { "description": "A short title (default: the start of the text).", "type": "string", "maxLength": TITLE_MAX },
        },
        "required": ["text"],
        "additionalProperties": false,
    })
}

/// memory_save's arguments as zod reads them: the text and the title, or each issue `path: message`.
fn save_args(args: &Value) -> std::result::Result<(String, Option<String>), String> {
    let args = if args.is_null() { &Value::Object(Default::default()) } else { args };
    let Some(o) = args.as_object() else {
        return Err(format!("input: Invalid input: expected object, received {}", type_name(args)));
    };
    let mut issues = Vec::new();
    let text = match o.get("text") {
        None => {
            issues.push("text: Invalid input: expected string, received undefined".to_string());
            None
        }
        Some(Value::String(s)) if s.encode_utf16().count() < 1 => {
            issues.push("text: Too small: expected string to have >=1 characters".to_string());
            None
        }
        Some(Value::String(s)) => Some(s.clone()),
        Some(v) => {
            issues.push(format!("text: Invalid input: expected string, received {}", type_name(v)));
            None
        }
    };
    let title = match o.get("title") {
        None => None,
        Some(Value::String(s)) if s.encode_utf16().count() > TITLE_MAX => {
            issues.push(format!("title: Too big: expected string to have <={TITLE_MAX} characters"));
            None
        }
        Some(Value::String(s)) => Some(s.clone()),
        Some(v) => {
            issues.push(format!("title: Invalid input: expected string, received {}", type_name(v)));
            None
        }
    };
    match (issues.is_empty(), text) {
        (true, Some(t)) => Ok((t, title)),
        _ => Err(issues.join("; ")),
    }
}

/// A provider no tool is ever run against: the tools' names, texts and schemas, listed before a memory is open.
struct Never;

#[async_trait::async_trait]
impl ToolProvider for Never {
    async fn search(&self, _q: SearchQuery) -> Result<Vec<RecalledItem>> {
        Err(crate::Error::Process("not a memory".into()))
    }
    async fn open(&self, _id: &str) -> Result<Option<Value>> {
        Err(crate::Error::Process("not a memory".into()))
    }
    async fn explore(&self, _q: ExploreQuery) -> Option<Result<Explored>> {
        None
    }
}

struct Server<M> {
    o: McpServerOptions<M>,
    memory: Mutex<Option<Arc<M>>>,
    busy: AtomicUsize,
    calls: AtomicU64,
    listed: Vec<Value>,
}

impl<M: McpMemory> Server<M> {
    fn say(&self, line: &str) {
        match &self.o.log {
            Some(log) => log(line),
            None => eprintln!("{line}"),
        }
    }

    /// The open memory, or opened now. Opened by one caller at a time; a failed open is tried again next call.
    async fn memory(&self) -> Result<Arc<M>> {
        let mut m = self.memory.lock().await;
        if let Some(m) = m.as_ref() {
            return Ok(m.clone());
        }
        let opened = Arc::new((self.o.open)().await?);
        *m = Some(opened.clone());
        Ok(opened)
    }

    async fn close_memory(&self) {
        let m = self.memory.lock().await.take();
        if let Some(m) = m {
            if let Err(e) = m.close().await {
                self.say(&format!("[memtrails] close: {e}"));
            }
        }
    }

    /// After a call: when nothing else runs, close the memory once `idle` passes with no call.
    fn idle_later(self: &Arc<Self>) {
        if self.o.idle.is_zero() || self.busy.load(Ordering::SeqCst) > 0 {
            return;
        }
        let at = self.calls.load(Ordering::SeqCst);
        let me = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(me.o.idle).await;
            if me.calls.load(Ordering::SeqCst) == at && me.busy.load(Ordering::SeqCst) == 0 {
                me.close_memory().await;
            }
        });
    }

    async fn call(self: &Arc<Self>, name: &str, args: Value) -> Value {
        let known = self.listed.iter().any(|t| t["name"] == name);
        if !known {
            return text(format!("Unknown tool: {name}"), true);
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.busy.fetch_add(1, Ordering::SeqCst);
        let answer = self.run(name, args).await;
        self.busy.fetch_sub(1, Ordering::SeqCst);
        self.idle_later();
        answer
    }

    async fn run(&self, name: &str, args: Value) -> Value {
        let m = match self.memory().await {
            Ok(m) => m,
            Err(e) => return text(e.to_string(), true),
        };
        if name == "memory_save" {
            let (body, title) = match save_args(&args) {
                Ok(x) => x,
                Err(why) => return text(format!("Invalid arguments: {why}"), true),
            };
            let stamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
            let mut nonce = [0u8; 4];
            rand::thread_rng().fill_bytes(&mut nonce);
            let id = format!("{stamp}-{}", hex::encode(nonce));
            let record = MemoryRecord {
                source: Some("mcp".into()),
                kind: Some("note".into()),
                title,
                at: Some(At::from(stamp.as_str())),
                ..MemoryRecord::new(id.clone(), body)
            };
            return match m.add(vec![record]).await {
                Ok(_) => text(format!("Saved ({}, id {id}).", &stamp[..10]), false),
                Err(e) => text(e.to_string(), true),
            };
        }
        match m.tools().into_iter().find(|t| t.name == name) {
            Some(t) => result_value((t.handler)(args).await),
            None => text(format!("Unknown tool: {name}"), true),
        }
    }

    async fn handle(self: &Arc<Self>, r: Value) -> Option<Value> {
        let empty = serde_json::Map::new();
        let o = r.as_object().unwrap_or(&empty);
        let id = o.get("id").cloned();
        let is_call = id.is_some();
        let method = o.get("method").and_then(Value::as_str);
        let (Some(method), true) = (method, o.get("jsonrpc") == Some(&json!("2.0"))) else {
            return is_call.then(|| json!({ "jsonrpc": "2.0", "id": id.clone().unwrap_or(Value::Null), "error": { "code": -32600, "message": "Invalid request" } }));
        };
        let params = o.get("params").and_then(Value::as_object).cloned().unwrap_or_default();
        let ok = |result: Value| {
            Some(json!({ "jsonrpc": "2.0", "id": id.clone().unwrap_or(Value::Null), "result": result }))
        };
        match method {
            "initialize" => {
                let asked = params.get("protocolVersion").and_then(Value::as_str);
                let version = asked.filter(|v| MCP_PROTOCOL_VERSIONS.contains(v)).unwrap_or(MCP_PROTOCOL_VERSIONS[0]);
                ok(json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "memtrails", "version": self.o.version },
                    "instructions": INSTRUCTIONS,
                }))
            }
            "ping" => is_call.then(|| json!({ "jsonrpc": "2.0", "id": id.clone().unwrap_or(Value::Null), "result": {} })),
            "tools/list" => ok(json!({ "tools": self.listed })),
            "tools/call" => {
                let Some(name) = params.get("name").and_then(Value::as_str) else {
                    return Some(json!({ "jsonrpc": "2.0", "id": id.clone().unwrap_or(Value::Null), "error": { "code": -32602, "message": "tools/call needs a tool name" } }));
                };
                let args = match params.get("arguments") {
                    None | Some(Value::Null) => json!({}),
                    Some(a) => a.clone(),
                };
                let result = self.call(name, args).await;
                ok(result)
            }
            // Notifications (initialized, cancelled, …) need no answer; an unknown request does.
            _ => is_call.then(|| json!({ "jsonrpc": "2.0", "id": id.clone().unwrap_or(Value::Null), "error": { "code": -32601, "message": format!("Method not found: {method}") } })),
        }
    }
}

/// Serves `input` until it ends, answering on `output`, then closes the memory.
pub async fn serve_mcp<M, R, W>(o: McpServerOptions<M>, input: R, mut output: W) -> Result<()>
where
    M: McpMemory,
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let mut listed: Vec<Value> = build_agent_tools(Arc::new(Never))
        .into_iter()
        .map(|t| json!({ "name": t.name, "description": t.description, "inputSchema": t.json_schema, "annotations": { "readOnlyHint": true, "openWorldHint": false } }))
        .collect();
    if !o.read_only {
        listed.push(json!({
            "name": "memory_save",
            "description": SAVE_DESCRIPTION,
            "inputSchema": save_schema(),
            "annotations": { "readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": false },
        }));
    }
    let server =
        Arc::new(Server { o, memory: Mutex::new(None), busy: AtomicUsize::new(0), calls: AtomicU64::new(0), listed });

    // One writer: answers go out whole, one a line, in the order they are finished.
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let writer = tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            if output.write_all(line.as_bytes()).await.is_err() || output.flush().await.is_err() {
                break;
            }
        }
    });
    let send = move |tx: &mpsc::UnboundedSender<String>, v: &Value| {
        let _ = tx.send(format!("{v}\n"));
    };

    let mut tasks = JoinSet::new();
    let mut lines = input.lines();
    while let Some(line) =
        lines.next_line().await.map_err(|e| crate::Error::Process(format!("reading the input: {e}")))?
    {
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                send(
                    &tx,
                    &json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": "Parse error" } }),
                );
                continue;
            }
        };
        // Answered as each finishes: a slow search does not hold up a ping.
        let (server, tx) = (server.clone(), tx.clone());
        tasks.spawn(async move {
            if let Value::Array(batch) = message {
                let mut answers = Vec::new();
                for m in batch {
                    if let Some(a) = server.handle(m).await {
                        answers.push(a);
                    }
                }
                if !answers.is_empty() {
                    send(&tx, &Value::Array(answers));
                }
            } else if let Some(a) = server.handle(message).await {
                send(&tx, &a);
            }
        });
        while tasks.try_join_next().is_some() {}
    }
    while tasks.join_next().await.is_some() {}
    drop(tx);
    let _ = writer.await;
    server.close_memory().await;
    Ok(())
}

/// Serves on stdin and stdout.
pub async fn serve_mcp_stdio<M: McpMemory>(o: McpServerOptions<M>) -> Result<()> {
    serve_mcp(o, tokio::io::BufReader::new(tokio::io::stdin()), tokio::io::stdout()).await
}
