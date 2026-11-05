//! The MCP server (mcp.rs; mcp.test.ts, ported): its answers are the reference's (`conformance/fixtures/mcp.json`), the
//! tools are listed before a memory is open, the memory is opened at the first call and closed when idle or when the
//! input ends, a memory that cannot be opened is a tool error, and memory_save adds a note. Over a stand-in memory.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use memtrails::mcp::{serve_mcp, McpMemory, McpServerOptions, MCP_PROTOCOL_VERSIONS};
use memtrails::tools::{ToolContent, ToolResult};
use memtrails::{Added, AgentTool, BoxFuture, MemoryRecord};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/fixtures").join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture is JSON")
}

#[derive(Default)]
struct Log {
    opened: AtomicUsize,
    closed: AtomicUsize,
    added: Mutex<Vec<MemoryRecord>>,
}

/// A stand-in memory: its tools answer with what they were asked; what is added and closed is kept.
struct Stand(Arc<Log>);

#[async_trait::async_trait]
impl McpMemory for Stand {
    fn tools(&self) -> Vec<AgentTool> {
        ["memory_search", "memory_open", "memory_explore"]
            .into_iter()
            .map(|name| AgentTool {
                name: name.into(),
                description: String::new(),
                json_schema: json!({}),
                handler: Arc::new(move |args: Value| {
                    Box::pin(async move {
                        ToolResult {
                            content: vec![ToolContent { kind: "text".into(), text: format!("{name} {args}") }],
                            is_error: false,
                        }
                    }) as BoxFuture<'static, ToolResult>
                }),
            })
            .collect()
    }
    async fn add(&self, records: Vec<MemoryRecord>) -> memtrails::Result<Added> {
        self.0.added.lock().unwrap().extend(records);
        Ok(Added { added: 1, unchanged: 0 })
    }
    async fn close(&self) -> memtrails::Result<()> {
        self.0.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// A server over in-memory pipes, and a way to talk to it one JSON line at a time.
struct Session {
    log: Arc<Log>,
    input: Option<DuplexStream>,
    answers: Lines<BufReader<DuplexStream>>,
    done: tokio::task::JoinHandle<memtrails::Result<()>>,
    id: u64,
}

impl Session {
    fn start(version: &str, read_only: bool, idle: Duration, fail_first_open: bool) -> Session {
        let log = Arc::new(Log::default());
        let opener = log.clone();
        let open = Arc::new(move || {
            let log = opener.clone();
            Box::pin(async move {
                let n = log.opened.fetch_add(1, Ordering::SeqCst);
                if fail_first_open && n == 0 {
                    return Err(memtrails::Error::Process(
                        "memory /m is open in process 4242: close it there first".into(),
                    ));
                }
                Ok(Stand(log))
            }) as BoxFuture<'static, memtrails::Result<Stand>>
        });
        let o = McpServerOptions {
            read_only,
            idle,
            log: Some(Arc::new(|_: &str| {})),
            ..McpServerOptions::new(open, version)
        };
        let (input, server_in) = tokio::io::duplex(1 << 16);
        let (server_out, output) = tokio::io::duplex(1 << 16);
        let done = tokio::spawn(serve_mcp(o, BufReader::new(server_in), server_out));
        Session { log, input: Some(input), answers: BufReader::new(output).lines(), done, id: 0 }
    }

    fn new() -> Session {
        Session::start("9.9.9", false, Duration::from_secs(300), false)
    }

    async fn send(&mut self, line: &str) {
        let input = self.input.as_mut().expect("open");
        input.write_all(format!("{line}\n").as_bytes()).await.unwrap();
    }

    async fn next(&mut self) -> Value {
        let line = tokio::time::timeout(Duration::from_secs(5), self.answers.next_line())
            .await
            .expect("an answer")
            .unwrap()
            .expect("a line");
        serde_json::from_str(&line).unwrap()
    }

    async fn ask(&mut self, method: &str, params: Value) -> Value {
        self.id += 1;
        let line = json!({ "jsonrpc": "2.0", "id": self.id, "method": method, "params": params }).to_string();
        self.send(&line).await;
        self.next().await
    }

    async fn end(mut self) -> Arc<Log> {
        drop(self.input.take());
        tokio::time::timeout(Duration::from_secs(5), self.done).await.expect("ends").unwrap().unwrap();
        self.log
    }
}

#[tokio::test]
async fn answers_every_message_as_the_reference_does() {
    let key = fixture("mcp");
    let mut s = Session::start(key["version"].as_str().unwrap(), false, Duration::from_secs(300), false);
    for x in key["session"].as_array().unwrap() {
        s.send(x["send"].as_str().unwrap()).await;
        if !x["answer"].is_null() {
            assert_eq!(s.next().await, x["answer"], "{}", x["send"]);
        }
    }
    s.end().await;
}

#[tokio::test]
async fn initialize_in_the_clients_version_when_it_speaks_it_else_its_newest() {
    let mut s = Session::new();
    let a = s.ask("initialize", json!({ "protocolVersion": "2025-03-26" })).await;
    assert_eq!(a["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(a["result"]["serverInfo"], json!({ "name": "memtrails", "version": "9.9.9" }));
    let b = s.ask("initialize", json!({ "protocolVersion": "2030-01-01" })).await;
    assert_eq!(b["result"]["protocolVersion"], MCP_PROTOCOL_VERSIONS[0]);
    s.end().await;
}

#[tokio::test]
async fn lists_tools_without_opening_the_memory_and_read_only_has_no_save() {
    let mut s = Session::new();
    let names: Vec<Value> = s.ask("tools/list", json!({})).await["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].clone())
        .collect();
    assert_eq!(
        names,
        vec![json!("memory_search"), json!("memory_open"), json!("memory_explore"), json!("memory_save")]
    );
    assert_eq!(s.log.opened.load(Ordering::SeqCst), 0);
    s.end().await;
    let mut r = Session::start("9.9.9", true, Duration::from_secs(300), false);
    let tools = r.ask("tools/list", json!({})).await["result"]["tools"].clone();
    assert!(tools.as_array().unwrap().iter().all(|t| t["name"] != "memory_save"));
    r.end().await;
}

#[tokio::test]
async fn opens_once_at_the_first_call_and_closes_when_the_input_ends() {
    let mut s = Session::new();
    s.send(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": { "name": "memory_search", "arguments": { "query": "renewal" } } }).to_string()).await;
    s.send(&json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": { "name": "memory_open", "arguments": { "id": "k:1" } } }).to_string()).await;
    let mut texts = vec![
        s.next().await["result"]["content"][0]["text"].clone(),
        s.next().await["result"]["content"][0]["text"].clone(),
    ];
    texts.sort_by_key(|t| t.to_string());
    assert_eq!(texts, vec![json!(r#"memory_open {"id":"k:1"}"#), json!(r#"memory_search {"query":"renewal"}"#)]);
    assert_eq!(s.log.opened.load(Ordering::SeqCst), 1);
    let log = s.end().await;
    assert_eq!(log.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn closes_when_idle_and_opens_again_at_the_next_call() {
    let mut s = Session::start("9.9.9", false, Duration::from_millis(30), false);
    s.ask("tools/call", json!({ "name": "memory_search", "arguments": { "query": "a" } })).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(s.log.closed.load(Ordering::SeqCst), 1);
    s.ask("tools/call", json!({ "name": "memory_search", "arguments": { "query": "b" } })).await;
    assert_eq!(s.log.opened.load(Ordering::SeqCst), 2);
    let log = s.end().await;
    assert_eq!(log.closed.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn says_why_the_memory_cannot_be_opened_and_tries_again() {
    let mut s = Session::start("9.9.9", false, Duration::from_secs(300), true);
    let a = s.ask("tools/call", json!({ "name": "memory_search", "arguments": { "query": "a" } })).await;
    assert_eq!(a["result"]["isError"], true);
    assert!(a["result"]["content"][0]["text"].as_str().unwrap().contains("memory /m is open in process 4242"));
    let b = s.ask("tools/call", json!({ "name": "memory_search", "arguments": { "query": "a" } })).await;
    assert!(b["result"].get("isError").is_none());
    s.end().await;
}

#[tokio::test]
async fn saves_a_note() {
    let mut s = Session::new();
    let a = s
        .ask("tools/call", json!({ "name": "memory_save", "arguments": { "text": "Ana prefers the Friday deploy window.", "title": "Deploy window" } }))
        .await;
    assert!(a["result"]["content"][0]["text"].as_str().unwrap().starts_with("Saved ("));
    let log = s.end().await;
    let added = log.added.lock().unwrap();
    assert_eq!(added.len(), 1);
    let r = &added[0];
    assert_eq!(
        (r.source.as_deref(), r.kind.as_deref(), r.title.as_deref()),
        (Some("mcp"), Some("note"), Some("Deploy window"))
    );
    assert_eq!(r.text, "Ana prefers the Friday deploy window.");
    assert!(r.id.ends_with(&r.id[r.id.len() - 8..]) && r.id.contains('T') && r.id.contains("Z-"));
}
