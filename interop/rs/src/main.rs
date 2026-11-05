//! The Rust side of the interop test (../run.mjs runs it): write a memory, add one record to it, or read it.
//!
//!   memtrails-interop write <home> <corpus.json>
//!   memtrails-interop add   <home> <corpus.json>
//!   memtrails-interop read  <home> <queries.json>   → JSON on stdout

use std::sync::Arc;

use memtrails::{
    Conversation, ExploreQuery, Memory, MemoryOptions, MemoryRecord, Order, RecalledItem, SearchOptions, SearchQuery,
};
use serde_json::{json, Value};

type Failure = Box<dyn std::error::Error>;

fn hit(h: &RecalledItem) -> Value {
    json!({
        "id": h.id,
        "kind": h.kind,
        "title": h.title,
        "project": h.project,
        "record": h.record.as_ref().map(|r| r.id.clone()),
        "summary": h.summary,
        "match": h.match_,
        "score": h.score,
    })
}

fn strings(v: &Value) -> Option<Vec<String>> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
}

fn search_query(q: &Value) -> SearchQuery {
    SearchQuery {
        query: q["query"].as_str().unwrap_or_default().to_string(),
        collections: strings(&q["collections"]),
        kinds: strings(&q["kinds"]),
        limit: q["limit"].as_u64().map(|n| n as usize),
        from: q["from"].as_str().map(String::from),
        to: q["to"].as_str().map(String::from),
    }
}

async fn read(memory: &Memory, queries: &Value) -> Result<Value, Failure> {
    let mut results = serde_json::Map::new();
    for query in queries.as_array().ok_or("the queries are not a list")? {
        let (name, op, q) =
            (query["name"].as_str().unwrap_or_default(), query["op"].as_str().unwrap_or_default(), &query["q"]);
        let out = match op {
            "search" => json!({ "hits": memory.search(&search_query(q)).await?.iter().map(hit).collect::<Vec<_>>() }),
            "searchText" => {
                let order = if q["order"] == "time" { Order::Time } else { Order::Relevance };
                json!({ "text": memory.search_text(&SearchOptions { query: search_query(q), order }).await? })
            }
            "room" => json!({ "text": memory.room(&search_query(q), q["chars"].as_u64().map(|n| n as usize)).await? }),
            "explore" => {
                let e = memory
                    .explore(&ExploreQuery {
                        topic: q["topic"].as_str().map(String::from),
                        from: strings(&q["from"]),
                        collections: strings(&q["collections"]),
                        limit: q["limit"].as_u64().map(|n| n as usize),
                    })
                    .await?;
                let related: Vec<Value> = e
                    .related
                    .iter()
                    .map(|r| {
                        let mut v = hit(&r.item);
                        v["via"] = json!(r.via);
                        v
                    })
                    .collect();
                json!({ "found": e.found.iter().map(hit).collect::<Vec<_>>(), "related": related, "concepts": e.concepts })
            }
            "open" => {
                let found = memory.search(&search_query(q)).await?;
                match found.first() {
                    Some(first) => json!({ "id": first.id, "item": memory.open_item(&first.id).await? }),
                    None => json!({ "id": null, "item": null }),
                }
            }
            other => return Err(format!("unknown op {other}").into()),
        };
        results.insert(name.to_string(), out);
    }
    let s = memory.stats();
    Ok(
        json!({ "lang": "rs", "stats": { "records": s.records, "items": s.items, "words": s.words }, "results": results }),
    )
}

async fn run(cmd: &str, memory: &Memory, input: &Value) -> Result<(), Failure> {
    match cmd {
        "write" => {
            let records: Vec<MemoryRecord> = serde_json::from_value(input["records"].clone())?;
            memory.add(records).await?;
            let conversations: Vec<Conversation> = serde_json::from_value(input["conversations"].clone())?;
            for c in &conversations {
                memory.add_conversation(c).await?;
            }
        }
        "add" => {
            let extra: MemoryRecord = serde_json::from_value(input["extra"].clone())?;
            memory.add(vec![extra]).await?;
        }
        "read" => println!("{}", serde_json::to_string(&read(memory, input).await?)?),
        other => return Err(format!("unknown command {other}").into()),
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cmd, home, file] = args.as_slice() else {
        eprintln!("usage: memtrails-interop write|add|read <home> <file>");
        std::process::exit(2);
    };
    let input: Value =
        match std::fs::read_to_string(file).map_err(Failure::from).and_then(|s| Ok(serde_json::from_str(&s)?)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[rs] {file}: {e}");
                std::process::exit(1);
            }
        };
    let options =
        MemoryOptions { log: Some(Arc::new(|line: &str| eprintln!("[rs] {line}"))), ..MemoryOptions::new(home) };
    let memory = match Memory::open(options).await {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[rs] open: {e}");
            std::process::exit(1);
        }
    };
    let outcome = run(cmd, &memory, &input).await;
    let closed = memory.close().await;
    if let Err(e) = outcome {
        eprintln!("[rs] {cmd}: {e}");
        std::process::exit(1);
    }
    if let Err(e) = closed {
        eprintln!("[rs] close: {e}");
        std::process::exit(1);
    }
}
