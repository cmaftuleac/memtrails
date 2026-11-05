//! The passage search shows as `match` is one found beyond the key's start: a short record's only chunk is its whole
//! key, the two points tie, and whichever Qdrant returns first must not repeat the key as a "match".
//! (recall-match.test.ts.)

mod common;

use std::sync::atomic::Ordering;

use common::fake_qdrant::{self, Fake};
use memtrails::recall::{memory_search, RecallDeps, SearchQuery};
use memtrails::store::{MemoryItem, StmEntry};
use memtrails::writer::{write_ltm, write_stm, WriterDeps};

/// A stand-in Qdrant whose tie between a key and its chunk goes to the chunk.
async fn chunk_first() -> (Fake, WriterDeps, RecallDeps) {
    let fake = fake_qdrant::start().await;
    fake.state.chunk_first.store(true, Ordering::SeqCst);
    let client = fake_qdrant::client(&fake).await;
    let w = WriterDeps { client: client.clone(), embed: fake_qdrant::embed() };
    let d = RecallDeps { client, embed: fake_qdrant::embed(), used: None, record: None, rerank: None };
    (fake, w, d)
}

fn record(uid: &str, request: &str, reply: &str) -> StmEntry {
    serde_json::from_value(serde_json::json!({
        "uid": uid, "fingerprint": uid, "engine": "email", "session": uid, "turn": 0, "cwd": null, "project": "-p",
        "title": "Mail", "ts": 1775088000000_i64, "synthetic": false, "typed": "human", "request": request,
        "reply": reply, "said": reply, "steps": [], "files": [], "done": false,
    }))
    .unwrap()
}

fn fact(uid: &str, summary: &str, body: &str) -> MemoryItem {
    serde_json::from_value(serde_json::json!({
        "uid": uid, "kind": "fact", "scope": "project", "project": "-p", "engine": "email", "session": "s",
        "cwd": null, "where": "w", "ts": 1, "firstTs": 1, "title": summary, "summary": summary, "body": body,
        "files": [], "appliesWhen": "", "confidence": "stated", "importance": 3, "status": "active",
        "provenance": [], "seen": 1, "fingerprint": uid,
    }))
    .unwrap()
}

fn query(q: &str) -> SearchQuery {
    SearchQuery { limit: Some(5), ..SearchQuery::new(q) }
}

#[tokio::test]
async fn is_never_the_key_again() {
    let (_fake, w, d) = chunk_first().await;
    write_stm(&w, &[record("mail:1", "Router password", "It is on the sticker under the router.")]).await.unwrap();
    let hits = memory_search(&d, &query("router password")).await.unwrap();
    assert_eq!(hits[0].id, "stm:mail:1");
    assert_eq!(hits[0].summary, "Router password It is on the sticker under the router.");
    assert_eq!(hits[0].match_, None);
}

#[tokio::test]
async fn is_never_the_start_of_the_key() {
    let (_fake, w, d) = chunk_first().await;
    let summary = "The backup server moved to rack seven.\nIt was on Monday.";
    write_ltm(&w, &[fact("mem:rack", summary, "The backup server moved to rack seven.")]).await.unwrap();
    let hits = memory_search(&d, &query("backup server")).await.unwrap();
    assert_eq!(hits[0].id, "mem:rack");
    assert_eq!(hits[0].match_, None);
}

#[tokio::test]
async fn is_still_a_passage_beyond_the_start() {
    let (_fake, w, d) = chunk_first().await;
    let body = "It now sits in rack seven, beside the switch.";
    write_ltm(&w, &[fact("mem:rack", "The backup server moved.", body)]).await.unwrap();
    let hits = memory_search(&d, &query("rack seven")).await.unwrap();
    assert_eq!(hits[0].match_.as_deref(), Some(body));
}
