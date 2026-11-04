//! A memory's life, end to end, over a stand-in Qdrant and a stand-in embedding server on loopback (and a stand-in
//! Qdrant binary): adding and finding, kinds and dates, opening and exploring, what an open that fails leaves, one
//! process a home, the index written again when its layout is not the logs', jobs one at a time, close() stopping
//! what runs, forgetting a record whole, the recall log.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::fake_qdrant::{self, Fake};
use memtrails::embedder::EmbedRoute;
use memtrails::llm::{Ask, LlmAnswer, LlmRequest};
use memtrails::qdrant_client::INDEX_LAYOUT;
use memtrails::{
    At, Conversation, EmbeddingOptions, Error, ExploreQuery, Llm, Memory, MemoryOptions, MemoryRecord, Order,
    QdrantOption, RecordId, Rerank, Role, SearchOptions, SearchQuery, Turn,
};
use serde_json::{json, Value};

struct Rig {
    home: tempfile::TempDir,
    fake: Fake,
    embed: common::Server,
}

async fn rig() -> Rig {
    Rig {
        home: tempfile::tempdir().unwrap(),
        fake: fake_qdrant::start().await,
        embed: fake_qdrant::embed_server().await,
    }
}

impl Rig {
    fn options(&self, ask: Option<Arc<dyn Ask>>) -> MemoryOptions {
        MemoryOptions {
            embedding: EmbeddingOptions {
                server: Some(EmbedRoute { url: self.embed.url.clone(), ..Default::default() }),
                ..Default::default()
            },
            qdrant: QdrantOption::Server { url: self.fake.server.url.clone(), api_key: None, collection: None },
            rerank: Rerank::Off,
            llm: ask.map_or(Llm::Env, Llm::Custom),
            ..MemoryOptions::new(self.home.path().join("mem"))
        }
    }

    async fn open(&self) -> Memory {
        Memory::open(self.options(None)).await.unwrap()
    }
}

fn mail(id: &str, title: &str, text: &str) -> MemoryRecord {
    MemoryRecord {
        kind: Some("email".into()),
        title: Some(title.into()),
        at: Some(At::from("2026-04-02T09:00:00Z")),
        ..MemoryRecord::new(id, text)
    }
}

struct Counting {
    now: AtomicUsize,
    most: AtomicUsize,
}

#[async_trait::async_trait]
impl Ask for Counting {
    async fn ask(&self, _req: LlmRequest) -> memtrails::Result<LlmAnswer> {
        let n = self.now.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(n, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(50)).await;
        self.now.fetch_sub(1, Ordering::SeqCst);
        Ok(LlmAnswer::Answer {
            answer: json!({ "segment_summary": "x", "episodes": [], "items": [], "observations": [] }),
            model: None,
        })
    }
}

#[tokio::test]
async fn adds_finds_opens_and_forgets() {
    let r = rig().await;
    let m = r.open().await;
    let added = m
        .add(vec![
            mail(
                "msg-1",
                "Proxy certificate expired",
                "The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.",
            ),
            MemoryRecord {
                kind: Some("commit".into()),
                author: Some("Ion".into()),
                ..MemoryRecord::new("c9f1e2", "Restore certificate renewal cron\nFiles: deploy/cron/renew.sh")
            },
        ])
        .await
        .unwrap();
    assert_eq!((added.added, added.unchanged), (2, 0));
    let again = m
        .add(vec![mail(
            "msg-1",
            "Proxy certificate expired",
            "The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.",
        )])
        .await
        .unwrap();
    assert_eq!((again.added, again.unchanged), (0, 1));
    let hits = m.search(&SearchQuery::new("why did the certificate renewal stop")).await.unwrap();
    assert_eq!(hits[0].record.as_ref().unwrap().id, "msg-1");
    assert_eq!(hits[0].kind, "email");
    let text = m
        .search_text(&SearchOptions { query: SearchQuery::new("certificate renewal cron"), order: Order::Time })
        .await
        .unwrap();
    assert!(text.starts_with("Oldest first"), "{text}");
    let opened = m.open_item(&hits[0].id).await.unwrap().unwrap();
    assert_eq!(opened["kind"], "email");
    assert!(m.open_item("stm:nothing").await.unwrap().is_none());
    assert_eq!(m.stats().records, 2);
    assert_eq!(m.remove(vec![RecordId::from("msg-1"), RecordId::from("nope")]).await.unwrap(), 1);
    assert!(m
        .search(&SearchQuery::new("certificate"))
        .await
        .unwrap()
        .iter()
        .all(|h| h.record.as_ref().is_none_or(|x| x.id != "msg-1")));
    assert!(matches!(m.add(vec![MemoryRecord::new("", "x")]).await, Err(Error::InvalidRecord(_))));
    m.close().await.unwrap();
    assert!(matches!(m.add(vec![MemoryRecord::new("a", "x")]).await, Err(Error::Closed)));
}

#[tokio::test]
async fn a_conversation_is_one_record_each_time_the_person_speaks() {
    let r = rig().await;
    let m = r.open().await;
    let turn = |role, text: &str| Turn { role, text: text.into(), author: None, at: None };
    let c = Conversation {
        id: "chat-42".into(),
        collection: Some("personal".into()),
        source: None,
        at: Some(At::from("2026-03-01T09:00:00Z")),
        turns: vec![
            turn(Role::User, "Where do we have lunch on Friday?"),
            turn(Role::Assistant, "At the Italian place near the office, at one."),
        ],
    };
    assert_eq!(m.add_conversation(&c).await.unwrap().added, 1);
    let hits = m
        .search(&SearchQuery { collections: Some(vec!["personal".into()]), ..SearchQuery::new("lunch on Friday") })
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(m
        .search(&SearchQuery { collections: Some(vec!["work".into()]), ..SearchQuery::new("lunch on Friday") })
        .await
        .unwrap()
        .is_empty());
    m.close().await.unwrap();
}

#[tokio::test]
async fn finds_a_record_of_a_kind_by_a_chunk_of_its_text_and_refuses_a_date_that_is_not_one() {
    let r = rig().await;
    let m = r.open().await;
    let filler = (0..300).map(|i| format!("agenda{} minutes{}", i % 41, i % 29)).collect::<Vec<_>>().join(" ");
    let said = format!("{filler} the backup server moved to rack seven {filler}");
    m.add(vec![
        MemoryRecord {
            kind: Some("email".into()),
            title: Some("Weekly notes".into()),
            ..MemoryRecord::new("mail-1", said.clone())
        },
        MemoryRecord {
            kind: Some("message".into()),
            title: Some("Weekly notes".into()),
            ..MemoryRecord::new("chat-1", said)
        },
    ])
    .await
    .unwrap();
    let hits = m
        .search(&SearchQuery { kinds: Some(vec!["email".into()]), ..SearchQuery::new("backup server rack seven") })
        .await
        .unwrap();
    assert_eq!(hits.iter().map(|h| h.record.as_ref().unwrap().id.as_str()).collect::<Vec<_>>(), ["mail-1"]);
    assert!(hits[0].match_.as_deref().unwrap_or("").contains("backup server moved to rack seven"));
    let all_points = r.fake.points();
    let mail: Vec<_> = all_points
        .iter()
        .filter(|p| p.payload["item"].as_str().unwrap_or("").starts_with("stm:") && p.payload["engine"] == "email")
        .collect();
    assert!(mail.len() > 1);
    for (q, says) in [
        (SearchQuery { from: Some("April 2026".into()), ..SearchQuery::new("x") }, "from \"April 2026\" is not a date"),
        (SearchQuery { to: Some("2026-13".into()), ..SearchQuery::new("x") }, "to \"2026-13\""),
        (SearchQuery { from: Some("2026-02-30".into()), ..SearchQuery::new("x") }, "is not a date"),
    ] {
        assert!(m.search(&q).await.unwrap_err().to_string().contains(says));
    }
    assert!(m
        .search(&SearchQuery {
            from: Some("2026-04".into()),
            to: Some("2026-04-30".into()),
            ..SearchQuery::new("backup")
        })
        .await
        .is_ok());
    m.close().await.unwrap();
}

#[tokio::test]
async fn records_what_a_search_showed_and_what_was_opened_a_line_each_in_order() {
    let r = rig().await;
    let m = r.open().await;
    m.add(vec![mail("m-1", "Boiler service", "The plumber serviced the boiler.")]).await.unwrap();
    let hits = m
        .search(&SearchQuery {
            collections: Some(vec!["default".into()]),
            ..SearchQuery::new("who serviced the boiler")
        })
        .await
        .unwrap();
    m.open_item(&hits[0].id).await.unwrap();
    m.close().await.unwrap();
    let lines: Vec<Value> = std::fs::read_to_string(r.home.path().join("mem/recall.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        (lines[0]["ev"].as_str(), lines[0]["project"].as_str(), lines[0]["query"].as_str()),
        (Some("search"), Some("default"), Some("who serviced the boiler"))
    );
    assert_eq!(lines[0]["shown"][0], json!(hits[0].id));
    assert_eq!((lines[1]["ev"].as_str(), lines[1]["item"].as_str()), (Some("open"), Some(hits[0].id.as_str())));
}

#[tokio::test]
async fn one_process_a_home_and_the_home_is_private() {
    let r = rig().await;
    let m = r.open().await;
    let err = Memory::open(r.options(None)).await.err().unwrap().to_string();
    assert!(err.contains("is open in process"), "{err}");
    m.close().await.unwrap();
    let m = r.open().await;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(r.home.path().join("mem")).unwrap().permissions().mode() & 0o777, 0o700);
    }
    m.close().await.unwrap();
    // A lock whose process is gone, or that was written before a reboot, is taken over.
    for stale in [
        json!({ "pid": 999_999_999, "boot": memtrails::js::now_ms() }),
        json!({ "pid": std::process::id(), "boot": 1000 }),
    ] {
        std::fs::write(r.home.path().join("mem/memory.lock"), stale.to_string()).unwrap();
        r.open().await.close().await.unwrap();
    }
    assert!(!r.home.path().join("mem/memory.lock").exists());
}

#[tokio::test]
async fn an_open_that_fails_leaves_the_home_free() {
    let r = rig().await;
    let mut o = r.options(None);
    o.qdrant = QdrantOption::Binary("memtrails-no-such-qdrant".into());
    assert!(Memory::open(o).await.is_err());
    r.open().await.close().await.unwrap();
}

#[tokio::test]
async fn writes_the_index_again_when_it_was_made_in_another_layout_or_one_not_written_down() {
    let r = rig().await;
    let m = r.open().await;
    m.add(vec![mail("m-1", "Kept", "Something kept in the logs.")]).await.unwrap();
    m.close().await.unwrap();
    for layout in [Some(INDEX_LAYOUT - 1), None] {
        let path = r.home.path().join("mem/index.json");
        match layout {
            Some(l) => std::fs::write(&path, json!({ "layout": l }).to_string()).unwrap(),
            None => std::fs::remove_file(&path).unwrap(),
        }
        let drops = r.fake.state.drops.load(Ordering::SeqCst);
        let m = r.open().await;
        assert_eq!(r.fake.state.drops.load(Ordering::SeqCst), drops + 1);
        assert!(!m.search(&SearchQuery::new("something kept")).await.unwrap().is_empty());
        let saved: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(saved["layout"], INDEX_LAYOUT);
        m.close().await.unwrap();
    }
}

#[tokio::test]
async fn forgets_a_record_whole_its_kept_episode_too_so_a_rebuild_does_not_bring_it_back() {
    let r = rig().await;
    let ask: Arc<dyn Ask> = Arc::new(Counting { now: AtomicUsize::new(0), most: AtomicUsize::new(0) });
    let m = Memory::open(r.options(Some(ask))).await.unwrap();
    m.add(vec![mail("m-1", "Secret plan", "The plan to forget.")]).await.unwrap();
    let uid = memtrails::records::record_uid("m-1", None);
    let episode: memtrails::store::MemoryItem = serde_json::from_value(json!({
        "uid": format!("ep:{uid}"), "kind": "episode", "scope": "project", "project": "default", "engine": "email", "session": uid, "cwd": null,
        "where": "Secret plan", "ts": 1, "firstTs": 1, "title": "Secret plan", "summary": "The plan to forget.", "body": "The plan to forget.",
        "files": [], "appliesWhen": "", "confidence": "observed", "importance": 2, "status": "active", "provenance": [uid], "seen": 1, "fingerprint": "f",
    }))
    .unwrap();
    m.knowledge().put(vec![episode]).await.unwrap();
    assert_eq!(m.remove(vec![RecordId::from("m-1")]).await.unwrap(), 1);
    assert!(m.knowledge().get(&format!("ep:{uid}")).await.is_none());
    m.rebuild().await.unwrap();
    assert!(r.fake.points().is_empty());
    m.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_consolidations_never_ask_the_model_at_once() {
    let r = rig().await;
    let counting = Arc::new(Counting { now: AtomicUsize::new(0), most: AtomicUsize::new(0) });
    let m = Memory::open(r.options(Some(counting.clone()))).await.unwrap();
    let old = MemoryRecord {
        at: Some(At::from("2026-01-01T00:00:00Z")),
        thread: Some("t".into()),
        ..MemoryRecord::new("a", "first thing")
    };
    let other = MemoryRecord {
        at: Some(At::from("2026-01-01T00:00:00Z")),
        thread: Some("u".into()),
        ..MemoryRecord::new("b", "second thing")
    };
    m.add(vec![old, other]).await.unwrap();
    let (a, b) = tokio::join!(m.consolidate(), m.consolidate());
    a.unwrap();
    b.unwrap();
    assert!(counting.most.load(Ordering::SeqCst) >= 1);
    // Each job reads its windows a few at once; never two jobs: at most the windows of one run.
    assert!(counting.most.load(Ordering::SeqCst) <= 2);
    m.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn close_stops_a_running_rebuild_at_its_next_step_and_the_next_open_finishes_it() {
    let r = rig().await;
    let m = r.open().await;
    let many: Vec<MemoryRecord> =
        (0..100).map(|i| MemoryRecord::new(format!("r{i}"), format!("record number {i} about the move"))).collect();
    m.add(many).await.unwrap();
    r.fake.state.delay.store(100, Ordering::SeqCst);
    let running = {
        let m = m.clone();
        tokio::spawn(async move { m.rebuild().await })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    m.close().await.unwrap();
    let _ = running.await.unwrap();
    assert!(r.home.path().join("mem/rebuild.json").exists());
    r.fake.state.delay.store(0, Ordering::SeqCst);
    let m = r.open().await;
    assert!(!r.home.path().join("mem/rebuild.json").exists());
    assert_eq!(r.fake.points().iter().filter(|p| p.payload["part"] == "key").count(), 100);
    m.close().await.unwrap();
}

#[tokio::test]
async fn explores_from_an_item_to_what_shares_its_concepts_or_is_linked_to_it() {
    let r = rig().await;
    let m = r.open().await;
    let item = |uid: &str, title: &str, summary: &str, concepts: Value| -> memtrails::store::MemoryItem {
        serde_json::from_value(json!({
            "uid": uid, "kind": "fact", "scope": "project", "project": "default", "engine": "email", "session": "s", "cwd": null, "where": "w",
            "ts": 1, "firstTs": 1, "title": title, "summary": summary, "body": summary, "files": [], "appliesWhen": "", "confidence": "stated",
            "importance": 3, "status": "active", "provenance": [], "seen": 1, "fingerprint": uid, "concepts": concepts,
        }))
        .unwrap()
    };
    let mut items = vec![
        item(
            "mem:image",
            "proxy config baked into the image",
            "the proxy config is baked into the image rebuild it after edits",
            json!(["proxy.conf", "acme-proxy:dev"]),
        ),
        item(
            "mem:reload",
            "a reload ignores host edits",
            "reloading inside the container keeps the old file",
            json!(["Proxy.conf", "caddy"]),
        ),
        item(
            "mem:recreate",
            "recreate after settings change",
            "a restart keeps the old settings, recreate it",
            json!(["docker compose"]),
        ),
        item("mem:invoice", "invoice pdf", "invoices render with the pdf service", json!(["pdf service"])),
        item("mem:alias", "proxy file location", "it lives in the project folder", json!(["proxy.conf file"])),
    ];
    items[0].related = Some(vec!["mem:recreate".into()]);
    items[2].related = Some(vec!["mem:image".into()]);
    let w = memtrails::writer::WriterDeps {
        client: memtrails::qdrant_client::QdrantClient::new(
            memtrails::qdrant_client::Endpoint::Fixed(memtrails::qdrant_process::QdrantEndpoint {
                url: r.fake.server.url.clone(),
                api_key: String::new(),
            }),
            None,
        ),
        embed: fake_qdrant::embed(),
    };
    memtrails::writer::write_ltm(&w, &items).await.unwrap();
    m.knowledge().put(items).await.unwrap();
    let e = m.explore(&ExploreQuery { from: Some(vec!["mem:image".into()]), ..Default::default() }).await.unwrap();
    let related: Vec<&str> = e.related.iter().map(|x| x.item.id.as_str()).collect();
    for want in ["mem:reload", "mem:recreate"] {
        assert!(related.contains(&want), "{related:?}");
    }
    for not in ["mem:image", "mem:invoice", "mem:alias"] {
        assert!(!related.contains(&not), "{related:?}");
    }
    let via = |id: &str| e.related.iter().find(|x| x.item.id == id).unwrap().via.clone();
    assert_eq!(via("mem:reload"), "shares \"proxy.conf\"");
    assert_eq!(via("mem:recreate"), "linked to \"proxy config baked into the image\"");
    assert!(e.concepts.contains(&"proxy.conf".to_string()));
    let tools = m.tools();
    let explore = tools.iter().find(|t| t.name == "memory_explore").unwrap();
    let out = (explore.handler)(json!({ "from": ["mem:image"] })).await;
    assert!(out.content[0].text.contains("Related — not found by search"), "{}", out.content[0].text);
    m.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn stops_the_qdrant_it_started_and_frees_the_home() {
    if std::process::Command::new("node").arg("-v").output().is_err() {
        return;
    }
    let r = rig().await;
    let bin = r.home.path().join("fake-qdrant.cjs");
    std::fs::write(
        &bin,
        r#"#!/usr/bin/env node
const http = require('node:http'); const fs = require('node:fs');
const args = process.argv.slice(2);
const port = Number(/http_port: (\d+)/.exec(fs.readFileSync(args[args.indexOf('--config-path') + 1], 'utf8'))[1]);
fs.writeFileSync('pid', String(process.pid));
const key = process.env.QDRANT__SERVICE__API_KEY;
http.createServer((req, res) => {
  if (req.url !== '/readyz' && req.headers['api-key'] !== key) { res.writeHead(401); return res.end('{}'); }
  res.writeHead(200); res.end(req.url.endsWith('/count') ? '{"result":{"count":0}}' : '{"result":{"collections":[]}}');
}).listen(port, '127.0.0.1');
process.on('SIGTERM', () => process.exit(0));
"#,
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut o = r.options(None);
    o.qdrant = QdrantOption::Binary(bin);
    let m = Memory::open(o).await.unwrap();
    let pid: i32 = std::fs::read_to_string(r.home.path().join("mem/qdrant/pid")).unwrap().trim().parse().unwrap();
    m.close().await.unwrap();
    // SAFETY: signal 0 only checks.
    assert_ne!(unsafe { libc::kill(pid, 0) }, 0);
    assert!(!r.home.path().join("mem/memory.lock").exists());
}
