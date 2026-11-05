//! Add a few records of different kinds and ask about them. No language model, no server.
//!
//!     cargo run --example quickstart

use std::sync::Arc;

use memtrails::{At, Conversation, Memory, MemoryOptions, MemoryRecord, Role, SearchOptions, SearchQuery, Turn};

fn record(id: &str, kind: &str, author: Option<&str>, title: &str, at: &str, text: &str) -> MemoryRecord {
    MemoryRecord {
        kind: Some(kind.into()),
        author: author.map(Into::into),
        title: Some(title.into()),
        at: Some(At::from(at)),
        ..MemoryRecord::new(id, text)
    }
}

fn turn(role: Role, text: &str) -> Turn {
    Turn { role, text: text.into(), author: None, at: None }
}

#[tokio::main]
async fn main() -> memtrails::Result<()> {
    let log = Arc::new(|line: &str| println!("{line}"));
    let memory = Memory::open(MemoryOptions { log: Some(log), ..MemoryOptions::new("./example-memory") }).await?;

    memory
        .add(vec![
            record(
                "mail-1",
                "email",
                Some("Ana"),
                "Proxy certificate expired",
                "2026-04-02",
                "The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.",
            ),
            record(
                "c9f1e2",
                "commit",
                Some("Ion"),
                "Restore certificate renewal cron",
                "2026-04-03",
                "Files: deploy/cron/renew.sh",
            ),
            record(
                "note-7",
                "note",
                None,
                "Router",
                "2026-04-04",
                "Parola routerului este în caietul albastru din sertarul de la birou.",
            ),
        ])
        .await?;
    memory
        .add_conversation(&Conversation {
            id: "chat-1".into(),
            collection: None,
            source: None,
            at: Some(At::from("2026-04-05")),
            turns: vec![
                turn(Role::User, "Where do we have lunch on Friday with Ana?"),
                turn(Role::Assistant, "At the small Italian place near the office, at one."),
            ],
        })
        .await?;

    let search = |query: &str, limit: usize| SearchOptions {
        query: SearchQuery { limit: Some(limit), ..SearchQuery::new(query) },
        ..Default::default()
    };
    println!("{}", memory.search_text(&search("why did the certificate stop renewing?", 3)).await?);
    println!("{}", memory.search_text(&search("where is the router password written down?", 1)).await?);
    println!("{}", memory.room(&SearchQuery::new("what made the renewal stop?"), Some(400)).await?);

    memory.close().await
}
