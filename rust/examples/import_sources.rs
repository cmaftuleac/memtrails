//! Import a mailbox, a Slack export, a repository's history and a folder of notes, then search across all of them.
//!
//!     cargo run --example import_sources -- ./archive.mbox ./slack-export ./my-repo ./notes

use std::io::Write;

use memtrails::importers::{
    from_files, from_git, from_mbox, from_slack_export, FilesOptions, GitOptions, MboxOptions, SlackOptions,
};
use memtrails::{Memory, MemoryOptions, SearchOptions, SearchQuery};

fn progress(what: &'static str) -> impl Fn(usize) + Send + Sync {
    move |n| {
        print!("\r{what}: {n}");
        let _ = std::io::stdout().flush();
    }
}

#[tokio::main]
async fn main() -> memtrails::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).filter(|a| !a.is_empty());
    let memory = Memory::open(MemoryOptions::new("./example-memory")).await?;

    if let Some(mbox) = arg(0) {
        let records = from_mbox(mbox, MboxOptions { collection: Some("mail".into()), ..Default::default() }).await?;
        println!("\n{:?}", memory.import(records, None, Some(&progress("mail"))).await?);
    }
    if let Some(slack) = arg(1) {
        let records = from_slack_export(slack, SlackOptions::default()).await?;
        println!("\n{:?}", memory.import(records, None, Some(&progress("slack"))).await?);
    }
    if let Some(repo) = arg(2) {
        let records = from_git(repo, GitOptions { max_count: Some(5000), ..Default::default() }).await?;
        println!("\n{:?}", memory.import(records, None, Some(&progress("commits"))).await?);
    }
    if let Some(notes) = arg(3) {
        let records =
            from_files(notes, FilesOptions { collection: Some("notes".into()), ..Default::default() }).await?;
        println!("\n{:?}", memory.import(records, None, Some(&progress("notes"))).await?);
    }

    println!("{:?}", memory.stats());
    let query = std::env::var("QUERY").unwrap_or_else(|_| "what did we decide about the release?".into());
    let q = SearchOptions { query: SearchQuery { limit: Some(10), ..SearchQuery::new(query) }, ..Default::default() };
    println!("{}", memory.search_text(&q).await?);
    memory.close().await
}
