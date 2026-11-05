//! memtrails on the command line (cli.ts, ported): `mcp` serves a memory to an MCP client, `add` fills it from files,
//! mail, Slack or git, `search` asks it. The memory is MEMTRAILS_HOME, or ~/.memtrails, unless --home names another.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use memtrails::importers::{
    from_eml, from_files, from_git, from_jsonl, from_mbox, from_slack_export, FilesOptions, GitOptions, JsonlOptions,
    MboxOptions, RecordStream, SlackOptions,
};
use memtrails::mcp::{serve_mcp_stdio, McpServerOptions};
use memtrails::{BoxFuture, Memory, MemoryOptions, SearchOptions, SearchQuery};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> String {
    format!(
        "memtrails {VERSION} — local memory for agents

  memtrails mcp [--read-only] [--idle MINUTES]   serve the memory to an MCP client over stdio
  memtrails add PATH... [--git REPO] [--collection NAME]
                                                 add folders of notes and documents, .mbox/.eml mail,
                                                 .jsonl records, a Slack export folder, a git history
  memtrails search QUERY... [--limit N]          search, best first

  --home DIR   the memory (default: MEMTRAILS_HOME, or ~/.memtrails)

MCP client configuration (Claude Desktop, Cursor, …), after cargo install memtrails:
  {{ \"mcpServers\": {{ \"memory\": {{ \"command\": \"memtrails\", \"args\": [\"mcp\"] }} }} }}
Claude Code:
  claude mcp add memory -- memtrails mcp
"
    )
}

#[derive(Default)]
struct Args {
    command: String,
    rest: Vec<String>,
    home: Option<String>,
    git: Vec<String>,
    collection: Option<String>,
    limit: Option<String>,
    idle: Option<String>,
    read_only: bool,
    help: bool,
    version: bool,
}

fn parse(argv: Vec<String>) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = argv.into_iter();
    let mut positional = Vec::new();
    while let Some(arg) = it.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| inline.clone().or_else(|| it.next()).ok_or(format!("{name} needs a value"));
        match name.as_str() {
            "--home" => a.home = Some(value("--home")?),
            "--git" => a.git.push(value("--git")?),
            "--collection" => a.collection = Some(value("--collection")?),
            "--limit" => a.limit = Some(value("--limit")?),
            "--idle" => a.idle = Some(value("--idle")?),
            "--read-only" => a.read_only = true,
            "--help" | "-h" => a.help = true,
            "--version" | "-v" => a.version = true,
            _ if arg.starts_with('-') && arg != "-" => return Err(format!("unknown option {arg}")),
            _ => positional.push(arg),
        }
    }
    let mut positional = positional.into_iter();
    a.command = positional.next().unwrap_or_default();
    a.rest = positional.collect();
    Ok(a)
}

fn say(line: &str) {
    eprintln!("{line}");
}

fn home_of(a: &Args) -> PathBuf {
    let raw = a
        .home
        .clone()
        .or_else(|| std::env::var("MEMTRAILS_HOME").ok().filter(|h| !h.is_empty()))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_default())
                .join(".memtrails")
        });
    std::path::absolute(&raw).unwrap_or(raw)
}

fn number(raw: Option<&str>, name: &str, fallback: f64, min: f64) -> Result<f64, String> {
    let Some(raw) = raw else { return Ok(fallback) };
    match raw.parse::<f64>() {
        Ok(n) if n.is_finite() && n >= min => Ok(n),
        _ => Err(format!("{name} must be a number of at least {min}")),
    }
}

fn options(home: PathBuf) -> MemoryOptions {
    MemoryOptions { log: Some(Arc::new(say)), ..MemoryOptions::new(home) }
}

/// What a path holds, as records: the importer its kind calls for.
async fn importer_for(path: &Path, collection: Option<String>) -> Result<RecordStream, String> {
    let err = |e: memtrails::Error| e.to_string();
    if !path.exists() {
        return Err(format!("{}: not found", path.display()));
    }
    if path.is_dir() {
        if path.join("channels.json").exists() && path.join("users.json").exists() {
            return from_slack_export(path, SlackOptions { collection, ..Default::default() }).await.map_err(err);
        }
        let on_skip = Arc::new(|p: &str, why: &str| say(&format!("  skipped {p}: {why}")));
        return from_files(path, FilesOptions { collection, on_skip: Some(on_skip), ..Default::default() })
            .await
            .map_err(err);
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "mbox" => from_mbox(path, MboxOptions { collection, ..Default::default() }).await.map_err(err),
        "eml" => from_eml(path, MboxOptions { collection, ..Default::default() }).await.map_err(err),
        "jsonl" => {
            let on_skip = Arc::new(|_line: &str, why: &str, n: usize| say(&format!("  skipped line {n}: {why}")));
            from_jsonl(path, JsonlOptions { collection, on_skip: Some(on_skip), ..Default::default() })
                .await
                .map_err(err)
        }
        _ => Err(format!(
            "{}: give a folder of notes or documents, an .mbox or .eml file, a .jsonl file, or a Slack export folder",
            path.display()
        )),
    }
}

async fn add(a: &Args) -> Result<(), String> {
    let mut sources: Vec<(String, RecordStream)> = Vec::new();
    for p in &a.rest {
        let path = std::path::absolute(p).map_err(|e| e.to_string())?;
        sources.push((p.clone(), importer_for(&path, a.collection.clone()).await?));
    }
    for p in &a.git {
        let path = std::path::absolute(p).map_err(|e| e.to_string())?;
        let records = from_git(&path, GitOptions { collection: a.collection.clone(), ..Default::default() })
            .await
            .map_err(|e| e.to_string())?;
        sources.push((format!("{p} (git)"), records));
    }
    if sources.is_empty() {
        return Err("add what? give one or more paths, or --git REPO".into());
    }
    let home = home_of(a);
    let memory = Memory::open(options(home.clone())).await.map_err(|e| e.to_string())?;
    let mut outcome = Ok(());
    for (name, records) in sources {
        say(&format!("adding {name}"));
        let progress = |n: usize| say(&format!("  {n} records"));
        match memory.import(records, None, Some(&progress)).await {
            Ok(r) => say(&format!("  {} added, {} unchanged", r.added, r.unchanged)),
            Err(e) => {
                outcome = Err(e.to_string());
                break;
            }
        }
    }
    if outcome.is_ok() {
        say(&format!("{}: {} records", home.display(), memory.stats().records));
    }
    memory.close().await.map_err(|e| e.to_string())?;
    outcome
}

async fn search(a: &Args) -> Result<(), String> {
    let query = a.rest.join(" ").trim().to_string();
    if query.is_empty() {
        return Err("search for what? give the question as words".into());
    }
    let limit = number(a.limit.as_deref(), "--limit", 10.0, 1.0)? as usize;
    let memory = Memory::open(options(home_of(a))).await.map_err(|e| e.to_string())?;
    let q =
        SearchOptions { query: SearchQuery { limit: Some(limit), ..SearchQuery::new(query) }, ..Default::default() };
    let found = memory.search_text(&q).await.map_err(|e| e.to_string());
    memory.close().await.map_err(|e| e.to_string())?;
    println!("{}", found?);
    Ok(())
}

async fn mcp(a: &Args) -> Result<(), String> {
    let home = home_of(a);
    let idle = number(a.idle.as_deref(), "--idle", 5.0, 0.0)?;
    let open = Arc::new(move || {
        Box::pin(Memory::open(options(home.clone()))) as BoxFuture<'static, memtrails::Result<Memory>>
    });
    let o = McpServerOptions {
        read_only: a.read_only,
        idle: Duration::from_secs_f64(idle * 60.0),
        ..McpServerOptions::new(open, VERSION)
    };
    serve_mcp_stdio(o).await.map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let a = match parse(std::env::args().skip(1).collect()) {
        Ok(a) => a,
        Err(e) => {
            say(&format!("memtrails: {e}"));
            return std::process::ExitCode::FAILURE;
        }
    };
    if a.version {
        println!("{VERSION}");
        return std::process::ExitCode::SUCCESS;
    }
    if a.help || a.command.is_empty() || a.command == "help" {
        print!("{}", usage());
        return std::process::ExitCode::SUCCESS;
    }
    let done = match a.command.as_str() {
        "mcp" => mcp(&a).await,
        "add" => add(&a).await,
        "search" => search(&a).await,
        other => Err(format!("unknown command {other}; try memtrails --help")),
    };
    match done {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            say(&format!("memtrails: {e}"));
            std::process::ExitCode::FAILURE
        }
    }
}
