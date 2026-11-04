//! A Git repository's history: one record a commit, read from `git log` as it runs (never the whole log at once).
//! Each commit is framed by markers carrying a random token made for the run, so no message text can be taken for
//! one.
//!
//! id: the full sha. title: the subject. text: the whole message, then `Files: …` (the paths it changed, at most
//! `max_files`). author: the author's name. at: the author date. kind `commit`; thread: the option's or the
//! repository's name; source: the option's or the repository's name; metadata { sha, email, parents }.

use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, LazyLock};

use rand::RngCore;
use regex::Regex;
use serde_json::Value;

use super::text::{crlf_to_lf, re, resolve_path, Draft};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdout, Command};
use tokio::task::JoinHandle;

use super::RecordStream;
use crate::error::{Error, Result};
use crate::js;
use crate::records::{At, MemoryRecord};

/// A time `git log --since` / `--until` reads: its own text (anything git reads), or milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub enum GitTime {
    /// As git reads it (`2025-01-01`, `2 weeks ago`).
    Text(String),
    /// Milliseconds since the epoch (given to git as an ISO time).
    Ms(f64),
}

impl GitTime {
    fn arg(&self) -> String {
        match self {
            GitTime::Text(s) => s.clone(),
            GitTime::Ms(ms) => js::iso_string(*ms),
        }
    }
}

/// Where to open a commit, from its sha.
pub type GitUrl = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// How a history is read.
#[derive(Clone, Default)]
pub struct GitOptions {
    /// Only commits after this.
    pub since: Option<GitTime>,
    /// Only commits before this.
    pub until: Option<GitTime>,
    /// The branch, tag or range to read (`main`, `v1.0..v2.0`). Default: HEAD. `rev` is the same.
    pub branch: Option<String>,
    /// The same as `branch`, and read first.
    pub rev: Option<String>,
    /// Every branch and tag.
    pub all: bool,
    /// At most this many commits.
    pub max_count: Option<usize>,
    /// Only commits that changed these paths.
    pub paths: Vec<String>,
    /// Leave out merge commits.
    pub no_merges: bool,
    /// At most this many changed paths listed in a record's text. Default 50.
    pub max_files: Option<usize>,
    /// Default: the repository's name.
    pub thread: Option<String>,
    /// Default: the repository's name.
    pub source: Option<String>,
    /// The records' collection.
    pub collection: Option<String>,
    /// Where to open a commit, from its sha.
    pub url: Option<GitUrl>,
    /// The git program. Default `git`.
    pub git: Option<String>,
}

/// A path as git writes it when it has unusual characters: `"a\tb\303\251"` back to its text.
fn unquote(path: &str) -> String {
    if !(path.len() > 1 && path.starts_with('"') && path.ends_with('"')) {
        return path.to_string();
    }
    let s: Vec<char> = path[1..path.len() - 1].chars().collect();
    let mut bytes: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let octal = s.get(i + 1..i + 4).filter(|d| d.iter().all(|c| ('0'..='7').contains(c)));
        let esc = s.get(i + 1).and_then(|c| match c {
            'a' => Some(7u8),
            'b' => Some(8),
            't' => Some(9),
            'n' => Some(10),
            'v' => Some(11),
            'f' => Some(12),
            'r' => Some(13),
            '"' => Some(34),
            '\\' => Some(92),
            _ => None,
        });
        if s[i] == '\\' && octal.is_some() {
            let digits: String = octal.unwrap_or_default().iter().collect();
            bytes.push(u32::from_str_radix(&digits, 8).unwrap_or(0) as u8);
            i += 4;
        } else if s[i] == '\\' && esc.is_some() {
            bytes.push(esc.unwrap_or(0));
            i += 2;
        } else {
            let mut buf = [0u8; 4];
            bytes.extend_from_slice(s[i].encode_utf8(&mut buf).as_bytes());
            i += 1;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

static SHA: LazyLock<Regex> = LazyLock::new(|| re("^[0-9a-f]{7,64}$"));
static PARAGRAPH: LazyLock<Regex> = LazyLock::new(|| re(&format!(r"\n[{}]*\n", js::SPACE_CLASS)));
static FOLD: LazyLock<Regex> = LazyLock::new(|| re(&format!(r"[{s}]*\n[{s}]*", s = js::SPACE_CLASS)));
static EMPTY_HISTORY: LazyLock<Regex> = LazyLock::new(|| re("does not have any commits|bad default revision 'HEAD'"));

struct Log {
    child: Option<Child>,
    stdout: ChildStdout,
    stderr: Option<JoinHandle<String>>,
    buffer: Vec<u8>,
    /// How far the buffer was searched for a marker already.
    scanned: usize,
    start: Vec<u8>,
    end: String,
    name: String,
    repo: String,
    max_files: usize,
    o: GitOptions,
    finished: bool,
}

impl Log {
    fn record(&self, chunk: &[u8]) -> Option<MemoryRecord> {
        let chunk = String::from_utf8_lossy(chunk);
        let close = chunk.rfind(&self.end);
        let head = close.map_or(&chunk[..], |c| &chunk[..c]);
        let mut fields = head.split('\u{0}');
        let sha = fields.next().unwrap_or("");
        let author = fields.next().unwrap_or("").to_string();
        let email = fields.next().unwrap_or("").to_string();
        let date = fields.next().unwrap_or("").to_string();
        let parents = fields.next().unwrap_or("").to_string();
        let message: Vec<&str> = fields.collect();
        if sha.is_empty() || !SHA.is_match(sha) {
            return None;
        }
        let body = js::trim(&crlf_to_lf(&message.join("\u{0}"))).to_string();
        let first = PARAGRAPH.split(&body).next().unwrap_or("");
        let subject = js::trim(&FOLD.replace_all(first, " ")).to_string();
        let files: Vec<String> = match close {
            Some(c) => {
                chunk[c + self.end.len()..].split('\n').filter(|l| !js::trim(l).is_empty()).map(unquote).collect()
            }
            None => vec![],
        };
        let mut listed = files.iter().take(self.max_files).cloned().collect::<Vec<_>>().join(", ");
        if files.len() > self.max_files {
            listed.push_str(&format!(", … and {} more", files.len() - self.max_files));
        }
        let files_line = if files.is_empty() { String::new() } else { format!("Files: {listed}") };
        let text = [body, files_line].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
        let title = if subject.is_empty() { sha.chars().take(12).collect() } else { subject };
        Some(
            Draft {
                id: sha.to_string(),
                title: Some(title),
                text,
                kind: Some("commit".into()),
                author: Some(author),
                at: (!date.is_empty()).then_some(At::Text(date)),
                thread: Some(self.o.thread.clone().unwrap_or_else(|| self.name.clone())),
                source: Some(self.o.source.clone().unwrap_or_else(|| self.name.clone())),
                collection: self.o.collection.clone(),
                url: self.o.url.as_ref().map(|u| u(sha)),
                metadata: vec![
                    ("sha".into(), Some(Value::String(sha.to_string()))),
                    ("email".into(), Some(Value::String(email))),
                    ("parents".into(), Some(Value::String(parents))),
                ],
            }
            .finish(),
        )
    }

    /// The next marker after the buffer's first byte; each byte looked at once, however the output comes in.
    fn find_start(&mut self) -> Option<usize> {
        let from = self.scanned.max(1);
        if self.buffer.len() < self.start.len() || from > self.buffer.len() - self.start.len() {
            self.scanned = from;
            return None;
        }
        let found = (from..=self.buffer.len() - self.start.len()).find(|&i| self.buffer[i..].starts_with(&self.start));
        self.scanned = found.unwrap_or(self.buffer.len() - self.start.len() + 1);
        found
    }

    /// The commit before the next marker, once one has arrived.
    fn next_complete(&mut self) -> Option<Option<MemoryRecord>> {
        let next = self.find_start()?;
        let skip = if self.buffer.starts_with(&self.start) { self.start.len() } else { 0 };
        let r = self.record(&self.buffer[skip..next]);
        self.buffer.drain(..next);
        self.scanned = 0;
        Some(r)
    }

    /// git's exit, after its output ended: an error unless it succeeded or the repository has no commits yet.
    async fn exit(&mut self) -> Result<()> {
        let Some(mut child) = self.child.take() else { return Ok(()) };
        let status = child.wait().await.map_err(|e| Error::Import(format!("git log failed in {}: {e}", self.repo)))?;
        let stderr = match self.stderr.take() {
            Some(h) => h.await.unwrap_or_default(),
            None => String::new(),
        };
        if !status.success() && !EMPTY_HISTORY.is_match(&stderr) {
            let code = status.code().map_or_else(|| "null".to_string(), |c| c.to_string());
            return Err(Error::Import(format!("git log failed ({code}) in {}: {}", self.repo, js::trim(&stderr))));
        }
        Ok(())
    }

    /// The next commit, as git's output comes in; at its end, git's exit.
    async fn next(&mut self) -> Option<Result<MemoryRecord>> {
        loop {
            if self.finished {
                return None;
            }
            match self.next_complete() {
                Some(Some(r)) => return Some(Ok(r)),
                Some(None) => continue,
                None => {}
            }
            let mut piece = vec![0u8; 64 * 1024];
            match self.stdout.read(&mut piece).await {
                Ok(0) => {
                    self.finished = true;
                    let last = if self.buffer.starts_with(&self.start) {
                        let rest = self.buffer.split_off(self.start.len());
                        self.record(&rest)
                    } else {
                        None
                    };
                    if let Err(e) = self.exit().await {
                        return Some(Err(e));
                    }
                    return last.map(Ok);
                }
                Ok(n) => self.buffer.extend_from_slice(&piece[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => {
                    self.finished = true;
                    return Some(Err(Error::Import(format!("git log failed in {}: {e}", self.repo))));
                }
            }
        }
    }
}

/// Records from a repository's commits, newest first (as `git log` gives them).
pub async fn from_git(repo_dir: impl AsRef<Path>, opts: GitOptions) -> Result<RecordStream> {
    let repo = resolve_path(repo_dir.as_ref()).map_err(|e| Error::io(repo_dir.as_ref(), e))?;
    let name = repo.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut raw = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut raw);
    let token = hex::encode(raw);
    let mut args: Vec<String> = [
        "-c",
        "core.quotePath=false",
        "-c",
        "log.showSignature=false",
        "log",
        "--encoding=UTF-8",
        "--no-color",
        "--no-ext-diff",
        "--name-only",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args.push(format!("--format=%x1e{token}%n%H%x00%an%x00%ae%x00%aI%x00%P%x00%B%x1f{token}"));
    if let Some(n) = opts.max_count {
        args.push(format!("--max-count={n}"));
    }
    if let Some(t) = &opts.since {
        args.push(format!("--since={}", t.arg()));
    }
    if let Some(t) = &opts.until {
        args.push(format!("--until={}", t.arg()));
    }
    if opts.no_merges {
        args.push("--no-merges".into());
    }
    if opts.all {
        args.push("--all".into());
    }
    if let Some(rev) = opts.rev.clone().or_else(|| opts.branch.clone()).filter(|r| !r.is_empty()) {
        if rev.starts_with('-') {
            return Err(Error::Import(format!("not a revision: {rev}")));
        }
        args.push(rev);
    }
    args.push("--".into());
    args.extend(opts.paths.iter().cloned());
    let program = opts.git.clone().unwrap_or_else(|| "git".into());
    // `-C`, not a working folder: on Windows a command is looked up in its working folder first, so running in the
    // repository would run a `git.exe` planted in the repository being imported.
    let mut child = Command::new(&program)
        .arg("-C")
        .arg(&repo)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // A reader that stops early stops git too: the child goes with the stream.
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Import(format!("{program} could not be run: {e}")))?;
    let stdout = child.stdout.take().ok_or_else(|| Error::Import("git gave no output".into()))?;
    let stderr = child.stderr.take().map(|mut err| {
        tokio::spawn(async move {
            // Kept to 4,000 characters; the rest is read and dropped so git never blocks on a full pipe.
            let mut buf = vec![0u8; 4096];
            let mut out = String::new();
            while let Ok(n) = err.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                if out.chars().count() < 4000 {
                    out.push_str(&String::from_utf8_lossy(&buf[..n]));
                }
            }
            out
        })
    });
    let log = Log {
        child: Some(child),
        stdout,
        stderr,
        buffer: Vec::new(),
        scanned: 0,
        start: format!("\u{1e}{token}\n").into_bytes(),
        end: format!("\u{1f}{token}"),
        name,
        repo: repo_dir.as_ref().display().to_string(),
        max_files: opts.max_files.unwrap_or(50),
        o: opts,
        finished: false,
    };
    Ok(Box::pin(futures::stream::unfold(log, |mut log| async move { log.next().await.map(|r| (r, log)) })))
}
