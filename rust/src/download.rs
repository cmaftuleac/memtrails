//! Fetching what memtrails runs on — the Qdrant binary and the embedding model — safely: every file into a private temp
//! folder of its own, checked as it is on disk against its pinned size and SHA-256, and one run at a time per target
//! (a lock file), so two first runs never fetch the same hundreds of MB or swap folders under each other.

use std::fs::{self, File, OpenOptions};
use std::future::Future;
use std::io::Read;

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::io::AsyncWriteExt;

use futures::StreamExt;
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::llm::http;
use crate::store::blocking;
use crate::sys;

/// Where memtrails keeps what it downloads (the Qdrant binary, the embedding model): `$MEMTRAILS_CACHE`, else
/// `$XDG_CACHE_HOME/memtrails`, else `~/.cache/memtrails`. Shared by every memory on the machine; safe to delete.
pub fn cache_dir() -> PathBuf {
    let env = |n: &str| std::env::var(n).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    if let Some(c) = env("MEMTRAILS_CACHE") {
        return PathBuf::from(c);
    }
    let base = env("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| home_dir().join(".cache"));
    base.join("memtrails")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// A download given up after this long without a byte (the answer's headers included).
pub const DOWNLOAD_STALL_MS: u64 = 60_000;

/// A file's pinned size and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    /// Bytes.
    pub size: u64,
    /// Hex.
    pub sha256: String,
}

/// How downloads run.
#[derive(Clone, Default)]
pub struct DownloadOptions {
    /// Given up after this long without a byte (default 60 s).
    pub stall_ms: Option<u64>,
    /// Where downloads live instead of `cache_dir()`.
    pub cache: Option<PathBuf>,
    /// Fetch from here instead of the pinned release address (a mirror, a test server); the file names stay.
    pub base_url: Option<String>,
    /// Progress lines.
    pub log: Option<crate::LogFn>,
}

impl std::fmt::Debug for DownloadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadOptions")
            .field("stall_ms", &self.stall_ms)
            .field("cache", &self.cache)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl DownloadOptions {
    pub(crate) fn say(&self, line: &str) {
        if let Some(l) = &self.log {
            l(line);
        }
    }

    pub(crate) fn cache_dir(&self) -> PathBuf {
        self.cache.clone().unwrap_or_else(cache_dir)
    }
}

/// The cache made (0700) or made private, and this user's alone — or refused. What is verified in it is loaded or run
/// later: a folder others could write to lets them swap a library between its check and its load, or a download
/// between its hash and its unpacking.
pub(crate) async fn private_cache(cache: &Path) -> Result<()> {
    let c = cache.to_path_buf();
    let safe = blocking(move || sys::make_dir(&c, 0o700).map(|()| sys::owned_safely(&c))).await?;
    match safe {
        Ok(true) => Ok(()),
        Ok(false) => Err(Error::Download(format!(
            "{} is not this user's alone (its owner, its permissions, or it is a link): set MEMTRAILS_CACHE to a folder of your own",
            cache.display()
        ))),
        Err(e) => Err(Error::io(cache, e)),
    }
}

fn random_hex(bytes: usize) -> String {
    let mut b = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut b);
    hex::encode(b)
}

/// `url` fetched into a fresh temp folder inside `dir`, never to a fixed name: two first runs at once wrote the same
/// `.part` and could rename each other's half-written file. The file is then checked as it is on disk, not as it
/// streamed past, against its pinned size and SHA-256. Returns the file and its folder; the caller renames the file
/// into place and removes the folder. Fails (leaving nothing) on an error, a mismatch, or `stall_ms` without a byte.
pub async fn download_verified(url: &str, dir: &Path, pin: &Pin, o: &DownloadOptions) -> Result<(PathBuf, PathBuf)> {
    tokio::fs::create_dir_all(dir).await.map_err(|e| Error::io(dir, e))?;
    let tmp = dir.join(format!(".download-{}", random_hex(6)));
    let made = tmp.clone();
    let folder = TempFolder::new(tmp.clone());
    blocking(move || sys::make_dir(&made, 0o700)).await?.map_err(|e| Error::io(&tmp, e))?;
    let file = tmp.join("file");
    let result = fetch_into(url, &file, pin, o.stall_ms.unwrap_or(DOWNLOAD_STALL_MS)).await;
    match result {
        Ok(()) => Ok((file, folder.keep())),
        Err(e) => {
            folder.remove().await;
            Err(e)
        }
    }
}

/// A temp folder removed when it is dropped — the work that made it done, failed or cancelled (a dropped future), so
/// a run given up half-way leaves no `.download-*` behind. `remove` does it off the runtime's threads on the way
/// out; `keep` hands the folder on.
pub(crate) struct TempFolder(Option<PathBuf>);

impl TempFolder {
    /// Removed when dropped, from now on.
    pub(crate) fn new(path: PathBuf) -> Self {
        TempFolder(Some(path))
    }

    /// Where it is.
    pub(crate) fn path(&self) -> &Path {
        self.0.as_deref().unwrap_or(Path::new(""))
    }

    /// The folder, no longer removed on drop.
    pub(crate) fn keep(mut self) -> PathBuf {
        self.0.take().unwrap_or_default()
    }

    /// Removed now (an error is ignored: it is a temp).
    pub(crate) async fn remove(mut self) {
        if let Some(p) = self.0.take() {
            let _ = tokio::fs::remove_dir_all(&p).await;
        }
    }
}

impl Drop for TempFolder {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = fs::remove_dir_all(p);
        }
    }
}

fn private_options() -> tokio::fs::OpenOptions {
    let mut o = tokio::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    o.mode(0o600);
    o
}

async fn fetch_into(url: &str, file: &Path, pin: &Pin, stall_ms: u64) -> Result<()> {
    let name = url.rsplit('/').next().unwrap_or(url).to_string();
    let stall = Duration::from_millis(stall_ms);
    let stalled = || Error::Download(format!("{name}: no data for {} s", (stall_ms as f64 / 1000.0).round()));
    let res = tokio::time::timeout(stall, http().get(url).send())
        .await
        .map_err(|_| stalled())?
        .map_err(|e| Error::Download(format!("{name}: {e}")))?;
    if !res.status().is_success() {
        return Err(Error::Download(format!("HTTP {} for {name}", res.status().as_u16())));
    }
    let mut out = private_options().open(file).await.map_err(|e| Error::io(file, e))?;
    let mut body = res.bytes_stream();
    let mut size = 0u64;
    loop {
        let chunk = match tokio::time::timeout(stall, body.next()).await.map_err(|_| stalled())? {
            None => break,
            Some(c) => c.map_err(|e| Error::Download(format!("{name}: {e}")))?,
        };
        size += chunk.len() as u64;
        if size > pin.size {
            return Err(Error::Download(format!("{name}: more than the expected {} bytes", pin.size)));
        }
        out.write_all(&chunk).await.map_err(|e| Error::io(file, e))?;
    }
    out.sync_all().await.map_err(|e| Error::io(file, e))?;
    drop(out);
    // Hashed as it is on disk, on a blocking thread: hundreds of MB.
    let path = file.to_path_buf();
    let (got_size, got_sha) = blocking(move || hash_file(&path)).await??;
    if got_size != pin.size || got_sha != pin.sha256 {
        return Err(Error::Download(format!(
            "{name}: got {got_size} bytes sha256 {got_sha}, expected {} {}",
            pin.size, pin.sha256
        )));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let mut f = File::open(path).map_err(|e| Error::io(path, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut size = 0u64;
    loop {
        let n = f.read(&mut buf).map_err(|e| Error::io(path, e))?;
        if n == 0 {
            break;
        }
        size += n as u64;
        h.update(&buf[..n]);
    }
    Ok((size, hex::encode(h.finalize())))
}

/// A lock not touched for this long is a dead holder's (the holder touches it every quarter of this).
pub const LOCK_STALE_MS: u64 = 120_000;

/// How a lock is waited for.
#[derive(Debug, Clone, Copy)]
pub struct LockOptions {
    /// Untouched this long, it is a dead holder's.
    pub stale_ms: u64,
    /// How often to look again.
    pub poll_ms: u64,
}

impl Default for LockOptions {
    fn default() -> Self {
        LockOptions { stale_ms: LOCK_STALE_MS, poll_ms: 250 }
    }
}

fn lock_stale(lock: &Path, stale_ms: u64) -> bool {
    let Ok(text) = fs::read_to_string(lock) else { return false };
    let mut parts = text.split(' ');
    let (pid, host) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    if host == sys::hostname() {
        if let Ok(pid) = pid.parse::<i64>() {
            if pid != i64::from(sys::pid()) && sys::gone(pid) {
                return true;
            }
        }
    }
    fs::metadata(lock)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| SystemTime::now().duration_since(t).ok())
        .is_some_and(|age| age.as_millis() > u128::from(stale_ms))
}

/// `f` run holding `<path>.lock`, created exclusively: a second first run, in another process or this one, waits and
/// then finds the first one's result. A lock whose process is gone, or untouched for `stale_ms`, is taken over.
/// Nothing rests on the lock for correctness (every file is verified in a private temp and renamed into place).
pub async fn with_lock<T, F, Fut>(path: &Path, o: LockOptions, f: F) -> Result<T>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let lock = PathBuf::from(format!("{}.lock", path.display()));
    let mine = format!("{} {} {}", sys::pid(), sys::hostname(), random_hex(8));
    if let Some(dir) = lock.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| Error::io(dir, e))?;
    }
    let mut denied: Option<std::time::Instant> = None;
    let held = loop {
        match private_options().open(&lock).await {
            Ok(mut fd) => {
                // Held from the moment it is made: a run cancelled while writing it still takes it away.
                let held = HeldLock { lock: lock.clone(), mine: mine.clone(), beat: None };
                fd.write_all(mine.as_bytes()).await.map_err(|e| Error::io(&lock, e))?;
                break held;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => denied = None,
            Err(e)
                if sys::being_removed(&e)
                    && denied.get_or_insert_with(std::time::Instant::now).elapsed().as_millis()
                        < u128::from(sys::DELETE_PENDING_MS) =>
            {
                tokio::time::sleep(Duration::from_millis(o.poll_ms)).await;
                continue;
            }
            Err(e) => return Err(Error::io(&lock, e)),
        }
        let at = lock.clone();
        if blocking(move || lock_stale(&at, o.stale_ms)).await? {
            let _ = tokio::fs::remove_file(&lock).await;
            continue;
        }
        tokio::time::sleep(Duration::from_millis(o.poll_ms)).await;
    };
    let mut held = held;
    let every = Duration::from_millis((o.stale_ms / 4).max(10));
    let at = lock.clone();
    held.beat = Some(tokio::spawn(async move {
        loop {
            tokio::time::sleep(every).await;
            // Taken over: nothing to keep alive.
            let at = at.clone();
            let _ = blocking(move || {
                if let Ok(f) = OpenOptions::new().write(true).open(&at) {
                    let _ = f.set_modified(SystemTime::now());
                }
            })
            .await;
        }
    }));
    let result = f().await;
    drop(held);
    result
}

/// A lock `with_lock` holds. Dropped — the work done, failed or cancelled (a dropped future) — it stops the heartbeat
/// and takes the lock away when it is still this holder's: a cancelled run once left a heartbeat touching the lock
/// for the life of the process, so no other run could ever take it over.
struct HeldLock {
    lock: PathBuf,
    mine: String,
    beat: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for HeldLock {
    fn drop(&mut self) {
        if let Some(b) = self.beat.take() {
            b.abort();
        }
        // Ours, whole or cut short while it was written — not one taken over meanwhile.
        if fs::read_to_string(&self.lock).is_ok_and(|t| self.mine.starts_with(&t)) {
            let _ = fs::remove_file(&self.lock);
        }
    }
}
