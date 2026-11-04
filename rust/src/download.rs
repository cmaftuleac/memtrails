//! Fetching what memtrails runs on — the Qdrant binary and the embedding model — safely: every file into a private temp
//! folder of its own, checked as it is on disk against its pinned size and SHA-256, and one run at a time per target
//! (a lock file), so two first runs never fetch the same hundreds of MB or swap folders under each other.

use std::fs::{self, File, OpenOptions};
use std::future::Future;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use futures::StreamExt;
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::llm::http;
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

fn random_hex(bytes: usize) -> String {
    let mut b = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut b);
    hex::encode(b)
}

fn private_file(path: &Path) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
}

/// `url` fetched into a fresh temp folder inside `dir`, never to a fixed name: two first runs at once wrote the same
/// `.part` and could rename each other's half-written file. The file is then checked as it is on disk, not as it
/// streamed past, against its pinned size and SHA-256. Returns the file and its folder; the caller renames the file
/// into place and removes the folder. Fails (leaving nothing) on an error, a mismatch, or `stall_ms` without a byte.
pub async fn download_verified(url: &str, dir: &Path, pin: &Pin, o: &DownloadOptions) -> Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let tmp = dir.join(format!(".download-{}", random_hex(6)));
    sys::make_dir(&tmp, 0o700).map_err(|e| Error::io(&tmp, e))?;
    let file = tmp.join("file");
    let result = fetch_into(url, &file, pin, o.stall_ms.unwrap_or(DOWNLOAD_STALL_MS)).await;
    match result {
        Ok(()) => Ok((file, tmp)),
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp);
            Err(e)
        }
    }
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
    let mut out = private_file(file).map_err(|e| Error::io(file, e))?;
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
        out.write_all(&chunk).map_err(|e| Error::io(file, e))?;
    }
    out.sync_all().map_err(|e| Error::io(file, e))?;
    drop(out);
    let (got_size, got_sha) = hash_file(file)?;
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
        fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    }
    loop {
        match private_file(&lock) {
            Ok(mut fd) => {
                fd.write_all(mine.as_bytes()).map_err(|e| Error::io(&lock, e))?;
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(Error::io(&lock, e)),
        }
        if lock_stale(&lock, o.stale_ms) {
            let _ = fs::remove_file(&lock);
            continue;
        }
        tokio::time::sleep(Duration::from_millis(o.poll_ms)).await;
    }
    let done = Arc::new(AtomicBool::new(false));
    let beat = {
        let done = done.clone();
        let lock = lock.clone();
        let every = Duration::from_millis((o.stale_ms / 4).max(10));
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                if done.load(Ordering::SeqCst) {
                    break;
                }
                // Taken over: nothing to keep alive.
                if let Ok(f) = OpenOptions::new().write(true).open(&lock) {
                    let _ = f.set_modified(SystemTime::now());
                }
            }
        })
    };
    let result = f().await;
    done.store(true, Ordering::SeqCst);
    beat.abort();
    // Not one taken over meanwhile.
    if fs::read_to_string(&lock).is_ok_and(|t| t == mine) {
        let _ = fs::remove_file(&lock);
    }
    result
}
