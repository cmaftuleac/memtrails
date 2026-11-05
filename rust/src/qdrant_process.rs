//! Qdrant as memory runs it when you give it no server of your own: one process for this memory, on a loopback port
//! picked at start, behind an API key made for that run, gRPC off, telemetry off, stopped with `close()`.
//!
//! Tuned for a laptop: the index on disk, memory-mapped, one optimisation thread, two search threads and ONE web
//! worker: Qdrant's default is one a core, each waking ~4 times a second, so an idle index woke ~83 times a second; with
//! one it is 6.5 wake-ups a second.
//!
//! A process that dies by itself is started again with a growing pause; five deaths in ten minutes stop the attempts
//! until the next `start()`. A start that fails leaves nothing running and nothing scheduled. It gets a minimal
//! environment, proves it is ours before it is handed anything, is stopped when the host exits (dropped, or the
//! process ending through `exit`), and one a crashed host left on the same storage is stopped before the next starts.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, LazyLock, Mutex, Once};
use std::time::{Duration, Instant};

use rand::RngCore;
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::error::{Error, Result};
use crate::llm::http;
use crate::sys;

/// Where a Qdrant answers, and its key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QdrantEndpoint {
    /// `http://127.0.0.1:<port>`.
    pub url: String,
    /// The API key.
    pub api_key: String,
}

/// The config file: what environment variables cannot say (gRPC off is `grpc_port: null`).
pub fn qdrant_config(storage: &Path, port: u16) -> String {
    let quote = |p: PathBuf| serde_json::to_string(&p.to_string_lossy()).unwrap_or_default();
    [
        "log_level: WARN".to_string(),
        "telemetry_disabled: true".into(),
        "service:".into(),
        "  host: 127.0.0.1".into(),
        format!("  http_port: {port}"),
        "  grpc_port: null".into(),
        "  enable_cors: false".into(),
        "  max_workers: 1".into(),
        "storage:".into(),
        format!("  storage_path: {}", quote(storage.join("storage"))),
        format!("  snapshots_path: {}", quote(storage.join("snapshots"))),
        "  on_disk_payload: true".into(),
        "  performance:".into(),
        "    max_search_threads: 2".into(),
        "  optimizers:".into(),
        "    max_optimization_threads: 1".into(),
        String::new(),
    ]
    .join("\n")
}

const LOG_MAX: u64 = 4 * 1024 * 1024;
/// Beside the storage: the pid of the Qdrant running on it, with its config and the machine's boot.
const PID_FILE: &str = "qdrant.pid";

fn random_key() -> String {
    let mut b = [0u8; 24];
    rand::thread_rng().fill_bytes(&mut b);
    hex::encode(b)
}

/// The process's log beside its data, rotated once at start when it has grown past 4 MB. Private: it names paths.
fn open_log(dir: &Path) -> Result<fs::File> {
    let path = dir.join("qdrant.log");
    if fs::metadata(&path).is_ok_and(|m| m.len() > LOG_MAX) {
        let _ = fs::rename(&path, dir.join("qdrant.log.1"));
    }
    sys::make_dir(dir, 0o700).map_err(|e| Error::io(dir, e))?;
    let mut o = OpenOptions::new();
    o.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let f = o.open(&path).map_err(|e| Error::io(&path, e))?;
    // Not ours to change: written all the same.
    let _ = sys::chmod(&path, 0o600);
    Ok(f)
}

/// Every Qdrant this process started and has not seen exit: stopped when the process exits without `close()`.
static RUNNING: LazyLock<Mutex<HashSet<u32>>> = LazyLock::new(|| Mutex::new(HashSet::new()));
static EXIT_HOOK: Once = Once::new();

extern "C" fn stop_running() {
    if let Ok(pids) = RUNNING.lock() {
        for pid in pids.iter() {
            sys::signal(i64::from(*pid), false);
        }
    }
}

fn track(pid: u32) {
    if let Ok(mut r) = RUNNING.lock() {
        r.insert(pid);
    }
    EXIT_HOOK.call_once(|| {
        #[cfg(unix)]
        // SAFETY: stop_running is a plain extern "C" function that only reads a static and sends signals.
        unsafe {
            libc::atexit(stop_running);
        }
    });
}

fn untrack(pid: u32) {
    if let Ok(mut r) = RUNNING.lock() {
        r.remove(&pid);
    }
}

/// The options a Qdrant process runs with.
#[derive(Clone)]
pub struct QdrantProcessOptions {
    /// The binary (a path, or a name looked up on PATH).
    pub binary: PathBuf,
    /// Its folder: storage, config, log, pid file.
    pub dir: PathBuf,
    /// Progress lines.
    pub log: crate::LogFn,
    /// Told when it is down for good (five deaths in ten minutes) or a restart failed.
    pub on_down: Option<crate::LogFn>,
}

struct Running {
    pid: u32,
    /// Becomes `Some(how)` when the process exits.
    exited: watch::Receiver<Option<String>>,
}

#[derive(Default)]
struct State {
    child: Option<Running>,
    endpoint: Option<QdrantEndpoint>,
    stopping: bool,
    deaths: Vec<Instant>,
    restart: Option<tokio::task::JoinHandle<()>>,
}

struct Inner {
    opts: QdrantProcessOptions,
    state: Mutex<State>,
    /// One start at a time.
    starting: tokio::sync::Mutex<()>,
}

/// Memory's own Qdrant.
#[derive(Clone)]
pub struct QdrantProcess {
    inner: Arc<Inner>,
}

fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

impl QdrantProcess {
    /// A Qdrant for `dir` (made absolute: Qdrant runs inside it), not yet started.
    pub fn new(mut opts: QdrantProcessOptions) -> Self {
        opts.dir = absolute(&opts.dir);
        let b = opts.binary.to_string_lossy();
        if b.contains('/') || b.contains('\\') {
            opts.binary = absolute(&opts.binary);
        }
        QdrantProcess {
            inner: Arc::new(Inner { opts, state: Mutex::new(State::default()), starting: tokio::sync::Mutex::new(()) }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Where it answers, once it does; `None` while it starts or after it stopped.
    pub fn current(&self) -> Option<QdrantEndpoint> {
        self.state().endpoint.clone()
    }

    fn log(&self, line: &str) {
        (self.inner.opts.log)(line)
    }

    /// Up and proven ours, or an error — and then nothing of this start keeps running or is started again: the
    /// caller was told, and may not be there to stop it.
    pub async fn start(&self) -> Result<QdrantEndpoint> {
        let _one = self.inner.starting.lock().await;
        {
            let mut s = self.state();
            s.stopping = false;
            if let (Some(e), Some(c)) = (&s.endpoint, &s.child) {
                if c.exited.borrow().is_none() {
                    return Ok(e.clone());
                }
            }
        }
        let dir = self.inner.opts.dir.clone();
        let made = dir.clone();
        crate::store::blocking(move || sys::make_dir(&made, 0o700)).await?.map_err(|e| Error::io(&dir, e))?;
        self.reap_leftover().await;
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map_err(|e| Error::Process(format!("no free port: {e}")))?
            .port();
        let api_key = random_key();
        let config = dir.join("config.yaml");
        crate::store::write_private(&config, qdrant_config(&dir, port).as_bytes())
            .map_err(|e| Error::io(&config, e))?;
        let log = open_log(&dir)?;
        let err_log = log.try_clone().map_err(|e| Error::io(&dir, e))?;
        let mut cmd = tokio::process::Command::new(&self.inner.opts.binary);
        cmd.arg("--config-path").arg(&config).arg("--disable-telemetry").current_dir(&dir);
        cmd.stdin(Stdio::null()).stdout(Stdio::from(log)).stderr(Stdio::from(err_log));
        // What Qdrant gets of the environment: what a process needs to run and the key. None of the host's secrets,
        // and none of its QDRANT__* variables — they override the config (QDRANT__SERVICE__HOST=0.0.0.0 would open it
        // to the network).
        cmd.env_clear();
        for k in ["PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SystemRoot", "USERPROFILE"] {
            if let Some(v) = std::env::var_os(k) {
                cmd.env(k, v);
            }
        }
        cmd.env("QDRANT__SERVICE__API_KEY", &api_key).env("QDRANT__TELEMETRY_DISABLED", "true");
        cmd.kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| {
            Error::Process(format!("Qdrant could not be started ({}): {e}", self.inner.opts.binary.display()))
        })?;
        let pid = child.id().unwrap_or(0);
        let (tx, rx) = watch::channel(None);
        track(pid);
        let pid_file = dir.join(PID_FILE);
        // Weak: a memory dropped without close() drops its Qdrant too (Inner's Drop stops it).
        let me = Arc::downgrade(&self.inner);
        tokio::spawn(async move {
            let how = match child.wait().await {
                Ok(s) => s.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
                Err(e) => e.to_string(),
            };
            untrack(pid);
            if tokio::fs::read_to_string(&pid_file)
                .await
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .is_some_and(|v| v["pid"].as_u64() == Some(u64::from(pid)))
            {
                let _ = tokio::fs::remove_file(&pid_file).await;
            }
            let _ = tx.send(Some(how.clone()));
            if let Some(inner) = me.upgrade() {
                QdrantProcess { inner }.exited(pid, &how);
            }
        });
        // Known before the first await: a start cancelled from here on leaves a process `stop` (or a drop) ends.
        self.state().child = Some(Running { pid, exited: rx.clone() });
        self.record_pid(pid, &config).await;
        let endpoint = QdrantEndpoint { url: format!("http://127.0.0.1:{port}"), api_key };
        let proven = async {
            self.ready(&endpoint, &rx).await?;
            self.owned(&endpoint, &rx).await?;
            if self.state().child.as_ref().map(|c| c.pid) != Some(pid) {
                return Err(Error::Process("Qdrant was stopped while it started".into()));
            }
            Ok(())
        };
        if let Err(e) = proven.await {
            {
                let mut s = self.state();
                if s.child.as_ref().map(|c| c.pid) == Some(pid) {
                    s.child = None; // its exit then schedules nothing
                }
            }
            if rx.borrow().is_none() {
                sys::signal(i64::from(pid), false);
            }
            return Err(e);
        }
        self.state().endpoint = Some(endpoint.clone());
        self.log(&format!("[memory] Qdrant up on 127.0.0.1:{port}"));
        Ok(endpoint)
    }

    async fn record_pid(&self, pid: u32, config: &Path) {
        let file = self.inner.opts.dir.join(PID_FILE);
        let text = crate::js::stringify(
            &json!({ "pid": pid, "config": config.to_string_lossy(), "boot": crate::js::ms_value(sys::boot_time()) }),
        );
        let _ = crate::store::blocking(move || crate::store::write_private(&file, text.as_bytes())).await;
    }

    /// A Qdrant left running on this storage by a host that died without `close()`: stopped before another starts on
    /// it. Only when the pid is still the process written down — this boot, its command naming this folder's config —
    /// never an unrelated process that has the pid since.
    async fn reap_leftover(&self) {
        let dir = &self.inner.opts.dir;
        let file = dir.join(PID_FILE);
        let Some(rec) =
            tokio::fs::read_to_string(&file).await.ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
        else {
            return;
        };
        let pid = rec["pid"].as_i64().unwrap_or(0);
        let config = dir.join("config.yaml").to_string_lossy().to_string();
        let mine = self.state().child.as_ref().map(|c| i64::from(c.pid));
        if rec["config"].as_str() == Some(config.as_str())
            && sys::same_boot(rec["boot"].as_f64())
            && Some(pid) != mine
            && sys::alive(pid)
            && sys::command_of(pid).await.is_some_and(|c| c.contains(&config))
        {
            self.log(&format!("[memory] stopping the Qdrant left running on {} (pid {pid})", dir.display()));
            for (kill, ms) in [(false, 10_000u64), (true, 2000)] {
                sys::signal(pid, kill);
                let until = Instant::now() + Duration::from_millis(ms);
                while sys::alive(pid) && Instant::now() < until {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                if !sys::alive(pid) {
                    break;
                }
            }
        }
        let _ = tokio::fs::remove_file(&file).await;
    }

    /// Answering `/readyz` within 30 s, or the start failed. `/readyz` asks no key, so none is sent: what answers may
    /// not be ours (`owned`).
    async fn ready(&self, e: &QdrantEndpoint, exited: &watch::Receiver<Option<String>>) -> Result<()> {
        let until = Instant::now() + Duration::from_secs(30);
        while Instant::now() < until {
            if let Some(how) = exited.borrow().clone() {
                return Err(Error::Process(format!(
                    "Qdrant exited ({how}) while starting — see {}",
                    self.inner.opts.dir.join("qdrant.log").display()
                )));
            }
            if let Ok(r) = http().get(format!("{}/readyz", e.url)).timeout(Duration::from_secs(1)).send().await {
                let ok = r.status().is_success();
                let _ = r.bytes().await;
                if ok {
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Err(Error::Process("Qdrant did not answer within 30 s".into()))
    }

    /// The port was free when picked, not when Qdrant bound it: another local process may have taken it, answered
    /// `/readyz`, and would be handed the key and every document. Ours is still running, refuses a wrong key and takes
    /// this run's — the wrong key asked first, so a process that takes any key never sees the real one.
    async fn owned(&self, e: &QdrantEndpoint, exited: &watch::Receiver<Option<String>>) -> Result<()> {
        let status = |key: String| async move {
            let r = http()
                .get(format!("{}/collections", e.url))
                .header("api-key", key)
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .map_err(|err| Error::Process(format!("what answers on {} could not be asked: {err}", e.url)))?;
            let s = r.status().as_u16();
            let _ = r.bytes().await;
            Ok::<u16, Error>(s)
        };
        let not_ours = format!("what answers on {} is not this memory's Qdrant", e.url);
        let log = self.inner.opts.dir.join("qdrant.log");
        let gone = || exited.borrow().is_some();
        if gone() {
            return Err(Error::Process(format!("{not_ours}: ours exited — see {}", log.display())));
        }
        let wrong = status(random_key()).await?;
        if wrong != 401 && wrong != 403 {
            return Err(Error::Process(format!("{not_ours}: it took a wrong key ({wrong})")));
        }
        let right = status(e.api_key.clone()).await?;
        if right != 200 {
            return Err(Error::Process(format!("{not_ours}: it refused this run's key ({right})")));
        }
        if gone() {
            return Err(Error::Process(format!("{not_ours}: ours exited — see {}", log.display())));
        }
        Ok(())
    }

    fn exited(&self, pid: u32, how: &str) {
        let restart = {
            let mut s = self.state();
            if s.child.as_ref().map(|c| c.pid) != Some(pid) {
                return;
            }
            // One that never came up is its start()'s to report: that start fails, and schedules nothing.
            let was_up = s.endpoint.is_some();
            s.child = None;
            s.endpoint = None;
            !s.stopping && was_up
        };
        if restart {
            self.restart_later(format!("Qdrant stopped ({how})"));
        }
    }

    /// Started again after a growing pause; a restart that fails counts as another death.
    fn restart_later(&self, why: String) {
        self.log(&format!("[memory] {why}"));
        let now = Instant::now();
        let wait = {
            let mut s = self.state();
            s.deaths.retain(|t| now.duration_since(*t) < Duration::from_secs(600));
            s.deaths.push(now);
            if s.deaths.len() >= 5 {
                drop(s);
                if let Some(down) = &self.inner.opts.on_down {
                    down(&format!("{why}; five times in ten minutes, so it is left stopped"));
                }
                return;
            }
            Duration::from_millis((1000u64 << (s.deaths.len() - 1)).min(60_000))
        };
        let Ok(rt) = tokio::runtime::Handle::try_current() else { return };
        let me = Arc::downgrade(&self.inner);
        let task = rt.spawn(async move {
            tokio::time::sleep(wait).await;
            let Some(inner) = me.upgrade() else { return };
            let me = QdrantProcess { inner };
            if me.state().stopping {
                return;
            }
            if let Err(e) = me.start().await {
                let msg = e.to_string();
                if let Some(down) = &me.inner.opts.on_down {
                    down(&msg);
                }
                if !me.state().stopping {
                    me.restart_later(msg);
                }
            }
        });
        self.state().restart = Some(task);
    }

    /// Stop it (no restart after).
    pub fn stop(&self) {
        let mut s = self.state();
        s.stopping = true;
        if let Some(t) = s.restart.take() {
            t.abort();
        }
        let c = s.child.take();
        s.endpoint = None;
        drop(s);
        if let Some(c) = c {
            if c.exited.borrow().is_none() {
                sys::signal(i64::from(c.pid), false);
            }
        }
    }

    /// Stop it and wait (up to `ms`) for the process to be gone — before its storage is deleted.
    pub async fn stop_and_wait(&self, ms: u64) {
        let running = self.state().child.as_ref().map(|c| c.exited.clone());
        self.stop();
        if let Some(mut rx) = running {
            let _ = tokio::time::timeout(Duration::from_millis(ms), rx.wait_for(|x| x.is_some())).await;
        }
    }

    /// The pid of the process running, if any.
    pub fn pid(&self) -> Option<u32> {
        self.state().child.as_ref().map(|c| c.pid)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let s = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = s.child.take() {
            if c.exited.borrow().is_none() {
                sys::signal(i64::from(c.pid), false);
            }
        }
    }
}
