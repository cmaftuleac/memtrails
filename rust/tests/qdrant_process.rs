//! Memory's own Qdrant, run safely: a binary that is not there fails the start, a start that fails leaves nothing
//! running or scheduled, the child sees none of the host's secrets, what answers on the port must prove it is ours
//! before it gets the key, its log is private, and a Qdrant a dead host left on the storage is stopped — an unrelated
//! process with its pid never. Over a stand-in Qdrant (a Node.js script; skipped where there is no `node`).
#![cfg(unix)]

use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use memtrails::qdrant_process::{QdrantProcess, QdrantProcessOptions};

fn have_node() -> bool {
    Command::new("node").arg("-v").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// A stand-in Qdrant: `ok` checks its key like Qdrant; `anykey` takes any key (not ours); `exit` dies at once. Each
/// start is a line in spawns.log, and its environment is env.json — both in the folder it runs in.
fn fake_qdrant(root: &Path, mode: &str) -> PathBuf {
    let path = root.join(format!("fake-qdrant-{mode}.cjs"));
    let script = format!(
        r#"#!/usr/bin/env node
const http = require('node:http'); const fs = require('node:fs');
const args = process.argv.slice(2);
const port = Number(/http_port: (\d+)/.exec(fs.readFileSync(args[args.indexOf('--config-path') + 1], 'utf8'))[1]);
fs.appendFileSync('spawns.log', process.pid + '\n');
fs.writeFileSync('env.json', JSON.stringify(process.env));
if ({mode:?} === 'exit') process.exit(1);
const key = process.env.QDRANT__SERVICE__API_KEY;
http.createServer((req, res) => {{
  if (req.url !== '/readyz' && {mode:?} === 'ok' && req.headers['api-key'] !== key) {{ res.writeHead(401); return res.end('{{}}'); }}
  res.writeHead(200); res.end('{{"result":{{"collections":[]}}}}');
}}).listen(port, '127.0.0.1');
process.on('SIGTERM', () => process.exit(0));
"#
    );
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn spawns(dir: &Path) -> Vec<i64> {
    fs::read_to_string(dir.join("spawns.log"))
        .map(|t| t.lines().filter_map(|l| l.parse().ok()).collect())
        .unwrap_or_default()
}

fn alive(pid: i64) -> bool {
    // SAFETY: signal 0 only checks.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

async fn until(ms: u64, mut ok: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_millis(ms);
    while !ok() {
        if Instant::now() > end {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    true
}

fn qdrant(binary: &Path, dir: &Path) -> QdrantProcess {
    QdrantProcess::new(QdrantProcessOptions {
        binary: binary.into(),
        dir: dir.into(),
        log: Arc::new(|_| {}),
        on_down: None,
    })
}

#[tokio::test]
async fn a_binary_that_is_not_there_fails_the_start_and_starts_nothing_later() {
    let root = tempfile::tempdir().unwrap();
    let p = qdrant(Path::new("memtrails-no-such-qdrant"), &root.path().join("q"));
    let err = p.start().await.unwrap_err().to_string();
    assert!(err.contains("could not be started"), "{err}");
    assert!(p.current().is_none());
}

#[tokio::test]
async fn no_restart_when_qdrant_dies_while_starting() {
    if !have_node() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("q");
    let p = qdrant(&fake_qdrant(root.path(), "exit"), &dir);
    let err = p.start().await.unwrap_err().to_string();
    assert!(err.contains("exited"), "{err}");
    tokio::time::sleep(Duration::from_millis(1500)).await; // the first restart would come after 1 s
    assert_eq!(spawns(&dir).len(), 1);
}

#[tokio::test]
async fn refuses_what_takes_any_key_and_stops_it() {
    if !have_node() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("q");
    let p = qdrant(&fake_qdrant(root.path(), "anykey"), &dir);
    let err = p.start().await.unwrap_err().to_string();
    assert!(err.contains("not this memory's Qdrant: it took a wrong key"), "{err}");
    let pid = spawns(&dir)[0];
    assert!(until(5000, || !alive(pid)).await);
    assert!(p.current().is_none());
}

#[tokio::test]
async fn started_again_when_it_dies_after_it_was_up() {
    if !have_node() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("q");
    let p = qdrant(&fake_qdrant(root.path(), "ok"), &dir);
    p.start().await.unwrap();
    // SAFETY: killing the stand-in this test started.
    unsafe { libc::kill(spawns(&dir)[0] as libc::pid_t, libc::SIGKILL) };
    assert!(until(5000, || spawns(&dir).len() == 2 && p.current().is_some()).await);
    p.stop_and_wait(5000).await;
}

#[tokio::test]
async fn gets_none_of_the_hosts_environment_but_what_it_needs_and_keeps_its_folder_and_log_private() {
    if !have_node() {
        return;
    }
    // The only test in this file that sets variables: the children of the others see them, and do not care.
    std::env::set_var("MEMTRAILS_TEST_SECRET", "hunter2");
    std::env::set_var("QDRANT__SERVICE__HOST", "0.0.0.0");
    std::env::remove_var("SSL_CERT_FILE");
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("q");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("qdrant.log"), "old\n").unwrap();
    fs::set_permissions(dir.join("qdrant.log"), fs::Permissions::from_mode(0o644)).unwrap();
    let p = qdrant(&fake_qdrant(root.path(), "ok"), &dir);
    let e = p.start().await.unwrap();
    std::env::remove_var("MEMTRAILS_TEST_SECRET");
    std::env::remove_var("QDRANT__SERVICE__HOST");
    let env: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("env.json")).unwrap()).unwrap();
    assert!(env.get("MEMTRAILS_TEST_SECRET").is_none());
    assert!(env.get("QDRANT__SERVICE__HOST").is_none());
    assert_eq!(env["QDRANT__SERVICE__API_KEY"], e.api_key.as_str());
    assert_eq!(env["PATH"].as_str(), std::env::var("PATH").ok().as_deref());
    assert_eq!(fs::metadata(dir.join("qdrant.log")).unwrap().permissions().mode() & 0o777, 0o600);
    p.stop_and_wait(5000).await;
    // Qdrant will not start without CA certificates, and a slim container has none: the ones this crate carries, or the
    // host's SSL_CERT_FILE.
    let pem = fs::read_to_string(env["SSL_CERT_FILE"].as_str().unwrap()).unwrap();
    assert!(pem.matches("-----BEGIN CERTIFICATE-----").count() > 50);
    assert_eq!(fs::metadata(dir.join("ca.pem")).unwrap().permissions().mode() & 0o777, 0o600);
    let fresh_dir = root.path().join("fresh");
    let fresh = qdrant(&fake_qdrant(root.path(), "ok"), &fresh_dir);
    std::env::set_var("SSL_CERT_FILE", "/etc/ssl/certs/the-hosts.pem");
    fresh.start().await.unwrap();
    std::env::remove_var("SSL_CERT_FILE");
    let env: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(fresh_dir.join("env.json")).unwrap()).unwrap();
    assert_eq!(env["SSL_CERT_FILE"], "/etc/ssl/certs/the-hosts.pem");
    assert_eq!(fs::metadata(&fresh_dir).unwrap().permissions().mode() & 0o777, 0o700);
    assert_eq!(fs::metadata(fresh_dir.join("qdrant.log")).unwrap().permissions().mode() & 0o777, 0o600);
    fresh.stop_and_wait(5000).await;
}

/// Run as a host by the tests below: start a Qdrant, say "up", then exit without stopping it or hang.
#[test]
fn host_child() {
    let (Ok(binary), Ok(dir), Ok(how)) =
        (std::env::var("MT_HOST_BINARY"), std::env::var("MT_HOST_DIR"), std::env::var("MT_HOST_HOW"))
    else {
        return;
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let p = qdrant(Path::new(&binary), Path::new(&dir));
    rt.block_on(p.start()).unwrap();
    println!("up");
    if how == "exit" {
        std::process::exit(0);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn host(binary: &Path, dir: &Path, how: &str) -> std::process::Child {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "host_child", "--nocapture", "--test-threads=1"])
        .env("MT_HOST_BINARY", binary)
        .env("MT_HOST_DIR", dir)
        .env("MT_HOST_HOW", how)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let out = child.stdout.take().unwrap();
    let mut lines = BufReader::new(out).lines();
    loop {
        match lines.next() {
            Some(Ok(l)) if l.contains("up") => break,
            Some(_) => continue,
            None => panic!("the host ended before its Qdrant was up"),
        }
    }
    std::mem::forget(lines);
    child
}

#[tokio::test]
async fn a_host_that_exits_without_close_stops_its_qdrant() {
    if !have_node() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("q");
    let mut child = host(&fake_qdrant(root.path(), "ok"), &dir, "exit");
    child.wait().unwrap();
    let pid = spawns(&dir)[0];
    assert!(until(5000, || !alive(pid)).await);
}

#[tokio::test]
async fn a_host_killed_outright_leaves_a_qdrant_the_next_start_stops_first() {
    if !have_node() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("q");
    let binary = fake_qdrant(root.path(), "ok");
    let mut child = host(&binary, &dir, "hang");
    let pid = spawns(&dir)[0];
    child.kill().unwrap();
    child.wait().unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(alive(pid), "orphaned, still on the storage");
    let next = qdrant(&binary, &dir);
    next.start().await.unwrap();
    assert!(!alive(pid));
    assert_eq!(spawns(&dir).len(), 2);
    next.stop_and_wait(5000).await;
}

#[tokio::test]
async fn never_stops_an_unrelated_process_that_has_the_recorded_pid() {
    if !have_node() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("q");
    let mut stranger = Command::new("sleep").arg("30").spawn().unwrap();
    fs::create_dir_all(&dir).unwrap();
    let rec = serde_json::json!({ "pid": stranger.id(), "config": dir.join("config.yaml").to_string_lossy(), "boot": memtrails_boot() });
    fs::write(dir.join("qdrant.pid"), rec.to_string()).unwrap();
    let p = qdrant(&fake_qdrant(root.path(), "ok"), &dir);
    p.start().await.unwrap();
    assert!(alive(i64::from(stranger.id())));
    assert!(stranger.try_wait().unwrap().is_none());
    stranger.kill().unwrap();
    p.stop_and_wait(5000).await;
}

/// This machine's boot as the library writes it (the pid file of a Qdrant it starts carries it).
fn memtrails_boot() -> f64 {
    let out = Command::new("sysctl").args(["-n", "kern.boottime"]).output();
    if let Ok(o) = out {
        let t = String::from_utf8_lossy(&o.stdout).to_string();
        if let Some(sec) =
            t.split("sec = ").nth(1).and_then(|r| r.split(',').next()).and_then(|x| x.trim().parse::<f64>().ok())
        {
            return sec * 1000.0;
        }
    }
    let up: f64 = fs::read_to_string("/proc/uptime").unwrap().split_whitespace().next().unwrap().parse().unwrap();
    memtrails::js::now_ms() - up * 1000.0
}
