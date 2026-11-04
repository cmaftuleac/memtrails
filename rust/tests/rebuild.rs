//! The index rebuilt from the logs: checkpoints survive a failed batch and a restart, batches in flight at once never
//! let the checkpoint pass one not written, a stop leaves the checkpoint where it was, and a rebuild begun again while
//! one runs stops that one and writes everything.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use indexmap::IndexMap;
use memtrails::error::Error;
use memtrails::rebuild::{MemoryRebuild, RebuildDeps, WriteBatch};
use memtrails::store::{ltm_log, stm_log, Keyed, MemoryItem, StmEntry};

const NOW: f64 = 1_800_000_000_000.0;

fn item(uid: &str, status: &str) -> MemoryItem {
    serde_json::from_value(serde_json::json!({
        "uid": uid, "kind": "fact", "scope": "project", "project": "-p", "engine": "conversation", "session": "s",
        "cwd": "/p", "where": "c", "ts": 1, "firstTs": 1, "title": uid, "summary": uid, "body": uid, "files": [],
        "appliesWhen": "", "confidence": "verified", "importance": 3, "status": status, "provenance": [], "seen": 1,
        "fingerprint": uid,
    }))
    .unwrap()
}

fn entry(uid: &str) -> StmEntry {
    StmEntry {
        uid: uid.into(),
        fingerprint: uid.into(),
        engine: "conversation".into(),
        session: "s".into(),
        turn: 1,
        cwd: Some("/p".into()),
        project: "-p".into(),
        title: uid.into(),
        ts: NOW - 1000.0,
        synthetic: false,
        typed: "human".into(),
        request: uid.into(),
        reply: String::new(),
        said: String::new(),
        steps: vec![],
        files: vec![],
        done: false,
        record: None,
        extra: IndexMap::new(),
    }
}

struct Rig {
    _dir: tempfile::TempDir,
    deps: RebuildDeps,
    written: Arc<Mutex<Vec<Vec<String>>>>,
    stop: Arc<AtomicBool>,
}

fn recorder<T: Keyed>(written: &Arc<Mutex<Vec<Vec<String>>>>) -> WriteBatch<T> {
    let w = written.clone();
    Arc::new(move |xs: Vec<T>| {
        let w = w.clone();
        Box::pin(async move {
            w.lock().unwrap().push(xs.iter().map(|x| x.uid().to_string()).collect());
            Ok(())
        })
    })
}

fn rig() -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let written = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    let deps = RebuildDeps {
        path: dir.path().join("rebuild.json"),
        ltm: ltm_log(dir.path().join("ltm.jsonl")),
        stm: stm_log(dir.path().join("stm.jsonl")),
        write_ltm: recorder(&written),
        write_stm: recorder(&written),
        aborted: Arc::new(move || s.load(Ordering::SeqCst)),
        concurrency: 1,
    };
    Rig { _dir: dir, deps, written, stop }
}

fn checkpoint(r: &Rig) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(&r.deps.path).unwrap()).unwrap()
}

fn flat(r: &Rig) -> Vec<String> {
    r.written.lock().unwrap().iter().flatten().cloned().collect()
}

#[tokio::test]
async fn resumes_after_a_failed_batch_and_a_restart_without_writing_finished_batches_again() {
    let r = rig();
    r.deps.ltm.put((0..41).map(|i| item(&format!("mem:{i}"), "active")).collect()).await.unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let inner = r.deps.write_ltm.clone();
    let c = calls.clone();
    let failing: WriteBatch<MemoryItem> = Arc::new(move |xs| {
        let inner = inner.clone();
        let n = c.fetch_add(1, Ordering::SeqCst) + 1;
        Box::pin(async move {
            if n == 2 {
                return Err(Error::Embedding("GPU service temporarily unavailable".into()));
            }
            inner(xs).await
        })
    });
    let first = MemoryRebuild::new(RebuildDeps { write_ltm: failing, ..r.deps.clone() });
    first.begin().await.unwrap();
    assert!(first.run().await.unwrap_err().to_string().contains("unavailable"));
    assert_eq!(checkpoint(&r), serde_json::json!({ "phase": "ltm", "after": "mem:19" }));
    assert!(first.pending().await);
    r.deps.ltm.load().await.unwrap();
    let resumed = MemoryRebuild::new(r.deps.clone());
    resumed.run().await.unwrap();
    assert!(!resumed.pending().await);
    assert_eq!(flat(&r), (0..41).map(|i| format!("mem:{i}")).collect::<Vec<_>>());
}

#[tokio::test]
async fn writes_several_batches_at_once_and_its_checkpoint_never_passes_one_in_flight_or_failed() {
    let r = rig();
    r.deps.stm.put((0..32 * 6).map(|i| entry(&format!("stm:{i:03}"))).collect()).await.unwrap();
    let n = Arc::new(AtomicUsize::new(0));
    let inner = r.deps.write_stm.clone();
    let slow: WriteBatch<StmEntry> = Arc::new(move |xs| {
        let k = n.fetch_add(1, Ordering::SeqCst);
        let inner = inner.clone();
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis([30, 10, 40, 1, 1, 1][k])).await;
            if k == 2 {
                return Err(Error::Embedding("GPU worker reset".into()));
            }
            inner(xs).await
        })
    });
    let job = MemoryRebuild::new(RebuildDeps { write_stm: slow, concurrency: 4, ..r.deps.clone() });
    job.begin().await.unwrap();
    assert!(job.run().await.unwrap_err().to_string().contains("reset"));
    assert_eq!(checkpoint(&r), serde_json::json!({ "phase": "stm", "after": "stm:063" }));
    r.deps.stm.load().await.unwrap();
    r.written.lock().unwrap().clear();
    MemoryRebuild::new(RebuildDeps { concurrency: 4, ..r.deps.clone() }).run().await.unwrap();
    let mut got = flat(&r);
    got.sort();
    assert_eq!(got, (0..32 * 4).map(|i| format!("stm:{:03}", i + 64)).collect::<Vec<_>>());
}

#[tokio::test]
async fn a_stopped_memory_does_no_work_and_leaves_the_checkpoint() {
    let r = rig();
    r.deps.ltm.put(vec![item("mem:a", "active")]).await.unwrap();
    let job = MemoryRebuild::new(r.deps.clone());
    job.begin().await.unwrap();
    r.stop.store(true, Ordering::SeqCst);
    job.run().await.unwrap();
    assert!(job.pending().await);
    assert!(flat(&r).is_empty());
}

#[tokio::test]
async fn indexes_active_knowledge_and_every_real_record_consolidated_or_not() {
    let r = rig();
    r.deps
        .ltm
        .put(vec![item("mem:active", "active"), item("mem:archived", "archived"), item("mem:old", "superseded")])
        .await
        .unwrap();
    let mut old = entry("expired");
    old.ts = NOW - 8.0 * 86_400_000.0;
    let mut done = entry("done");
    done.done = true;
    let mut synthetic = entry("synthetic");
    synthetic.synthetic = true;
    r.deps.stm.put(vec![entry("recent"), old, done, synthetic]).await.unwrap();
    let job = MemoryRebuild::new(r.deps.clone());
    job.begin().await.unwrap();
    job.run().await.unwrap();
    assert_eq!(
        *r.written.lock().unwrap(),
        vec![vec!["mem:active".to_string()], vec!["recent".into(), "expired".into(), "done".into()]]
    );
    assert!(!job.pending().await);
}

#[tokio::test]
async fn simultaneous_runs_make_one_rebuild() {
    let r = rig();
    r.deps.ltm.put(vec![item("mem:a", "active")]).await.unwrap();
    let job = MemoryRebuild::new(r.deps.clone());
    job.begin().await.unwrap();
    let (a, b, c) = tokio::join!(job.run(), job.run(), job.run());
    a.unwrap();
    b.unwrap();
    c.unwrap();
    assert_eq!(*r.written.lock().unwrap(), vec![vec!["mem:a".to_string()]]);
}

#[tokio::test]
async fn replays_a_phase_when_its_checkpoint_is_damaged_or_names_a_missing_uid() {
    for cp in ["broken", r#"{"phase":"ltm","after":"not-here"}"#] {
        let r = rig();
        r.deps.ltm.put(vec![item("mem:a", "active")]).await.unwrap();
        std::fs::write(&r.deps.path, cp).unwrap();
        let job = MemoryRebuild::new(r.deps.clone());
        job.run().await.unwrap();
        assert_eq!(*r.written.lock().unwrap(), vec![vec!["mem:a".to_string()]]);
        assert!(!job.pending().await);
    }
}

#[tokio::test]
async fn a_cancelled_run_does_not_move_the_checkpoint() {
    let r = rig();
    r.deps.ltm.put(vec![item("mem:a", "active")]).await.unwrap();
    let gate = Arc::new(tokio::sync::Notify::new());
    let g = gate.clone();
    let inner = r.deps.write_ltm.clone();
    let blocked: WriteBatch<MemoryItem> = Arc::new(move |xs| {
        let g = g.clone();
        let inner = inner.clone();
        Box::pin(async move {
            g.notified().await;
            inner(xs).await
        })
    });
    let job = Arc::new(MemoryRebuild::new(RebuildDeps { write_ltm: blocked, ..r.deps.clone() }));
    job.begin().await.unwrap();
    let j = job.clone();
    let running = tokio::spawn(async move { j.run().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    job.cancel();
    gate.notify_one();
    running.await.unwrap().unwrap();
    assert_eq!(checkpoint(&r), serde_json::json!({ "phase": "ltm", "after": null }));
    MemoryRebuild::new(r.deps.clone()).run().await.unwrap();
    assert_eq!(flat(&r), ["mem:a", "mem:a"]);
}

#[tokio::test]
async fn a_rebuild_begun_again_while_one_runs_stops_it_and_writes_everything() {
    let r = rig();
    r.deps.stm.put((0..32 * 3).map(|i| entry(&format!("r{i:03}"))).collect()).await.unwrap();
    let index = Arc::new(Mutex::new(std::collections::HashSet::new()));
    let ix = index.clone();
    let write: WriteBatch<StmEntry> = Arc::new(move |xs| {
        let ix = ix.clone();
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            ix.lock().unwrap().extend(xs.into_iter().map(|x| x.uid));
            Ok(())
        })
    });
    let job = Arc::new(MemoryRebuild::new(RebuildDeps { write_stm: write, ..r.deps.clone() }));
    job.begin().await.unwrap();
    let j = job.clone();
    let first = tokio::spawn(async move { j.run().await });
    while index.lock().unwrap().is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    job.begin().await.unwrap();
    index.lock().unwrap().clear();
    let second = job.run().await;
    first.await.unwrap().unwrap();
    second.unwrap();
    assert_eq!(index.lock().unwrap().len(), 32 * 3);
    assert!(!r.deps.path.exists());
}
