//! Memory's logs: the truth the index is rebuilt from. Each case is a way a rebuild or a sweep would go wrong — a
//! replaced record coming back, a record read at the wrong offset after an append, a rewrite or a crash, a
//! compaction that would drop what it cannot read, usage counts lost.

use std::fs;
use std::io::Write;

use memtrails::store::{fingerprint, FpPart, JsonlLog, Keyed, UsageBook};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Rec {
    uid: String,
    text: String,
    n: i64,
}

impl Keyed for Rec {
    fn uid(&self) -> &str {
        &self.uid
    }
}

fn rec(uid: &str, text: &str, n: i64) -> Rec {
    Rec { uid: uid.into(), text: text.into(), n }
}

fn log(path: &std::path::Path) -> JsonlLog<Rec, i64> {
    JsonlLog::new(path, |r: &Rec| r.n)
}

async fn uids(l: &JsonlLog<Rec, i64>) -> Vec<String> {
    l.all().await.into_iter().map(|r| r.uid).collect()
}

fn append(path: &std::path::Path, text: &str) {
    fs::OpenOptions::new().append(true).open(path).unwrap().write_all(text.as_bytes()).unwrap();
}

#[tokio::test]
async fn reads_each_record_back_by_offset_latest_line_winning_across_a_reload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let a = log(&path);
    a.put(vec![rec("a", "первый — ünïcödé", 1), rec("b", "second", 2)]).await.unwrap();
    a.put(vec![rec("a", "replaced", 3)]).await.unwrap();
    assert_eq!(a.get("a").await.unwrap().text, "replaced");
    assert_eq!(a.get("b").await.unwrap().text, "second");
    assert_eq!(a.meta("a"), Some(3));
    let b = log(&path);
    b.load().await.unwrap();
    assert_eq!(b.size(), 2);
    assert_eq!(b.get("a").await.unwrap().text, "replaced");
    assert_eq!(uids(&b).await, ["b", "a"]);
}

#[tokio::test]
async fn compacts_once_twice_what_is_live_offsets_still_right() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    for i in 0..1200 {
        l.put(vec![rec(&format!("u{}", i % 3), &format!("v{i} é"), i)]).await.unwrap();
    }
    l.compact().await.unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap().trim().split('\n').count(), 3);
    assert_eq!(l.get("u0").await.unwrap().text, "v1197 é");
    l.put(vec![rec("u3", "after", 0)]).await.unwrap();
    assert_eq!(l.get("u3").await.unwrap().text, "after");
    assert_eq!(l.get("u2").await.unwrap().text, "v1199 é");
}

#[tokio::test]
async fn removes_and_stays_removed_after_a_reload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    l.put(vec![rec("a", "one", 1), rec("b", "two", 2)]).await.unwrap();
    l.remove(vec!["a".into(), "nope".into()]).await.unwrap();
    assert!(l.get("a").await.is_none());
    assert!(fs::read_to_string(&path).unwrap().contains(r#"{"uid":"a","removed":true}"#));
    let r = log(&path);
    r.load().await.unwrap();
    assert_eq!(uids(&r).await, ["b"]);
}

#[tokio::test]
async fn drops_a_cut_off_last_line_and_keeps_what_comes_after_through_reopen_and_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    log(&path).put(vec![rec("a", "ok é", 1)]).await.unwrap();
    append(&path, r#"{"uid":"b","text":"torn"#);
    let l = log(&path);
    l.load().await.unwrap();
    l.put(vec![rec("c", "after the crash", 2), rec("d", "and this", 3)]).await.unwrap();
    assert_eq!(l.get("c").await.unwrap().text, "after the crash");
    assert_eq!(l.get("d").await.unwrap().text, "and this");
    assert!(!fs::read_to_string(&path).unwrap().contains("torn"));
    let r = log(&path);
    r.load().await.unwrap();
    assert_eq!(uids(&r).await, ["a", "c", "d"]);
    for i in 0..1100 {
        r.put(vec![rec("a", &format!("v{i}"), i)]).await.unwrap();
    }
    r.compact().await.unwrap();
    let s = log(&path);
    s.load().await.unwrap();
    let got: Vec<(String, String)> = s.all().await.into_iter().map(|x| (x.uid, x.text)).collect();
    assert_eq!(
        got,
        [("c".into(), "after the crash".into()), ("d".into(), "and this".into()), ("a".into(), "v1099".into())]
    );
}

#[tokio::test]
async fn keeps_a_last_line_whole_but_for_its_newline() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    fs::write(&path, "{\"uid\":\"a\",\"text\":\"one\",\"n\":1}\n{\"uid\":\"b\",\"text\":\"two\",\"n\":2}").unwrap();
    let l = log(&path);
    l.load().await.unwrap();
    l.put(vec![rec("c", "three", 3)]).await.unwrap();
    assert_eq!(l.get("b").await.unwrap().text, "two");
    assert_eq!(l.get("c").await.unwrap().text, "three");
    let r = log(&path);
    r.load().await.unwrap();
    assert_eq!(uids(&r).await, ["a", "b", "c"]);
}

#[tokio::test]
async fn refuses_to_compact_when_a_record_cannot_be_read_leaving_the_file_as_it_was() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    l.put(vec![rec("keep", "precious", 0)]).await.unwrap();
    for i in 0..1100 {
        l.put(vec![rec("u", &format!("v{i}"), i)]).await.unwrap();
    }
    // Damaged in place (same length): the index still points at it.
    let before = fs::read_to_string(&path).unwrap().replacen("precious", "prec\"ous", 1);
    fs::write(&path, &before).unwrap();
    assert!(l.get("keep").await.is_none());
    let err = l.compact().await.unwrap_err().to_string();
    assert!(err.contains("keep"), "{err}");
    assert_eq!(fs::read_to_string(&path).unwrap(), before);
    assert_eq!(l.get("u").await.unwrap().text, "v1099");
}

#[tokio::test]
async fn compacts_a_log_larger_than_one_write_every_offset_right_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    let big = "ü".repeat(3000);
    let live = 800;
    for round in 0..4 {
        let batch: Vec<Rec> = (0..live).map(|i| rec(&format!("u{i}"), &format!("{round}:{i}:{big}"), round)).collect();
        l.put(batch).await.unwrap();
    }
    l.compact().await.unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap().trim().split('\n').count(), live);
    assert_eq!(l.get("u799").await.unwrap().text, format!("3:799:{big}"));
    l.put(vec![rec("new", "after", 9)]).await.unwrap();
    let r = log(&path);
    r.load().await.unwrap();
    assert_eq!(r.size(), live + 1);
    for i in 0..live {
        assert_eq!(r.get(&format!("u{i}")).await.unwrap().text, format!("3:{i}:{big}"));
    }
    assert_eq!(r.get("new").await.unwrap().text, "after");
}

#[tokio::test]
async fn usage_counts_what_was_shown_and_opened_and_keeps_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("usage.json");
    let u = UsageBook::open(&path).await;
    u.note(&["m1".into(), "m2".into()], false, 100.0);
    u.note(&["m1".into()], true, 200.0);
    u.flush().await;
    let again = UsageBook::open(&path).await;
    let m1 = again.of("m1");
    assert_eq!((m1.shown, m1.opened, m1.last_used), (1, 1, 200.0));
    assert_eq!(again.of("m3").shown, 0);
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        r#"{"m1":{"shown":1,"opened":1,"lastUsed":200},"m2":{"shown":1,"opened":0,"lastUsed":100}}"#
    );
}

#[test]
fn fingerprint_changes_with_any_part_and_does_not_confuse_a_list_with_its_joined_text() {
    let bc = vec!["b".to_string(), "c".to_string()];
    assert_ne!(
        fingerprint(&[FpPart::Text("a"), FpPart::List(&bc)]),
        fingerprint(&[FpPart::Text("a"), FpPart::Text("b,c")])
    );
    assert_eq!(fingerprint(&[FpPart::Text("a"), FpPart::Null]), fingerprint(&[FpPart::Text("a"), FpPart::Text("")]));
    assert_ne!(fingerprint(&[FpPart::Text("a")]), fingerprint(&[FpPart::Text("b")]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_read_during_a_compaction_gets_the_record_it_asked_for() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    l.put((0..50).map(|i| rec(&format!("k{i}"), &format!("kept {i}"), i)).collect()).await.unwrap();
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    // Readers of the kept records, many at once, while replaced lines are compacted away again and again.
    let readers: Vec<_> = (0..8)
        .map(|r| {
            let (l, stop) = (l.clone(), stop.clone());
            tokio::spawn(async move {
                let mut n = r;
                while !stop.load(Ordering::SeqCst) {
                    let i = n % 50;
                    let got = l.get_many(vec![format!("k{i}"), format!("k{}", (i + 1) % 50)]).await;
                    let texts: Vec<Option<String>> = got.into_iter().map(|g| g.map(|x| x.text)).collect();
                    assert_eq!(texts, [Some(format!("kept {i}")), Some(format!("kept {}", (i + 1) % 50))]);
                    n += 1;
                }
            })
        })
        .collect();
    for round in 0..12 {
        l.put((0..1100).map(|i| rec("churn", &format!("{round}:{i}"), i)).collect::<Vec<_>>()).await.unwrap();
        for _ in 0..1100 {
            l.put(vec![rec("churn", "again", round)]).await.unwrap();
            if readers.iter().any(|r| r.is_finished()) {
                break;
            }
        }
        l.compact().await.unwrap();
    }
    stop.store(true, Ordering::SeqCst);
    for r in readers {
        r.await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_compaction_does_not_hold_up_what_reads_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    let big = "ü".repeat(4000);
    for round in 0..3 {
        l.put((0..1500).map(|i| rec(&format!("u{i}"), &format!("{round}:{big}"), round)).collect()).await.unwrap();
    }
    let c = l.clone();
    let started = std::time::Instant::now();
    let compacting = tokio::spawn(async move { c.compact().await });
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let mut slowest = std::time::Duration::ZERO;
    while !compacting.is_finished() {
        let t = std::time::Instant::now();
        assert_eq!(l.meta("u7"), Some(2));
        assert!(l.size() >= 1500);
        l.put(vec![rec("during", "a put meanwhile", 9)]).await.unwrap();
        slowest = slowest.max(t.elapsed());
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    compacting.await.unwrap().unwrap();
    let took = started.elapsed();
    assert!(took > std::time::Duration::from_millis(100), "the compaction was too quick to tell: {took:?}");
    assert!(slowest < took / 3, "a read of the index waited {slowest:?} of a {took:?} compaction");
    // What was put meanwhile is there, at the right offset, and after a reload.
    assert_eq!(l.get("during").await.unwrap().text, "a put meanwhile");
    assert_eq!(l.get("u1499").await.unwrap().text, format!("2:{big}"));
    let r = log(&path);
    r.load().await.unwrap();
    assert_eq!(r.size(), 1501);
    assert_eq!(r.get("during").await.unwrap().text, "a put meanwhile");
    assert_eq!(r.get("u0").await.unwrap().text, format!("2:{big}"));
}

#[tokio::test]
async fn a_line_the_reference_reads_is_never_dropped_by_a_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    // A record another version wrote without a field this one needs, a lone surrogate, a byte that is not UTF-8.
    let mut raw =
        b"{\"uid\":\"partial\",\"text\":\"no number\"}\n{\"uid\":\"lone\",\"text\":\"a\\ud800b\",\"n\":1}\n".to_vec();
    raw.extend_from_slice(b"{\"uid\":\"bytes\",\"text\":\"a\xffb\",\"n\":2}\n");
    fs::write(&path, &raw).unwrap();
    let l = log(&path);
    l.load().await.unwrap();
    assert_eq!(l.get("lone").await.unwrap().text, "a\u{fffd}b");
    assert_eq!(l.get("bytes").await.unwrap().text, "a\u{fffd}b");
    for i in 0..1100 {
        l.put(vec![rec("u", &format!("v{i}"), i)]).await.unwrap();
    }
    l.compact().await.unwrap();
    let text = fs::read_to_string(&path).unwrap();
    assert_eq!(text.lines().count(), 4, "{text}");
    assert!(text.starts_with("{\"uid\":\"partial\",\"text\":\"no number\"}\n"), "{text}");
    // A later line for it replaces it, as any other.
    l.put(vec![rec("partial", "whole now", 5)]).await.unwrap();
    for i in 0..1100 {
        l.put(vec![rec("u", &format!("w{i}"), i)]).await.unwrap();
    }
    l.compact().await.unwrap();
    let r = log(&path);
    r.load().await.unwrap();
    assert_eq!(r.get("partial").await.unwrap().text, "whole now");
    assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 4);
}
