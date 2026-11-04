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

fn uids(l: &JsonlLog<Rec, i64>) -> Vec<String> {
    l.all().map(|r| r.uid).collect()
}

fn append(path: &std::path::Path, text: &str) {
    fs::OpenOptions::new().append(true).open(path).unwrap().write_all(text.as_bytes()).unwrap();
}

#[test]
fn reads_each_record_back_by_offset_latest_line_winning_across_a_reload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let a = log(&path);
    a.put(&[rec("a", "первый — ünïcödé", 1), rec("b", "second", 2)]).unwrap();
    a.put(&[rec("a", "replaced", 3)]).unwrap();
    assert_eq!(a.get("a").unwrap().text, "replaced");
    assert_eq!(a.get("b").unwrap().text, "second");
    assert_eq!(a.meta("a"), Some(3));
    let b = log(&path);
    b.load().unwrap();
    assert_eq!(b.size(), 2);
    assert_eq!(b.get("a").unwrap().text, "replaced");
    assert_eq!(uids(&b), ["b", "a"]);
}

#[test]
fn compacts_once_twice_what_is_live_offsets_still_right() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    for i in 0..1200 {
        l.put(&[rec(&format!("u{}", i % 3), &format!("v{i} é"), i)]).unwrap();
    }
    l.compact().unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap().trim().split('\n').count(), 3);
    assert_eq!(l.get("u0").unwrap().text, "v1197 é");
    l.put(&[rec("u3", "after", 0)]).unwrap();
    assert_eq!(l.get("u3").unwrap().text, "after");
    assert_eq!(l.get("u2").unwrap().text, "v1199 é");
}

#[test]
fn removes_and_stays_removed_after_a_reload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    l.put(&[rec("a", "one", 1), rec("b", "two", 2)]).unwrap();
    l.remove(&["a".into(), "nope".into()]).unwrap();
    assert!(l.get("a").is_none());
    assert!(fs::read_to_string(&path).unwrap().contains(r#"{"uid":"a","removed":true}"#));
    let r = log(&path);
    r.load().unwrap();
    assert_eq!(uids(&r), ["b"]);
}

#[test]
fn drops_a_cut_off_last_line_and_keeps_what_comes_after_through_reopen_and_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    log(&path).put(&[rec("a", "ok é", 1)]).unwrap();
    append(&path, r#"{"uid":"b","text":"torn"#);
    let l = log(&path);
    l.load().unwrap();
    l.put(&[rec("c", "after the crash", 2), rec("d", "and this", 3)]).unwrap();
    assert_eq!(l.get("c").unwrap().text, "after the crash");
    assert_eq!(l.get("d").unwrap().text, "and this");
    assert!(!fs::read_to_string(&path).unwrap().contains("torn"));
    let r = log(&path);
    r.load().unwrap();
    assert_eq!(uids(&r), ["a", "c", "d"]);
    for i in 0..1100 {
        r.put(&[rec("a", &format!("v{i}"), i)]).unwrap();
    }
    r.compact().unwrap();
    let s = log(&path);
    s.load().unwrap();
    let got: Vec<(String, String)> = s.all().map(|x| (x.uid, x.text)).collect();
    assert_eq!(
        got,
        [("c".into(), "after the crash".into()), ("d".into(), "and this".into()), ("a".into(), "v1099".into())]
    );
}

#[test]
fn keeps_a_last_line_whole_but_for_its_newline() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    fs::write(&path, "{\"uid\":\"a\",\"text\":\"one\",\"n\":1}\n{\"uid\":\"b\",\"text\":\"two\",\"n\":2}").unwrap();
    let l = log(&path);
    l.load().unwrap();
    l.put(&[rec("c", "three", 3)]).unwrap();
    assert_eq!(l.get("b").unwrap().text, "two");
    assert_eq!(l.get("c").unwrap().text, "three");
    let r = log(&path);
    r.load().unwrap();
    assert_eq!(uids(&r), ["a", "b", "c"]);
}

#[test]
fn refuses_to_compact_when_a_record_cannot_be_read_leaving_the_file_as_it_was() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    l.put(&[rec("keep", "precious", 0)]).unwrap();
    for i in 0..1100 {
        l.put(&[rec("u", &format!("v{i}"), i)]).unwrap();
    }
    // Damaged in place (same length): the index still points at it.
    let before = fs::read_to_string(&path).unwrap().replacen("precious", "prec\"ous", 1);
    fs::write(&path, &before).unwrap();
    assert!(l.get("keep").is_none());
    let err = l.compact().unwrap_err().to_string();
    assert!(err.contains("keep"), "{err}");
    assert_eq!(fs::read_to_string(&path).unwrap(), before);
    assert_eq!(l.get("u").unwrap().text, "v1099");
}

#[test]
fn compacts_a_log_larger_than_one_write_every_offset_right_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.jsonl");
    let l = log(&path);
    let big = "ü".repeat(3000);
    let live = 800;
    for round in 0..4 {
        let batch: Vec<Rec> = (0..live).map(|i| rec(&format!("u{i}"), &format!("{round}:{i}:{big}"), round)).collect();
        l.put(&batch).unwrap();
    }
    l.compact().unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap().trim().split('\n').count(), live);
    assert_eq!(l.get("u799").unwrap().text, format!("3:799:{big}"));
    l.put(&[rec("new", "after", 9)]).unwrap();
    let r = log(&path);
    r.load().unwrap();
    assert_eq!(r.size(), live + 1);
    for i in 0..live {
        assert_eq!(r.get(&format!("u{i}")).unwrap().text, format!("3:{i}:{big}"));
    }
    assert_eq!(r.get("new").unwrap().text, "after");
}

#[test]
fn usage_counts_what_was_shown_and_opened_and_keeps_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("usage.json");
    let u = UsageBook::new(&path);
    u.note(&["m1".into(), "m2".into()], false, 100.0);
    u.note(&["m1".into()], true, 200.0);
    u.flush();
    let again = UsageBook::new(&path);
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
