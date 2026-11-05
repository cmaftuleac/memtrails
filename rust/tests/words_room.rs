//! The word-vector cache after a crash, the sentence room, and the recall log's trim.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memtrails::recall_log::{RecallBook, RecallEvent, RECALL_QUERY_CHARS};
use memtrails::room::{sentence_room, sentence_spans, RoomItem};
use memtrails::words::{PackedWords, WordEncoder, WordStore, WordVectors};
use serde_json::Value;

fn packed(seed: u8) -> PackedWords {
    PackedWords { words: 2, dims: 16, bits: vec![seed, seed + 1, seed + 2, seed + 3] }
}

#[tokio::test]
async fn a_cut_off_entry_goes_so_what_is_added_after_it_loads_on_the_next_start() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("words.bin");
    WordStore::new(Some(file.clone())).put(vec![("first".into(), packed(1))]).await.unwrap();
    // Half an entry: a whole head, half its bits.
    let mut head = vec![0u8; 12];
    head[8..10].copy_from_slice(&2u16.to_le_bytes());
    head[10..12].copy_from_slice(&16u16.to_le_bytes());
    head.extend([9, 9]);
    use std::io::Write;
    std::fs::OpenOptions::new().append(true).open(&file).unwrap().write_all(&head).unwrap();
    let s = WordStore::new(Some(file.clone()));
    s.load().await.unwrap();
    assert_eq!(s.size(), 1);
    s.put(vec![("second".into(), packed(5))]).await.unwrap();
    let again = WordStore::new(Some(file));
    again.load().await.unwrap();
    assert_eq!(again.size(), 2);
    assert_eq!(*again.get("first").unwrap(), packed(1));
    assert_eq!(*again.get("second").unwrap(), packed(5));
}

const DIMS: usize = 16;

fn word_vector(word: &str) -> Vec<f32> {
    let mut h: u32 = 2_166_136_261;
    for c in word.to_lowercase().chars() {
        h = (h ^ c as u32).wrapping_mul(16_777_619);
    }
    (0..DIMS)
        .map(|_| {
            h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
            (if h & 1 == 1 { 1.0 } else { -1.0 }) / (DIMS as f32).sqrt()
        })
        .collect()
}

fn encode() -> WordEncoder {
    Arc::new(|texts: Vec<String>, _| {
        Box::pin(async move {
            Ok(texts
                .iter()
                .map(|t| {
                    let words: Vec<&str> = t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
                    WordVectors {
                        words: words.len(),
                        dims: DIMS,
                        data: words.iter().flat_map(|w| word_vector(w)).collect(),
                    }
                })
                .collect())
        })
    })
}

#[test]
fn sentences_end_at_any_scripts_marks_and_at_line_breaks_never_inside_a_decimal() {
    let text = "Version 2.5 shipped today. Did it work? Yes!\nNext line here。下一句在这里。";
    let got: Vec<&str> = sentence_spans(text).into_iter().map(|(a, b)| memtrails_slice(text, a, b)).collect();
    assert_eq!(got, ["Version 2.5 shipped today.", "Did it work?", "Next line here。下一句在这里。"]);
    assert!(sentence_spans(&"word ".repeat(300)).len() > 2);
}

/// `text.slice(a, b)` in UTF-16 units.
fn memtrails_slice(text: &str, a: usize, b: usize) -> &str {
    let mut units = 0;
    let (mut start, mut end) = (text.len(), text.len());
    for (i, c) in text.char_indices() {
        if units == a {
            start = i;
        }
        if units == b {
            end = i;
            break;
        }
        units += c.len_utf16();
    }
    &text[start.min(end)..end]
}

#[tokio::test]
async fn the_room_keeps_the_sentence_that_answers_from_deep_in_a_memory() {
    let filler = "We talked about many different and unrelated things for a long while that afternoon. ";
    let items = vec![
        RoomItem {
            id: "a".into(),
            head: "(1 Mar)".into(),
            ts: 2.0,
            text: format!("{}The proxy certificate expired because the renewal cron was removed.", filler.repeat(6)),
        },
        RoomItem { id: "b".into(), head: "(2 Mar)".into(), ts: 1.0, text: filler.repeat(8) },
    ];
    let room = sentence_room("why did the proxy certificate expire", &items, &encode(), &WordStore::new(None), 160, 30)
        .await
        .unwrap();
    assert!(room.contains("certificate expired"), "{room}");
    assert!(memtrails::js::len16(&room) <= 160);
    assert!(room.starts_with("(1 Mar)"));
}

#[tokio::test]
async fn the_room_shows_each_memory_once_oldest_first_its_sentences_in_their_order() {
    let items = vec![
        RoomItem {
            id: "new".into(),
            head: "(new)".into(),
            ts: 9.0,
            text: "Ana likes the Italian place. Ana goes there on Fridays.".into(),
        },
        RoomItem {
            id: "old".into(),
            head: "(old)".into(),
            ts: 1.0,
            text: "Ana once mentioned the Italian place near the office.".into(),
        },
    ];
    let room = sentence_room("where does Ana eat on Fridays", &items, &encode(), &WordStore::new(None), 1000, 30)
        .await
        .unwrap();
    assert_eq!(
        room.split('\n').collect::<Vec<_>>(),
        [
            "(old) Ana once mentioned the Italian place near the office.",
            "(new) Ana likes the Italian place. … Ana goes there on Fridays."
        ]
    );
}

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[tokio::test]
async fn the_recall_log_keeps_a_long_query_up_to_its_start_and_stays_private() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recall.jsonl");
    let book = RecallBook::with(&path, Arc::new(|| 1_800_000_000_000.0), 1e12, 1 << 20);
    book.record(&RecallEvent::Search { project: None, query: "x".repeat(RECALL_QUERY_CHARS + 100), shown: vec![] });
    book.record(&RecallEvent::Open { item: "mem:1".into() });
    book.settled().await;
    let l = lines(&path);
    assert_eq!(l[0]["query"].as_str().unwrap().len(), RECALL_QUERY_CHARS);
    assert_eq!(l[1], serde_json::json!({ "t": 1_800_000_000_000u64, "ev": "open", "item": "mem:1" }));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }
}

#[tokio::test]
async fn the_recall_log_is_trimmed_to_its_last_days_and_newest_bytes_losing_nothing_appended_meanwhile() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recall.jsonl");
    let now = 1_800_000_000_000u64;
    let clock = Arc::new(AtomicU64::new(now - 200 * 86_400_000));
    let c = clock.clone();
    let book = RecallBook::with(&path, Arc::new(move || c.load(Ordering::SeqCst) as f64), 180.0 * 86_400_000.0, 400);
    book.record(&RecallEvent::Open { item: "mem:ancient".into() });
    clock.store(now, Ordering::SeqCst);
    for i in 0..6 {
        book.record(&RecallEvent::Search {
            project: None,
            query: format!("q{i} {}", "x".repeat(60)),
            shown: vec![format!("mem:{i}")],
        });
    }
    let trimming = book.trim();
    book.record(&RecallEvent::Open { item: "mem:after".into() });
    assert!(trimming.await >= 2);
    book.settled().await;
    let l = lines(&path);
    assert!(l.iter().all(|x| x["item"] != "mem:ancient"));
    assert_eq!(l.last().unwrap()["item"], "mem:after");
    assert!(l[l.len() - 2]["query"].as_str().unwrap().starts_with("q5 "));
    assert!(std::fs::metadata(&path).unwrap().len() <= 600);
    book.trim().await;
    assert!(std::fs::metadata(&path).unwrap().len() <= 400);
    assert_eq!(book.trim().await, 0);
}

#[tokio::test]
async fn the_recall_log_is_trimmed_even_with_a_line_that_is_not_utf8() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recall.jsonl");
    let now = 1_800_000_000_000.0;
    let mut raw = format!("{{\"t\":{},\"ev\":\"open\",\"item\":\"old\"}}\n", now - 400.0 * 86_400_000.0).into_bytes();
    raw.extend_from_slice(format!("{{\"t\":{now},\"ev\":\"search\",\"query\":\"caf").as_bytes());
    raw.extend_from_slice(b"\xe9\"}\n");
    raw.extend_from_slice(format!("{{\"t\":\"{now}\",\"ev\":\"open\",\"item\":\"as text\"}}\n").as_bytes());
    std::fs::write(&path, raw).unwrap();
    let book = RecallBook::with(&path, Arc::new(move || now), 180.0 * 86_400_000.0, 1 << 20);
    assert_eq!(book.trim().await, 1);
    let text = String::from_utf8(std::fs::read(&path).unwrap()).unwrap();
    assert!(!text.contains("\"old\"") && text.contains("caf\u{fffd}") && text.contains("as text"), "{text}");
}
