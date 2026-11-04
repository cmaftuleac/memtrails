//! The answer key every port is held to: `conformance/fixtures/*.json`, written by the TypeScript reference
//! (`conformance/generate.ts`). Every case must come out the same here (floating point to 1e-6).

use std::fs;
use std::path::PathBuf;

use memtrails::episodes::{calendar_of, episode_body, render_for_extraction};
use memtrails::qdrant_client::point_id;
use memtrails::recall_types::RecalledItem;
use memtrails::records::{
    collection_key, conversation_entries, entry_of, invalid_record, record_uid, time_of, At, Conversation, MemoryRecord,
};
use memtrails::redact::redact_string;
use memtrails::room::sentence_spans;
use memtrails::store::{fingerprint, FpPart};
use memtrails::text::{children_with, concept_key, cut, doc_sparse, first_words, fold, query_sparse, term_id, tokens};
use memtrails::tools::{lasted_text, readable_date, search_text, Order};
use memtrails::words::{max_sim, pack_words, PackedWords, WordVectors};
use memtrails::writer::{date_span, raw_key, stm_title};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/fixtures").join(format!("{name}.json"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture is JSON")
}

fn s(v: &Value) -> &str {
    v.as_str().expect("a string")
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().expect("an array").iter().map(|x| s(x).to_string()).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1.0)
}

fn sparse_same(got: &memtrails::text::SparseVector, want: &Value) -> bool {
    let idx: Vec<u32> = want["indices"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let vals: Vec<f64> = want["values"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    got.indices == idx && got.values.len() == vals.len() && got.values.iter().zip(&vals).all(|(a, b)| close(*a, *b))
}

/// Where the ICU4X word segmenter splits differently from the ICU4C one Node.js carries: the text's start and the
/// fields that differ — the known differences, said in rust/README.md. Everything else must match exactly.
const KNOWN_SEGMENTATION_DIFFERENCES: &[(&str, &[&str])] = &[
    // ICU4X's Japanese dictionary splits "がいっぱい" as "がい" + "っぱい"; ICU4C as "が" + "いっぱい".
    ("データベースのバックアップは失敗しました", &["tokens", "docSparse"]),
    // Folded Thai (sara am split in two by NFKD): ICU4X "การ" + "สํ" + "า", ICU4C "กา" + "รสํา".
    ("การสำรองข้อมูลล้มเหลว", &["tokens", "docSparse"]),
];

fn differs(text: &str, field: &str) -> bool {
    KNOWN_SEGMENTATION_DIFFERENCES.iter().any(|(t, fields)| text.starts_with(t) && fields.contains(&field))
}

#[test]
fn text() {
    let f = fixture("text");
    for case in f["cases"].as_array().unwrap() {
        let t = s(&case["text"]);
        assert_eq!(fold(t), s(&case["fold"]), "fold {t:?}");
        assert_eq!(cut(t, 20), s(&case["cut20"]), "cut20 {t:?}");
        assert_eq!(cut(t, 1), s(&case["cut1"]), "cut1 {t:?}");
        assert_eq!(concept_key(t), s(&case["conceptKey"]), "conceptKey {t:?}");
        let check = |field: &str, same: bool| {
            if differs(t, field) {
                assert!(!same, "{field} {t:?} now matches: take it off the known differences");
            } else {
                assert!(same, "{field} {t:?}");
            }
        };
        check("tokens", tokens(t) == strings(&case["tokens"]));
        check("firstWords", first_words(t, 5) == s(&case["firstWords5"]));
        check("children", children_with(t, 12, 9) == strings(&case["children"]));
        check("docSparse", sparse_same(&doc_sparse(t), &case["docSparse"]));
    }
    for c in f["termIds"].as_array().unwrap() {
        assert_eq!(u64::from(term_id(s(&c["term"]))), c["id"].as_u64().unwrap(), "termId {}", c["term"]);
    }
    assert!(sparse_same(&query_sparse(&strings(&f["querySparse"]["texts"])), &f["querySparse"]["vector"]));
    for c in f["conceptKeys"].as_array().unwrap() {
        assert_eq!(concept_key(s(&c["name"])), s(&c["key"]));
    }
    for c in f["sentenceSpans"].as_array().unwrap() {
        let want: Vec<(usize, usize)> = c["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize))
            .collect();
        assert_eq!(sentence_spans(s(&c["text"])), want, "sentenceSpans {}", c["text"]);
    }
}

#[test]
fn redact() {
    for c in fixture("redact").as_array().unwrap() {
        assert_eq!(redact_string(s(&c["input"])), s(&c["output"]), "{}", c["input"]);
    }
}

#[test]
fn ids() {
    let f = fixture("ids");
    for c in f["fingerprint"].as_array().unwrap() {
        let raw = c["parts"].as_array().unwrap();
        let lists: Vec<Option<Vec<String>>> = raw.iter().map(|p| p.as_array().map(|_| strings(p))).collect();
        let fp: Vec<FpPart> = raw
            .iter()
            .zip(&lists)
            .map(|(p, l)| match (p, l) {
                (_, Some(l)) => FpPart::List(l),
                (Value::String(x), _) => FpPart::Text(x),
                _ => FpPart::Null,
            })
            .collect();
        assert_eq!(fingerprint(&fp), s(&c["fp"]), "{}", c["parts"]);
    }
    for c in f["pointId"].as_array().unwrap() {
        assert_eq!(point_id(s(&c["item"]), c["n"].as_i64().unwrap()), s(&c["id"]));
    }
    for c in f["recordUid"].as_array().unwrap() {
        assert_eq!(record_uid(s(&c["record"]["id"]), c["record"]["source"].as_str()), s(&c["uid"]));
    }
    for c in f["collectionKey"].as_array().unwrap() {
        assert_eq!(collection_key(c["collection"].as_str()), s(&c["key"]));
    }
}

fn at_of(v: &Value) -> Option<At> {
    match v {
        Value::Number(n) => Some(At::Ms(n.as_f64().unwrap())),
        Value::String(x) => Some(At::Text(x.clone())),
        _ => None,
    }
}

#[test]
fn dates() {
    let f = fixture("dates");
    for c in f["dateSpan"].as_array().unwrap() {
        let got = date_span(s(&c["date"]));
        match c["span"].as_object() {
            None => assert!(got.is_none(), "{}", c["date"]),
            Some(span) => {
                let got = got.unwrap_or_else(|| panic!("{} is a date", c["date"]));
                assert_eq!(got.from, span["from"].as_f64().unwrap());
                assert_eq!(got.to, span["to"].as_f64().unwrap());
            }
        }
    }
    for c in f["readableDate"].as_array().unwrap() {
        assert_eq!(readable_date(s(&c["date"])), s(&c["text"]));
    }
    for c in f["lastedText"].as_array().unwrap() {
        assert_eq!(lasted_text(c["hit"]["startAt"].as_str(), c["hit"]["endAt"].as_str()), s(&c["text"]));
    }
    for c in f["timeOf"].as_array().unwrap() {
        assert_eq!(time_of(at_of(&c["at"]).as_ref()), c["ms"].as_f64(), "{}", c["at"]);
    }
    let times: Vec<f64> = f["calendarOf"]["times"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(calendar_of(&times), s(&f["calendarOf"]["text"]));
}

#[test]
fn records() {
    let f = fixture("records");
    let now = f["now"].as_f64().unwrap();
    for c in f["invalid"].as_array().unwrap() {
        let r: MemoryRecord = serde_json::from_value(c["record"].clone()).unwrap();
        assert_eq!(invalid_record(&r).as_deref(), c["error"].as_str(), "{}", c["record"]);
    }
    let mut entries = Vec::new();
    for c in f["entries"].as_array().unwrap() {
        let r: MemoryRecord = serde_json::from_value(c["record"].clone()).unwrap();
        let e = entry_of(&r, now, c["turn"].as_u64().unwrap() as usize);
        assert_eq!(serde_json::to_value(&e).unwrap(), c["entry"], "{}", c["record"]);
        entries.push(e);
    }
    let conv: Conversation = serde_json::from_value(f["conversation"]["input"].clone()).unwrap();
    let conv = conversation_entries(&conv, now);
    assert_eq!(serde_json::to_value(&conv).unwrap(), f["conversation"]["entries"]);
    let all: Vec<_> = entries.iter().chain(conv.iter()).collect();
    for (n, (e, want)) in all.iter().zip(f["rendered"].as_array().unwrap()).enumerate() {
        assert_eq!(e.uid, s(&want["uid"]));
        assert_eq!(raw_key(e), s(&want["rawKey"]));
        assert_eq!(stm_title(e), s(&want["stmTitle"]));
        assert_eq!(episode_body(e), s(&want["episodeBody"]));
        assert_eq!(render_for_extraction(e, n + 1), s(&want["extraction"]));
    }
}

fn vectors(v: &Value) -> WordVectors {
    WordVectors {
        words: v["words"].as_u64().unwrap() as usize,
        dims: v["dims"].as_u64().unwrap() as usize,
        data: v["data"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect(),
    }
}

#[test]
fn words() {
    let f = fixture("words");
    let q = vectors(&f["query"]);
    for d in f["docs"].as_array().unwrap() {
        let packed = pack_words(&vectors(&d["vectors"]));
        let want = PackedWords {
            words: d["packed"]["words"].as_u64().unwrap() as usize,
            dims: d["packed"]["dims"].as_u64().unwrap() as usize,
            bits: d["packed"]["bits"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect(),
        };
        assert_eq!(packed, want);
        assert!(close(max_sim(&q, &packed), d["maxSim"].as_f64().unwrap()));
    }
}

#[test]
fn search_texts() {
    let f = fixture("search-text");
    let hits: Vec<RecalledItem> = serde_json::from_value(f["hits"].clone()).unwrap();
    assert_eq!(search_text(&hits, Order::Relevance), s(&f["relevance"]));
    assert_eq!(search_text(&hits, Order::Time), s(&f["time"]));
    let many: Vec<RecalledItem> =
        (0..22).map(|i| RecalledItem { id: format!("x:{i}"), ..hits[i % hits.len()].clone() }).collect();
    assert_eq!(search_text(&many, Order::Relevance), s(&f["compact"]));
}
