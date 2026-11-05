//! With the real model and a real Qdrant: the vectors the reference's local embedder gives for fixed texts
//! (`conformance/fixtures/python-embeddings.json`), and a memory opened, filled, searched and closed. Downloads the
//! model (218 MB) and Qdrant (~30 MB) into the cache on first run, so it is ignored by default:
//!
//!   cargo test --test integration -- --ignored

use std::path::PathBuf;

use memtrails::download::DownloadOptions;
use memtrails::local_embedder::{load_local_embedder, LocalEmbedderOptions};
use memtrails::model::ensure_model;
use memtrails::words::WordKind;
use memtrails::{At, Memory, MemoryOptions, MemoryRecord, SearchQuery};
use serde_json::Value;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/fixtures/python-embeddings.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The same ONNX Runtime build as the reference: the same numbers, to float32 rounding.
const CLOSE: f64 = 1e-6;

#[tokio::test]
#[ignore = "downloads the model; run with --ignored"]
async fn the_local_model_gives_the_references_vectors() {
    let dir = ensure_model(&DownloadOptions::default(), None).await.unwrap();
    let e = load_local_embedder(&dir, LocalEmbedderOptions::default()).await.unwrap();
    let f = fixture();
    let texts: Vec<String> = serde_json::from_value(f["texts"].clone()).unwrap();
    let got = e.embed(texts).await.unwrap();
    for (g, w) in got.iter().zip(f["vectors"].as_array().unwrap()) {
        let w: Vec<f64> = w.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(g.len(), w.len());
        let worst = g.iter().zip(&w).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
        assert!(worst < CLOSE, "a vector differs by {worst}");
    }
    let texts: Vec<String> = serde_json::from_value(f["words"]["texts"].clone()).unwrap();
    let words = e.words(texts, WordKind::Query).await.unwrap();
    for (g, w) in words.iter().zip(f["words"]["vectors"].as_array().unwrap()) {
        assert_eq!(g.words as u64, w["words"].as_u64().unwrap(), "words a text");
        for (k, head) in w["head"].as_array().unwrap().iter().enumerate() {
            for (d, x) in head.as_array().unwrap().iter().enumerate() {
                let diff = (f64::from(g.data[k * g.dims + d]) - x.as_f64().unwrap()).abs();
                assert!(diff < CLOSE, "word {k} number {d} differs by {diff}");
            }
        }
    }
    e.close().await;
}

#[tokio::test]
#[ignore = "downloads the model and Qdrant; run with --ignored"]
async fn a_memory_on_the_real_model_and_qdrant_finds_what_was_added() {
    let home = tempfile::tempdir().unwrap();
    let m = Memory::open(MemoryOptions::new(home.path().join("memory"))).await.unwrap();
    m.add(vec![
        MemoryRecord {
            kind: Some("email".into()),
            author: Some("Ana".into()),
            title: Some("Proxy certificate expired".into()),
            at: Some(At::from("2026-04-02")),
            ..MemoryRecord::new(
                "msg-1",
                "The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.",
            )
        },
        MemoryRecord {
            kind: Some("commit".into()),
            author: Some("Ion".into()),
            title: Some("Restore certificate renewal cron".into()),
            at: Some(At::from("2026-04-03")),
            ..MemoryRecord::new("c9f1e2", "Files: deploy/cron/renew.sh")
        },
        MemoryRecord::new("note-1", "Lunch on Friday is at the Italian place near the office, at one."),
    ])
    .await
    .unwrap();
    let hits = m.search(&SearchQuery::new("why did the certificate stop renewing?")).await.unwrap();
    assert_eq!(hits[0].record.as_ref().unwrap().id, "msg-1", "{hits:?}");
    let room = m.room(&SearchQuery::new("what made the renewal stop?"), Some(600)).await.unwrap();
    assert!(room.contains("cron job was removed"), "{room}");
    assert!(m.stats().words > 0);
    m.close().await.unwrap();
    // Opened again: the same index, nothing written again.
    let m = Memory::open(MemoryOptions::new(home.path().join("memory"))).await.unwrap();
    assert_eq!(m.search(&SearchQuery::new("lunch on Friday")).await.unwrap()[0].record.as_ref().unwrap().id, "note-1");
    m.close().await.unwrap();
}
