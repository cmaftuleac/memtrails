//! The importers against the reference: `conformance/fixtures/rust-importers.json`, written by
//! `conformance/generate-rust.ts` from the inputs in `conformance/inputs/importers`. The same bytes in must give the
//! same records out.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use futures::TryStreamExt;
use memtrails::importers::{
    decode_bytes, decode_encoded_words, decode_entities, from_eml, from_files, from_jsonl, from_mbox,
    from_slack_export, front_matter, html_title, html_to_text, slack_text, split_text, FilesOptions, JsonlOptions,
    MboxOptions, RecordStream, SlackOptions,
};
use memtrails::records::MemoryRecord;
use serde_json::{json, Value};

fn conformance() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance")
}

fn inputs() -> PathBuf {
    conformance().join("inputs/importers")
}

fn fixture() -> Value {
    let path = conformance().join("fixtures/rust-importers.json");
    serde_json::from_str(&fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

/// Whole numbers as integers, wherever they are: `1714557600000.0` and `1714557600000` are the same time.
fn norm(v: Value) -> Value {
    match v {
        Value::Number(n) => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && f.abs() < 9e15 => json!(f as i64),
            _ => Value::Number(n),
        },
        Value::Array(a) => Value::Array(a.into_iter().map(norm).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, norm(v))).collect()),
        other => other,
    }
}

async fn values(r: memtrails::Result<RecordStream>) -> Vec<Value> {
    let records: Vec<MemoryRecord> = r.expect("the source opens").try_collect().await.expect("every record reads");
    records.into_iter().map(|x| norm(serde_json::to_value(x).unwrap())).collect()
}

fn same(got: Vec<Value>, want: &Value, what: &str) {
    let want = want.as_array().expect("a list");
    assert_eq!(got.len(), want.len(), "{what}: {got:#?}");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(*g, norm(w.clone()), "{what}");
    }
}

#[tokio::test]
async fn jsonl_matches_the_reference() {
    let f = fixture();
    let skipped = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let seen = skipped.clone();
    let got = values(
        from_jsonl(
            inputs().join("records.jsonl"),
            JsonlOptions {
                source: Some("notes".into()),
                kind: Some("record".into()),
                on_skip: Some(std::sync::Arc::new(move |_l: &str, reason: &str, n: usize| {
                    let reason = if reason.starts_with("not JSON") { "not JSON" } else { reason };
                    seen.lock().unwrap().push(json!({ "line": n, "reason": reason }));
                })),
                ..Default::default()
            },
        )
        .await,
    )
    .await;
    same(got, &f["jsonl"]["records"], "jsonl");
    assert_eq!(Value::Array(skipped.lock().unwrap().clone()), f["jsonl"]["skipped"]);
}

#[tokio::test]
async fn files_match_the_reference() {
    let f = fixture();
    let dir = inputs().join("files");
    let skipped = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let seen = skipped.clone();
    let records: Vec<MemoryRecord> = from_files(
        &dir,
        FilesOptions {
            collection: Some("docs".into()),
            max_chars: Some(1500),
            on_skip: Some(std::sync::Arc::new(move |p: &str, r: &str| {
                seen.lock().unwrap().push(json!({ "path": p, "reason": r }))
            })),
            ..Default::default()
        },
    )
    .await
    .unwrap()
    .try_collect()
    .await
    .unwrap();
    let want = f["files"]["records"].as_array().unwrap();
    let mut roots: Vec<String> = Vec::new();
    let got: Vec<Value> = records
        .iter()
        .zip(want)
        .map(|(r, w)| {
            let mut v = norm(serde_json::to_value(r).unwrap());
            // The inputs' folder differs by machine: the URL after it must match, and the folder be one for all.
            let suffix = s(&w["url"]).strip_prefix("file://<root>").unwrap_or("");
            if let Some(root) = r.url.as_deref().and_then(|u| u.strip_suffix(suffix)) {
                roots.push(root.to_string());
                v["url"] = w["url"].clone();
            }
            let file = dir.join(r.thread.as_deref().unwrap_or(&r.id));
            let mtime = fs::metadata(&file)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| (d.as_secs_f64() * 1000.0 + 0.5).floor());
            if mtime.is_some() && v["at"].as_f64() == mtime {
                v["at"] = json!("<mtime>");
            }
            v
        })
        .collect();
    same(got, &f["files"]["records"], "files");
    roots.dedup();
    assert_eq!(roots.len(), 1, "{roots:?}");
    assert!(roots[0].starts_with("file:///") && roots[0].ends_with("/conformance/inputs/importers"), "{}", roots[0]);
    assert_eq!(Value::Array(skipped.lock().unwrap().clone()), f["files"]["skipped"]);
}

#[tokio::test]
async fn slack_matches_the_reference() {
    let f = fixture();
    let dir = inputs().join("slack");
    same(
        values(
            from_slack_export(
                &dir,
                SlackOptions { workspace_url: Some("https://example.slack.com/".into()), ..Default::default() },
            )
            .await,
        )
        .await,
        &f["slack"]["records"],
        "slack",
    );
    same(
        values(
            from_slack_export(
                &dir,
                SlackOptions {
                    channels: Some(vec!["#general".into(), "D1".into()]),
                    collection: Some("team".into()),
                    skip_bots: true,
                    source: Some("ws".into()),
                    ..Default::default()
                },
            )
            .await,
        )
        .await,
        &f["slack"]["filtered"],
        "slack filtered",
    );
}

#[tokio::test]
async fn mail_matches_the_reference() {
    let f = fixture();
    let mbox = inputs().join("mail/Inbox.mbox");
    let skipped = std::sync::Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
    let seen = skipped.clone();
    same(
        values(
            from_mbox(
                &mbox,
                MboxOptions {
                    on_skip: Some(std::sync::Arc::new(move |_r: &str, i: usize| seen.lock().unwrap().push(i))),
                    ..Default::default()
                },
            )
            .await,
        )
        .await,
        &f["mbox"]["records"],
        "mbox",
    );
    assert_eq!(json!(*skipped.lock().unwrap()), f["mbox"]["skipped"]);
    same(
        values(
            from_mbox(
                &mbox,
                MboxOptions {
                    keep_quoted: true,
                    keep_signature: true,
                    collection: Some("work".into()),
                    source: Some("box".into()),
                    ..Default::default()
                },
            )
            .await,
        )
        .await,
        &f["mbox"]["kept"],
        "mbox kept",
    );
    let mut eml = values(from_eml(inputs().join("mail/note.eml"), MboxOptions::default()).await).await;
    eml.extend(values(from_eml(inputs().join("mail/bare.eml"), MboxOptions::default()).await).await);
    same(eml, &f["eml"], "eml");
}

fn s(v: &Value) -> &str {
    v.as_str().expect("a string")
}

#[tokio::test]
async fn text_tools_match_the_reference() {
    let f = fixture();
    for c in f["html"].as_array().unwrap() {
        let html = s(&c["html"]);
        assert_eq!(html_to_text(html, false), s(&c["text"]), "{html:?}");
        assert_eq!(html_to_text(html, true), s(&c["dropQuotes"]), "{html:?}");
        assert_eq!(html_title(html).map(Value::String).unwrap_or(Value::Null), c["title"], "{html:?}");
    }
    for c in f["titles"].as_array().unwrap() {
        assert_eq!(html_title(s(&c["html"])).map(Value::String).unwrap_or(Value::Null), c["title"]);
    }
    for c in f["entities"].as_array().unwrap() {
        assert_eq!(decode_entities(s(&c["s"])), s(&c["text"]));
    }
    for c in f["bytes"].as_array().unwrap() {
        let bytes: Vec<u8> = c["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8).collect();
        assert_eq!(decode_bytes(&bytes, c["charset"].as_str()), s(&c["text"]), "{c}");
    }
    for c in f["split"].as_array().unwrap() {
        let parts: Vec<String> = c["parts"].as_array().unwrap().iter().map(|p| s(p).to_string()).collect();
        assert_eq!(split_text(s(&c["text"]), c["max"].as_u64().unwrap() as usize), parts);
    }
    for c in f["encodedWords"].as_array().unwrap() {
        assert_eq!(decode_encoded_words(s(&c["s"])), s(&c["text"]));
    }
    let users: HashMap<String, String> = [("U1".to_string(), "Ana".to_string())].into();
    let channels: HashMap<String, String> = [("C1".to_string(), "general".to_string())].into();
    for c in f["slackText"].as_array().unwrap() {
        assert_eq!(slack_text(s(&c["s"]), &users, &channels), s(&c["text"]));
    }
    for c in f["frontMatter"].as_array().unwrap() {
        let (fields, body) = front_matter(s(&c["s"]));
        assert_eq!(norm(serde_json::to_value(fields).unwrap()), norm(c["fields"].clone()), "{}", c["s"]);
        assert_eq!(body, s(&c["body"]));
    }
}
