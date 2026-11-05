//! The core held to the reference: every language-model ask built from fixed inputs — system text, prompt, schema,
//! room, wait and consequence, byte for byte — and what is kept of fixed answers, as `conformance/generate-rust.ts`
//! recorded them from the TypeScript library (`conformance/fixtures/rust-core.json`). And the agent tools: their JSON
//! Schemas, their checks and what they answer.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use memtrails::consolidate::{ConsolidateDeps, Consolidator, Pair};
use memtrails::embedder::EmbedFn;
use memtrails::extract::{
    extract_window, observe_window, parts_of, windows_of, ExtractOptions, Extraction, Read, WindowMeta,
};
use memtrails::gather::{memory_gather, GatherOptions, Gathered};
use memtrails::js;
use memtrails::llm::{Ask, LlmAnswer, LlmRequest};
use memtrails::profiles::{profile_subjects, write_profiles};
use memtrails::qdrant_client::{point_id, Endpoint, QdrantClient};
use memtrails::qdrant_process::QdrantEndpoint;
use memtrails::recall::{ExploreQuery, Explored, RecallDeps, RecalledItem, SearchQuery};
use memtrails::records::{entry_of, MemoryRecord};
use memtrails::store::{ltm_log, stm_log, MemoryItem, StmEntry};
use memtrails::tools::{build_agent_tools, ToolProvider};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/fixtures/rust-core.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A model that answers with the answers given, in order (the last again), and keeps every ask.
struct Recorder {
    answers: Vec<Value>,
    n: AtomicUsize,
    seen: Mutex<Vec<LlmRequest>>,
}

impl Recorder {
    fn new(answers: Vec<Value>) -> Arc<Self> {
        Arc::new(Recorder { answers, n: AtomicUsize::new(0), seen: Mutex::new(vec![]) })
    }

    fn seen(&self) -> Vec<LlmRequest> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl Ask for Recorder {
    async fn ask(&self, req: LlmRequest) -> memtrails::Result<LlmAnswer> {
        self.seen.lock().unwrap().push(req);
        let n = self.n.fetch_add(1, Ordering::SeqCst).min(self.answers.len() - 1);
        Ok(match &self.answers[n] {
            Value::String(s) if s == "unavailable" => LlmAnswer::Unavailable("the model is away".into()),
            a => LlmAnswer::Answer { answer: a.clone(), model: None },
        })
    }
}

/// Each ask as the fixture recorded it; a long prompt by its hash, length and start.
fn same_asks(got: &[LlmRequest], want: &Value) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len(), "how many asks");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.purpose, w["purpose"].as_str().unwrap());
        assert_eq!(g.system, w["system"].as_str().unwrap(), "system");
        match &w["prompt"] {
            Value::String(p) => assert_eq!(&g.prompt, p, "prompt"),
            p => {
                assert_eq!(
                    hex::encode(Sha256::digest(g.prompt.as_bytes())),
                    p["sha256"].as_str().unwrap(),
                    "prompt hash"
                );
                assert_eq!(js::len16(&g.prompt) as u64, p["length"].as_u64().unwrap());
            }
        }
        assert_eq!(js::stringify(&g.schema), js::stringify(&w["schema"]), "schema");
        assert_eq!(g.name.as_deref(), w["name"].as_str());
        assert_eq!(g.max_tokens, w["maxTokens"].as_u64(), "max tokens");
        assert_eq!(g.timeout_ms, w["timeoutMs"].as_u64());
        assert_eq!(g.consequence.as_deref(), w["consequence"].as_str());
    }
}

fn extraction_json(x: &Read<Extraction>) -> Value {
    let x = match x {
        Read::Read(x) => x,
        Read::Unavailable(w) => return json!({ "unavailable": w }),
        Read::Busy => return json!({ "busy": true }),
    };
    let candidates: Vec<Value> = x
        .candidates
        .iter()
        .map(|c| {
            json!({ "kind": c.kind, "title": c.title, "summary": c.summary, "body": c.body, "appliesWhen": c.applies_when, "scope": c.scope,
                "provenance": c.provenance, "confidence": c.confidence, "importance": js::ms_value(c.importance), "cues": c.cues, "concepts": c.concepts })
        })
        .collect();
    let episodes: Vec<Value> = x
        .episodes
        .iter()
        .map(|e| {
            let mut v =
                json!({ "uid": e.uid, "title": e.title, "summary": e.summary, "cues": e.cues, "concepts": e.concepts });
            if let Some(o) = &e.outcome {
                v["outcome"] = json!(o);
            }
            v
        })
        .collect();
    json!({ "summary": x.summary, "candidates": candidates, "episodes": episodes, "observations": observations_json(&x.observations), "dropped": x.dropped })
}

fn observations_json(os: &[memtrails::extract::Observation]) -> Vec<Value> {
    os.iter()
        .map(|o| {
            let mut v = json!({ "text": o.text, "about": o.about, "eventAt": o.event_at });
            if let Some(s) = &o.start_at {
                v["startAt"] = json!(s);
            }
            if let Some(e) = &o.end_at {
                v["endAt"] = json!(e);
            }
            v["phase"] = json!(o.phase);
            v["provenance"] = json!(o.provenance);
            v["concepts"] = json!(o.concepts);
            v["cues"] = json!(o.cues);
            v
        })
        .collect()
}

fn entries(v: &Value) -> Vec<StmEntry> {
    serde_json::from_value(v.clone()).unwrap()
}

fn meta(v: &Value) -> WindowMeta {
    WindowMeta {
        cwd: v["cwd"].as_str().map(str::to_string),
        engine: v["engine"].as_str().unwrap().into(),
        title: v["title"].as_str().unwrap().into(),
    }
}

fn answers(v: &Value) -> Vec<Value> {
    v.as_array().unwrap().clone()
}

#[tokio::test]
async fn extraction_asks_and_keeps_as_the_reference_does() {
    let f = fixture();
    let win = entries(&f["window"]);
    let conv = entries(&f["conversationEntries"]);
    let m = meta(&f["meta"]);
    let a = Recorder::new(vec![f["extract"]["answer"].clone()]);
    let known = Some(vec!["(2026-03-01) The old rack was rack three.".to_string()]);
    let x =
        extract_window(&win, &m, &ExtractOptions { ask: Some(a.as_ref()), observations: true, known }).await.unwrap();
    same_asks(&a.seen(), &f["extract"]["asked"]);
    assert_eq!(extraction_json(&x), f["extract"]["result"]);

    let both: Vec<StmEntry> = win.iter().chain(conv.iter()).cloned().collect();
    let a = Recorder::new(vec![json!({ "segment_summary": "", "episodes": [], "items": [] })]);
    let mixed = WindowMeta { cwd: Some("/srv".into()), engine: "email".into(), title: "Mixed".into() };
    extract_window(&both, &mixed, &ExtractOptions { ask: Some(a.as_ref()), observations: false, known: None })
        .await
        .unwrap();
    same_asks(&a.seen(), &f["extract"]["withoutObservations"]);

    let a = Recorder::new(vec![json!("unavailable")]);
    let x = extract_window(&win, &m, &ExtractOptions { ask: Some(a.as_ref()), observations: true, known: None })
        .await
        .unwrap();
    assert_eq!(a.seen().len() as u64, f["extract"]["unavailable"]["asked"].as_u64().unwrap());
    assert_eq!(extraction_json(&x), f["extract"]["unavailable"]["result"]);
}

#[tokio::test]
async fn a_request_too_long_for_a_window_is_read_in_parts_each_folded_into_the_last() {
    let f = fixture();
    let lines = f["long"]["lines"].as_u64().unwrap();
    let text = (0..lines)
        .map(|i| {
            format!(
                "Step {i}: check the rack {}, the cable {} and the label on disk {} before moving on.",
                i % 9,
                i % 5,
                i % 13
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let record = MemoryRecord {
        kind: Some("document".into()),
        title: Some("Runbook".into()),
        at: Some("2026-03-20T00:00:00Z".into()),
        ..MemoryRecord::new("long-1", text)
    };
    let long = entry_of(&record, f["now"].as_f64().unwrap(), 0);
    assert_eq!(long.uid, f["long"]["uid"].as_str().unwrap());
    let a = Recorder::new(answers(&f["long"]["answers"]));
    let x = extract_window(
        &[long],
        &meta(&f["meta"]),
        &ExtractOptions { ask: Some(a.as_ref()), observations: true, known: None },
    )
    .await
    .unwrap();
    same_asks(&a.seen(), &f["long"]["asked"]);
    let mut got = extraction_json(&x);
    // The reference builds this result with its fields in another order; they are compared as values.
    got.as_object_mut().unwrap().sort_keys();
    let mut want = f["long"]["result"].clone();
    want.as_object_mut().unwrap().sort_keys();
    assert_eq!(got, want);
}

#[tokio::test]
async fn windows_and_parts_as_the_reference_cuts_them() {
    let f = fixture();
    let both: Vec<StmEntry> = entries(&f["window"]).into_iter().chain(entries(&f["conversationEntries"])).collect();
    let of: Vec<Vec<String>> =
        windows_of(&both, 300).iter().map(|w| w.iter().map(|e| e.uid.clone()).collect()).collect();
    assert_eq!(json!(of), f["windows"]["of"]);
    let text = format!("a\n{}\nccc\n\ndd", "b".repeat(25));
    assert_eq!(json!(parts_of(&text, 10)), f["windows"]["parts"]);
}

#[tokio::test]
async fn observations_are_read_apart_with_what_came_just_before() {
    let f = fixture();
    let win = entries(&f["window"]);
    let conv = entries(&f["conversationEntries"]);
    let cat = WindowMeta { cwd: None, engine: "conversation".into(), title: "Cat".into() };
    for (key, before, chars) in [("observe", &win[..2], 300), ("one", &win[..], 7)] {
        let a = Recorder::new(vec![f["observe"]["answer"].clone()]);
        let got = observe_window(&conv, before, &cat, Some(a.as_ref()), chars, None).await.unwrap();
        let (asked, result) = if key == "observe" {
            (&f["observe"]["asked"], &f["observe"]["result"])
        } else {
            (&f["observe"]["one"]["asked"], &f["observe"]["one"]["result"])
        };
        same_asks(&a.seen(), asked);
        let Read::Read(os) = got else { panic!("read") };
        assert_eq!(json!({ "observations": observations_json(&os) }), *result);
    }
}

fn nowhere() -> QdrantClient {
    QdrantClient::new(
        Endpoint::Fixed(QdrantEndpoint { url: "http://127.0.0.1:9".into(), api_key: String::new() }),
        None,
    )
}

fn zero_embed() -> EmbedFn {
    Arc::new(|texts: Vec<String>, _| Box::pin(async move { Ok(texts.iter().map(|_| vec![0.0f32; 768]).collect()) }))
}

fn deps(client: QdrantClient, ask: Arc<dyn Ask>, dir: &std::path::Path) -> ConsolidateDeps {
    ConsolidateDeps::new(
        client,
        zero_embed(),
        stm_log(dir.join("records.jsonl")),
        ltm_log(dir.join("knowledge.jsonl")),
        ask,
    )
}

fn items(v: &Value) -> Vec<MemoryItem> {
    serde_json::from_value(v.clone()).unwrap()
}

#[tokio::test]
async fn reconciling_asks_about_pairs_and_reads_the_labels_back() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let a = Recorder::new(vec![f["relate"]["answer"].clone()]);
    let worker = Consolidator::new(deps(nowhere(), a.clone(), dir.path()));
    let pairs: Vec<Pair> = f["relate"]["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Pair {
            it: serde_json::from_value(p["it"].clone()).unwrap(),
            known: serde_json::from_value(p["known"].clone()).unwrap(),
        })
        .collect();
    let decided = worker.relate(&pairs).await.unwrap().unwrap();
    same_asks(&a.seen(), &f["relate"]["asked"]);
    let got: serde_json::Map<String, Value> = decided
        .into_iter()
        .map(|(k, d)| (k, json!({ "relation": d.relation, "title": d.title, "summary": d.summary, "body": d.body })))
        .collect();
    assert_eq!(Value::Object(got), f["relate"]["decided"]);
}

#[tokio::test]
async fn cues_are_asked_for_what_has_none_and_kept_as_the_reference_keeps_them() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let a = Recorder::new(vec![f["cues"]["answer"].clone()]);
    let worker = Consolidator::new(deps(nowhere(), a.clone(), dir.path()));
    let got = worker.cue_batch(&items(&f["cues"]["items"])).await.unwrap().unwrap();
    same_asks(&a.seen(), &f["cues"]["asked"]);
    assert_eq!(serde_json::to_value(&got).unwrap(), f["cues"]["result"]);
}

#[tokio::test]
async fn profiles_of_the_subjects_enough_items_share() {
    let f = fixture();
    let all = items(&f["profiles"]["items"]);
    let mut sizes = IndexMap::new();
    sizes.insert("home".to_string(), 35usize);
    let subjects: Vec<Value> = profile_subjects(&all, &sizes)
        .iter()
        .map(|s| json!({ "uid": s.uid, "concept": s.concept, "scope": s.scope, "project": s.project, "read": s.read.iter().map(|it| it.uid.clone()).collect::<Vec<_>>(), "fingerprint": s.fingerprint }))
        .collect();
    assert_eq!(json!(subjects), f["profiles"]["subjects"]);

    let accepting = common::serve(|_| common::Response::json(200, &json!({ "result": true }))).await;
    let client =
        QdrantClient::new(Endpoint::Fixed(QdrantEndpoint { url: accepting.url.clone(), api_key: String::new() }), None);
    let dir = tempfile::tempdir().unwrap();
    let a = Recorder::new(
        answers(&f["profiles"]["asked"])
            .iter()
            .map(|_| json!({ "title": "  Miso the cat  ", "facts": ["  2026-03: adopted ", "", 7, "vet booked"] }))
            .collect(),
    );
    let mut d = deps(client, a.clone(), dir.path());
    let now = f["now"].as_f64().unwrap();
    d.now = Arc::new(move || now);
    d.ltm.put(all).await.unwrap();
    let r = write_profiles(&d, 400).await.unwrap();
    let mut asked = a.seen();
    let mut want: Vec<Value> = answers(&f["profiles"]["asked"]);
    // Asked at once: compared in the order of their prompts.
    asked.sort_by(|x, y| x.prompt.cmp(&y.prompt));
    want.sort_by(|x, y| x["prompt"].as_str().cmp(&y["prompt"].as_str()));
    same_asks(&asked, &json!(want));
    assert_eq!(json!({ "profiled": r.profiled, "unprofiled": r.unprofiled }), f["profiles"]["report"]);
    let mut written: Vec<Value> = d
        .ltm
        .all()
        .await
        .into_iter()
        .filter(|it| it.kind == "profile")
        .map(|it| {
            let mut v = serde_json::to_value(MemoryItem { learned_at: None, ..it }).unwrap();
            v["learnedAt"] = Value::Null;
            v
        })
        .collect();
    let mut want: Vec<Value> = answers(&f["profiles"]["written"]);
    written.sort_by(|x, y| x["uid"].as_str().cmp(&y["uid"].as_str()));
    want.sort_by(|x, y| x["uid"].as_str().cmp(&y["uid"].as_str()));
    assert_eq!(written, want);
}

/// A stand-in index for gathering: the first search's observations and other memories, then what its own searches
/// find — as the generator's stand-in gave them.
async fn gather_index(f: &Value) -> common::Server {
    let obs = answers(&f["gather"]["obsHits"]);
    let rest = answers(&f["gather"]["restHits"]);
    let extra = answers(&f["gather"]["extraHits"]);
    let payloads: IndexMap<String, Value> =
        obs.iter().chain(&rest).chain(&extra).map(|h| (point_id(h["item"].as_str().unwrap(), 0), h.clone())).collect();
    let rest_calls = AtomicUsize::new(0);
    common::serve(move |r| {
        let body = r.json();
        if r.path.ends_with("/points/query/groups") {
            let is_obs = body["filter"]["must"].as_array().into_iter().flatten().any(|c| c["key"] == "kind");
            let list: Vec<Value> = if is_obs {
                obs.clone()
            } else if rest_calls.fetch_add(1, Ordering::SeqCst) >= 3 {
                rest.iter().chain(&extra).cloned().collect()
            } else {
                rest.clone()
            };
            let groups: Vec<Value> = list
                .iter()
                .enumerate()
                .map(|(i, h)| json!({ "id": h["item"], "hits": [{ "id": point_id(h["item"].as_str().unwrap(), 0), "score": 1.0 - i as f64 / 10.0 }] }))
                .collect();
            return common::Response::json(200, &json!({ "result": { "groups": groups } }));
        }
        if r.path.ends_with("/points") {
            let found: Vec<Value> = body["ids"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|id| payloads.get(id.as_str().unwrap_or("")).map(|p| json!({ "id": id, "payload": p })))
                .collect();
            return common::Response::json(200, &json!({ "result": found }));
        }
        common::Response::json(200, &json!({ "result": true }))
    })
    .await
}

fn gathered_json(g: &Gathered) -> Value {
    json!({
        "found": g.found.iter().map(|h| h.id.clone()).collect::<Vec<_>>(), "kept": g.kept, "searches": g.searches, "notes": g.notes,
        "paused": g.paused,
    })
}

fn want_gathered(v: &Value) -> Value {
    json!({
        "found": v["found"].as_array().unwrap().iter().map(|h| h["id"].clone()).collect::<Vec<_>>(), "kept": v["kept"], "searches": v["searches"],
        "notes": v["notes"], "paused": v.get("paused").cloned().unwrap_or(Value::Null),
    })
}

#[tokio::test]
async fn gathering_reads_every_candidate_keeps_what_the_model_kept_and_searches_again() {
    let f = fixture();
    for (case, limit, notes) in [("notes", None, true), ("plain", Some(40), false)] {
        let index = gather_index(&f).await;
        let client =
            QdrantClient::new(Endpoint::Fixed(QdrantEndpoint { url: index.url.clone(), api_key: String::new() }), None);
        let d = RecallDeps { client, embed: zero_embed(), used: None, record: None, rerank: None };
        let a = Recorder::new(answers(&f["gather"]["answers"]));
        let q = SearchQuery { limit, ..SearchQuery::new("how many plumbers did we call?") };
        let g = memory_gather(&d, &q, GatherOptions { ask: Some(a.as_ref()), limit, notes }).await.unwrap();
        same_asks(&a.seen(), &f["gather"][case]["asked"]);
        assert_eq!(gathered_json(&g), want_gathered(&f["gather"][case]["result"]), "{case}");
        // The items as the search made them, but for the score (the stand-in fused its own way).
        let found: Vec<Value> = g
            .found
            .iter()
            .map(|h| {
                let mut v = serde_json::to_value(h).unwrap();
                v.as_object_mut().unwrap().remove("score");
                v
            })
            .collect();
        let want: Vec<Value> = f["gather"][case]["result"]["found"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| {
                let mut v = h.clone();
                v.as_object_mut().unwrap().remove("score");
                v
            })
            .collect();
        assert_eq!(found, want, "{case}");
    }
    let index = gather_index(&f).await;
    let client =
        QdrantClient::new(Endpoint::Fixed(QdrantEndpoint { url: index.url.clone(), api_key: String::new() }), None);
    let d = RecallDeps { client, embed: zero_embed(), used: None, record: None, rerank: None };
    let a = Recorder::new(vec![json!("unavailable")]);
    let g = memory_gather(
        &d,
        &SearchQuery::new("plumbers"),
        GatherOptions { ask: Some(a.as_ref()), limit: None, notes: false },
    )
    .await
    .unwrap();
    assert_eq!(a.seen().len() as u64, f["gather"]["unavailable"]["asked"].as_u64().unwrap());
    assert_eq!(gathered_json(&g), want_gathered(&f["gather"]["unavailable"]["result"]));
}

struct Provider {
    hits: Vec<RecalledItem>,
    opened: Value,
    explored: Explored,
    searched: Mutex<Vec<Value>>,
}

#[async_trait::async_trait]
impl ToolProvider for Provider {
    async fn search(&self, q: SearchQuery) -> memtrails::Result<Vec<RecalledItem>> {
        let mut v = json!({ "query": q.query });
        if let Some(c) = &q.collections {
            v["collections"] = json!(c);
        }
        v["limit"] = json!(q.limit);
        if let Some(x) = &q.from {
            v["from"] = json!(x);
        }
        if let Some(x) = &q.to {
            v["to"] = json!(x);
        }
        self.searched.lock().unwrap().push(v);
        match q.query.as_str() {
            "fail" => Err(memtrails::Error::Other("index down".into())),
            "none" => Ok(vec![]),
            _ => Ok(self.hits.clone()),
        }
    }

    async fn open(&self, id: &str) -> memtrails::Result<Option<Value>> {
        match id {
            "fail" => Err(memtrails::Error::Other("index down".into())),
            "mem:1" => Ok(Some(self.opened.clone())),
            _ => Ok(None),
        }
    }

    async fn explore(&self, q: ExploreQuery) -> Option<memtrails::Result<Explored>> {
        Some(Ok(if q.topic.as_deref() == Some("nothing") { Explored::default() } else { self.explored.clone() }))
    }
}

#[tokio::test]
async fn the_agent_tools_declare_check_and_answer_as_the_reference_does() {
    let f = fixture();
    let t = &f["tools"];
    let hits: Vec<RecalledItem> = serde_json::from_value(t["hits"].clone()).unwrap();
    let related = t["explored"]["related"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| memtrails::recall::RelatedItem {
            item: serde_json::from_value(r.clone()).unwrap(),
            via: r["via"].as_str().unwrap().into(),
        })
        .collect();
    let explored = Explored {
        found: hits.clone(),
        related,
        concepts: serde_json::from_value(t["explored"]["concepts"].clone()).unwrap(),
    };
    let provider = Arc::new(Provider { hits, opened: t["opened"].clone(), explored, searched: Mutex::new(vec![]) });
    let tools = build_agent_tools(provider.clone());
    for (tool, want) in tools.iter().zip(t["schemas"].as_array().unwrap()) {
        assert_eq!(tool.name, want["name"].as_str().unwrap());
        assert_eq!(tool.description, want["description"].as_str().unwrap());
        assert_eq!(js::stringify(&tool.json_schema), js::stringify(&want["jsonSchema"]), "{}", tool.name);
    }
    for call in t["calls"].as_array().unwrap() {
        let tool = tools.iter().find(|x| x.name == call["name"].as_str().unwrap()).unwrap();
        let got = (tool.handler)(call["args"].clone()).await;
        assert_eq!(serde_json::to_value(&got).unwrap(), call["result"], "{} {}", call["name"], call["args"]);
    }
    assert_eq!(json!(*provider.searched.lock().unwrap()), t["searched"]);
}
