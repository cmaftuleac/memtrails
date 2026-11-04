//! Consolidation and sleep over a stand-in index (common::fake_qdrant, spoken to by the real client) and an embedder
//! whose vectors are the words of a text, with a stand-in for the fast model: the same knowledge never stored twice, a
//! changed decision superseded, junk dropped, records changed while the model reads them read again, a thread that
//! never goes quiet still read, quoted labels read, links capped, imported history neither skipped nor archived.

mod common;

use std::sync::Arc;

use common::fake_qdrant::{self, embed, Fake};
use memtrails::consolidate::{label_no, relate_batches, ConsolidateDeps, Consolidator, Pair, MAX_LINKS};
use memtrails::gather::{memory_gather, GatherOptions};
use memtrails::llm::{Ask, LlmAnswer, LlmRequest};
use memtrails::qdrant_client::point_id;
use memtrails::recall::{RecallDeps, SearchQuery};
use memtrails::sleep::{sleep, SleepDeps};
use memtrails::store::{ltm_log, stm_log, MemoryItem, StmEntry, UsageBook};
use memtrails::text::children;
use memtrails::writer::{write_cues, write_ltm, write_stm, WriterDeps};
use memtrails::BoxFuture;
use serde_json::{json, Value};

const DAY: f64 = 86_400_000.0;

fn old() -> f64 {
    memtrails::js::date_parse("2026-01-01T00:00:00Z").unwrap()
}

fn now() -> f64 {
    memtrails::js::date_parse("2026-10-23T12:00:00Z").unwrap()
}

/// A model that is a function of the ask.
struct FnAsk(Arc<dyn Fn(LlmRequest) -> BoxFuture<'static, LlmAnswer> + Send + Sync>);

#[async_trait::async_trait]
impl Ask for FnAsk {
    async fn ask(&self, req: LlmRequest) -> memtrails::Result<LlmAnswer> {
        Ok((self.0)(req).await)
    }
}

fn answering(f: impl Fn(LlmRequest) -> LlmAnswer + Send + Sync + 'static) -> Arc<dyn Ask> {
    let f = Arc::new(f);
    Arc::new(FnAsk(Arc::new(move |r| {
        let f = f.clone();
        Box::pin(async move { f(r) })
    })))
}

fn answer(v: Value) -> LlmAnswer {
    LlmAnswer::Answer { answer: v, model: None }
}

fn nothing() -> Arc<dyn Ask> {
    answering(|_| answer(json!({ "segment_summary": "x", "episodes": [], "items": [], "observations": [] })))
}

fn item(uid: &str, title: &str, summary: &str, over: Value) -> MemoryItem {
    let mut v = json!({
        "uid": uid, "kind": "fact", "scope": "project", "project": "-p", "engine": "conversation", "session": "s0", "cwd": null, "where": "conv",
        "ts": old(), "firstTs": old(), "title": title, "summary": summary, "body": summary, "files": [], "appliesWhen": "", "confidence": "verified",
        "importance": 3, "status": "active", "provenance": [], "seen": 1, "fingerprint": uid,
    });
    for (k, x) in over.as_object().unwrap() {
        v[k] = x.clone();
    }
    serde_json::from_value(v).unwrap()
}

fn entry(uid: &str, request: &str, over: Value) -> StmEntry {
    let mut v = json!({
        "uid": uid, "fingerprint": uid, "engine": "conversation", "session": "sess-1", "turn": 0, "cwd": null, "project": "-p", "title": "Deploy work",
        "ts": old(), "synthetic": false, "typed": "human", "request": request, "reply": "ok", "said": "ok", "steps": [], "files": [], "done": false,
    });
    for (k, x) in over.as_object().unwrap() {
        v[k] = x.clone();
    }
    serde_json::from_value(v).unwrap()
}

struct S {
    dir: tempfile::TempDir,
    fake: Fake,
    base: ConsolidateDeps,
}

impl S {
    fn with(&self, ask: Arc<dyn Ask>) -> ConsolidateDeps {
        ConsolidateDeps { ask, ..self.base.clone() }
    }

    fn writer(&self) -> WriterDeps {
        WriterDeps { client: self.base.client.clone(), embed: self.base.embed.clone() }
    }

    async fn put_items(&self, items: Vec<MemoryItem>) {
        write_ltm(&self.writer(), &items).await.unwrap();
        self.base.ltm.put(items).await.unwrap();
    }

    fn usage(&self) -> UsageBook {
        futures::executor::block_on(UsageBook::open(self.dir.path().join("usage.json")))
    }
}

async fn setup() -> S {
    let dir = tempfile::tempdir().unwrap();
    let fake = fake_qdrant::start().await;
    let client = fake_qdrant::client(&fake).await;
    let mut base = ConsolidateDeps::new(
        client,
        embed(),
        stm_log(dir.path().join("stm.jsonl")),
        ltm_log(dir.path().join("ltm.jsonl")),
        nothing(),
    );
    let n = now();
    base.now = Arc::new(move || n);
    S { dir, fake, base }
}

#[tokio::test]
async fn a_record_removed_while_the_model_read_it_stays_removed_and_one_changed_is_read_again() {
    let s = setup().await;
    let stm = s.base.stm.clone();
    stm.put(vec![
        entry("c:1", "rotate the staging certificate", json!({ "turn": 0 })),
        entry("c:2", "renew the token", json!({ "turn": 1 })),
    ])
    .await
    .unwrap();
    let st = stm.clone();
    let ask: Arc<dyn Ask> = Arc::new(FnAsk(Arc::new(move |r: LlmRequest| {
        let st = st.clone();
        Box::pin(async move {
            if r.name.as_deref() == Some("memory_consolidation") {
                st.remove(vec!["c:1".into()]).await.unwrap();
                let mut c2 = st.get("c:2").await.unwrap();
                c2.fingerprint = "c:2-new".into();
                c2.request = "renew the token, and the key too".into();
                st.put(vec![c2]).await.unwrap();
            }
            answer(json!({ "segment_summary": "x", "episodes": [], "items": [], "observations": [] }))
        })
    })));
    let r = Consolidator::new(s.with(ask)).run(20).await.unwrap();
    assert_eq!(r.windows, 1);
    assert!(stm.get("c:1").await.is_none());
    let c2 = stm.get("c:2").await.unwrap();
    assert_eq!((c2.request.as_str(), c2.done), ("renew the token, and the key too", false));
}

#[tokio::test]
async fn marks_done_a_record_left_as_it_was() {
    let s = setup().await;
    s.base.stm.put(vec![entry("c:1", "rotate the staging certificate", json!({}))]).await.unwrap();
    Consolidator::new(s.base.clone()).run(20).await.unwrap();
    assert!(s.base.stm.get("c:1").await.unwrap().done);
}

#[tokio::test]
async fn a_thread_that_never_goes_quiet_is_read_once_its_oldest_entry_is_a_day_old() {
    let s = setup().await;
    let n = now();
    let busy: Vec<StmEntry> = (0..51)
        .map(|i| {
            entry(
                &format!("busy:{i}"),
                &format!("update {i}"),
                json!({ "session": "busy", "turn": i, "ts": n - 25.0 * 3_600_000.0 + i as f64 * 1_800_000.0 }),
            )
        })
        .collect();
    s.base.stm.put(busy).await.unwrap();
    s.base
        .stm
        .put(vec![entry("fresh:0", "just now", json!({ "session": "fresh", "ts": n - 600_000.0 }))])
        .await
        .unwrap();
    let pending: Vec<String> = Consolidator::new(s.base.clone()).pending(n).into_iter().map(|p| p.session).collect();
    assert_eq!(pending, ["busy"]);
}

#[tokio::test]
async fn labels_a_model_quotes_are_read() {
    assert_eq!(label_no("p3", 'p'), 3);
    assert_eq!(label_no("[p12]\nNEW (fact): x", 'p'), 12);
    assert_eq!(label_no("[i2] (fact) deploy the api", 'i'), 2);
    assert_eq!(label_no("item", 'i'), 0);
    let s = setup().await;
    let ask = answering(|r| {
        let first: Vec<&str> = r.prompt.split("\n\n").next().unwrap().split('\n').take(2).collect();
        answer(
            json!({ "pairs": [{ "id": first.join("\n"), "relation": "related", "title": "", "summary": "", "body": "" }] }),
        )
    });
    let pair = Pair {
        it: item("mem:new", "new", "deploy through jenkins", json!({})),
        known: item("mem:old", "old", "deploy by hand", json!({})),
    };
    let got = Consolidator::new(s.with(ask)).relate(&[pair.clone()]).await.unwrap();
    assert_eq!(got["p1"].relation, "related");
    let away = answering(|_| LlmAnswer::Unavailable("no model is set".into()));
    assert_eq!(Consolidator::new(s.with(away)).relate(&[pair]).await.unwrap_err(), "no model is set");
}

#[tokio::test]
async fn an_item_gets_its_cues_when_the_model_quotes_its_label_line() {
    let s = setup().await;
    s.put_items(vec![item("mem:api", "deploy the api", "push main and jenkins deploys", json!({}))]).await;
    let ask = answering(|r| {
        answer(
            json!({ "items": [{ "id": r.prompt.split('\n').next().unwrap(), "cues": ["how do I ship the api"], "concepts": ["jenkins"] }] }),
        )
    });
    assert_eq!(Consolidator::new(s.with(ask)).fill_cues(40).await.unwrap(), (1, None));
    let it = s.base.ltm.get("mem:api").await.unwrap();
    assert_eq!(
        (it.cue_list(), it.concept_list()),
        (&["how do I ship the api".to_string()][..], &["jenkins".to_string()][..])
    );
    assert_eq!(s.fake.of("mem:api").iter().filter(|p| p.payload["part"] == "cue").count(), 1);
}

#[tokio::test]
async fn links_keep_the_newest_max_links() {
    let s = setup().await;
    let related: Vec<String> = (0..MAX_LINKS).map(|n| format!("mem:r{n}")).collect();
    let busy = item("mem:busy", "busy", "busy", json!({ "related": related }));
    let other = item("mem:new", "new", "new", json!({}));
    s.put_items(vec![busy.clone(), other.clone()]).await;
    let w = Consolidator::new(s.base.clone());
    w.link(busy, other.clone()).await.unwrap();
    let got = s.base.ltm.get("mem:busy").await.unwrap();
    assert_eq!(got.related_list().len(), MAX_LINKS);
    assert_eq!(got.related_list().last().unwrap(), "mem:new");
    assert!(!got.related_list().contains(&"mem:r0".to_string()));
    w.link(got, other).await.unwrap();
    assert_eq!(s.base.ltm.get("mem:busy").await.unwrap().related_list().iter().filter(|u| *u == "mem:new").count(), 1);
}

#[tokio::test]
async fn what_consolidation_writes_is_stamped_with_when_it_was_learned_and_carries_its_records_files() {
    let s = setup().await;
    s.base
        .stm
        .put(vec![entry(
            "c:1",
            "deploy the api through jenkins",
            json!({ "ts": old(), "files": ["/p/deploy.sh", "/p/Buildfile"] }),
        )])
        .await
        .unwrap();
    let ask = answering(|_| {
        answer(json!({ "segment_summary": "x", "episodes": [], "observations": [], "items": [{
            "kind": "procedure", "title": "deploy through jenkins", "summary": "push main and jenkins deploys", "body": "b", "applies_when": "", "scope": "project",
            "requests": [1], "confidence": "verified", "importance": 4, "contested": false, "ambiguous": false, "cues": [], "concepts": [] }] }))
    });
    Consolidator::new(s.with(ask)).run(20).await.unwrap();
    let made = s.base.ltm.all().await.into_iter().find(|x| x.title == "deploy through jenkins").unwrap();
    assert_eq!(made.ts, old());
    assert_eq!(made.learned_at, Some(now()));
    assert_eq!(made.files, ["/p/deploy.sh", "/p/Buildfile"]);
}

#[tokio::test]
async fn keeps_what_matters_once_merges_supersedes_adds_drops_and_empties_the_queue() {
    let s = setup().await;
    let known = item(
        "mem:known",
        "deploy api buildbot",
        "deploy the api by pushing main buildbot builds and deploys it",
        json!({}),
    );
    let old_decision = item(
        "mem:old",
        "database choice postgres",
        "alpha beta gamma delta epsilon zeta eta theta",
        json!({ "kind": "decision" }),
    );
    s.put_items(vec![known, old_decision]).await;
    let pending = vec![
        entry("conv:sess-1:0", "how do we deploy", json!({})),
        entry("conv:sess-1:1", "switch the database", json!({ "turn": 1 })),
    ];
    write_stm(&s.writer(), &pending).await.unwrap();
    s.base.stm.put(pending.clone()).await.unwrap();
    let ask = answering(|r| {
        if r.name.as_deref() == Some("memory_consolidation") {
            let base = |kind: &str, title: &str, summary: &str| {
                json!({ "kind": kind, "title": title, "summary": summary, "body": "b", "applies_when": "", "scope": "project", "requests": [1],
                    "confidence": "verified", "importance": 4, "contested": false, "ambiguous": false })
            };
            let mut pitfall =
                base("pitfall", "flaky e2e on wifi", "the e2e suite times out on hotel wifi run it wired");
            pitfall["cues"] =
                json!(["the e2e tests keep timing out", "testele e2e cad pe wifi", "run the suite wired"]);
            let mut guess = base("fact", "a guess", "probably redis");
            guess["confidence"] = json!("inferred");
            return answer(json!({
                "segment_summary": "Deploy and database work.",
                "episodes": [{ "request": 1, "keep": false, "title": "", "summary": "", "cues": [] },
                    { "request": 2, "keep": true, "title": "database switched", "summary": "the database was switched to sqlite after load tests", "cues": ["de ce am trecut pe sqlite", "which database do we use"] }],
                "items": [base("fact", "deploy api buildbot", "deploy the api by pushing main buildbot builds and deploys it"),
                    base("decision", "database choice postgres", "alpha beta gamma delta epsilon zeta eta iota"), pitfall, guess],
            }));
        }
        answer(json!({ "pairs": [{ "id": "p1", "relation": "contradicts", "title": "", "summary": "", "body": "" }] }))
    });
    let r = Consolidator::new(s.with(ask)).run(20).await.unwrap();
    assert_eq!((r.windows, r.merged, r.superseded, r.dropped, r.episodes), (1, 1, 1, 1, 1));
    let known = s.base.ltm.get("mem:known").await.unwrap();
    assert_eq!((known.seen, known.status.as_str()), (2.0, "active"));
    assert!(known.provenance.contains(&"conv:sess-1:0".to_string()));
    assert_eq!(s.base.ltm.get("mem:old").await.unwrap().status, "superseded");
    assert!(s.fake.of("mem:old").is_empty());
    let all = s.base.ltm.all().await;
    let fresh: Vec<&str> = all.iter().filter(|x| x.status == "active").map(|x| x.title.as_str()).collect();
    for t in ["database choice postgres", "flaky e2e on wifi", "database switched"] {
        assert!(fresh.contains(&t), "{t}");
    }
    assert!(!fresh.contains(&"a guess"));
    let cues_of = |title: &str| {
        let uid = &all.iter().find(|x| x.title == title).unwrap().uid;
        s.fake.of(uid).iter().filter(|p| p.payload["part"] == "cue").count()
    };
    assert_eq!(cues_of("flaky e2e on wifi"), 3);
    assert_eq!(cues_of("database switched"), 2);
    assert!(s.base.stm.entries().iter().all(|(_, m)| m.done));
    assert!(pending.iter().all(|e| !s.fake.of(&format!("stm:{}", e.uid)).is_empty()));
}

#[tokio::test]
async fn a_request_typed_memories_came_from_is_not_kept_again_as_an_episode() {
    let s = setup().await;
    s.base
        .stm
        .put(vec![
            entry("conv:sess-1:0", "fix the flaky e2e", json!({})),
            entry("conv:sess-1:1", "rename the job", json!({ "turn": 1 })),
        ])
        .await
        .unwrap();
    let ask = answering(|_| {
        answer(json!({
            "segment_summary": "e2e and job work.",
            "episodes": [{ "request": 1, "keep": true, "title": "e2e fixed", "summary": "the e2e suite was fixed by running it wired", "cues": [] },
                { "request": 2, "keep": true, "title": "job renamed", "summary": "the deploy job was renamed to ship", "cues": [] }],
            "items": [{ "kind": "pitfall", "title": "flaky e2e on wifi", "summary": "the e2e suite times out on hotel wifi run it wired", "requests": [1],
                "body": "b", "applies_when": "", "scope": "project", "confidence": "verified", "importance": 4, "contested": false, "ambiguous": false }],
        }))
    });
    let r = Consolidator::new(s.with(ask)).run(20).await.unwrap();
    assert_eq!(r.episodes, 1);
    let mut titles: Vec<String> =
        s.base.ltm.all().await.into_iter().filter(|x| x.status == "active").map(|x| x.title).collect();
    titles.sort();
    assert_eq!(titles, ["flaky e2e on wifi", "job renamed"]);
    assert!(s.base.ltm.get("ep:conv:sess-1:0").await.is_none());
}

#[tokio::test]
async fn pauses_leaving_the_queue_as_it_was_when_the_model_cannot_be_asked() {
    let s = setup().await;
    s.base.stm.put(vec![entry("c:1", "something", json!({}))]).await.unwrap();
    let away = answering(|_| LlmAnswer::Unavailable("no model is set".into()));
    let r = Consolidator::new(s.with(away)).run(20).await.unwrap();
    assert_eq!((r.windows, r.paused.as_deref()), (0, Some("no model is set")));
    assert!(!s.base.stm.get("c:1").await.unwrap().done);
}

#[tokio::test]
async fn a_merge_keeps_both_texts_as_records_the_item_lists() {
    let s = setup().await;
    let known = item(
        "mem:k",
        "deploy api buildbot",
        "deploy the api by pushing main buildbot builds and deploys it",
        json!({ "body": "the first text" }),
    );
    s.put_items(vec![known.clone()]).await;
    let fresh = item(
        "mem:n",
        "deploy api buildbot",
        "deploy the api with buildbot",
        json!({ "body": "the second text, with a detail" }),
    );
    let w = Consolidator::new(s.base.clone());
    w.merge(
        s.base.ltm.get("mem:k").await.unwrap(),
        fresh,
        Some(("Deploy the API through Buildbot".into(), "merged".into(), "merged text".into())),
    )
    .await
    .unwrap();
    let merged = s.base.ltm.get("mem:k").await.unwrap();
    assert_eq!(merged.body, "merged text");
    let mut absorbed = merged.absorbed.clone().unwrap();
    absorbed.sort();
    assert_eq!(absorbed, [format!("mem:k@{}", known.fingerprint), "mem:n".to_string()]);
    let n = s.base.ltm.get("mem:n").await.unwrap();
    assert_eq!(
        (n.status.as_str(), n.merged_into.as_deref(), n.body.as_str()),
        ("merged", Some("mem:k"), "the second text, with a detail")
    );
    assert_eq!(s.base.ltm.get(&format!("mem:k@{}", known.fingerprint)).await.unwrap().body, "the first text");
    assert!(s.fake.of("mem:n").is_empty());
}

#[tokio::test]
async fn every_chunk_of_a_long_body_is_indexed_its_cues_apart() {
    let s = setup().await;
    let body = (0..3000).map(|i| format!("detail{i}")).collect::<Vec<_>>().join(" ");
    let cues: Vec<String> = (0..10).map(|i| format!("cue question {i}")).collect();
    write_ltm(&s.writer(), &[item("mem:long", "long", "a long one", json!({ "body": body, "cues": cues }))])
        .await
        .unwrap();
    let parts =
        |part: &str| s.fake.of("mem:long").into_iter().filter(|p| p.payload["part"] == part).collect::<Vec<_>>();
    assert_eq!(parts("child").len(), children(&body).len());
    assert_eq!(parts("cue").len(), 10);
    assert!(s.fake.state.points.lock().unwrap().contains_key(&point_id("mem:long", -10)));
    assert_eq!(parts("key")[0].payload["body"], json!(body));
}

#[tokio::test]
async fn a_failed_embedding_keeps_the_old_item_and_its_cues_searchable() {
    let s = setup().await;
    let known = item("mem:known", "deploy", "push main", json!({ "cues": ["deploy the api"] }));
    write_ltm(&s.writer(), std::slice::from_ref(&known)).await.unwrap();
    let before: Vec<String> = s.fake.state.points.lock().unwrap().keys().cloned().collect();
    let failing = WriterDeps {
        client: s.base.client.clone(),
        embed: Arc::new(|_, _| Box::pin(async { Err(memtrails::Error::Embedding("GPU unavailable".into())) })),
    };
    let changed = MemoryItem { summary: "new deployment procedure".into(), ..known.clone() };
    assert!(write_ltm(&failing, &[changed]).await.unwrap_err().to_string().contains("GPU unavailable"));
    assert!(write_cues(&failing, &[MemoryItem { cues: Some(vec!["new question".into()]), ..known.clone() }])
        .await
        .is_err());
    assert_eq!(s.fake.state.points.lock().unwrap().keys().cloned().collect::<Vec<_>>(), before);
    write_cues(&s.writer(), &[MemoryItem { cues: Some(vec![]), ..known.clone() }]).await.unwrap();
    assert!(s.fake.points().iter().all(|p| p.payload["part"] != "cue"));
    write_ltm(&s.writer(), &[MemoryItem { status: "archived".into(), ..known }]).await.unwrap();
    assert!(s.fake.points().is_empty());
}

#[test]
fn pairs_are_asked_about_in_batches_of_at_most_eight_and_at_most_a_size() {
    let size = |n: &usize| *n;
    assert_eq!(relate_batches(vec![10usize; 20], size).iter().map(Vec::len).collect::<Vec<_>>(), [8, 8, 4]);
    assert_eq!(relate_batches(vec![10, 70_000, 10], size).iter().map(Vec::len).collect::<Vec<_>>(), [1, 1, 1]);
}

#[tokio::test]
async fn sleep_relates_and_merges_imported_history_the_night_after_and_does_not_archive_it() {
    let s = setup().await;
    let learned = now() - 3_600_000.0;
    let a = item(
        "mem:a",
        "ports",
        "the proxy listens on port 8080 behind caddy",
        json!({ "ts": old() - 30.0 * DAY, "learnedAt": learned }),
    );
    let b = item(
        "mem:b",
        "ports",
        "the proxy listens on port 8080 behind caddy",
        json!({ "ts": old(), "learnedAt": learned }),
    );
    let old_episode = item(
        "ep:old",
        "one-off fix",
        "fixed a typo in a readme long ago",
        json!({ "kind": "episode", "importance": 2, "ts": now() - 400.0 * DAY, "learnedAt": learned }),
    );
    s.put_items(vec![a, b, old_episode]).await;
    let state = s.dir.path().join("sleep.json");
    std::fs::write(&state, json!({ "last": now() - 2.0 * 3_600_000.0 }).to_string()).unwrap();
    let r = sleep(&SleepDeps::new(s.base.clone(), s.usage(), state)).await.unwrap();
    assert_eq!((r.merged, r.archived), (1, 0));
    assert_eq!(s.base.ltm.get("ep:old").await.unwrap().status, "active");
}

#[tokio::test]
async fn sleep_archives_an_item_without_learned_at_by_its_own_time() {
    let s = setup().await;
    s.put_items(vec![item(
        "ep:old",
        "one-off fix",
        "fixed a typo in a readme long ago",
        json!({ "kind": "episode", "importance": 2, "ts": now() - 400.0 * DAY }),
    )])
    .await;
    let r = sleep(&SleepDeps::new(s.base.clone(), s.usage(), s.dir.path().join("sleep.json"))).await.unwrap();
    assert_eq!(r.archived, 1);
    assert_eq!(s.base.ltm.get("ep:old").await.unwrap().status, "archived");
    let state: Value =
        serde_json::from_str(&std::fs::read_to_string(s.dir.path().join("sleep.json")).unwrap()).unwrap();
    assert_eq!(state["last"].as_f64(), Some(now()));
}

#[tokio::test]
async fn a_model_that_is_not_there_shows_in_what_sleep_and_gathering_return() {
    let s = setup().await;
    let words = "deploy jenkins pipeline main branch build docker image registry";
    s.put_items(vec![
        item("mem:alpha", "alpha", &format!("{words} one"), json!({})),
        item("mem:bravo", "bravo", &format!("{words} three"), json!({ "ts": old() + DAY })),
    ])
    .await;
    let away = answering(|_| LlmAnswer::Unavailable("no model is set".into()));
    let r = sleep(&SleepDeps::new(s.with(away.clone()), s.usage(), s.dir.path().join("sleep.json"))).await.unwrap();
    assert_eq!(r.paused.as_deref(), Some("no model is set"));

    let g = setup().await;
    let obs: Vec<MemoryItem> = ["bought an evening gown", "went camping"]
        .iter()
        .enumerate()
        .map(|(n, t)| {
            item(
                &format!("obs:{n}"),
                &format!("Nadia {t}"),
                &format!("Nadia {t}"),
                json!({ "kind": "observation", "scope": "person" }),
            )
        })
        .collect();
    write_ltm(&g.writer(), &obs).await.unwrap();
    let deps = RecallDeps { client: g.base.client.clone(), embed: embed(), used: None, record: None, rerank: None };
    let q = SearchQuery::new("what did Nadia buy");
    let got =
        memory_gather(&deps, &q, GatherOptions { ask: Some(away.as_ref()), limit: None, notes: false }).await.unwrap();
    assert_eq!(got.paused.as_deref(), Some("no model is set"));
    let mut kept = got.kept.clone();
    kept.sort();
    assert_eq!(kept, ["obs:0", "obs:1"]);
    let one = answering(|_| answer(json!({ "keep": ["m1"], "searches": [] })));
    let read =
        memory_gather(&deps, &q, GatherOptions { ask: Some(one.as_ref()), limit: None, notes: false }).await.unwrap();
    assert!(read.paused.is_none());
    assert_eq!(read.kept.len(), 1);
}

#[tokio::test]
async fn observations_are_only_added_the_same_told_again_once_and_another_day_twice() {
    let s = setup().await;
    s.base
        .stm
        .put(vec![
            entry("c:1", "we had dinner at the harbour", json!({ "turn": 0 })),
            entry("c:2", "and again, dinner at the harbour", json!({ "turn": 1 })),
        ])
        .await
        .unwrap();
    let ask = answering(|_| {
        let obs = |text: &str, date: &str, req: u32| json!({ "text": text, "about": "the person", "event_date": date, "start_date": "", "end_date": "", "phase": "done", "requests": [req], "concepts": ["harbour"], "cues": [] });
        answer(json!({ "segment_summary": "x", "episodes": [], "items": [], "observations": [
            obs("They had dinner at the harbour restaurant", "2026-01-10", 1),
            obs("They had dinner at the harbour restaurant", "2026-01-10", 2),
            obs("They had dinner at the harbour restaurant", "2026-01-17", 2),
        ] }))
    });
    let r = Consolidator::new(s.with(ask)).run(20).await.unwrap();
    assert_eq!((r.observed, r.merged), (2, 1));
    let obs: Vec<MemoryItem> = s.base.ltm.all().await.into_iter().filter(|x| x.kind == "observation").collect();
    assert_eq!(obs.len(), 2);
    let told_twice = obs.iter().find(|o| o.event_at.as_deref() == Some("2026-01-10")).unwrap();
    assert_eq!(told_twice.seen, 2.0);
    assert_eq!(told_twice.provenance, ["c:1", "c:2"]);
}

#[tokio::test]
async fn profiles_are_written_for_a_subject_enough_items_share_and_only_again_when_they_change() {
    let s = setup().await;
    let mut items: Vec<MemoryItem> = (0..5)
        .map(|i| {
            item(
                &format!("obs:m{i}"),
                &format!("Miso note {i}"),
                &format!("Miso did thing {i}."),
                json!({ "kind": "observation", "scope": "person", "concepts": ["Miso"] }),
            )
        })
        .collect();
    items.extend(
        (0..30).map(|i| item(&format!("mem:f{i}"), "filler", "filler", json!({ "concepts": [format!("topic {i}")] }))),
    );
    s.put_items(items).await;
    let asks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let n = asks.clone();
    let ask = answering(move |_| {
        n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        answer(json!({ "title": "Miso", "facts": ["2026: adopted"] }))
    });
    let d = s.with(ask);
    let r = memtrails::profiles::write_profiles(&d, 400).await.unwrap();
    assert_eq!((r.profiled, r.unprofiled), (1, 0));
    let again = memtrails::profiles::write_profiles(&d, 400).await.unwrap();
    assert_eq!((again.profiled, asks.load(std::sync::atomic::Ordering::SeqCst)), (0, 1));
    let profile = s.base.ltm.all().await.into_iter().find(|x| x.kind == "profile").unwrap();
    assert_eq!((profile.summary.as_str(), profile.concept_list()), ("2026: adopted", &["Miso".to_string()][..]));
}
