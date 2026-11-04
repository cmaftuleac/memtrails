//! The HTTP clients memory talks through — the language model, an embeddings server, a rerank server — over a fake
//! server on loopback: what they send, what they accept, what they refuse.

mod common;

use base64::Engine;
use common::{serve, Response};
use memtrails::embedder::{doc_text, query_text, remote_embedder, EmbedRoute, MEMORY_EMBEDDER};
use memtrails::llm::{openai_ask, LlmAnswer, LlmRequest, OpenAiAskOptions};
use memtrails::reranker::{remote_reranker, reranked_order, RerankRoute};
use serde_json::{json, Value};

fn req() -> LlmRequest {
    LlmRequest {
        purpose: "memory".into(),
        system: "s".into(),
        prompt: "p".into(),
        schema: json!({ "type": "object" }),
        name: None,
        model: None,
        max_tokens: None,
        timeout_ms: Some(5000),
        consequence: Some("so it waits".into()),
    }
}

#[tokio::test]
async fn the_model_is_asked_for_strict_structured_output_and_its_answer_read() {
    let s = serve(|_| Response::json(200, &json!({ "model": "m-1", "output": [{ "content": [{ "text": "{\"ok\":" }, { "text": "true}" }] }], "usage": { "input_tokens": 7, "output_tokens": 3, "input_tokens_details": { "cached_tokens": 2 } } }))).await;
    let used = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let u = used.clone();
    let ask = openai_ask(OpenAiAskOptions {
        model: Some("m".into()),
        api_key: Some("k".into()),
        base_url: Some(format!("{}/v1/", s.url)),
        on_usage: Some(std::sync::Arc::new(move |x| u.lock().unwrap().push(x))),
        ..Default::default()
    });
    let a = ask(req()).await;
    assert_eq!(a, LlmAnswer::Answer { answer: json!({ "ok": true }), model: Some("m-1".into()) });
    let r = &s.requests()[0];
    assert_eq!(r.path, "/v1/responses");
    assert_eq!(r.header("authorization"), Some("Bearer k"));
    let body = r.json();
    assert_eq!(
        body["text"]["format"],
        json!({ "type": "json_schema", "name": "memory", "strict": true, "schema": { "type": "object" } })
    );
    assert_eq!(
        (body["instructions"].as_str(), body["input"].as_str(), body["max_output_tokens"].as_u64()),
        (Some("s"), Some("p"), Some(12_000))
    );
    let u = used.lock().unwrap();
    assert_eq!((u[0].input_tokens, u[0].cached_tokens, u[0].output_tokens), (7, 2, 3));
}

#[tokio::test]
async fn says_what_the_provider_said_when_it_refuses_secrets_taken_out() {
    let secret = ["sk-", "abcdefghijklmnop1234"].concat();
    let body = format!(r#"{{"error":{{"message":"model gpt-x does not exist; key {secret}"}}}}"#);
    let s = serve(move |_| Response::new(400, body.clone())).await;
    let ask = openai_ask(OpenAiAskOptions {
        model: Some("m".into()),
        api_key: Some("k".into()),
        base_url: Some(s.url.clone()),
        ..Default::default()
    });
    let LlmAnswer::Unavailable(why) = ask(req()).await else { panic!("refused") };
    assert!(why.contains("model gpt-x does not exist"), "{why}");
    assert!(why.ends_with(", so it waits"), "{why}");
    assert!(!why.contains(&secret));
}

#[tokio::test]
async fn retries_a_busy_model_then_says_it_did_not_answer() {
    let s = serve(|_| Response::new(503, "busy")).await;
    let ask = openai_ask(OpenAiAskOptions {
        model: Some("m".into()),
        api_key: Some("k".into()),
        base_url: Some(s.url.clone()),
        retries: Some(1),
        ..Default::default()
    });
    let LlmAnswer::Unavailable(why) = ask(req()).await else { panic!("not answered") };
    assert!(why.contains("did not answer (HTTP 503)"), "{why}");
    assert_eq!(s.requests().len(), 2);
}

const DIMS: usize = MEMORY_EMBEDDER.dims;

fn vec_of(k: usize) -> Vec<f32> {
    (0..DIMS).map(|i| if i == k % DIMS { 3.0 } else { 0.0 }).collect()
}

fn ok(input: &Value, base64: bool) -> Response {
    let data: Vec<Value> = input
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let v = vec_of(i);
            let e = if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(v)
            };
            json!({ "index": i, "embedding": e })
        })
        .collect();
    Response::json(200, &json!({ "data": data }))
}

#[tokio::test]
async fn embeddings_use_the_models_prompts_and_come_back_as_unit_vectors_in_order() {
    assert_eq!(query_text("why"), "task: search result | query: why");
    assert_eq!(doc_text("", "text"), "title: none | text: text");
    let s = serve(|r| ok(&r.json()["input"], false)).await;
    let embed =
        remote_embedder(EmbedRoute { url: format!("{}/v1", s.url), api_key: Some("k".into()), ..Default::default() });
    let out = embed(vec!["a".into(), "b".into(), "c".into()], None).await.unwrap();
    let r = &s.requests()[0];
    assert_eq!(r.path, "/v1/embeddings");
    assert_eq!(r.header("authorization"), Some("Bearer k"));
    assert_eq!(r.json(), json!({ "model": "embeddinggemma-300m", "input": ["a", "b", "c"] }));
    assert_eq!(out.iter().map(|v| v.iter().position(|x| *x == 1.0)).collect::<Vec<_>>(), [Some(0), Some(1), Some(2)]);
}

#[tokio::test]
async fn embeddings_accept_base64_a_model_name_gzip_and_no_key() {
    let s = serve(|r| ok(&r.json()["input"], true)).await;
    let embed = remote_embedder(EmbedRoute {
        url: format!("{}/v1/embeddings", s.url),
        api_key: None,
        model: Some("embeddinggemma".into()),
        gzip: true,
    });
    let v = embed(vec!["a".into()], None).await.unwrap();
    assert_eq!(v[0][0], 1.0);
    let r = &s.requests()[0];
    assert_eq!(r.path, "/v1/embeddings");
    assert_eq!(r.header("content-encoding"), Some("gzip"));
    assert_eq!(r.header("authorization"), None);
    assert_eq!(r.json()["model"], "embeddinggemma");
}

#[tokio::test]
async fn embeddings_split_a_large_batch_and_keep_the_order() {
    let s = serve(|r| ok(&r.json()["input"], false)).await;
    let embed = remote_embedder(EmbedRoute { url: s.url.clone(), ..Default::default() });
    let texts: Vec<String> = (0..600).map(|i| format!("text {i}")).collect();
    let out = embed(texts.clone(), None).await.unwrap();
    assert_eq!(out.len(), 600);
    let reqs = s.requests();
    assert!(reqs.len() > 1);
    let mut sent: Vec<(usize, Vec<String>)> = reqs
        .iter()
        .map(|r| {
            let input: Vec<String> = serde_json::from_value(r.json()["input"].clone()).unwrap();
            (texts.iter().position(|t| *t == input[0]).unwrap(), input)
        })
        .collect();
    sent.sort_by_key(|x| x.0);
    assert_eq!(sent.into_iter().flat_map(|x| x.1).collect::<Vec<_>>(), texts);
}

#[tokio::test]
async fn embeddings_retry_a_busy_server_and_refuse_what_is_wrong() {
    let n = std::sync::atomic::AtomicUsize::new(0);
    let busy = serve(move |r| {
        if n.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            Response::new(503, "busy")
        } else {
            ok(&r.json()["input"], false)
        }
    })
    .await;
    let embed = remote_embedder(EmbedRoute { url: busy.url.clone(), ..Default::default() });
    assert_eq!(embed(vec!["a".into()], None).await.unwrap().len(), 1);
    for (body, says) in [
        (json!({ "data": [{ "index": 0, "embedding": [1, 2] }] }), "dimensions"),
        (json!({ "data": [{ "index": 0, "embedding": vec![0.0; DIMS] }] }), "zero"),
        (json!({ "data": [] }), "0 vectors for 1"),
    ] {
        let s = serve(move |_| Response::json(200, &body)).await;
        let embed = remote_embedder(EmbedRoute { url: s.url.clone(), ..Default::default() });
        let err = embed(vec!["a".into()], None).await.unwrap_err().to_string();
        assert!(err.contains(says), "{err}");
    }
}

#[tokio::test]
async fn a_rerank_server_orders_and_a_failing_one_leaves_the_order() {
    let s = serve(|_| {
        Response::json(
            200,
            &json!({ "results": [{ "index": 1, "relevance_score": 0.9 }, { "index": 0, "relevance_score": 0.1 }] }),
        )
    })
    .await;
    let rerank = remote_reranker(RerankRoute { url: s.url.clone(), ..Default::default() });
    let scores = rerank("q".into(), vec!["a".into(), "b".into()], 3000).await.unwrap();
    assert_eq!(scores, [0.1, 0.9]);
    assert_eq!(s.requests()[0].json(), json!({ "model": "bge-reranker-v2-m3", "query": "q", "documents": ["a", "b"] }));
    assert_eq!(reranked_order(&["a", "b"], Some(&scores)), ["b", "a"]);
    assert_eq!(reranked_order(&["a", "b"], None), ["a", "b"]);
    let down = serve(|_| Response::new(500, "")).await;
    let rerank = remote_reranker(RerankRoute { url: down.url.clone(), ..Default::default() });
    assert!(rerank("q".into(), vec!["a".into()], 3000).await.is_none());
}
