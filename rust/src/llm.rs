//! The language model memory asks, for consolidation, sleep and gathering — never for adding or searching. One
//! structured ask: a system text, a prompt and a JSON Schema the answer must follow (strict structured output).
//!
//! `openai_ask` speaks the OpenAI Responses API, so any provider or local server that serves it works (`base_url`).
//! Bring any other model by passing your own [`Ask`]. Every call's tokens can be counted (`on_usage`).

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use rand::Rng;
use serde_json::{json, Value};

use crate::error::Result;
use crate::redact::redact_string;

/// One structured ask.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmRequest {
    /// What this is for, in a word (`memory`).
    pub purpose: String,
    /// The system text.
    pub system: String,
    /// The prompt.
    pub prompt: String,
    /// The answer's JSON Schema.
    pub schema: Value,
    /// What the structured answer is called.
    pub name: Option<String>,
    /// A model for this ask only.
    pub model: Option<String>,
    /// The answer's room, in tokens.
    pub max_tokens: Option<u64>,
    /// How long to wait.
    pub timeout_ms: Option<u64>,
    /// What a failure cost, said after its reason ("so no summary was written").
    pub consequence: Option<String>,
}

/// What an ask gives.
#[derive(Debug, Clone, PartialEq)]
pub enum LlmAnswer {
    /// The answer, as the schema shapes it.
    Answer {
        /// The parsed answer.
        answer: Value,
        /// The model that answered.
        model: Option<String>,
    },
    /// Why it could not be asked, or did not answer, in words a person can read.
    Unavailable(String),
    /// One for this purpose is already being asked.
    Busy,
}

/// A language model: one structured ask in, an answer out. Implement it to bring any model; [`openai_ask`] speaks
/// the OpenAI-compatible Responses API. An error is read as the model being unavailable, with the error as the reason.
#[async_trait::async_trait]
pub trait Ask: Send + Sync {
    /// Ask once.
    async fn ask(&self, req: LlmRequest) -> Result<LlmAnswer>;
}

/// Ask, an error read as the model being unavailable (the work waits for the next run).
pub(crate) async fn ask_or_pause(ask: &dyn Ask, req: LlmRequest) -> LlmAnswer {
    let consequence = req.consequence.clone();
    match ask.ask(req).await {
        Ok(a) => a,
        Err(e) => LlmAnswer::Unavailable(match consequence {
            Some(c) if !c.is_empty() => format!("{e}, {c}"),
            _ => e.to_string(),
        }),
    }
}

/// The OpenAI-compatible model ([`openai_ask`]).
#[derive(Debug, Clone)]
pub struct OpenAiAsk {
    options: OpenAiAskOptions,
}

#[async_trait::async_trait]
impl Ask for OpenAiAsk {
    async fn ask(&self, req: LlmRequest) -> Result<LlmAnswer> {
        Ok(ask_openai(&self.options, req).await)
    }
}

/// One answered call's tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmUsage {
    /// The model that answered.
    pub model: String,
    /// The ask's name.
    pub name: Option<String>,
    /// Tokens read.
    pub input_tokens: u64,
    /// Of those, read from the provider's cache.
    pub cached_tokens: u64,
    /// Tokens written.
    pub output_tokens: u64,
}

/// How hard a reasoning model thinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// `minimal`
    Minimal,
    /// `low`
    Low,
    /// `medium`
    Medium,
    /// `high`
    High,
}

impl ReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }
    }
}

/// The OpenAI-compatible model's settings.
#[derive(Clone, Default)]
pub struct OpenAiAskOptions {
    /// Default: `$MEMTRAILS_LLM_API_KEY`, else `$OPENAI_API_KEY` — the latter only for OpenAI's own address.
    pub api_key: Option<String>,
    /// Default: `$MEMTRAILS_MODEL`.
    pub model: Option<String>,
    /// Default: `$MEMTRAILS_LLM_BASE_URL`, else `https://api.openai.com/v1`.
    pub base_url: Option<String>,
    /// For reasoning models: how hard to think. Not sent when not given.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Each answered call's tokens.
    pub on_usage: Option<Arc<dyn Fn(LlmUsage) + Send + Sync>>,
    /// Tries after a rate limit, a server error or a dropped connection (default 5), waiting longer each time.
    pub retries: Option<u32>,
}

impl std::fmt::Debug for OpenAiAskOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiAskOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<set>"))
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("retries", &self.retries)
            .finish()
    }
}

const OPENAI: &str = "https://api.openai.com/v1";

/// The shared HTTP client (rustls, connection reuse). It goes to each address directly, never through a proxy the
/// environment names (`HTTP_PROXY`, `ALL_PROXY`…), as the reference's fetch does: through one, the local Qdrant's key
/// went to whoever serves the proxy, and a Qdrant on loopback could not be reached at all.
pub(crate) fn http() -> &'static reqwest::Client {
    static CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
        // The builder fails only when TLS cannot start, and then `Client::new` fails the same way.
        reqwest::Client::builder().no_proxy().build().unwrap_or_else(|_| reqwest::Client::new())
    });
    &CLIENT
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn why(reason: &str, req: &LlmRequest) -> String {
    match &req.consequence {
        Some(c) if !c.is_empty() => format!("{reason}, {c}"),
        _ => reason.to_string(),
    }
}

/// An [`Ask`] that speaks the OpenAI Responses API with strict JSON Schema output and retries rate limits, server
/// errors and dropped connections.
pub fn openai_ask(o: OpenAiAskOptions) -> Arc<dyn Ask> {
    Arc::new(OpenAiAsk { options: o })
}

/// The model the environment names (`MEMTRAILS_MODEL` and a key), asked when no model is given.
pub fn default_ask() -> Arc<dyn Ask> {
    openai_ask(OpenAiAskOptions::default())
}

/// The model, the address and the key an ask goes with. An OpenAI key goes to OpenAI only: with another address set,
/// it would hand the key to whoever serves that.
fn route(o: &OpenAiAskOptions, req: &LlmRequest) -> (Option<String>, String, Option<String>) {
    let model = req.model.clone().or_else(|| o.model.clone()).or_else(|| env("MEMTRAILS_MODEL"));
    let base = o.base_url.clone().or_else(|| env("MEMTRAILS_LLM_BASE_URL")).unwrap_or_else(|| OPENAI.to_string());
    let base = base.trim_end_matches('/').to_string();
    let key = o.api_key.clone().or_else(|| env("MEMTRAILS_LLM_API_KEY")).or_else(|| {
        if base == OPENAI {
            env("OPENAI_API_KEY")
        } else {
            None
        }
    });
    (model, base, key)
}

async fn ask_openai(o: &OpenAiAskOptions, req: LlmRequest) -> LlmAnswer {
    let (model, base, key) = route(o, &req);
    let Some(model) = model else {
        return LlmAnswer::Unavailable(why("no model is set (the model option, or MEMTRAILS_MODEL)", &req));
    };
    let Some(key) = key else {
        let or_openai = if base == OPENAI { ", or OPENAI_API_KEY" } else { "" };
        return LlmAnswer::Unavailable(why(
            &format!("no API key is set (the apiKey option or MEMTRAILS_LLM_API_KEY{or_openai})"),
            &req,
        ));
    };
    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(model));
    body.insert("instructions".into(), json!(req.system));
    body.insert("input".into(), json!(req.prompt));
    body.insert("max_output_tokens".into(), json!(req.max_tokens.unwrap_or(12_000)));
    if let Some(effort) = o.reasoning_effort {
        body.insert("reasoning".into(), json!({ "effort": effort.as_str() }));
    }
    body.insert(
        "text".into(),
        json!({ "format": { "type": "json_schema", "name": req.name.clone().unwrap_or_else(|| "memory".into()), "strict": true, "schema": req.schema } }),
    );
    let body = crate::js::stringify(&Value::Object(body));
    let tries = o.retries.unwrap_or(5) + 1;
    let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(180_000));
    let mut last = String::new();
    for attempt in 0..tries {
        if attempt > 0 {
            let base_ms = (1000u64 << (attempt - 1).min(16)).min(60_000) as f64;
            let jitter = 0.5 + rand::thread_rng().gen::<f64>();
            tokio::time::sleep(Duration::from_millis((base_ms * jitter) as u64)).await;
        }
        let sent = http()
            .post(format!("{base}/responses"))
            .timeout(timeout)
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await;
        let res = match sent {
            Ok(r) => r,
            Err(e) => {
                last = e.to_string();
                continue;
            }
        };
        let status = res.status().as_u16();
        if status == 429 || status >= 500 {
            last = format!("HTTP {status}");
            continue;
        }
        if !res.status().is_success() {
            // What the provider said (a wrong model name, a schema it refuses) — not only its status.
            let text = res.text().await.unwrap_or_default();
            let said = redact_string(crate::js::trim(&crate::js::collapse_spaces(&text)));
            let said = crate::js::head16(&said, 300);
            let said = if said.is_empty() { String::new() } else { format!(": {said}") };
            return LlmAnswer::Unavailable(why(&format!("the model answered HTTP {status}{said}"), &req));
        }
        let read = async {
            let r: Value = res.json().await.map_err(|e| e.to_string())?;
            let usage = &r["usage"];
            if let Some(f) = &o.on_usage {
                f(LlmUsage {
                    model: r["model"].as_str().unwrap_or(&model).to_string(),
                    name: req.name.clone(),
                    input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
                    cached_tokens: usage["input_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
                    output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
                });
            }
            let text: String = r["output"]
                .as_array()
                .into_iter()
                .flatten()
                .flat_map(|x| x["content"].as_array().cloned().unwrap_or_default())
                .map(|c| c["text"].as_str().unwrap_or("").to_string())
                .collect();
            let answer: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            Ok::<_, String>(LlmAnswer::Answer {
                answer,
                model: Some(r["model"].as_str().unwrap_or(&model).to_string()),
            })
        };
        return match read.await {
            Ok(a) => a,
            Err(e) => LlmAnswer::Unavailable(why(&format!("the model's answer could not be read ({e})"), &req)),
        };
    }
    LlmAnswer::Unavailable(why(&format!("the model did not answer ({last})"), &req))
}

/// A request for memory's own asks: `purpose: memory`, with its name, room, wait and consequence.
pub(crate) fn memory_request(
    system: &str,
    prompt: &str,
    schema: Value,
    name: &str,
    max_tokens: u64,
    timeout_ms: u64,
    consequence: &str,
) -> LlmRequest {
    LlmRequest {
        purpose: "memory".into(),
        system: system.to_string(),
        prompt: redact_string(prompt),
        schema,
        name: Some(name.to_string()),
        model: None,
        max_tokens: Some(max_tokens),
        timeout_ms: Some(timeout_ms),
        consequence: Some(consequence.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> LlmRequest {
        memory_request("s", "p", json!({ "type": "object" }), "memory", 10, 1000, "")
    }

    #[test]
    fn an_openai_key_goes_to_openai_never_to_another_address() {
        // The only test here that reads these variables: nothing else races it.
        std::env::set_var("OPENAI_API_KEY", "openai-key");
        std::env::remove_var("MEMTRAILS_LLM_API_KEY");
        std::env::remove_var("MEMTRAILS_LLM_BASE_URL");
        let o = OpenAiAskOptions { model: Some("m".into()), ..Default::default() };
        assert_eq!(route(&o, &req()), (Some("m".into()), OPENAI.into(), Some("openai-key".into())));
        let other = OpenAiAskOptions { base_url: Some("https://llm.example/v1/".into()), ..o.clone() };
        assert_eq!(route(&other, &req()).2, None);
        std::env::set_var("MEMTRAILS_LLM_API_KEY", "own-key");
        assert_eq!(route(&other, &req()), (Some("m".into()), "https://llm.example/v1".into(), Some("own-key".into())));
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("MEMTRAILS_LLM_API_KEY");
    }
}
