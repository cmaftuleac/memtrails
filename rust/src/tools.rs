//! An agent's tools over a memory: `memory_search` (find), `memory_open` (read one whole, with the records it was
//! learned from) and `memory_explore` (from a topic or items to what is related but not found by search), and the
//! text a reader is shown of a search.

use serde_json::{json, Value};

use crate::js;
use crate::profiles::PROFILE_SHOWN;
use crate::recall::{ExploreQuery, Explored, SearchQuery};
use crate::recall_types::RecalledItem;
use crate::text::cut;
use crate::writer::date_span;

pub(crate) fn day(ts: f64) -> String {
    if ts != 0.0 && ts.is_finite() {
        js::iso_day(ts)
    } else {
        String::new()
    }
}

pub(crate) fn hit_line(h: &RecalledItem, n: usize) -> String {
    let kind = if h.kind == "episode" { "earlier work" } else { h.kind.as_str() };
    let author = h.record.as_ref().and_then(|r| r.author.as_deref()).filter(|a| !a.is_empty());
    format!(
        "{}. {} ({kind} · {}{} · id {})",
        n + 1,
        h.title,
        day(h.ts),
        author.map(|a| format!(" · {a}")).unwrap_or_default(),
        h.id
    )
}

/// A date as a reader reckons with it — the weekday too ("the Friday before …"): 2023-07-14 → "Fri 14 Jul 2023".
/// Not a date (2023-13, 2023-02-30): shown as written.
pub fn readable_date(d: &str) -> String {
    if date_span(d).is_none() {
        return d.to_string();
    }
    let parts: Vec<i64> = d.split('-').map(|p| p.parse().unwrap_or(0)).collect();
    let (y, m, dd) = (parts[0], parts.get(1).copied().unwrap_or(0), parts.get(2).copied().unwrap_or(0));
    if y == 0 {
        return String::new();
    }
    if m == 0 {
        return y.to_string();
    }
    let month = js::MONTHS[(m - 1) as usize];
    if dd == 0 {
        return format!("{month} {y}");
    }
    let wd = js::WEEKDAYS[js::utc_of(js::date_utc(y, m - 1, dd, 0, 0, 0, 0)).weekday as usize];
    format!("{wd} {dd} {month} {y}")
}

/// How long something lasted, as a reader reckons with it: "since Feb 2022", "Nov 2021 – May 2022".
pub fn lasted_text(start_at: Option<&str>, end_at: Option<&str>) -> String {
    let show = |d: Option<&str>| d.filter(|d| !d.is_empty()).map(readable_date).unwrap_or_default();
    let (from, to) = (show(start_at), show(end_at));
    if !from.is_empty() && !to.is_empty() {
        format!("{from} – {to}")
    } else if !from.is_empty() {
        format!("since {from}")
    } else if !to.is_empty() {
        format!("until {to}")
    } else {
        String::new()
    }
}

/// The order `search_text` lists hits in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Order {
    /// Best first.
    #[default]
    Relevance,
    /// Oldest first, by when it happened.
    Time,
}

/// When a hit happened, for ordering by time: an observation's own date, else when it was learned.
fn when(h: &RecalledItem) -> String {
    h.event_at.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| day(h.ts))
}

/// What memory_search shows: an observation as one dated line (when it happened, and when it was told when that
/// differs), a profile whole (up to `PROFILE_SHOWN`), anything else as its title line and summary — shorter when many
/// are asked for. Best first, or by time.
pub fn search_text(hits: &[RecalledItem], order: Order) -> String {
    let mut list: Vec<&RecalledItem> = hits.iter().collect();
    if order == Order::Time {
        list.sort_by_key(|h| when(h));
    }
    // Many hits: one line each — the date, what it is, its summary — so a list question sees every one clearly.
    let compact = list.len() > 20;
    let lines: Vec<String> = list
        .iter()
        .enumerate()
        .map(|(n, h)| {
            if h.kind == "profile" {
                return format!("{}\n   {}", hit_line(h, n), cut(&h.summary, PROFILE_SHOWN).replace('\n', "\n   "));
            }
            if h.kind != "observation" {
                // The passage that matched, when it was not the start: a fact said deep in a long request is what
                // was found.
                let matched = h.match_.as_deref().filter(|m| !m.is_empty());
                let shown = matched.map(|m| format!(" … {}", js::collapse_spaces(m))).unwrap_or_default();
                if !compact {
                    let n_chars = if matched.is_some() { 200 } else { 500 };
                    return format!("{}\n   {}{shown}", hit_line(h, n), js::head16(&h.summary, n_chars));
                }
                let what = if h.kind == "episode" { "earlier work" } else { h.kind.as_str() };
                let n_chars = if matched.is_some() { 150 } else { 300 };
                return format!(
                    "{}. ({} · {what}) {}{shown} (id {})",
                    n + 1,
                    readable_date(&day(h.ts)),
                    js::head16(&js::collapse_spaces(&h.summary), n_chars),
                    h.id
                );
            }
            let told = day(h.ts);
            let lasted = lasted_text(h.start_at.as_deref(), h.end_at.as_deref());
            let event = h.event_at.as_deref().filter(|e| !e.is_empty());
            let happened = [event.map(readable_date).unwrap_or_default(), lasted.clone()]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            let at = if happened.is_empty() {
                format!("told {}", readable_date(&told))
            } else if h.event_at.as_deref() != Some(told.as_str()) || !lasted.is_empty() {
                format!("{happened}; told {}", readable_date(&told))
            } else {
                happened
            };
            format!("{}. ({at}) {} (id {})", n + 1, h.summary, h.id)
        })
        .collect();
    let head = if order == Order::Time {
        "Oldest first — the date is when it happened (and when it was told, when that differs).\n"
    } else {
        ""
    };
    format!("{head}{}", lines.join("\n"))
}

/// A tool's answer, as MCP servers and function-calling APIs take it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ToolResult {
    /// The text blocks (one).
    pub content: Vec<ToolContent>,
    /// An error the agent should read as one.
    #[serde(rename = "isError", skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

/// One block of a tool's answer.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ToolContent {
    /// Always `text`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The text.
    pub text: String,
}

fn text(t: impl Into<String>, is_error: bool) -> ToolResult {
    ToolResult { content: vec![ToolContent { kind: "text".into(), text: t.into() }], is_error }
}

/// A tool's handler: the model's arguments in, the answer out.
pub type ToolHandler = std::sync::Arc<dyn Fn(Value) -> crate::BoxFuture<'static, ToolResult> + Send + Sync>;

/// One tool an agent can be handed.
#[derive(Clone)]
pub struct AgentTool {
    /// `memory_search`, `memory_open` or `memory_explore`.
    pub name: String,
    /// What it does, for the model.
    pub description: String,
    /// Its input as JSON Schema (draft 2020-12), for function calling and MCP.
    pub json_schema: Value,
    /// Runs it. The arguments are checked here too: a caller doing function calling hands the model's arguments
    /// straight to the handler.
    pub handler: ToolHandler,
}

impl std::fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTool").field("name", &self.name).field("json_schema", &self.json_schema).finish()
    }
}

/// What the tools read: a memory gives these.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync {
    /// A search.
    async fn search(&self, q: SearchQuery) -> crate::Result<Vec<RecalledItem>>;
    /// One item whole.
    async fn open(&self, id: &str) -> crate::Result<Option<Value>>;
    /// Exploring; `None` when this provider does not explore.
    async fn explore(&self, q: ExploreQuery) -> Option<crate::Result<Explored>>;
    /// Whether it explores (memory_explore is offered only then).
    fn explores(&self) -> bool {
        true
    }
}

/// What a model may send: a question, a topic or an id is a line, not a document; a few dozen collections at most.
const TEXT_MAX: usize = 2000;
const COLLECTIONS_MAX: usize = 50;
const DATE_PATTERN: &str = r"^\d{4}(-\d{2}(-\d{2})?)?$";

const SEARCH_DESCRIPTION: &str = "Search memory: everything it was given (e-mails, messages, notes, documents, conversations, commits, …) and what was learned from them — facts, decisions, dated events, profiles of people and things. Best first; each result with its date and id. Ask in plain words, as you would ask a colleague who read it all. Name dates when the question names a time. Open a result with memory_open for all of it.";
const OPEN_DESCRIPTION: &str = "Read one memory item whole, by the id memory_search or memory_explore gave: everything it holds, when it happened, and for something learned, the records it was learned from.";
const EXPLORE_DESCRIPTION: &str = "Explore memory around a topic, or around items you already found: it follows what they share — the same people, places, systems, names — and links between memories, to what a search for the words does not find. Returns what search found, what is related, and the concepts around it to explore next.";

/// One field of a tool's input, as the checks and the JSON Schema read it.
enum Field {
    Text { max: usize, pattern: bool },
    Texts { max_items: usize },
    Int { min: i64, max: i64 },
    OneOf(&'static [&'static str]),
}

struct Spec {
    name: &'static str,
    field: Field,
    required: bool,
    description: &'static str,
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// The issues zod would find, each `path: message`, in the order of the fields.
fn check(specs: &[Spec], args: &Value) -> Vec<String> {
    let Some(obj) = args.as_object() else {
        return vec![format!("input: Invalid input: expected object, received {}", type_name(args))];
    };
    let mut issues = Vec::new();
    let text_issue = |v: &Value, max: usize, pattern: bool| -> Option<String> {
        let Some(s) = v.as_str() else {
            return Some(format!("Invalid input: expected string, received {}", type_name(v)));
        };
        if js::len16(s) > max {
            return Some(format!("Too big: expected string to have <={max} characters"));
        }
        if pattern && !crate::writer::is_date_shape(s) {
            return Some(format!("Invalid string: must match pattern /{DATE_PATTERN}/"));
        }
        None
    };
    for s in specs {
        let v = obj.get(s.name);
        let Some(v) = v else {
            if s.required {
                issues.push(format!("{}: Invalid input: expected {}, received undefined", s.name, expected(&s.field)));
            }
            continue;
        };
        match &s.field {
            Field::Text { max, pattern } => {
                if let Some(m) = text_issue(v, *max, *pattern) {
                    issues.push(format!("{}: {m}", s.name));
                }
            }
            Field::Texts { max_items } => match v.as_array() {
                None => issues.push(format!("{}: Invalid input: expected array, received {}", s.name, type_name(v))),
                Some(a) => {
                    if a.len() > *max_items {
                        issues.push(format!("{}: Too big: expected array to have <={max_items} items", s.name));
                    }
                    for (i, x) in a.iter().enumerate() {
                        if let Some(m) = text_issue(x, TEXT_MAX, false) {
                            issues.push(format!("{}.{i}: {m}", s.name));
                        }
                    }
                }
            },
            Field::Int { min, max } => match v.as_f64() {
                None => issues.push(format!("{}: Invalid input: expected number, received {}", s.name, type_name(v))),
                Some(n) if n.fract() != 0.0 => {
                    issues.push(format!("{}: Invalid input: expected int, received number", s.name))
                }
                Some(n) if n < *min as f64 => {
                    issues.push(format!("{}: Too small: expected number to be >={min}", s.name))
                }
                Some(n) if n > *max as f64 => {
                    issues.push(format!("{}: Too big: expected number to be <={max}", s.name))
                }
                Some(_) => {}
            },
            Field::OneOf(options) => {
                if !v.as_str().is_some_and(|x| options.contains(&x)) {
                    let list = options.iter().map(|o| format!("\"{o}\"")).collect::<Vec<_>>().join("|");
                    issues.push(format!("{}: Invalid option: expected one of {list}", s.name));
                }
            }
        }
    }
    issues
}

fn expected(f: &Field) -> &'static str {
    match f {
        Field::Text { .. } | Field::OneOf(_) => "string",
        Field::Texts { .. } => "array",
        Field::Int { .. } => "number",
    }
}

/// The input's JSON Schema, as zod writes it for the reference.
fn json_schema(specs: &[Spec]) -> Value {
    let mut properties = serde_json::Map::new();
    for s in specs {
        let mut p = serde_json::Map::new();
        // zod writes a required field's own keys first, an optional one's description first.
        let described = |p: &mut serde_json::Map<String, Value>| {
            p.insert("description".into(), json!(s.description));
        };
        if !s.required {
            described(&mut p);
        }
        match &s.field {
            Field::Text { max, pattern } => {
                p.insert("type".into(), json!("string"));
                p.insert("maxLength".into(), json!(max));
                if *pattern {
                    p.remove("maxLength");
                    p.insert("pattern".into(), json!(DATE_PATTERN));
                }
            }
            Field::Texts { max_items } => {
                p.insert("maxItems".into(), json!(max_items));
                p.insert("type".into(), json!("array"));
                p.insert("items".into(), json!({ "type": "string", "maxLength": TEXT_MAX }));
            }
            Field::Int { min, max } => {
                p.insert("type".into(), json!("integer"));
                p.insert("minimum".into(), json!(min));
                p.insert("maximum".into(), json!(max));
            }
            Field::OneOf(options) => {
                p.insert("type".into(), json!("string"));
                p.insert("enum".into(), json!(options));
            }
        }
        if s.required {
            described(&mut p);
        }
        properties.insert(s.name.into(), Value::Object(p));
    }
    let mut schema = serde_json::Map::new();
    schema.insert("$schema".into(), json!("https://json-schema.org/draft/2020-12/schema"));
    schema.insert("type".into(), json!("object"));
    schema.insert("properties".into(), Value::Object(properties));
    let required: Vec<&str> = specs.iter().filter(|s| s.required).map(|s| s.name).collect();
    if !required.is_empty() {
        schema.insert("required".into(), json!(required));
    }
    schema.insert("additionalProperties".into(), json!(false));
    Value::Object(schema)
}

fn tool<F>(name: &str, description: &str, specs: Vec<Spec>, run: F) -> AgentTool
where
    F: Fn(Value) -> crate::BoxFuture<'static, ToolResult> + Send + Sync + 'static,
{
    let json_schema = json_schema(&specs);
    let specs = std::sync::Arc::new(specs);
    let run = std::sync::Arc::new(run);
    AgentTool {
        name: name.into(),
        description: description.into(),
        json_schema,
        handler: std::sync::Arc::new(move |args: Value| {
            let args = if args.is_null() { json!({}) } else { args };
            let issues = check(&specs, &args);
            if !issues.is_empty() {
                let t = text(format!("Invalid arguments: {}", issues.join("; ")), true);
                return Box::pin(async move { t });
            }
            run(args)
        }),
    }
}

fn strings_of(v: &Value) -> Option<Vec<String>> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
}

/// memory_search, memory_open and memory_explore (when the provider explores) over a memory.
pub fn build_agent_tools(p: std::sync::Arc<dyn ToolProvider>) -> Vec<AgentTool> {
    let search_specs = vec![
        Spec { name: "query", field: Field::Text { max: TEXT_MAX, pattern: false }, required: true, description: "What you need to know, in plain words." },
        Spec { name: "collections", field: Field::Texts { max_items: COLLECTIONS_MAX }, required: false, description: "Only these collections (default: all)." },
        Spec { name: "limit", field: Field::Int { min: 1, max: 200 }, required: false, description: "How many of each to return — dated observations, and other memories (default 20; 100 to 200 when you gather everything on a topic, list or count things, or need their dates)." },
        Spec { name: "order", field: Field::OneOf(&["relevance", "time"]), required: false, description: "Best first (default), or oldest first — for what happened when, and in what order." },
        Spec { name: "from", field: Field::Text { max: usize::MAX, pattern: true }, required: false, description: "Only what happened on or after this date (YYYY-MM-DD, YYYY-MM or YYYY) — when the request names a time (\"in June 2023\", \"last week\")." },
        Spec { name: "to", field: Field::Text { max: usize::MAX, pattern: true }, required: false, description: "Only what happened on or before this date (YYYY-MM-DD, YYYY-MM or YYYY)." },
    ];
    let sp = p.clone();
    let mut tools = vec![tool("memory_search", SEARCH_DESCRIPTION, search_specs, move |args| {
        let p = sp.clone();
        Box::pin(async move {
            let q = SearchQuery {
                query: args["query"].as_str().unwrap_or("").to_string(),
                collections: strings_of(&args["collections"]),
                kinds: None,
                limit: Some(args["limit"].as_u64().map_or(20, |n| n as usize)),
                from: args["from"].as_str().map(str::to_string),
                to: args["to"].as_str().map(str::to_string),
            };
            let order = if args["order"] == "time" { Order::Time } else { Order::Relevance };
            match p.search(q).await {
                Ok(hits) if hits.is_empty() => text("Nothing in memory matches that.", false),
                Ok(hits) => text(
                    format!(
                        "{}\n\nOpen one with memory_open and its id. It can be stale: check it against the current state before relying on it.",
                        search_text(&hits, order)
                    ),
                    false,
                ),
                Err(e) => text(e.to_string(), true),
            }
        })
    })];
    let op = p.clone();
    tools.push(tool(
        "memory_open",
        OPEN_DESCRIPTION,
        vec![Spec {
            name: "id",
            field: Field::Text { max: TEXT_MAX, pattern: false },
            required: true,
            description: "The memory id, as memory_search or memory_explore gave it.",
        }],
        move |args| {
            let p = op.clone();
            Box::pin(async move {
                let id = args["id"].as_str().unwrap_or("").to_string();
                // An index that failed is not an id that is missing: the agent should not conclude the memory is gone.
                let it = match p.open(&id).await {
                    Ok(it) => it,
                    Err(e) => return text(format!("Memory could not be read: {e}"), true),
                };
                let Some(it) = it else {
                    return text(format!("There is no memory \"{id}\" (it may have been merged or cleared)."), true);
                };
                text(open_text(&it), false)
            })
        },
    ));
    if p.explores() {
        let ep = p.clone();
        let specs = vec![
            Spec {
                name: "topic",
                field: Field::Text { max: TEXT_MAX, pattern: false },
                required: false,
                description: "What to explore, in plain words — a system, a project, a problem, a name.",
            },
            Spec {
                name: "from",
                field: Field::Texts { max_items: 5 },
                required: false,
                description: "Memory ids to explore from, as memory_search or memory_explore gave them.",
            },
            Spec {
                name: "collections",
                field: Field::Texts { max_items: COLLECTIONS_MAX },
                required: false,
                description: "Only these collections (default: all).",
            },
            Spec {
                name: "limit",
                field: Field::Int { min: 1, max: 20 },
                required: false,
                description: "How many related items to return (default 10).",
            },
        ];
        tools.push(tool("memory_explore", EXPLORE_DESCRIPTION, specs, move |args| {
            let p = ep.clone();
            Box::pin(async move {
                let topic = args["topic"].as_str().map(str::to_string);
                let from = strings_of(&args["from"]);
                if topic.as_deref().is_none_or(|t| js::trim(t).is_empty()) && from.as_ref().is_none_or(|f| f.is_empty())
                {
                    return text("Give a topic, or memory ids to explore from.", true);
                }
                let q = ExploreQuery {
                    topic,
                    from,
                    collections: strings_of(&args["collections"]),
                    limit: Some(args["limit"].as_u64().map_or(10, |n| n as usize)),
                };
                match p.explore(q).await {
                    None => text("Nothing in memory reaches that.", false),
                    Some(Err(e)) => text(e.to_string(), true),
                    Some(Ok(r)) => text(explore_text(&r), false),
                }
            })
        }));
    }
    tools
}

/// What memory_open shows of an item.
fn open_text(it: &Value) -> String {
    let files = it["files"]
        .as_array()
        .filter(|f| !f.is_empty())
        .map(|f| {
            format!("\nFiles: {}", f.iter().map(crate::qdrant_client::value_string).collect::<Vec<_>>().join(", "))
        })
        .unwrap_or_default();
    let told = day(it["ts"].as_f64().unwrap_or(0.0));
    let event = it["eventAt"].as_str().filter(|e| !e.is_empty()).map(readable_date).unwrap_or_default();
    let lasted = lasted_text(it["startAt"].as_str(), it["endAt"].as_str());
    let happened = [event, lasted].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", ");
    let when = if happened.is_empty() { told } else { format!("happened {happened}; told {}", readable_date(&told)) };
    // The raw requests it was learned from, whole (the truth it is a key to).
    let sources = it["sources"]
        .as_array()
        .filter(|s| !s.is_empty())
        .map(|s| {
            let parts: Vec<String> = s
                .iter()
                .map(|x| {
                    format!(
                        "### {} · {} (id {})\n{}",
                        day(x["when"].as_f64().unwrap_or(0.0)),
                        crate::qdrant_client::value_string(&x["title"]),
                        crate::qdrant_client::value_string(&x["id"]),
                        crate::qdrant_client::value_string(&x["body"])
                    )
                })
                .collect();
            format!("\n\n## Where it came from\n{}", parts.join("\n\n"))
        })
        .unwrap_or_default();
    let s = |k: &str| if it[k].is_null() { String::new() } else { crate::qdrant_client::value_string(&it[k]) };
    format!(
        "# {}\n({} · {when} · {})\n\n{}\n\n{}{files}{sources}",
        s("title"),
        s("kind"),
        s("project"),
        s("summary"),
        s("body")
    )
}

/// What memory_explore shows.
fn explore_text(r: &Explored) -> String {
    let mut parts = Vec::new();
    if !r.found.is_empty() {
        parts.push(format!(
            "Found by search:\n{}",
            r.found.iter().enumerate().map(|(n, h)| hit_line(h, n)).collect::<Vec<_>>().join("\n")
        ));
    }
    if !r.related.is_empty() {
        parts.push(format!(
            "Related — not found by search, reached through what they share:\n{}",
            r.related
                .iter()
                .enumerate()
                .map(|(n, x)| format!("{} — {}\n   {}", hit_line(&x.item, n), x.via, js::head16(&x.item.summary, 300)))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    if !r.concepts.is_empty() {
        parts.push(format!("Concepts around it: {}", r.concepts.join(", ")));
    }
    if parts.is_empty() {
        return "Nothing in memory reaches that.".into();
    }
    format!(
        "{}\n\nOpen one with memory_open; explore further from an item (from: [its id]) or with one of the concepts as the topic. It can be stale: check it against the current state before relying on it.",
        parts.join("\n\n")
    )
}
