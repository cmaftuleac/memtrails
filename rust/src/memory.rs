//! A memory: open it, add anything with text, ask it things.
//!
//! What is added is searchable at once, with no language model and no server: the embedding model runs here (fetched
//! on first use) and orders results word by word (words.rs). A language model is needed only for `consolidate`,
//! `sleep` and `gather`, which turn records into typed knowledge, dated observations and profiles — and memory works
//! without them.
//!
//! The JSONL logs in `home` are the truth. The Qdrant index and the word vectors are derived: lost or out of date,
//! they are written again from the logs.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Once};

use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use crate::consolidate::{ConsolidateDeps, ConsolidationReport, Consolidator};
use crate::download::DownloadOptions;
use crate::embedder::{remote_embedder, resolve_embed_route, EmbedFn, EmbedRoute};
use crate::episodes::episode_body;
use crate::error::{Error, Result};
use crate::gather::{memory_gather, GatherOptions, Gathered};
use crate::js;
use crate::llm::{default_ask, openai_ask, Ask, OpenAiAskOptions};
use crate::local_embedder::{load_local_embedder, LocalEmbedder, LocalEmbedderOptions};
use crate::model::ensure_model;
use crate::qdrant_binary::ensure_qdrant;
use crate::qdrant_client::{Endpoint, QdrantClient, COLLECTION, INDEX_LAYOUT};
use crate::qdrant_process::{QdrantEndpoint, QdrantProcess, QdrantProcessOptions};
use crate::rebuild::{MemoryRebuild, RebuildDeps};
use crate::recall::{
    memory_explore, memory_open, memory_search, ExploreQuery, Explored, RecallDeps, RecalledItem, SearchQuery,
};
use crate::recall_log::RecallBook;
use crate::records::{conversation_entries, entry_of, invalid_record, record_uid, Conversation, MemoryRecord};
use crate::reranker::{remote_reranker, resolve_rerank_route, RerankRoute};
use crate::room::{sentence_room, sentence_spans, RoomItem};
use crate::sleep::{sleep, SleepDeps, SleepReport};
use crate::store::{blocking, ltm_log, stm_log, write_private, LtmStore, StmEntry, StmStore, UsageBook};
use crate::sys;
use crate::tools::{build_agent_tools, lasted_text, readable_date, search_text, AgentTool, Order, ToolProvider};
use crate::words::{pack_words, word_reranker, WordEncoder, WordKind, WordStore};
use crate::writer::{raw_key, stm_item, stm_title, write_ltm, write_stm, WriterDeps};
use crate::{LogFn, RerankFn};

const BATCH: usize = 32;
const ROOM_TEXT: usize = 1500;
const ROOM_CHARS: usize = 6000;
const RECORD_LIMIT: usize = 2_000_000;
/// In the home: who has it open, and the index layout its Qdrant collection was written in.
const LOCK_FILE: &str = "memory.lock";
const LAYOUT_FILE: &str = "index.json";

/// How texts are embedded (EmbeddingGemma-300m in every case: the index, prompts and measured thresholds are its).
#[derive(Debug, Clone, Default)]
pub struct EmbeddingOptions {
    /// On this machine, from a directory you provide (default: the model fetched into the cache, 218 MB).
    pub model_dir: Option<PathBuf>,
    /// A server that serves it over the OpenAI embeddings API. Word by word then needs `model_dir` too, or it is off.
    /// With neither, `MEMTRAILS_EMBED_URL` (+ `_KEY`, `_MODEL`) names a server when it is set.
    pub server: Option<EmbedRoute>,
    /// CPU threads for the local model (default 2).
    pub threads: Option<usize>,
}

/// How the best 30 of a search are reordered.
#[derive(Clone, Default)]
pub enum Rerank {
    /// Word by word with a local model, else a server `MEMTRAILS_RERANK_URL` names.
    #[default]
    Default,
    /// Word by word (needs the local model).
    Words,
    /// A cross-encoder server.
    Server(RerankRoute),
    /// Your own function.
    Custom(RerankFn),
    /// The search's own order.
    Off,
}

/// The language model for consolidate, sleep and gather.
#[derive(Clone, Default)]
pub enum Llm {
    /// The one the environment names (`MEMTRAILS_MODEL` and a key).
    #[default]
    Env,
    /// The OpenAI-compatible one, with these options.
    OpenAi(OpenAiAskOptions),
    /// Your own.
    Custom(Arc<dyn Ask>),
}

/// The index.
#[derive(Debug, Clone, Default)]
pub enum QdrantOption {
    /// A process of its own, the pinned binary fetched on first use.
    #[default]
    Own,
    /// A process of its own, from this binary.
    Binary(PathBuf),
    /// A server you run.
    Server {
        /// Its address.
        url: String,
        /// Its key.
        api_key: Option<String>,
        /// The collection (default `memory`).
        collection: Option<String>,
    },
}

/// What a memory is opened with. Only `home` is required.
#[derive(Clone, Default)]
pub struct MemoryOptions {
    /// Where this memory lives: its logs, its index, its word vectors.
    pub home: PathBuf,
    /// How texts are embedded.
    pub embedding: EmbeddingOptions,
    /// How the best 30 of a search are reordered.
    pub rerank: Rerank,
    /// The language model for consolidate, sleep and gather.
    pub llm: Llm,
    /// The index.
    pub qdrant: QdrantOption,
    /// Windows the language model reads at once during consolidation (default 4).
    pub concurrency: Option<usize>,
    /// Progress lines.
    pub log: Option<LogFn>,
    /// How downloads run (where they are kept, a mirror).
    pub download: DownloadOptions,
}

impl MemoryOptions {
    /// Options for a memory in `home`, everything else by default.
    pub fn new(home: impl Into<PathBuf>) -> Self {
        MemoryOptions { home: home.into(), ..Default::default() }
    }
}

/// A search, with how its text is ordered.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// The search.
    pub query: SearchQuery,
    /// For `search_text`: best first (default) or oldest first.
    pub order: Order,
}

/// What an add did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Added {
    /// Records new or changed.
    pub added: usize,
    /// Records already there as they are.
    pub unchanged: usize,
}

/// What a memory holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Records.
    pub records: usize,
    /// Active long-term items.
    pub items: usize,
    /// Texts with word vectors.
    pub words: usize,
}

/// A record to forget: its id, and its source when it has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordId {
    /// Its id.
    pub id: String,
    /// Its source.
    pub source: Option<String>,
}

impl From<&str> for RecordId {
    fn from(id: &str) -> Self {
        RecordId { id: id.into(), source: None }
    }
}

impl From<String> for RecordId {
    fn from(id: String) -> Self {
        RecordId { id, source: None }
    }
}

/// The homes this process holds, released if it exits without close().
static HOLDING: LazyLock<Mutex<HashSet<PathBuf>>> = LazyLock::new(|| Mutex::new(HashSet::new()));
static EXIT_HOOK: Once = Once::new();

extern "C" fn release_all() {
    if let Ok(held) = HOLDING.lock() {
        for path in held.iter() {
            release_lock_file(path);
        }
    }
}

fn release_lock_file(path: &Path) {
    let ours = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .is_some_and(|v| v["pid"].as_u64() == Some(u64::from(sys::pid())));
    if ours {
        let _ = std::fs::remove_file(path);
    }
}

/// Who holds a lock: its live pid, 0 when it is being written this instant, `None` when nobody (stale or gone).
fn lock_holder(path: &Path) -> (String, Option<i64>) {
    let Ok(text) = std::fs::read_to_string(path) else { return (String::new(), None) };
    match serde_json::from_str::<Value>(&text) {
        Ok(l) => {
            let pid = l["pid"].as_i64().unwrap_or(0);
            let held = sys::same_boot(l["boot"].as_f64()) && sys::alive(pid);
            (text, held.then_some(pid))
        }
        Err(_) => {
            // Not written yet by the opener that just made it — or torn by a crash, and then old.
            let fresh = std::fs::metadata(path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| std::time::SystemTime::now().duration_since(t).ok())
                .is_some_and(|age| age.as_millis() < 10_000);
            (text, fresh.then_some(0))
        }
    }
}

/// One process at a time in a home: two would run two Qdrants on one storage and append to the logs at offsets the
/// other does not know. The lock is made exclusively with its pid and the boot; one whose process is gone, or ran
/// before a reboot (its pid may be another's now), is taken over.
fn lock_home(dir: &Path) -> Result<PathBuf> {
    let path = dir.join(LOCK_FILE);
    for attempt in 0.. {
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        match o.open(&path) {
            Ok(mut f) => {
                use std::io::Write;
                let text = js::stringify(&json!({ "pid": sys::pid(), "boot": js::ms_value(sys::boot_time()) }));
                f.write_all(text.as_bytes()).map_err(|e| Error::io(&path, e))?;
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let (text, pid) = lock_holder(&path);
                if pid.is_some() || attempt > 0 {
                    let by = match pid {
                        Some(p) if p > 0 => format!(" in process {p}"),
                        _ => String::new(),
                    };
                    return Err(Error::Locked(format!(
                        "memory {} is open{by}: close it there first (if no process has it, delete {})",
                        dir.display(),
                        path.display()
                    )));
                }
                // Stale: taken over — unless another opener took it over since it was read.
                if std::fs::read_to_string(&path).is_ok_and(|t| t == text) {
                    let _ = std::fs::remove_file(&path);
                }
            }
            Err(e) => return Err(Error::io(&path, e)),
        }
    }
    if let Ok(mut h) = HOLDING.lock() {
        h.insert(path.clone());
    }
    EXIT_HOOK.call_once(|| {
        #[cfg(unix)]
        // SAFETY: release_all only reads a static and removes files.
        unsafe {
            libc::atexit(release_all);
        }
    });
    Ok(path)
}

fn unlock_home(path: &Path) {
    let held = HOLDING.lock().map(|mut h| h.remove(path)).unwrap_or(false);
    if held {
        release_lock_file(path);
    }
}

struct Inner {
    dir: PathBuf,
    log: LogFn,
    client: QdrantClient,
    embed: EmbedFn,
    encode: Option<WordEncoder>,
    words: WordStore,
    local: Option<LocalEmbedder>,
    proc: Option<QdrantProcess>,
    stm: StmStore,
    ltm: LtmStore,
    usage: UsageBook,
    book: RecallBook,
    ask: Arc<dyn Ask>,
    rerank: Option<RerankFn>,
    rebuild: MemoryRebuild,
    concurrency: usize,
    closing: Arc<AtomicBool>,
    /// consolidate, sleep, rebuild, index_words and remove one at a time: each reads the logs and writes the index from
    /// them — two consolidations pay the model twice and write stale copies over each other, a rebuild clears what
    /// another is writing, a remove during a consolidation would see its record's episode written back after it.
    jobs: tokio::sync::Mutex<()>,
    closed: tokio::sync::OnceCell<()>,
    lock: PathBuf,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Best effort for a memory dropped without close(): Qdrant stopped (its own Drop), the home released.
        if let Some(p) = &self.proc {
            p.stop();
        }
        unlock_home(&self.lock);
    }
}

/// An open memory. Cheap to clone; every clone is the same memory. Send and Sync: use it from any task.
#[derive(Clone)]
pub struct Memory {
    inner: Arc<Inner>,
}

/// What an open holds while it runs: the home's lock and what it has started. An open that fails releases them on
/// its way out (waiting for Qdrant to stop); one that is cancelled (its future dropped) releases them here, on drop —
/// it once left the home locked for the life of the process. An open that completes hands them to the memory.
struct Started {
    lock: PathBuf,
    proc: Option<QdrantProcess>,
    local: Option<LocalEmbedder>,
    open: bool,
}

impl Started {
    /// Released now, waiting for Qdrant to stop (an open that failed).
    async fn release(mut self) {
        if let Some(p) = self.proc.take() {
            p.stop_and_wait(10_000).await;
        }
        if let Some(l) = self.local.take() {
            l.close().await;
        }
    }
}

impl Drop for Started {
    fn drop(&mut self) {
        if self.open {
            return;
        }
        if let Some(p) = self.proc.take() {
            p.stop();
        }
        if let Some(l) = self.local.take() {
            // The model is freed on a blocking thread when there is a runtime to run it; else here.
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move { l.close().await });
            }
        }
        unlock_home(&self.lock);
    }
}

impl Memory {
    /// Open the memory in `o.home` (made, private, if it is not there), fetching the model and Qdrant on first use.
    pub async fn open(o: MemoryOptions) -> Result<Memory> {
        let dir = std::path::absolute(&o.home).map_err(|e| Error::io(&o.home, e))?;
        let log: LogFn = o.log.clone().unwrap_or_else(|| Arc::new(|_| {}));
        // Private: the logs, the index and the word vectors are everything that was added (a shared machine).
        let made = dir.clone();
        blocking(move || sys::make_dir(&made, 0o700)).await?.map_err(|e| Error::io(&dir, e))?;
        if let Err(e) = sys::chmod(&dir, 0o700) {
            log(&format!("[memtrails] {} could not be made private: {e}", dir.display()));
        }
        let at = dir.clone();
        // Held from inside the file task: an open cancelled while the lock is taken still lets it go when the task
        // ends.
        let mut started =
            blocking(move || lock_home(&at).map(|lock| Started { lock, proc: None, local: None, open: false }))
                .await??;
        let lock = started.lock.clone();
        match Self::open_in(&dir, &o, &log, &mut started, lock).await {
            Ok(m) => {
                started.open = true;
                Ok(m)
            }
            Err(e) => {
                started.release().await;
                Err(e)
            }
        }
    }

    async fn open_in(
        dir: &Path,
        o: &MemoryOptions,
        log: &LogFn,
        started: &mut Started,
        lock: PathBuf,
    ) -> Result<Memory> {
        let mut downloads = o.download.clone();
        if downloads.log.is_none() {
            downloads.log = Some(log.clone());
        }
        // Embedding: here, or on a server — the options' own, else the one the environment names.
        let e = &o.embedding;
        let server = e.server.clone().or_else(|| if e.model_dir.is_none() { resolve_embed_route() } else { None });
        let local = if server.is_none() || e.model_dir.is_some() {
            let model = match &e.model_dir {
                Some(d) => d.clone(),
                None => ensure_model(&downloads, None).await?,
            };
            let onnxruntime = Some(crate::runtime::ensure_runtime(&downloads).await?);
            let options = LocalEmbedderOptions { onnxruntime, threads: e.threads.unwrap_or(2), ..Default::default() };
            Some(load_local_embedder(&model, options).await?)
        } else {
            None
        };
        started.local = local.clone();
        let embed: EmbedFn = match (&server, &local) {
            (Some(s), _) => remote_embedder(s.clone()),
            (None, Some(l)) => l.embed_fn(),
            (None, None) => {
                return Err(Error::EmbeddingUnavailable("no embedder: give a model folder or a server".into()))
            }
        };
        let encode = local.as_ref().map(LocalEmbedder::word_encoder);
        let words = WordStore::new(Some(dir.join("words.bin")));
        words.load().await?;

        // The index: a Qdrant of its own, or yours.
        let (endpoint, collection, proc) = match &o.qdrant {
            QdrantOption::Server { url, api_key, collection } => (
                Endpoint::Fixed(QdrantEndpoint {
                    url: url.trim_end_matches('/').to_string(),
                    api_key: api_key.clone().unwrap_or_default(),
                }),
                collection.clone().unwrap_or_else(|| COLLECTION.into()),
                None,
            ),
            other => {
                let binary = match other {
                    QdrantOption::Binary(b) => b.clone(),
                    _ => ensure_qdrant(&downloads, None).await?,
                };
                let p = QdrantProcess::new(QdrantProcessOptions {
                    binary,
                    dir: dir.join("qdrant"),
                    log: log.clone(),
                    on_down: None,
                });
                started.proc = Some(p.clone());
                p.start().await?;
                (Endpoint::Process(p.clone()), COLLECTION.to_string(), Some(p))
            }
        };
        let client = QdrantClient::new(endpoint, Some(&collection));
        client.ensure_collection().await?;

        let stm = stm_log(dir.join("records.jsonl"));
        let ltm = ltm_log(dir.join("knowledge.jsonl"));
        stm.load().await?;
        ltm.load().await?;
        let usage = UsageBook::open(dir.join("usage.json")).await;
        let book = RecallBook::new(dir.join("recall.jsonl"));
        // The documented cap: once an open, and every so many events while open.
        let trimming = book.clone();
        tokio::spawn(async move {
            trimming.trim().await;
        });
        let ask: Arc<dyn Ask> = match &o.llm {
            Llm::Env => default_ask(),
            Llm::OpenAi(x) => openai_ask(x.clone()),
            Llm::Custom(a) => a.clone(),
        };
        let rerank: Option<RerankFn> = match &o.rerank {
            Rerank::Off => None,
            Rerank::Custom(f) => Some(f.clone()),
            Rerank::Server(r) => Some(remote_reranker(r.clone())),
            Rerank::Words => encode.clone().map(|e| word_reranker(e, words.clone())),
            Rerank::Default => match resolve_rerank_route() {
                Some(r) => Some(remote_reranker(r)),
                None => encode.clone().map(|e| word_reranker(e, words.clone())),
            },
        };
        let closing = Arc::new(AtomicBool::new(false));
        let writer = WriterDeps { client: client.clone(), embed: embed.clone() };
        let (w1, w2) = (writer.clone(), writer);
        let c = closing.clone();
        let rebuild = MemoryRebuild::new(RebuildDeps {
            path: dir.join("rebuild.json"),
            ltm: ltm.clone(),
            stm: stm.clone(),
            write_ltm: Arc::new(move |xs| {
                let w = w1.clone();
                Box::pin(async move { write_ltm(&w, &xs).await })
            }),
            write_stm: Arc::new(move |xs| {
                let w = w2.clone();
                Box::pin(async move { write_stm(&w, &xs).await })
            }),
            aborted: Arc::new(move || c.load(Ordering::SeqCst)),
            concurrency: 1,
        });

        // The index is derived: written again from the logs when it is empty while they are not (a new Qdrant, a
        // deleted folder), or was written in another layout — or in one not written down.
        let indexed = client.count(None).await?;
        let layout = tokio::fs::read_to_string(dir.join(LAYOUT_FILE))
            .await
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v["layout"].as_f64());
        if layout != Some(f64::from(INDEX_LAYOUT)) {
            if indexed > 0 {
                let was = layout.map_or_else(|| "unknown".to_string(), js::number);
                log(&format!("[memtrails] the index was written in layout {was}, not {INDEX_LAYOUT}: writing it again from the logs"));
                rebuild.begin().await?; // before the drop: a crash after it resumes
                client.drop_collection().await;
                client.ensure_collection().await?;
            }
            let path = dir.join(LAYOUT_FILE);
            blocking(move || {
                let tmp = PathBuf::from(format!("{}.tmp", path.display()));
                write_private(&tmp, js::stringify(&json!({ "layout": INDEX_LAYOUT })).as_bytes())?;
                std::fs::rename(&tmp, &path)
            })
            .await?
            .map_err(|e| Error::io(dir, e))?;
        }
        if rebuild.pending().await || (indexed == 0 && (stm.size() > 0 || ltm.size() > 0)) {
            log("[memtrails] writing the index from the logs");
            if !rebuild.pending().await {
                rebuild.begin().await?;
            }
            rebuild.run().await?;
        }
        Ok(Memory {
            inner: Arc::new(Inner {
                dir: dir.to_path_buf(),
                log: log.clone(),
                client,
                embed,
                encode,
                words,
                local,
                proc,
                stm,
                ltm,
                usage,
                book,
                ask,
                rerank,
                rebuild,
                concurrency: o.concurrency.unwrap_or(4),
                closing,
                jobs: tokio::sync::Mutex::new(()),
                closed: tokio::sync::OnceCell::new(),
                lock,
            }),
        })
    }

    fn closing(&self) -> bool {
        self.inner.closing.load(Ordering::SeqCst)
    }

    fn writer(&self) -> WriterDeps {
        WriterDeps { client: self.inner.client.clone(), embed: self.inner.embed.clone() }
    }

    fn recall(&self) -> RecallDeps {
        let usage = self.inner.usage.clone();
        let book = self.inner.book.clone();
        RecallDeps {
            client: self.inner.client.clone(),
            embed: self.inner.embed.clone(),
            used: Some(Arc::new(move |ids, opened| usage.note(ids, opened, js::now_ms()))),
            record: Some(Arc::new(move |e| book.record(e))),
            rerank: self.inner.rerank.clone(),
        }
    }

    fn base(&self) -> ConsolidateDeps {
        let i = &self.inner;
        let c = i.closing.clone();
        ConsolidateDeps {
            log: i.log.clone(),
            aborted: Arc::new(move || c.load(Ordering::SeqCst)),
            concurrency: i.concurrency,
            ..ConsolidateDeps::new(i.client.clone(), i.embed.clone(), i.stm.clone(), i.ltm.clone(), i.ask.clone())
        }
    }

    /// One job at a time; refused once the memory is closing.
    async fn exclusive<T>(&self, job: impl std::future::Future<Output = Result<T>>) -> Result<T> {
        let _turn = self.inner.jobs.lock().await;
        if self.closing() {
            return Err(Error::Closed);
        }
        job.await
    }

    /// The text word by word scores an item by: its key, as search returns it.
    fn key_text(title: &str, summary: &str) -> String {
        format!("{title}\n{summary}")
    }

    fn words_of_entries(es: &[StmEntry]) -> Vec<String> {
        let mut out = Vec::new();
        for x in es {
            let joined = format!("{}\n{}", x.request, x.reply);
            let room = js::head16(&joined, ROOM_TEXT);
            out.push(Self::key_text(&stm_title(x), &raw_key(x)));
            out.extend(sentence_spans(room).into_iter().map(|(a, b)| js::slice16(room, a, b).to_string()));
        }
        out
    }

    async fn encode_missing(&self, texts: Vec<String>) -> Result<usize> {
        let Some(encode) = &self.inner.encode else { return Ok(0) };
        let mut seen = HashSet::new();
        let todo: Vec<String> = texts
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .filter(|t| !js::trim(t).is_empty() && self.inner.words.get(t).is_none())
            .collect();
        for part in todo.chunks(BATCH) {
            if self.closing() {
                break;
            }
            let encoded = encode(part.to_vec(), WordKind::Document).await?;
            self.inner.words.put(part.iter().cloned().zip(encoded.iter().map(pack_words)).collect()).await?;
        }
        Ok(todo.len())
    }

    async fn write(&self, entries: Vec<StmEntry>, total: usize) -> Result<Added> {
        if self.closing() {
            return Err(Error::Closed);
        }
        let fresh: Vec<StmEntry> =
            entries.into_iter().filter(|x| self.inner.stm.meta(&x.uid).is_none_or(|m| m.fp != x.fingerprint)).collect();
        let w = self.writer();
        for part in fresh.chunks(BATCH) {
            write_stm(&w, part).await?;
            self.inner.stm.put(part.to_vec()).await?;
            self.encode_missing(Self::words_of_entries(part)).await?;
        }
        self.inner.stm.compact().await?;
        Ok(Added { added: fresh.len(), unchanged: total - fresh.len() })
    }

    /// Add or update records. Unchanged ones cost nothing. Searchable when this returns. An invalid record fails the
    /// whole call before anything is written.
    pub async fn add(&self, records: Vec<MemoryRecord>) -> Result<Added> {
        let now = js::now_ms();
        for r in &records {
            if let Some(bad) = invalid_record(r) {
                return Err(Error::InvalidRecord(bad));
            }
            if js::len16(&r.text) > RECORD_LIMIT {
                return Err(Error::InvalidRecord(format!(
                    "record {}: longer than {RECORD_LIMIT} characters; split it",
                    r.id
                )));
            }
        }
        let total = records.len();
        self.write(records.iter().map(|r| entry_of(r, now, 0)).collect(), total).await
    }

    /// Add or update a conversation: one record for each time the person speaks, with what was answered.
    pub async fn add_conversation(&self, c: &Conversation) -> Result<Added> {
        if c.id.is_empty() {
            return Err(Error::InvalidRecord("a conversation needs an id and turns".into()));
        }
        let entries = conversation_entries(c, js::now_ms());
        let total = entries.len();
        self.write(entries, total).await
    }

    /// Add records from any stream (an importer, a database cursor) in batches (256 by default), streaming;
    /// `on_progress` is told how many were read so far.
    pub async fn import<S>(
        &self,
        records: S,
        batch: Option<usize>,
        on_progress: Option<&(dyn Fn(usize) + Send + Sync)>,
    ) -> Result<Added>
    where
        S: Stream<Item = Result<MemoryRecord>> + Send,
    {
        let size = batch.unwrap_or(256).max(1);
        let mut total = Added::default();
        let mut records = std::pin::pin!(records.chunks(size));
        while let Some(chunk) = records.next().await {
            let batch: Vec<MemoryRecord> = chunk.into_iter().collect::<Result<_>>()?;
            let r = self.add(batch).await?;
            total.added += r.added;
            total.unchanged += r.unchanged;
            if let Some(f) = on_progress {
                f(total.added + total.unchanged);
            }
        }
        Ok(total)
    }

    /// Add records from anything iterable, in batches (see `import`).
    pub async fn import_iter<I>(&self, records: I, batch: Option<usize>) -> Result<Added>
    where
        I: IntoIterator<Item = MemoryRecord>,
        I::IntoIter: Send,
    {
        self.import(futures::stream::iter(records.into_iter().map(Ok)), batch, None).await
    }

    /// Forget records: gone from search. What consolidation learned from them stays until sleep finds it unsupported;
    /// a kept episode — the record's own text — goes with it. How many were found.
    pub async fn remove(&self, ids: Vec<RecordId>) -> Result<usize> {
        self.exclusive(async {
            let uids: Vec<String> = ids
                .iter()
                .map(|x| record_uid(&x.id, x.source.as_deref()))
                .filter(|u| self.inner.stm.meta(u).is_some())
                .collect();
            if uids.is_empty() {
                return Ok(0);
            }
            let episodes: Vec<String> = uids.iter().map(|u| format!("ep:{u}")).collect();
            let items: Vec<String> = uids.iter().map(|u| stm_item(u)).chain(episodes.iter().cloned()).collect();
            self.inner.client.delete_items(&items).await?;
            self.inner.stm.remove(uids.clone()).await?;
            // A kept episode is the record's own text: gone from the long-term log too, or a rebuild indexes it again
            // and the profiles go on reading it.
            self.inner.ltm.remove(episodes).await?;
            Ok(uids.len())
        })
        .await
    }

    /// Best first: records and what consolidation learned, each with its key text, dates, and the passage that matched.
    pub async fn search(&self, q: &SearchQuery) -> Result<Vec<RecalledItem>> {
        memory_search(&self.recall(), q).await
    }

    /// The same search as a reader is shown it: one line a memory, best first or oldest first.
    pub async fn search_text(&self, q: &SearchOptions) -> Result<String> {
        Ok(search_text(&memory_search(&self.recall(), &q.query).await?, q.order))
    }

    /// What a small room should hold for a question: the best sentences of what search found, oldest first, in
    /// `chars` (6,000). Needs the local model.
    pub async fn room(&self, q: &SearchQuery, chars: Option<usize>) -> Result<String> {
        let Some(encode) = &self.inner.encode else {
            return Err(Error::Model(
                "the room needs the local embedding model (embedding.model_dir, or no embedding.server)".into(),
            ));
        };
        let found =
            memory_search(&self.recall(), &SearchQuery { limit: Some(q.limit.unwrap_or(20)), ..q.clone() }).await?;
        let mut items = Vec::new();
        for h in &found {
            items.push(self.room_item(h).await);
        }
        sentence_room(&q.query, &items, encode, &self.inner.words, chars.unwrap_or(ROOM_CHARS), 30).await
    }

    /// A found item's whole text for the room: a record's own, or what consolidation wrote.
    async fn room_item(&self, h: &RecalledItem) -> RoomItem {
        let r = match h.id.strip_prefix("stm:") {
            Some(uid) => self.inner.stm.get(uid).await,
            None => None,
        };
        let it = if r.is_some() { None } else { self.inner.ltm.get(&h.id).await };
        let text = match (&r, &it) {
            (Some(r), _) => format!("{}\n{}", r.request, r.reply),
            (None, Some(it)) => format!("{}\n{}\n{}", it.title, it.summary, it.body),
            _ => format!("{}\n{}", h.title, h.summary),
        };
        let told = crate::tools::day(h.ts);
        let lasted = lasted_text(h.start_at.as_deref(), h.end_at.as_deref());
        let event = h.event_at.as_deref().filter(|e| !e.is_empty());
        let happened = [event.map(readable_date).unwrap_or_default(), lasted.clone()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        let when = if happened.is_empty() {
            readable_date(&told)
        } else if h.event_at.as_deref() != Some(told.as_str()) || !lasted.is_empty() {
            format!("{happened}; told {}", readable_date(&told))
        } else {
            happened
        };
        RoomItem {
            id: h.id.clone(),
            head: format!("({when} · {})", h.kind),
            text: js::head16(&text, ROOM_TEXT).to_string(),
            ts: event.and_then(js::date_parse).unwrap_or(h.ts),
        }
    }

    /// One item whole, by the id search returned; for something learned, `sources`: the records it was learned from.
    pub async fn open_item(&self, id: &str) -> Result<Option<Value>> {
        let Some(mut p) = memory_open(&self.recall(), id).await? else { return Ok(None) };
        // A record: its kind as given. Something learned: the records it was learned from, whole.
        if p["tier"] == "stm" {
            p["kind"] = p["engine"].clone();
            return Ok(Some(p));
        }
        let provenance = self.inner.ltm.get(id).await.map(|it| it.provenance).unwrap_or_default();
        let sources: Vec<Value> = self
            .inner
            .stm
            .get_many(provenance)
            .await
            .into_iter()
            .flatten()
            .map(|x| {
                let mut s = json!({ "id": stm_item(&x.uid), "when": js::ms_value(x.ts), "title": x.title, "body": episode_body(&x) });
                if let Some(r) = &x.record {
                    s["record"] = serde_json::to_value(r).unwrap_or(Value::Null);
                }
                s
            })
            .collect();
        if !sources.is_empty() {
            p["sources"] = Value::Array(sources);
        }
        Ok(Some(p))
    }

    /// Around a topic, or around items you found: what the concept graph reaches that search did not, and its
    /// concepts.
    pub async fn explore(&self, q: &ExploreQuery) -> Result<Explored> {
        memory_explore(&self.recall(), q).await
    }

    /// A set for a question that counts, lists or totals: a wide search, then the language model keeps every
    /// instance. With `notes`, it also writes what the kept memories say that answers the question.
    pub async fn gather(&self, q: &SearchQuery, notes: bool) -> Result<Gathered> {
        let o = GatherOptions { ask: Some(self.inner.ask.as_ref()), limit: q.limit, notes };
        memory_gather(&self.recall(), q, o).await
    }

    /// Read what was added since last time and keep typed knowledge, dated observations and kept episodes. Needs a
    /// language model; without one, `paused` says why.
    pub async fn consolidate(&self) -> Result<ConsolidationReport> {
        self.exclusive(async {
            let worker = Consolidator::new(self.base());
            let mut total = ConsolidationReport::default();
            for _ in 0..1000 {
                if self.closing() || worker.pending(js::now_ms()).is_empty() {
                    break;
                }
                let r = worker.run(64).await?;
                total.add(&r);
                if r.paused.is_some() {
                    total.paused = r.paused;
                    break;
                }
                if r.windows == 0 {
                    break;
                }
            }
            self.index_words_now().await?;
            Ok(total)
        })
        .await
    }

    /// The offline pass: merge near-duplicates, retire what was superseded, link, write profiles. Once a day.
    pub async fn sleep(&self) -> Result<SleepReport> {
        self.exclusive(async {
            let d = SleepDeps::new(self.base(), self.inner.usage.clone(), self.inner.dir.join("sleep.json"));
            let r = sleep(&d).await?;
            self.index_words_now().await?;
            Ok(r)
        })
        .await
    }

    async fn index_words_now(&self) -> Result<usize> {
        if self.inner.encode.is_none() {
            return Ok(0);
        }
        let mut texts = Self::words_of_entries(&self.inner.stm.all().await);
        let active: Vec<String> =
            self.inner.ltm.entries().into_iter().filter(|(_, m)| m.status == "active").map(|(u, _)| u).collect();
        for it in self.inner.ltm.get_many(active).await.into_iter().flatten() {
            let joined = format!("{}\n{}\n{}", it.title, it.summary, it.body);
            let room = js::head16(&joined, ROOM_TEXT);
            texts.push(Self::key_text(&it.title, &it.summary));
            texts.extend(sentence_spans(room).into_iter().map(|(a, b)| js::slice16(room, a, b).to_string()));
        }
        self.encode_missing(texts).await
    }

    /// Encode what word by word and the room do not hold yet (done after every add, consolidate and sleep).
    pub async fn index_words(&self) -> Result<usize> {
        self.exclusive(self.index_words_now()).await
    }

    /// Write the index again from the logs.
    pub async fn rebuild(&self) -> Result<()> {
        self.exclusive(async {
            // Before the drop: a crash, or a close, after it resumes at the next open.
            self.inner.rebuild.begin().await?;
            self.inner.client.drop_collection().await;
            self.inner.client.ensure_collection().await?;
            self.inner.rebuild.run().await
        })
        .await
    }

    /// Tool definitions an agent can be handed: memory_search, memory_open, memory_explore.
    pub fn tools(&self) -> Vec<AgentTool> {
        build_agent_tools(Arc::new(self.clone()))
    }

    /// What it holds.
    pub fn stats(&self) -> Stats {
        Stats {
            records: self.inner.stm.size(),
            items: self.inner.ltm.entries().iter().filter(|(_, m)| m.status == "active").count(),
            words: self.inner.words.size(),
        }
    }

    /// The records log (for reading what is kept).
    pub fn records(&self) -> &StmStore {
        &self.inner.stm
    }

    /// The knowledge log (for reading what was learned).
    pub fn knowledge(&self) -> &LtmStore {
        &self.inner.ltm
    }

    /// Where it lives.
    pub fn home(&self) -> &Path {
        &self.inner.dir
    }

    /// Stops a consolidate, sleep, rebuild or index_words at its next step (a rebuild resumes at the next open) and
    /// waits for it, then releases Qdrant, the model and the home. Calling it again waits for the first.
    pub async fn close(&self) -> Result<()> {
        self.inner
            .closed
            .get_or_init(|| async {
                self.inner.closing.store(true, Ordering::SeqCst);
                // The job running stops at its next step; those waiting are refused.
                let _turn = self.inner.jobs.lock().await;
                self.inner.book.settled().await;
                self.inner.usage.flush().await;
                if let Some(p) = &self.inner.proc {
                    p.stop_and_wait(10_000).await;
                }
                if let Some(l) = &self.inner.local {
                    l.close().await;
                }
                let lock = self.inner.lock.clone();
                let _ = blocking(move || unlock_home(&lock)).await;
            })
            .await;
        Ok(())
    }
}

#[async_trait::async_trait]
impl ToolProvider for Memory {
    async fn search(&self, q: SearchQuery) -> Result<Vec<RecalledItem>> {
        memory_search(&self.recall(), &q).await
    }

    async fn open(&self, id: &str) -> Result<Option<Value>> {
        self.open_item(id).await
    }

    async fn explore(&self, q: ExploreQuery) -> Option<Result<Explored>> {
        Some(memory_explore(&self.recall(), &q).await)
    }
}
