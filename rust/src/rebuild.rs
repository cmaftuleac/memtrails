//! A derived index rebuild that survives unavailable embeddings, a failed batch and a restart. The logs stay the truth:
//! a checkpoint (`rebuild.json`) says which phase (long-term items, then records) and the last uid written, moved only
//! over batches already written, so a crash never skips one still in flight.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::store::{blocking, write_private, Keyed, LtmStore, MemoryItem, StmEntry, StmStore};
use crate::BoxFuture;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Checkpoint {
    phase: String,
    after: Option<String>,
}

/// Writes a batch of items into the index.
pub type WriteBatch<T> = Arc<dyn Fn(Vec<T>) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// What a rebuild reads and writes.
#[derive(Clone)]
pub struct RebuildDeps {
    /// The checkpoint file.
    pub path: PathBuf,
    /// The knowledge log.
    pub ltm: LtmStore,
    /// The records log.
    pub stm: StmStore,
    /// Writes long-term items.
    pub write_ltm: WriteBatch<MemoryItem>,
    /// Writes records.
    pub write_stm: WriteBatch<StmEntry>,
    /// The memory is closing: stop at the next step.
    pub aborted: Arc<dyn Fn() -> bool + Send + Sync>,
    /// Batches written at once (1 when not given).
    pub concurrency: usize,
}

/// The rebuild of one memory's index.
pub struct MemoryRebuild {
    d: RebuildDeps,
    generation: AtomicU64,
    /// One run at a time: a run of the same `begin` joins the one before it (finding nothing left), a newer one waits
    /// for the cancelled one to stop.
    running: tokio::sync::Mutex<()>,
}

impl MemoryRebuild {
    /// A rebuild over these logs.
    pub fn new(d: RebuildDeps) -> Self {
        MemoryRebuild { d, generation: AtomicU64::new(0), running: tokio::sync::Mutex::new(()) }
    }

    /// The run in flight stops at its next step, without moving or removing the checkpoint.
    pub fn cancel(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Whether a rebuild is begun and not finished.
    pub async fn pending(&self) -> bool {
        tokio::fs::try_exists(&self.d.path).await.unwrap_or(false)
    }

    /// Written BEFORE clearing an old index: a crash at any later point leaves a resumable rebuild. A run in flight is
    /// cancelled: it wrote into the index about to be cleared, and must neither move nor delete the new checkpoint.
    pub async fn begin(&self) -> Result<()> {
        self.generation.fetch_add(1, Ordering::SeqCst);
        save(&self.d.path, &Checkpoint { phase: "ltm".into(), after: None }).await
    }

    /// Run (or resume) the rebuild begun last; nothing when none is pending.
    pub async fn run(&self) -> Result<()> {
        let generation = self.generation.load(Ordering::SeqCst);
        let _one = self.running.lock().await;
        self.resume(generation).await
    }

    /// Wait for a run in flight to stop.
    pub async fn settled(&self) {
        let _ = self.running.lock().await;
    }

    fn cancelled(&self, generation: u64) -> bool {
        generation != self.generation.load(Ordering::SeqCst) || (self.d.aborted)()
    }

    async fn resume(&self, generation: u64) -> Result<()> {
        if !self.pending().await || self.cancelled(generation) {
            return Ok(());
        }
        let mut checkpoint = tokio::fs::read_to_string(&self.d.path)
            .await
            .ok()
            .and_then(|t| serde_json::from_str::<Checkpoint>(&t).ok())
            .filter(|c| c.phase == "ltm" || c.phase == "stm")
            .unwrap_or(Checkpoint { phase: "ltm".into(), after: None });
        // While rebuilding, nothing else writes to the logs, so their order stays stable; a checkpoint whose item is
        // absent (a repaired or compacted log) replays its phase instead of silently skipping records.
        if let Some(after) = &checkpoint.after {
            let known = if checkpoint.phase == "ltm" {
                self.d.ltm.meta(after).is_some()
            } else {
                self.d.stm.meta(after).is_some()
            };
            if !known {
                checkpoint.after = None;
            }
        }
        if checkpoint.phase == "ltm" {
            let ltm = self.d.ltm.clone();
            let done = self
                .phase(
                    ltm.order(),
                    move |uids| {
                        let ltm = ltm.clone();
                        Box::pin(async move { ltm.get_many(uids).await })
                    },
                    checkpoint.after.clone(),
                    |it: &MemoryItem| it.status == "active",
                    20,
                    self.d.write_ltm.clone(),
                    "ltm",
                    generation,
                )
                .await?;
            if !done {
                return Ok(());
            }
            checkpoint = Checkpoint { phase: "stm".into(), after: None };
            save(&self.d.path, &checkpoint).await?;
        }
        // Every record stays searchable, consolidated or not: the raw record is the truth, a memory a key to it. Only
        // throwaway runs (synthetic) stay out.
        let stm = self.d.stm.clone();
        let done = self
            .phase(
                stm.order(),
                move |uids| {
                    let stm = stm.clone();
                    Box::pin(async move { stm.get_many(uids).await })
                },
                checkpoint.after,
                |it: &StmEntry| !it.synthetic,
                32,
                self.d.write_stm.clone(),
                "stm",
                generation,
            )
            .await?;
        if done && !self.cancelled(generation) {
            let _ = tokio::fs::remove_file(&self.d.path).await;
        }
        Ok(())
    }

    /// One phase from its checkpoint: batches of `size`, up to `concurrency` written at once, the checkpoint moved over
    /// the written batches at the front. False when cancelled. A failed batch fails the run, the checkpoint before it.
    #[allow(clippy::too_many_arguments)]
    async fn phase<T: Keyed>(
        &self,
        order: Vec<String>,
        read: impl Fn(Vec<String>) -> BoxFuture<'static, Vec<Option<T>>>,
        after: Option<String>,
        keep: impl Fn(&T) -> bool,
        size: usize,
        write: WriteBatch<T>,
        phase: &str,
        generation: u64,
    ) -> Result<bool> {
        let limit = self.d.concurrency.max(1);
        // Batches in flight, in order: their last uid and whether they are written.
        let mut queue: VecDeque<(String, bool, usize)> = VecDeque::new();
        let mut flying: FuturesUnordered<BoxFuture<'static, (usize, Result<()>)>> = FuturesUnordered::new();
        let mut next_id = 0usize;
        let mut failure: Option<Error> = None;
        let advance = |queue: &mut VecDeque<(String, bool, usize)>| -> Option<String> {
            let mut last = None;
            while queue.front().is_some_and(|q| q.1) {
                last = queue.pop_front().map(|q| q.0);
            }
            last
        };
        let mut skip = after.is_some();
        let mut batch: Vec<T> = Vec::new();
        let mut uids = order.into_iter().peekable();
        'reading: while uids.peek().is_some() {
            let chunk: Vec<String> = uids.by_ref().take(256).collect();
            for it in read(chunk).await.into_iter().flatten() {
                if self.cancelled(generation) {
                    break 'reading;
                }
                if skip {
                    if Some(it.uid()) == after.as_deref() {
                        skip = false;
                    }
                    continue;
                }
                if !keep(&it) {
                    continue;
                }
                batch.push(it);
                if batch.len() >= size {
                    let items = std::mem::take(&mut batch);
                    let last = items.last().map(|x| x.uid().to_string()).unwrap_or_default();
                    let id = next_id;
                    next_id += 1;
                    queue.push_back((last, false, id));
                    let w = write.clone();
                    flying.push(Box::pin(async move { (id, w(items).await) }));
                    while flying.len() >= limit {
                        if let Some((id, r)) = flying.next().await {
                            self.settle(&mut queue, id, r, &mut failure);
                            if failure.is_some() {
                                break 'reading;
                            }
                            if let Some(last) = advance(&mut queue) {
                                if !self.cancelled(generation) {
                                    save(&self.d.path, &Checkpoint { phase: phase.into(), after: Some(last) }).await?;
                                }
                            }
                        }
                    }
                }
            }
        }
        if failure.is_none() && !batch.is_empty() && !self.cancelled(generation) {
            let last = batch.last().map(|x| x.uid().to_string()).unwrap_or_default();
            let id = next_id;
            queue.push_back((last, false, id));
            let w = write.clone();
            let items = std::mem::take(&mut batch);
            flying.push(Box::pin(async move { (id, w(items).await) }));
        }
        // On a failure, the batches still in flight settle first, and the checkpoint takes every written one at the
        // front.
        while let Some((id, r)) = flying.next().await {
            self.settle(&mut queue, id, r, &mut failure);
        }
        if let Some(last) = advance(&mut queue) {
            if !self.cancelled(generation) {
                save(&self.d.path, &Checkpoint { phase: phase.into(), after: Some(last) }).await?;
            }
        }
        if let Some(e) = failure {
            return Err(e);
        }
        Ok(!self.cancelled(generation))
    }

    fn settle(
        &self,
        queue: &mut VecDeque<(String, bool, usize)>,
        id: usize,
        r: Result<()>,
        failure: &mut Option<Error>,
    ) {
        match r {
            Ok(()) => {
                if let Some(q) = queue.iter_mut().find(|q| q.2 == id) {
                    q.1 = true;
                }
            }
            Err(e) => {
                if failure.is_none() {
                    *failure = Some(e);
                }
            }
        }
    }
}

async fn save(path: &std::path::Path, c: &Checkpoint) -> Result<()> {
    let (path, text) = (path.to_path_buf(), crate::js::stringify(&serde_json::to_value(c).map_err(Error::json)?));
    blocking(move || {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        }
        let tmp = PathBuf::from(format!("{}.tmp", path.display()));
        write_private(&tmp, text.as_bytes()).map_err(|e| Error::io(&tmp, e))?;
        std::fs::rename(&tmp, &path).map_err(|e| Error::io(&path, e))
    })
    .await?
}
