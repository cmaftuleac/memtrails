//! EmbeddingGemma on this machine, for a memory that needs no server: the same quantised model and the same prompts as
//! a server would run, so vectors and an index move between the two. One run gives both what the search keeps (the
//! memory's vector, `sentence_embedding`) and what word by word keeps (a vector per word, the step before it is
//! averaged: `last_hidden_state`).
//!
//! The model and its tokenizer are a folder (onnx/model_q4.onnx, tokenizer.json). ONNX Runtime runs one inference at a
//! time, on a blocking thread, so the async runtime is never held up by it.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::embedder::{doc_text, query_text, EmbedFn};
use crate::error::{Error, Result};
use crate::words::{WordEncoder, WordKind, WordVectors, WORD_LIMIT};
use crate::BoxFuture;

const BOS_EOS: usize = 2;

struct Model {
    session: Session,
    tokenizer: Tokenizer,
    batch: usize,
    max_tokens: usize,
}

/// The local model, shared: `embed` for the search, `words` for word by word.
#[derive(Clone)]
pub struct LocalEmbedder {
    model: Arc<Mutex<Option<Model>>>,
}

fn model_err(e: impl std::fmt::Display) -> Error {
    Error::Model(e.to_string())
}

fn unit(v: &[f32]) -> Vec<f32> {
    let s: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    let n = s.sqrt();
    let n = if n == 0.0 { 1.0 } else { n };
    v.iter().map(|x| (f64::from(*x) / n) as f32).collect()
}

/// One encoded text: its token ids, and whether the text was cut before its end.
struct Encoded {
    ids: Vec<i64>,
    cut: bool,
}

/// The outputs a batch needs, flat, with their shapes.
struct Out {
    sentence: Option<(Vec<usize>, Vec<f32>)>,
    hidden: Option<(Vec<usize>, Vec<f32>)>,
}

impl Model {
    fn tokens(&self, text: &str, limit: usize) -> Result<Encoded> {
        let e = self.tokenizer.encode(text, true).map_err(model_err)?;
        let ids: Vec<i64> = e.get_ids().iter().map(|x| i64::from(*x)).collect();
        let cut = ids.len() > limit;
        Ok(Encoded { ids: ids.into_iter().take(limit).collect(), cut })
    }

    /// Each text padded to the run's longest, a mask over what is real.
    fn run(&mut self, encoded: &[&Encoded]) -> Result<Out> {
        let width = encoded.iter().map(|e| e.ids.len()).max().unwrap_or(0).max(1);
        let mut ids = vec![0i64; encoded.len() * width];
        let mut mask = vec![0i64; encoded.len() * width];
        for (b, e) in encoded.iter().enumerate() {
            for (i, id) in e.ids.iter().enumerate() {
                ids[b * width + i] = *id;
                mask[b * width + i] = 1;
            }
        }
        let shape = [encoded.len(), width];
        let ids = Tensor::from_array((shape, ids)).map_err(model_err)?;
        let mask = Tensor::from_array((shape, mask)).map_err(model_err)?;
        let last_name = self.session.outputs.last().map(|o| o.name.clone()).unwrap_or_default();
        let outputs =
            self.session.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask]).map_err(model_err)?;
        let take = |name: &str| -> Option<(Vec<usize>, Vec<f32>)> {
            let v = outputs.get(name)?;
            let (shape, data) = v.try_extract_tensor::<f32>().ok()?;
            Some((shape.iter().map(|d| *d as usize).collect(), data.to_vec()))
        };
        Ok(Out { sentence: take("sentence_embedding").or_else(|| take(&last_name)), hidden: take("last_hidden_state") })
    }

    /// Texts in batches of like lengths (sorted by length, then put back in order).
    fn batches<R>(
        &mut self,
        items: &[Encoded],
        mut read: impl FnMut(&Out, usize, &Encoded) -> Result<R>,
    ) -> Result<Vec<R>> {
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&i| items[i].ids.len());
        let mut results: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
        for part in order.chunks(self.batch) {
            let encoded: Vec<&Encoded> = part.iter().map(|&j| &items[j]).collect();
            let out = self.run(&encoded)?;
            for (b, &j) in part.iter().enumerate() {
                results[j] = Some(read(&out, b, &items[j])?);
            }
        }
        results.into_iter().map(|r| r.ok_or_else(|| Error::Model("a text was not embedded".into()))).collect()
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let items = texts.iter().map(|t| self.tokens(t, self.max_tokens)).collect::<Result<Vec<_>>>()?;
        self.batches(&items, |out, b, _| {
            let (shape, data) =
                out.sentence.as_ref().ok_or_else(|| Error::Model("the model gave no embedding".into()))?;
            let d = *shape.last().unwrap_or(&0);
            Ok(unit(&data[b * d..(b + 1) * d]))
        })
    }

    fn words(&mut self, texts: &[String], kind: WordKind) -> Result<Vec<WordVectors>> {
        let items = texts
            .iter()
            .map(|t| {
                let prompted = if kind == WordKind::Query { query_text(t) } else { doc_text("", t) };
                self.tokens(&prompted, WORD_LIMIT + BOS_EOS)
            })
            .collect::<Result<Vec<_>>>()?;
        self.batches(&items, |out, b, it| {
            let (shape, data) = out
                .hidden
                .as_ref()
                .ok_or_else(|| Error::Model("This model gives no vector per word (last_hidden_state).".into()))?;
            let (width, dims) = (shape.get(1).copied().unwrap_or(0), shape.get(2).copied().unwrap_or(0));
            // Without its first token, and its last unless the text was cut before it.
            let words = it.ids.len().saturating_sub(if it.cut { 1 } else { BOS_EOS });
            let mut v = Vec::with_capacity(words * dims);
            for w in 0..words {
                let from = (b * width + w + 1) * dims;
                v.extend(unit(&data[from..from + dims]));
            }
            Ok(WordVectors { words, dims, data: v })
        })
    }
}

/// How the local model runs.
#[derive(Debug, Clone, Copy)]
pub struct LocalEmbedderOptions {
    /// CPU threads (default 1; a memory opens it with 2).
    pub threads: usize,
    /// Texts a run (default 8).
    pub batch: usize,
    /// Tokens read of a text (default 512; the model's own limit is longer, memory's keys are short).
    pub max_tokens: usize,
}

impl Default for LocalEmbedderOptions {
    fn default() -> Self {
        LocalEmbedderOptions { threads: 1, batch: 8, max_tokens: 512 }
    }
}

/// Load the model from a folder (onnx/model_q4.onnx and tokenizer.json).
pub async fn load_local_embedder(dir: &Path, o: LocalEmbedderOptions) -> Result<LocalEmbedder> {
    let dir = dir.to_path_buf();
    let model = tokio::task::spawn_blocking(move || -> Result<Model> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(model_err)?;
        let session = Session::builder()
            .map_err(model_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(model_err)?
            .with_parallel_execution(false)
            .map_err(model_err)?
            .with_intra_threads(o.threads.max(1))
            .map_err(model_err)?
            .commit_from_file(dir.join("onnx").join("model_q4.onnx"))
            .map_err(model_err)?;
        Ok(Model { session, tokenizer, batch: o.batch.max(1), max_tokens: o.max_tokens })
    })
    .await
    .map_err(model_err)??;
    Ok(LocalEmbedder { model: Arc::new(Mutex::new(Some(model))) })
}

impl LocalEmbedder {
    async fn with<R: Send + 'static>(&self, f: impl FnOnce(&mut Model) -> Result<R> + Send + 'static) -> Result<R> {
        // One inference at a time: the lock is the queue (tokio's, fair), held while the blocking thread runs it.
        let mut g = self.model.clone().lock_owned().await;
        tokio::task::spawn_blocking(move || match g.as_mut() {
            Some(m) => f(m),
            None => Err(Error::Closed),
        })
        .await
        .map_err(model_err)?
    }

    /// Already prompted texts → unit vectors, in order.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.with(move |m| m.embed(&texts)).await
    }

    /// Texts → a vector per word, prompted as a question or a document.
    pub async fn words(&self, texts: Vec<String>, kind: WordKind) -> Result<Vec<WordVectors>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.with(move |m| m.words(&texts, kind)).await
    }

    /// `embed` as the search takes it.
    pub fn embed_fn(&self) -> EmbedFn {
        let me = self.clone();
        Arc::new(move |texts, _timeout| {
            let me = me.clone();
            Box::pin(async move { me.embed(texts).await }) as BoxFuture<'static, Result<Vec<Vec<f32>>>>
        })
    }

    /// `words` as word by word and the room take it.
    pub fn word_encoder(&self) -> WordEncoder {
        let me = self.clone();
        Arc::new(move |texts, kind| {
            let me = me.clone();
            Box::pin(async move { me.words(texts, kind).await }) as BoxFuture<'static, Result<Vec<WordVectors>>>
        })
    }

    /// Wait for the inference running, then release the model.
    pub async fn close(&self) {
        let model = self.model.lock().await.take();
        // Released on a blocking thread: ONNX Runtime frees its session there.
        let _ = tokio::task::spawn_blocking(move || drop(model)).await;
    }
}
