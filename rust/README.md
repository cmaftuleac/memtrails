# memtrails for Rust

**Retrieval and exploration for any text — e-mail, chat, notes, documents, commit messages, conversations — natively in
Rust and async (tokio) from top to bottom.**

This is the Rust port of [memtrails](https://github.com/cmaftuleac/memtrails). The TypeScript library is the reference
and this port follows it: the same algorithms and constants, the same prompts byte for byte, the same files on disk and
the same Qdrant index. A memory folder written by either library opens and searches in the other with the same
results.

Everything you add can be searched straight away, on your own machine. There is no server to run and no language model
to call: a local embedding model, a keyword channel and a word-by-word reordering do the work. Give it a language model
too, and it consolidates what it holds into typed knowledge, dated events and profiles, and an agent can explore it.

## Install

```toml
[dependencies]
memtrails = { git = "https://github.com/cmaftuleac/memtrails" }   # crates.io: `memtrails = "0.4"` once published
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

It needs Rust 1.88 or newer, on macOS or Linux (x64 or arm64). Windows has the pieces it needs but is not tested yet.

The first `Memory::open` downloads three things into `~/.cache/memtrails`, and checks each against a pinned SHA-256:

- the embedding model, EmbeddingGemma-300m at 4 bits (218 MB, from Hugging Face);
- ONNX Runtime 1.23.2 (about 10 MB, Microsoft's own release);
- the Qdrant binary (about 30 MB, from Qdrant's GitHub release).

Set `MEMTRAILS_CACHE` to keep them somewhere else.

## Quick start

```rust,no_run
use memtrails::{At, Memory, MemoryOptions, MemoryRecord, SearchQuery};

#[tokio::main]
async fn main() -> memtrails::Result<()> {
    let memory = Memory::open(MemoryOptions::new("./memory")).await?;

    memory.add(vec![
        MemoryRecord {
            kind: Some("email".into()), author: Some("Ana".into()), title: Some("Proxy certificate expired".into()),
            at: Some(At::from("2026-04-02")),
            ..MemoryRecord::new("msg-1", "The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.")
        },
        MemoryRecord {
            kind: Some("commit".into()), author: Some("Ion".into()), title: Some("Restore certificate renewal cron".into()),
            ..MemoryRecord::new("c9f1e2", "Files: deploy/cron/renew.sh")
        },
    ]).await?;

    let hits = memory.search(&SearchQuery::new("why did the certificate stop renewing?")).await?;
    println!("{} {}", hits[0].id, hits[0].title);

    memory.close().await
}
```

Every method that does I/O is `async`. `Memory` is `Clone + Send + Sync`: every clone is the same memory, so you can
hand it to any task. Blocking work never runs on the runtime's own threads. That covers model inference, tokenising,
word-by-word scoring over many items, hashing a download, unpacking an archive, loading and compacting the logs, and
the word-vector cache.

## What it does

| Method | |
|---|---|
| `add(records)`, `add_conversation(&c)` | Adds or updates records. Unchanged records cost nothing, and new ones can be searched once the call returns. |
| `import(stream, batch, on_progress)`, `import_iter(iter, batch)` | Takes any `Stream<Item = Result<MemoryRecord>>`, such as an importer or a database cursor, or any iterator. |
| `remove(ids)` | Forgets records, together with any episode that was kept from them. |
| `search(&q)`, `search_text(&q)` | Best results first. `kinds`, `collections`, `limit`, and `from`/`to` dates (what falls within the dates comes first). |
| `room(&q, chars)` | The best sentences of what was found, fitted into a small context. |
| `open_item(id)` | One item in full. For something learned, also the records it was learned from. |
| `explore(&q)` | What the concept graph reaches from a topic or from items, beyond what search found. |
| `gather(&q, notes)` | A language model reads what was found and keeps every instance the question needs. |
| `consolidate()`, `sleep()` | With a language model: typed knowledge, dated observations, merging and superseding, profiles. |
| `index_words()`, `rebuild()`, `stats()`, `tools()`, `close()` | Upkeep, and the agent tools. |

The language model is the async trait `memtrails::Ask`. `openai_ask(OpenAiAskOptions { .. })` speaks the
OpenAI-compatible Responses API with strict JSON Schema output, and you can implement `Ask` to bring any other model.
`OPENAI_API_KEY` is only ever sent to OpenAI's own address.

```rust,no_run
use memtrails::{openai_ask, Llm, MemoryOptions, OpenAiAskOptions};

let options = MemoryOptions {
    llm: Llm::Custom(openai_ask(OpenAiAskOptions { model: Some("gpt-6-luna".into()), ..Default::default() })),
    ..MemoryOptions::new("./memory")
};
```

Importers in `memtrails::importers` stream records: `from_mbox`, `from_eml`, `from_slack_export`, `from_git`,
`from_files`, `from_jsonl` and `batches`. Each is an `async fn` that returns a `RecordStream`, so you can pass it to
`memory.import(...)`.

`memory.tools()` gives `memory_search`, `memory_open` and `memory_explore`. Each comes with its JSON Schema (draft
2020-12, the same as the reference's) and an async handler that checks its own arguments.

## Configuration

| `MemoryOptions` | Default | |
|---|---|---|
| `home` | (required) | Where the memory lives: its logs, its index and its word vectors. The folder is private (0700). |
| `embedding` | the local model | `model_dir`, `server: EmbedRoute` (an OpenAI-compatible embeddings API), `threads` (default 2). |
| `rerank` | words, with the local model | `Rerank::Words`, `Server(RerankRoute)`, `Custom(fn)` or `Off`. |
| `llm` | from the environment | `Llm::Env`, `OpenAi(options)` or `Custom(Arc<dyn Ask>)`. |
| `qdrant` | its own process | `QdrantOption::Own`, `Binary(path)` or `Server { url, api_key, collection }`. |
| `concurrency` | 4 | How many windows the model reads at once during consolidation. |
| `download` | | Where downloads are kept, a mirror, a stall timeout. |

| Environment variable | |
|---|---|
| `MEMTRAILS_CACHE` | Where downloads live. |
| `MEMTRAILS_MODEL`, `MEMTRAILS_LLM_API_KEY` or `OPENAI_API_KEY`, `MEMTRAILS_LLM_BASE_URL` | The language model when `llm` is `Env`. |
| `MEMTRAILS_EMBED_URL`, `MEMTRAILS_RERANK_URL` (+ `_KEY`, `_MODEL`) | Servers, when the options name none. |
| `MEMTRAILS_ONNXRUNTIME` | An ONNX Runtime library of your own (nothing is downloaded then). |

## Safety

- Every download is fetched into a private temporary folder of its own and checked on disk against its pinned size
  and SHA-256. Only one process fetches a given download at a time (a lock file), and a stalled download is given up.
- Qdrant runs on 127.0.0.1, behind a random key made for each run. Before it is given the key, the process has to
  prove it is ours. It gets a minimal environment and is stopped by `close()`, when it is dropped, or when the process
  exits. One left behind by a crash is stopped by the next start.
- Only one process at a time can open a home (`memory.lock`).
- Secrets are scrubbed from everything sent to a language model and from everything consolidation keeps. The patterns
  are the reference's, written so that none takes more than linear time on hostile input.

## Parity with the reference

`cargo test` holds this port to the reference's answer key in `../conformance/fixtures`:

- **Text, dates, keys and records:** folding, keywords, children, BM25 terms, term ids, `cut`, concept keys, sentence
  spans, the secret filter, fingerprints, record uids, point ids, dates (including 54 forms that V8 reads), records and
  conversations, word packing and MaxSim, and the text of a search.
- **Prompts and what is kept:** every prompt consolidation, sleep, profiles and gathering send, compared byte for byte
  along with its JSON Schema, room and wait. What the gate keeps of fixed answers.
- **Agent tools:** each tool's schema, its argument checks and its answers.
- **Importers:** what each importer reads from committed inputs.
- **Embeddings:** the local model's vectors for fixed texts, equal to the reference's to 1e-6 on the processor family
  the fixture was made on (macOS, arm64), because both run Microsoft's ONNX Runtime 1.23.2. On x64, ONNX Runtime's own
  CPU kernels round differently, by up to about 0.003 a number (cosine above 0.999). This one runs with `--ignored`,
  because it needs the model.

`conformance/generate-rust.ts` writes the fixtures that are only this port's.

## Known differences

- **Japanese and folded Thai word segmentation.** ICU4X's dictionaries split a few words differently from the ICU4C
  that Node.js carries. In Japanese, `がいっぱい` becomes `がい` + `っぱい`. In Thai, NFKD splits sara am, and the result
  is cut differently. This affects keyword tokens only; the conformance tests list the cases. ICU4X also gives no word
  type to a word that ends in a combining mark, as in Devanagari, so here a segment with a letter or a number in it
  counts as a word, as ICU4C does.
- **Half a character.** A Rust string cannot hold half of a surrogate pair. Where the reference would cut between the
  two halves of an emoji, this port cuts before it. In the secret filter, a quoted value counts characters, not UTF-16
  units. This only matters for a secret value longer than 512 units that contains emoji.
- **Text comparison.** The reference's `localeCompare` (ICU collation) breaks ties between profile names. Here the
  comparison uses letters first, then accents, then lower case before upper case. Names that differ only in
  punctuation may order differently.
- **Importers.**
  - JSON Lines: a line that is not JSON is skipped with serde's wording. A field of the wrong type is skipped with a
    reason, where the reference would keep it or fail.
  - Slack: an export with no `channels.json` is read in folder-name order. A message whose `ts` is a number is
    accepted.
  - Front matter: keys keep the order they have in the file in the record the importer gives. Memory keeps and
    fingerprints a record's metadata in JavaScript's order (integer-like keys first), as the reference does.
- **Log lines that are not whole records.** A line of `records.jsonl` or `knowledge.jsonl` that is JSON with a uid but
  lacks a field this port needs (one another version wrote) is not read here, where the reference would read it with
  the field missing. It still counts as that uid's latest line, and a compaction keeps it as it is.
- **Windows.** The check that a process is still alive is not implemented, so locks and leftover Qdrants are taken over
  only once they go stale.

MIT © Corneliu Maftuleac. EmbeddingGemma is Google's, under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
It is downloaded from Hugging Face, not shipped with this crate. Qdrant (Apache-2.0) and ONNX Runtime (MIT) are
downloaded from their own releases.
