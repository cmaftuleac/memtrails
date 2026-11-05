# memtrails

**Retrieval and exploration for any text — e-mail, chat, notes, memos, documents, news, commit messages, conversations.**

[![CI](https://github.com/cmaftuleac/memtrails/actions/workflows/ci.yml/badge.svg)](https://github.com/cmaftuleac/memtrails/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/memtrails.svg)](https://www.npmjs.com/package/memtrails)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

memtrails keeps what you give it and finds it again. Everything you add is searchable the moment it is added, on your
own machine, with no server and no language model: a local embedding model, a keyword channel, and a word-by-word
reordering that needs no reranker. In any language. Give it a language model too, and it consolidates what it holds
into typed knowledge, dated events and profiles of people and things — and an agent can explore it.

```ts
import { openMemory } from 'memtrails';

const memory = await openMemory({ home: './memory' });

await memory.add([
  { id: 'msg-1', kind: 'email', author: 'Ana', title: 'Proxy certificate expired', at: '2026-04-02',
    text: 'The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.' },
  { id: 'c9f1e2', kind: 'commit', author: 'Ion', title: 'Restore certificate renewal cron', at: '2026-04-03',
    text: 'Files: deploy/cron/renew.sh' },
]);

const hits = await memory.search({ query: 'why did the certificate stop renewing?' });
// [{ id, kind: 'email', title, summary, match, ts, record: { id: 'msg-1', author: 'Ana' }, score }, …]

await memory.close();
```

More to run in [`examples/`](https://github.com/cmaftuleac/memtrails/blob/main/examples): a quick start, importing mail, Slack, git and notes, the agent tools
(an MCP server, or JSON Schemas for function calling), and consolidation with a language model. The same four are in
[Python](https://github.com/cmaftuleac/memtrails/blob/main/python/examples) and [Rust](https://github.com/cmaftuleac/memtrails/blob/main/rust/examples).

## Why memtrails

- **Searchable at once.** No ingestion pipeline to wait for and no model call per record. Add a mailbox and search it.
- **On your machine.** The embedding model (EmbeddingGemma-300m, 4-bit, ~220 MB) and the index (Qdrant) run locally
  and are fetched on first use. Nothing leaves the machine unless you turn on a language model.
- **Any language.** Multilingual embeddings, keyword folding for any script, no word lists. A question in Romanian
  finds an English memory and the other way round.
- **Little to read, much found.** Search reorders its best 30 word by word (MaxSim over the model's own token vectors,
  stored at one bit a number): on 2,011 benchmark questions the answer reached the reader in 53.1% of top-5 lists
  against 50.3% for plain search, and most of a cross-encoder reranker's gain, in about 3 ms on one CPU thread.
- **Measured, honestly.** On all 1,307 held-out LoCoMo questions with 6,000 characters to read, memtrails alone on a
  CPU scores 77.4 (strict judge) against Mem0's 76.5 — a tie, reported as one; with `gather` reading what search
  found and noting what it says, 80.9 — ahead beyond chance; 84.8 against 79.9 without the questions whose official
  answer an independent audit found wrong. [Benchmarks](docs/BENCHMARKS.md).
- **Explorable.** A concept graph links what records share — people, places, systems, names — so an agent can walk
  from what it found to what a search for the words does not find.
- **Built for agents.** `npx memtrails mcp` gives Claude, Cursor or any MCP app a memory it can search and save to;
  `memory.tools()` gives the same tools with zod and JSON schemas, for your own server or any function-calling API.

## Install

```bash
npm install memtrails
```

Node.js 22 or newer, on macOS or Linux (x64 or arm64) or Windows (x64). The first `openMemory` downloads the embedding model
(218 MB, from Hugging Face) and the Qdrant binary (~30 MB, from Qdrant's GitHub release) into `~/.cache/memtrails`,
each checked against a pinned SHA-256. Set `MEMTRAILS_CACHE` to put them elsewhere.

On Windows, keep a memory's path under 90 characters: Qdrant's files sit up to about 170 characters deeper, and
Windows stops a path at 260 (memtrails says so when one is longer).

On Linux x64, `ONNXRUNTIME_NODE_INSTALL=skip npm install memtrails` skips the CUDA libraries ONNX Runtime's installer
downloads otherwise (about 350 MB; memtrails runs on the CPU and never uses them). Use it always on a Linux machine
shared with other users: that download unpacks through a guessable temp folder ([SECURITY.md](SECURITY.md)). In
an MCP client's configuration the same goes in `"env": { "ONNXRUNTIME_NODE_INSTALL": "skip" }`.

### Python and Rust

The same library, written natively in each language, fully async, with the same results and the same files: a memory
folder written by one opens in the others (checked on every change by [`interop/`](https://github.com/cmaftuleac/memtrails/tree/main/interop) and held to the shared
answers in [`conformance/`](https://github.com/cmaftuleac/memtrails/tree/main/conformance)).

```bash
pip install memtrails          # Python 3.10–3.13, asyncio — see python/README.md
cargo add memtrails            # Rust 1.88+, tokio — see rust/README.md
```

## Memory for Claude, Cursor and other MCP apps

One line gives an MCP client a long-term memory on your machine: `memory_search`, `memory_open`, `memory_explore`,
and `memory_save` for what the agent should remember.

```bash
claude mcp add memory -- npx -y memtrails mcp                                  # Claude Code
```

```json
{ "mcpServers": { "memory": { "command": "npx", "args": ["-y", "memtrails", "mcp"] } } }
```

The second is for Claude Desktop (`claude_desktop_config.json`), Cursor (`.cursor/mcp.json`) and most other clients.
The same server comes with the Python and Rust versions: `uvx memtrails mcp`, or `cargo install memtrails` and then
`memtrails mcp`. Fill the memory from the command line, and try it:

```bash
npx memtrails add ~/notes ~/mail/archive.mbox ./slack-export      # notes and documents, mail, a Slack export
npx memtrails add --git ~/code/my-repo                             # commit messages and changed files
npx memtrails search "why did the certificate stop renewing?"
```

The memory lives in `~/.memtrails` (`--home` or `MEMTRAILS_HOME` for another). The server opens it at the first tool
call and closes it after five minutes without one (`--idle`), so `add` can run in between; `--read-only` leaves out
`memory_save`. From code: `serveMcp({ open: () => openMemory({ home }), version })`.

## LangChain and LlamaIndex

A memory is a LangChain retriever (install `@langchain/core` beside memtrails):

```ts
import { MemtrailsRetriever } from 'memtrails/langchain';
const docs = await new MemtrailsRetriever({ memory, search: { limit: 8 } }).invoke('why did renewal stop?');
```

Each document is a found item: its summary (and the passage that matched) as the text, its id, and flat metadata
(`id`, `kind`, `title`, `date`, `ts`, `score`, and when it has them `eventAt`/`startAt`/`endAt`, `recordId`, `source`,
`author`, `url`); `memory.open(id)` gives all of it. `search` takes every search option but the query (`collections`,
`kinds`, `limit`, `from`, `to`; `limit` counts dated observations and other memories each). In Python, LangChain and
LlamaIndex both: `memtrails.integrations.langchain` and `memtrails.integrations.llama_index`. (LlamaIndex's TypeScript
packages are no longer maintained, so there is no TypeScript LlamaIndex adapter.)

## Adding data

A **record** is anything with text. Only `id` and `text` are required.

```ts
await memory.add({
  id: '<CAF8w@mail.example>',      // unique within its source; adding the same id again updates it
  source: 'work-mailbox',          // part of the identity
  kind: 'email',                   // any word: email, message, note, memo, article, commit, ticket, …
  collection: 'ops',               // search can be scoped to collections
  thread: 'proxy-cert',          // records that belong together
  author: 'Ana Popescu',
  title: 'Proxy certificate expired',
  text: '…',
  at: '2026-04-02T09:14:00Z',      // ms, ISO string or Date
  url: 'https://mail.example/…',
  metadata: { to: 'ops@example.com' },
});
```

Conversations (a person and an assistant, or any two parties) are added as turns; each time the person speaks becomes
one record, with what was answered:

```ts
await memory.addConversation({ id: 'chat-42', turns: [
  { role: 'user', text: 'Where do we have lunch on Friday?' },
  { role: 'assistant', text: 'At the Italian place near the office, at one.' },
] });
```

### Importers

Streaming importers for common sources, in `memtrails/importers`:

```ts
import { fromMbox, fromSlackExport, fromGit, fromFiles, fromJsonl } from 'memtrails/importers';

await memory.import(fromMbox('./archive.mbox'));                 // e-mail (mbox / .eml)
await memory.import(fromSlackExport('./slack-export'));          // a Slack workspace export
await memory.import(fromGit('./my-repo', { since: '2025-01-01' })); // commit messages and changed files
await memory.import(fromFiles('./notes'));                       // Markdown, text and HTML files
await memory.import(fromJsonl('./records.jsonl'));               // one record per line
```

`memory.import` takes any async iterable of records, so a database cursor or an API client works the same way.
Unchanged records cost nothing; `memory.remove([...ids])` forgets.

## Searching

```ts
await memory.search({ query: 'who renewed the certificate?', limit: 8 });
await memory.search({ query: 'standup notes', collections: ['ops'], kinds: ['message'], from: '2026-04', to: '2026-04' });

// The same results as a reader is shown them: one line a memory, best first or oldest first.
await memory.searchText({ query: 'what happened with the proxy?', order: 'time' });

// The best sentences of what was found, for a small context window.
await memory.room({ query: 'what made the renewal stop?', chars: 6000 });

// One item whole — for something learned, with the records it was learned from.
await memory.open(hits[0].id);

// What the concept graph reaches from a topic or from items, that search did not find.
await memory.explore({ topic: 'the proxy' });
```

With dates, what happened within them comes first and the rest follows: a question that names a time is rarely only
about that time.

## Consolidation and sleep (optional, with a language model)

Records are kept as they are, for good. With a language model, memtrails also reads threads in windows and keeps what
matters as typed items — facts, decisions, procedures, pitfalls, preferences, dated observations ("Ana renewed the
certificate on 2026-04-03"), kept episodes — reconciled against what it already knows, so nothing is stored twice. A
nightly `sleep` merges near-duplicates, retires what was superseded, links related items and writes a profile of each
person or thing that many items mention.

```ts
const memory = await openMemory({
  home: './memory',
  llm: { model: 'gpt-6-luna', reasoningEffort: 'low', apiKey: process.env.OPENAI_API_KEY },   // or any OpenAI-compatible endpoint: baseUrl
});
await memory.consolidate();   // after adding
await memory.sleep();         // once a day
await memory.gather({ query: 'every plumber we called this year' });   // a set, for counting and listing questions
```

Any provider works: pass `llm: { baseUrl, model, apiKey }` for an OpenAI-compatible server (a hosted model, vLLM,
Ollama, …), or your own function `(request) => Promise<answer>`. Every call asks for strict structured output, and
`onUsage` reports each call's tokens. On the LoCoMo benchmark, consolidating and sleeping a conversation of ~700
messages cost about $0.07 with GPT-6 Luna (the model every benchmark here used).

## For agents

```ts
const tools = memory.tools();
// [{ name: 'memory_search', description, inputSchema (zod), jsonSchema, handler }, memory_open, memory_explore]

// With the MCP TypeScript SDK:
for (const t of tools) server.tool(t.name, t.description, t.inputSchema, t.handler);
```

## Configuration

| Option | Default | |
|---|---|---|
| `home` | — | Where this memory lives: its logs, index and word vectors. |
| `embedding.modelDir` | fetched into the cache | A directory with EmbeddingGemma-300m ONNX (`onnx/model_q4.onnx`, tokenizer). |
| `embedding.server` | — | `{ url, apiKey?, model? }`: an OpenAI-compatible embeddings server serving EmbeddingGemma (text-embeddings-inference, Ollama, …). |
| `embedding.threads` | 2 | CPU threads for the local model. |
| `rerank` | `'words'` | `'words'`, a rerank server `{ url, apiKey?, model? }` (any `/rerank` API), your own function, or `false`. |
| `llm` | from the environment | `{ model, apiKey?, baseUrl?, reasoningEffort?, onUsage? }` or your own `Ask` function. |
| `qdrant` | a process of its own | `{ url, apiKey?, collection? }` for your Qdrant server, or `{ binary }`. |

| Environment | |
|---|---|
| `MEMTRAILS_CACHE` | Where downloads live (default `~/.cache/memtrails`). |
| `MEMTRAILS_MODEL`, `MEMTRAILS_LLM_API_KEY` or `OPENAI_API_KEY`, `MEMTRAILS_LLM_BASE_URL` | The language model when `llm` is not given. `OPENAI_API_KEY` is sent only to OpenAI's own address. |
| `MEMTRAILS_EMBED_URL`, `MEMTRAILS_RERANK_URL` (+ `_KEY`, `_MODEL`) | Servers, when not given in options. |

## How it works

Every record goes to an append-only log (the truth) and to a Qdrant index (derived, rebuilt from the logs whenever it
is lost or out of date): a key vector of its start, its whole text in ~96-word chunks, and a BM25 keyword vector.
A search runs every channel at once, fuses them by score at the level of items (dense 0.7, keywords 0.3), and reorders
the best 30 word by word. Consolidation, sleep and the concept graph sit on top. See [Architecture](docs/ARCHITECTURE.md),
the [API reference](docs/API.md), and [what was tried and why](docs/RESEARCH.md).

## Benchmarks

| | memtrails | Mem0 | Graphiti |
|---|---|---|---|
| LoCoMo, all 1,307 held-out questions, 6,000 characters, strict judge, `gather` at 40 with notes | **80.9** | 76.5 | — |
| the same, Mem0's own lenient judge | **91.9** | 87.8 | — |
| the same, strict, without the 80 questions whose golden answer is wrong³ | **84.8** | 79.9 | — |
| the same, `gather` without notes, strict / lenient | 79.5 / 91.4 | 76.5 / 87.8 | — |
| the same with `search` alone, no model when searching, strict / lenient | 77.4 / 89.3 | 76.5 / 87.8 | — |
| LoCoMo, 1,307 held-out questions, 28,000 characters, one search¹ | 79.9 | 80.6 | 42.1 |
| LoCoMo, the same with an agent using each system's own tools¹ | **79.0** | 77.1 | 47.0 |
| LongMemEval-S, 60-question sample, 6,000 characters, strict judge, `gather` at 40 with notes (a tie) | 90.0 | 88.3 | — |
| LongMemEval-S retrieval, strict all sessions @5 (467)² | **96.15** | — | — |

The same reader and judges for every system. With `gather` (a model reads what search found, keeps what the answer
needs and notes what it says) memtrails leads Mem0 beyond chance: +113/−55 questions strict, +78/−25 lenient (both
p < 0.001). With `search` alone,
the small-room and one-search comparisons are ties; the agent lead is
small and was judged once. ¹ An earlier build (2026-10-27). ² gbrain-evals publishes 95.96 on the same count: a tie. ³ As found by an
independent audit of LoCoMo ([locomo-audit](https://github.com/dial481/locomo-audit)), dropped for both systems;
+109/−49 (p < 0.001), `bench/audited.ts` reproduces it. [Full results, protocol and limitations](docs/BENCHMARKS.md).

## Limitations

- English benchmarks dominate the evidence. Other languages were measured with translated questions over English memories, not memories written in other languages.
- The word-by-word reorder was trained on nothing; a small per-person table taught on a person's own use did better in tests but is not in this release.
- Questions whose answer is only implied — relevant through outside knowledge, with no word in common — are rarely found by any retrieval system, this one included.
- Consolidation's prompts were tuned on conversations; on other records they are reasonable but unmeasured.
- The index stores 768-dimensional EmbeddingGemma vectors; another embedding model needs its own index and is not measured.

## Privacy

Records stay in `home`. Nothing is sent anywhere unless you configure a language model, an embedding server or a
rerank server; then what they need is sent to them. Secrets (keys, tokens, credentials in URLs, private keys) are
scrubbed from what is sent to a language model and from what consolidation keeps.

## Licence

MIT © Corneliu Maftuleac. Third parties: EmbeddingGemma is Google's, under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms), downloaded from Hugging Face, not shipped; Qdrant is Apache-2.0,
downloaded from its release; LoCoMo (CC BY-NC 4.0) and LongMemEval are linked, never redistributed.
