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
  { id: 'msg-1', kind: 'email', author: 'Ana', title: 'Gateway certificate expired', at: '2026-04-02',
    text: 'The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.' },
  { id: 'c9f1e2', kind: 'commit', author: 'Ion', title: 'Restore certificate renewal cron', at: '2026-04-03',
    text: 'Files: deploy/cron/renew.sh' },
]);

const hits = await memory.search({ query: 'why did the certificate stop renewing?' });
// [{ id, kind: 'email', title, summary, match, ts, record: { id: 'msg-1', author: 'Ana' }, score }, …]

await memory.close();
```

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
  found, 79.5 — ahead beyond chance. [Benchmarks](docs/BENCHMARKS.md).
- **Explorable.** A concept graph links what records share — people, places, systems, names — so an agent can walk
  from what it found to what a search for the words does not find.
- **Built for agents.** `memory.tools()` gives `memory_search`, `memory_open` and `memory_explore` with zod and JSON
  schemas, ready for an MCP server or any function-calling API.

## Install

```bash
npm install memtrails
```

Node.js 22 or newer, on macOS or Linux (x64 or arm64). Windows has the pieces it needs but is not tested yet. The first `openMemory` downloads the embedding model
(218 MB, from Hugging Face) and the Qdrant binary (~30 MB, from Qdrant's GitHub release) into `~/.cache/memtrails`,
each checked against a pinned SHA-256. Set `MEMTRAILS_CACHE` to put them elsewhere.

## Adding data

A **record** is anything with text. Only `id` and `text` are required.

```ts
await memory.add({
  id: '<CAF8w@mail.example>',      // unique within its source; adding the same id again updates it
  source: 'work-mailbox',          // part of the identity
  kind: 'email',                   // any word: email, message, note, memo, article, commit, ticket, …
  collection: 'ops',               // search can be scoped to collections
  thread: 'gateway-cert',          // records that belong together
  author: 'Ana Popescu',
  title: 'Gateway certificate expired',
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
await memory.searchText({ query: 'what happened with the gateway?', order: 'time' });

// The best sentences of what was found, for a small context window.
await memory.room({ query: 'what made the renewal stop?', chars: 6000 });

// One item whole — for something learned, with the records it was learned from.
await memory.open(hits[0].id);

// What the concept graph reaches from a topic or from items, that search did not find.
await memory.explore({ topic: 'the gateway' });
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
await memory.gather({ query: 'every doctor I saw this year' });   // a set, for counting and listing questions
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
| `MEMTRAILS_MODEL`, `MEMTRAILS_LLM_API_KEY` or `OPENAI_API_KEY`, `MEMTRAILS_LLM_BASE_URL` | The language model when `llm` is not given. |
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
| LoCoMo, all 1,307 held-out questions, 6,000 characters, strict judge, `gather` at 40 | **79.5** | 76.5 | — |
| the same, Mem0's own lenient judge | **91.4** | 87.8 | — |
| the same with `search` alone, no model when searching, strict / lenient | 77.4 / 89.3 | 76.5 / 87.8 | — |
| LoCoMo, 1,307 held-out questions, 28,000 characters, one search¹ | 79.9 | 80.6 | 42.1 |
| LoCoMo, the same with an agent using each system's own tools¹ | **79.0** | 77.1 | 47.0 |
| LongMemEval-S retrieval, strict all sessions @5 (467)² | **96.15** | — | — |

The same reader and judges for every system. With `gather` (a model reads what search found and keeps what the answer
needs) memtrails leads Mem0 beyond chance: +110/−71 questions strict (p 0.005), +78/−32 lenient. With `search` alone,
the small-room and one-search comparisons are ties; the agent lead is
small and was judged once. ¹ An earlier build (2026-10-27). ² gbrain-evals publishes 95.96 on the same count: a tie. [Full results, protocol and limitations](docs/BENCHMARKS.md).

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
