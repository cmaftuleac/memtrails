# API

```ts
import { openMemory } from 'memtrails';
import { fromMbox, fromSlackExport, fromGit, fromFiles, fromJsonl, fromEml, batches } from 'memtrails/importers';
```

## `openMemory(options): Promise<Memory>`

| Option | Type | Default |
|---|---|---|
| `home` | `string` | required: the memory's directory |
| `embedding` | `{ modelDir?: string; server?: EmbedRoute; threads?: number }` | the model fetched into the cache, 2 threads |
| `rerank` | `'words' \| RerankRoute \| ((query, documents, timeoutMs) => Promise<number[] \| null>) \| false` | `'words'` when the model is local |
| `llm` | `Ask \| OpenAiAskOptions` | `MEMTRAILS_MODEL` and `MEMTRAILS_LLM_API_KEY` / `OPENAI_API_KEY` |
| `qdrant` | `{ url: string; apiKey?: string; collection?: string } \| { binary: string }` | a Qdrant process of its own |
| `concurrency` | `number` | 4 windows read at once |
| `log` | `(line: string) => void` | silent |

`EmbedRoute`: `{ url, apiKey?, model?, gzip? }` — an OpenAI-compatible embeddings API (`POST {url}/embeddings`) serving
EmbeddingGemma-300m. `RerankRoute`: `{ url, apiKey?, model?, gzip? }` — the common rerank API (`POST {url}/rerank`).

## `Memory`

### Adding

| Method | |
|---|---|
| `add(records: MemoryRecord \| MemoryRecord[])` | Add or update. Resolves `{ added, unchanged }` once searchable. Throws `TypeError` for an invalid record. |
| `addConversation(conversation: Conversation)` | One record each time the person speaks, with what was answered. |
| `import(source: AsyncIterable<MemoryRecord> \| Iterable<MemoryRecord>, { batch?, onProgress? })` | Streams a source in batches (256). |
| `remove(ids: Array<string \| { id, source? }>)` | Forget records. Returns how many were found. |

```ts
interface MemoryRecord {
  id: string;            // unique within `source`
  text: string;
  title?: string;        // default: the text's first line
  kind?: string;         // email, message, note, memo, article, commit, ticket, … (default 'record')
  source?: string;       // part of the identity
  collection?: string;   // for scoped searches (default 'default')
  thread?: string;       // records that belong together
  author?: string;
  at?: number | string | Date;   // default: when added
  url?: string;
  metadata?: Record<string, string | number | boolean | null>;   // returned, not searched
}

interface Conversation {
  id: string; collection?: string; source?: string; at?: number | string | Date;
  turns: Array<{ role: 'user' | 'assistant'; text: string; author?: string; at?: number | string | Date }>;
}
```

### Finding

| Method | |
|---|---|
| `search(q: SearchOptions): Promise<RecalledItem[]>` | Best first. |
| `searchText(q: SearchOptions): Promise<string>` | The same, as a reader is shown it: one line a memory; `order: 'time'` for oldest first. |
| `room(q: SearchQuery & { chars?: number }): Promise<string>` | The best sentences of what was found, in `chars` (6,000). Needs the local model. |
| `open(id: string)` | One item whole; for something learned, `sources`: the records it was learned from. |
| `explore({ topic?, from?, collections?, limit? }): Promise<Explored>` | What the concept graph reaches: `found`, `related` (with `via`), `concepts`. |
| `gather(q: SearchQuery): Promise<Gathered>` | For counting and listing questions. Needs a language model. |

```ts
interface SearchQuery {
  query: string;
  collections?: string[];   // only these
  kinds?: string[];         // only these kinds: record kinds, or learned ones (fact, observation, profile, …)
  limit?: number;           // default 8, at most 200
  from?: string; to?: string;   // YYYY-MM-DD, YYYY-MM or YYYY: within them first, then the rest
}

interface RecalledItem {
  id: string;              // pass to open()
  kind: string;            // the record's kind, or what was learned
  title: string; summary: string;
  ts: number;              // when it was written or learned
  eventAt?: string; startAt?: string; endAt?: string;   // when it happened, for observations
  match?: string;          // the passage that matched, when it was not the start
  record?: { id: string; source?: string; author?: string; url?: string; metadata?: … };
  project: string;         // its collection key
  score: number;
}
```

### Knowledge (optional, with a language model)

| Method | |
|---|---|
| `consolidate(): Promise<ConsolidationReport>` | Reads every settled thread not yet read. `{ windows, added, merged, updated, superseded, linked, dropped, episodes, observed, cued, paused? }`. |
| `sleep(): Promise<SleepReport>` | One offline pass. Call once a day. |

A thread is read once it has settled: quiet for an hour, or a day old.

### Upkeep

| Method | |
|---|---|
| `indexWords()` | Encodes word vectors not stored yet (runs after add, consolidate and sleep). |
| `rebuild()` | Writes the index again from the logs. |
| `stats()` | `{ records, items, words }`. |
| `tools(): AgentTool[]` | `memory_search`, `memory_open`, `memory_explore`: `{ name, description, inputSchema (zod), jsonSchema, handler }`. |
| `close()` | Stops what it started. |

## Language models

```ts
import { openAiAsk } from 'memtrails';
const llm = openAiAsk({ model: 'gpt-6-luna', apiKey, baseUrl: 'https://api.openai.com/v1', reasoningEffort: 'low',
  onUsage: (u) => console.log(u.inputTokens, u.outputTokens) });
```

`openAiAsk` speaks the OpenAI Responses API with strict JSON Schema output and retries rate limits and server errors.
Any other model: an `Ask` is `(req: { system, prompt, schema, name?, maxTokens?, timeoutMs? }) =>
Promise<{ answer } | { unavailable: string }>`.

## Importers (`memtrails/importers`)

Each returns an `AsyncIterable<MemoryRecord>` and streams.

| Importer | Reads |
|---|---|
| `fromMbox(path, opts?)`, `fromEml(path)` | e-mail: MIME, encodings, threads by References / In-Reply-To |
| `fromSlackExport(dir, opts?)` | a Slack workspace export: names, mentions, links, threads |
| `fromGit(repoDir, opts?)` | commits: message, author, date, changed files |
| `fromFiles(dir, opts?)` | Markdown, text and HTML files: front matter, titles, large files in parts |
| `fromJsonl(path, map?)` | one record (or anything `map` turns into one) per line |
| `batches(iterable, size)` | groups any iterable |

## Lower level

`sentenceRoom`, `sentenceSpans`, `maxSim`, `packWords`, `WordStore`, `ensureModel`, `ensureQdrant`, `searchText`,
`buildAgentTools` are exported for building your own pipeline.
