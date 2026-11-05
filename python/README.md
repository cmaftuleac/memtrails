# memtrails for Python

**Retrieval and exploration for any text — e-mail, chat, notes, memos, documents, news, commit messages, conversations.**

A native Python port of [memtrails](https://github.com/cmaftuleac/memtrails) (npm `memtrails` 0.5.4): the same algorithms,
constants, prompts, JSON Schemas and files, written in Python and natively async. Everything you add is searchable the
moment it is added, on your own machine, with no server and no language model: a local embedding model, a keyword
channel, and a word-by-word reordering that needs no reranker. Give it a language model too, and it consolidates what it
holds into typed knowledge, dated events and profiles — and an agent can explore it.

```python
import asyncio
from memtrails import open_memory

async def main() -> None:
    async with await open_memory(home="./memory") as memory:
        await memory.add([
            {"id": "msg-1", "kind": "email", "author": "Ana", "title": "Proxy certificate expired", "at": "2026-04-02",
             "text": "The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March."},
            {"id": "c9f1e2", "kind": "commit", "author": "Ion", "title": "Restore certificate renewal cron", "at": "2026-04-03",
             "text": "Files: deploy/cron/renew.sh"},
        ])
        hits = await memory.search({"query": "why did the certificate stop renewing?"})
        # [{"id": "stm:…", "kind": "email", "title": …, "summary": …, "ts": …, "record": {"id": "msg-1", "author": "Ana"}, "score": …}, …]

asyncio.run(main())
```

More to run in [`examples/`](https://github.com/cmaftuleac/memtrails/blob/main/python/examples): `quickstart.py`, `import_sources.py` (mail, Slack, git,
notes), `agent_tools.py` (an MCP server, or JSON Schemas for function calling) and `consolidate.py` (with a language
model).

## Install

```bash
pip install memtrails
```

Python 3.10 to 3.13, on macOS or Linux (x64 or arm64) or Windows (x64). The first `open_memory` downloads the embedding model (218 MB, from
Hugging Face) and the Qdrant binary (~30 MB, from Qdrant's GitHub release) into `~/.cache/memtrails`, each checked against
a pinned SHA-256 — the same files, in the same folder, as the TypeScript library, so the two share one download. Set
`MEMTRAILS_CACHE` to put them elsewhere.

On Windows, keep a memory's path under 90 characters: Qdrant's files sit up to about 170 characters deeper, and
Windows stops a path at 260 (memtrails says so when one is longer).

## One memory, two languages

A memory folder written by the TypeScript library opens and searches in Python with the same results, and the other way
round: the same `records.jsonl` and `knowledge.jsonl` logs (crash recovery included), `words.bin`, `usage.json`,
`recall.jsonl`, `index.json`, `memory.lock`, `rebuild.json` and `sleep.json`; the same Qdrant collection (named vectors
`dense` and `bm25`, payload fields, point ids, index layout 7); the same record ids and fingerprints; the same keyword
tokens (ICU word segmentation, through `icu4py`, which ships ICU 78 — the reference's); and, on the same ONNX Runtime
(1.23.2, pinned) and the same machine, the very same vectors (its 4-bit kernels differ slightly between CPU
architectures, in both libraries). The one-process-a-home lock is shared: a memory open in Node cannot be opened in
Python at the same time, and the other way round.

Strings are measured as JavaScript measures them (UTF-16 code units), JSON is written as `JSON.stringify` writes it, and
dates are read as V8's `Date.parse` reads them, local time and mail dates included. Every language-model request — system
text, prompt and JSON Schema — is the reference's, byte for byte.

The port is held to the conformance fixtures in [`../conformance`](../conformance): what the TypeScript library gives for
fixed inputs (text, words, ids, dates, records, the secret filter, search text, importers, prompts, fused search, vectors).

## Adding data

A record is anything with text. Only `id` and `text` are required.

```python
await memory.add({
    "id": "<CAF8w@mail.example>",   # unique within its source; adding the same id again updates it
    "source": "work-mailbox",       # part of the identity
    "kind": "email",                # any word: email, message, note, memo, article, commit, ticket, …
    "collection": "ops",            # search can be scoped to collections
    "thread": "proxy-cert",         # records that belong together
    "author": "Ana Popescu",
    "title": "Proxy certificate expired",
    "text": "…",
    "at": "2026-04-02T09:14:00Z",   # ms, an ISO or RFC 2822 string, or a datetime
    "url": "https://mail.example/…",
    "metadata": {"to": "ops@example.com"},
})

await memory.add_conversation({"id": "chat-42", "turns": [
    {"role": "user", "text": "Where do we have lunch on Friday?"},
    {"role": "assistant", "text": "At the Italian place near the office, at one."},
]})
```

### Importers

Each importer is an async generator: records stream one at a time, never the whole source in memory.

```python
from memtrails.importers import from_mbox, from_eml, from_slack_export, from_git, from_files, from_jsonl, batches

await memory.import_(from_mbox("./archive.mbox"))                  # e-mail (mbox / .eml)
await memory.import_(from_slack_export("./slack-export"))          # a Slack workspace export
await memory.import_(from_git("./my-repo", since="2025-01-01"))    # commit messages and changed files
await memory.import_(from_files("./notes"))                        # Markdown, text and HTML files
await memory.import_(from_jsonl("./records.jsonl"))                # one record per line

async for record in from_mbox("./archive.mbox"):
    ...
```

`import_` takes any async iterable of records (a plain iterable too), so a database cursor or an API client works the
same way. Unchanged records cost nothing; `await memory.remove([...ids])` forgets.

## Searching

```python
await memory.search({"query": "who renewed the certificate?", "limit": 8})
await memory.search({"query": "standup notes", "collections": ["ops"], "kinds": ["message"], "from": "2026-04", "to": "2026-04"})
await memory.search_text({"query": "what happened with the proxy?", "order": "time"})   # as a reader is shown it
await memory.room({"query": "what made the renewal stop?", "chars": 6000})             # the best sentences
await memory.open(hits[0]["id"])                                                         # one item whole
await memory.explore({"topic": "the proxy"})                                             # what the concept graph reaches
```

Results are plain dicts keyed as the reference keys them (`eventAt`, `startAt`, `endAt`, `match`, `record`), so they
serialise as they are and read the same in either language.

## Consolidation and sleep (optional, with a language model)

```python
memory = await open_memory(
    home="./memory",
    llm={"model": "gpt-6-luna", "reasoning_effort": "low", "api_key": os.environ["OPENAI_API_KEY"]},  # or base_url for any OpenAI-compatible server
)
await memory.consolidate()   # after adding
await memory.sleep()         # once a day
await memory.gather({"query": "every plumber we called this year"}, notes=True)
```

Any model works: pass `llm={"base_url": …, "model": …, "api_key": …}` for an OpenAI-compatible server, or your own async
function:

```python
from memtrails import LlmRequest

async def ask(req: LlmRequest) -> dict:
    ...   # req.system, req.prompt, req.schema (strict JSON Schema), req.name, req.max_tokens, req.timeout_ms
    return {"answer": {...}}   # or {"unavailable": "why"}
```

Secrets (keys, tokens, credentials in URLs, private keys) are scrubbed from everything sent to a language model and from
what consolidation keeps, exactly as the reference scrubs them. `OPENAI_API_KEY` is sent only to OpenAI's own address.

## Memory for Claude, Cursor and other MCP apps

One line gives an MCP client a long-term memory on your machine: `memory_search`, `memory_open`, `memory_explore`,
and `memory_save` for what the agent should remember.

```bash
claude mcp add memory -- uvx memtrails mcp                                  # Claude Code
```

```json
{ "mcpServers": { "memory": { "command": "uvx", "args": ["memtrails", "mcp"] } } }
```

The second is for Claude Desktop (`claude_desktop_config.json`), Cursor (`.cursor/mcp.json`) and most other clients.
Fill the memory from the command line (`memtrails` once installed, or `python -m memtrails`), and try it:

```bash
uvx memtrails add ~/notes ~/mail/archive.mbox ./slack-export      # notes and documents, mail, a Slack export
uvx memtrails add --git ~/code/my-repo                             # commit messages and changed files
uvx memtrails search "why did the certificate stop renewing?"
```

The memory lives in `~/.memtrails` (`--home` or `MEMTRAILS_HOME` for another), the same files as the TypeScript
server's: either can serve a memory the other filled. The server opens it at the first tool call and closes it after five
minutes without one (`--idle`), so `add` can run in between; `--read-only` leaves out `memory_save`. From code:
`await serve_mcp(lambda: open_memory(home), version=...)`, answering as the reference does
([`conformance/fixtures/mcp.json`](https://github.com/cmaftuleac/memtrails/blob/main/conformance/fixtures/mcp.json)).

## LangChain and LlamaIndex

A memory is a retriever in either framework, async like the memory (`ainvoke`, `aretrieve`, async chains and query
engines):

```bash
pip install "memtrails[langchain]"      # or memtrails[llama-index]
```

```python
from memtrails.integrations.langchain import MemtrailsRetriever

docs = await MemtrailsRetriever(memory=memory, search={"limit": 8}).ainvoke("why did renewal stop?")

from memtrails.integrations.llama_index import MemtrailsRetriever

nodes = await MemtrailsRetriever(memory).aretrieve("why did renewal stop?")
```

Each document is a found item: its summary (and the passage that matched) as the text, its id, and flat metadata
(`id`, `kind`, `title`, `date`, `ts`, `score`, and when it has them `eventAt`/`startAt`/`endAt`, `recordId`, `source`,
`author`, `url`), the same as the TypeScript LangChain adapter gives
([`conformance/fixtures/retriever.json`](https://github.com/cmaftuleac/memtrails/blob/main/conformance/fixtures/retriever.json)).
`search` takes every search option but the query. The sync calls (`invoke`, `retrieve`) say to use the async ones: the
memory belongs to the event loop it was opened on.

## For agents

```python
for tool in memory.tools():          # memory_search, memory_open, memory_explore
    tool.name, tool.description, tool.json_schema
    result = await tool.handler({"query": "proxy certificate"})   # {"content": [{"type": "text", "text": …}]}
```

The schemas are the reference's, key for key; the handler checks its arguments itself (texts up to 2,000 characters, up
to 50 collections) and answers a bad call with zod's words.

## Configuration

| Option of `open_memory` | Default | |
|---|---|---|
| `home` | — | Where this memory lives: its logs, index and word vectors (made private, 0700). |
| `embedding` | the model fetched into the cache | `{"model_dir": …}`, `{"server": {"url", "api_key"?, "model"?, "gzip"?}}`, `{"threads": 2}`. |
| `rerank` | `"words"` | `"words"`, a rerank server `{"url", …}`, your own async function, or `False`. |
| `llm` | from the environment | Your own async `Ask`, or `{"model", "api_key"?, "base_url"?, "reasoning_effort"?, "on_usage"?, "retries"?}`. |
| `qdrant` | a process of its own | `{"url", "api_key"?, "collection"?}` for your Qdrant server, or `{"binary": …}`. |
| `concurrency` | 4 | Windows the language model reads at once. |
| `http` | one of its own | The `httpx.AsyncClient` every call goes through (yours stays open when the memory closes). |
| `log` | silent | A function of one line. |

The environment variables are the reference's: `MEMTRAILS_CACHE`, `MEMTRAILS_MODEL`, `MEMTRAILS_LLM_API_KEY` or
`OPENAI_API_KEY`, `MEMTRAILS_LLM_BASE_URL`, `MEMTRAILS_EMBED_URL` / `_KEY` / `_MODEL`, `MEMTRAILS_RERANK_URL` / `_KEY` / `_MODEL`.

## Async all the way down

Every method that does I/O is `async def`, and a memory is an async context manager. A memory belongs to the event loop
that opened it: use it, and close it, from that loop. One `httpx.AsyncClient` reaches a Qdrant of yours, the embedding
and rerank servers, the language model and the downloads (streamed, given up after 60 s without a byte); memory's own
Qdrant is reached through a client of its own that takes no proxy from the environment. Qdrant, `tar` and `git` run as
asyncio subprocesses; Qdrant is supervised and restarted by an asyncio task, bound to 127.0.0.1 with a random key,
proven to be ours before it gets the key, and given a minimal environment. The model's inference (one at a time, under
an `asyncio.Lock`), word-by-word scoring, and the reading or rewriting of whole files run on threads, so the event loop
never waits on them. Consolidate, sleep, rebuild, index_words and remove take turns under an `asyncio.Lock`, and an add
waits while a rebuild runs; `close()` stops the running one at its next step and waits for it and for the adds and
searches in flight.

## Known differences

- **Word-like segments**: ICU tells JavaScript whether a segment is a word (its rule status); `icu4py` gives the segments
  but not the status, so it is read off the segment. On adversarial mixtures of every script it agrees with Node except
  for a combining mark left alone after a Hangul syllable or a `‿` joiner — text no one writes. Boundaries agree everywhere.
- **Unicode versions**: `fold` uses Python's `unicodedata` (Unicode 13 on Python 3.10, 15.1 on 3.13) where Node 24 has
  Unicode 17: characters added since fold differently or not at all.
- **Collation**: `localeCompare` breaks two ties (a profile's name among equally common spellings, items of the same time);
  Python has no ICU collator here, so the root collation is approximated (punctuation, digits, letters by base, then
  accents, then case). It agrees on every name memory compares in the tests.
- **ONNX Runtime**: pinned to 1.23.2 for identical vectors; Python 3.14 waits for a runtime that both libraries pin.

## Licence

MIT © Corneliu Maftuleac. Third parties: EmbeddingGemma is Google's, under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms), downloaded from Hugging Face, not shipped; Qdrant is Apache-2.0,
downloaded from its release; ICU (in `icu4py`) is under the Unicode licence.
