# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Windows (x64), in all three languages, tested on every change (the tests, and a real memory searched with the real
  model and Qdrant). Fixed on the way: TypeScript and Python rewrote a log over a file still open; Python checked a
  process with `os.kill(pid, 0)`, which on Windows sends Ctrl+C to its console group, wrote `\r\n` where the files say
  `\n` (records then read back as missing), removed lock files while they were open, took a time-zone offset 2³³
  seconds off for some years, and printed in the console's code page; Rust took every process for alive, never stopped
  its Qdrant at exit, and unpacked the Windows ONNX Runtime as gzip. A memory whose path is too long for Qdrant's files
  on Windows is now said to be so.

## [0.5.3] — 2026-11-05

- LangChain and LlamaIndex retrievers: `memtrails/langchain` (with `@langchain/core` beside it, an optional peer), and
  in Python `memtrails.integrations.langchain` and `memtrails.integrations.llama_index` (`pip install
  memtrails[langchain]` / `[llama-index]`). Each found item becomes the framework's document with the same id, text and
  flat metadata in both languages (`conformance/fixtures/retriever.json`). No TypeScript LlamaIndex adapter: its
  packages are no longer maintained.
  The Rust version is released at the same version, unchanged (LangChain and LlamaIndex are Python and JavaScript).

## [0.5.2] — 2026-11-05

- Examples in Python and Rust, the same four as in TypeScript (a quick start, importing sources, the agent tools,
  consolidation), each run in CI against the real model; every README links them. No change to the libraries.

## [0.5.1] — 2026-11-05

- Rust: the MCP server and the command line, as in TypeScript and Python — `cargo install memtrails`, then
  `memtrails mcp`, `add`, `search`; `memtrails::mcp::serve_mcp` from code, held to the same answers
  (`conformance/fixtures/mcp.json`). The TypeScript and Python versions are released at the same version, unchanged.

## [0.5.0] — 2026-11-05

- An MCP server and a command line: `npx memtrails mcp` serves a memory to Claude Desktop, Claude Code, Cursor and any
  MCP client (`memory_search`, `memory_open`, `memory_explore`, and `memory_save` unless `--read-only`), opening it at
  the first tool call and closing it when idle; `memtrails add` fills it from folders, mail, JSONL, a Slack export or
  git; `memtrails search` asks it. `serveMcp` serves one from code. The same in Python (`uvx memtrails mcp`,
  `serve_mcp`), held to the same answers (`conformance/fixtures/mcp.json`). The Rust version has the same library at
  this version, without the server and command line for now: its `Memory::tools()` serve an MCP server of your own.

## [0.4.3] — 2026-11-05

- Fixes: the first start no longer fails with "Qdrant exited (101) while starting" where the system has no CA
  certificates, as in the official slim Node image. Qdrant will not start without them, though it makes no outside
  call here: it now gets the host's `SSL_CERT_FILE`, or the certificates the library carries. All three languages.

## [0.4.2] — 2026-11-05

- Fixes: the first start no longer fails when run as root (as in most containers) with "is not this user's alone":
  the downloaded Qdrant binary was unpacked with the archive's owner and then refused. TypeScript and Python; the Rust
  version was not affected and is released at the same version.

## [0.4.1] — 2026-11-05

- Native Python (`pip install memtrails`, asyncio) and Rust (`cargo add memtrails`, tokio) versions, with the same
  results and the same files as this library: held to shared answers (`conformance/`), and a memory folder written by
  any of the three is searched the same by the others (`interop/`).
- Fixes: search no longer shows a short record's own summary again as the passage that matched (`match`, and after
  " … " in `searchText`); a match is now only ever a passage beyond the start of what is shown.
- Fixes: the downloaded Qdrant binary and embedding model are only run or loaded from folders that are this user's
  alone. The cache folder (and the model's folder) is made private when it is created or found open to others, and
  refused with a clear error when it belongs to someone else or is a link. `fromFiles` no longer reads through a note
  that was replaced by a link after the folder was listed, and `fromJsonl` and `fromMbox` skip a line longer than
  64 MiB (with the reason) instead of holding it in memory.
- Fixes: a rebuild that is stopped (a close, a newer rebuild) returns only once the batches it was writing are written,
  so none of them lands in the index after it.
- Fixes: `close()` while memory's own Qdrant is being restarted (after it died) stops that restart and waits for the
  Qdrant it began; one could come up after the close and stay running.
- Fixes: an add no longer runs beside a rebuild (a record could be logged but never indexed, then skipped as unchanged
  when added again); `close()` waits for the adds and searches in flight, and every call after it is refused with
  "memory is closed".
- `LlmRequest.signal` (optional): aborted when no one will read the answer any more; `openAiAsk` stops its call and its
  retries. Asks made side by side (consolidation's windows, cues and reconcile batches, gathering's parts, sleep's
  relate batches, profiles) stop the others when one fails. Consolidation that fails or is stopped starts no further
  extraction and returns only once the ones being read are over; before, they ran on, paid for, after it returned.

## [0.4.0] — 2026-11-04

- Prompt examples are invented now. The extraction and gathering prompts had a few example sentences taken from LoCoMo
  conversations, one of them (in `gather`'s notes rule) from a held-out conversation. Their effect on the published
  results, measured on the judged answers: leaving out every held-out question that mentions any of those words or
  people (48 questions) gives 81.1 against Mem0's 76.6 (+111/−55), against 80.9 and 76.5 on all 1,307. The question
  the example came from was answered wrong by both systems.
- The search index layout is 7: `kinds` now selects records by their kind on every point and learned items by theirs,
  so an existing index is rebuilt from the logs on the next open.
- Fixes: a crash in the middle of a write no longer loses the records added after it; consolidation no longer
  overwrites records changed or removed while the model answers; a title cut through an emoji no longer fails a write;
  a thread that never goes quiet is consolidated; imported history is related and kept like new knowledge (`learnedAt`);
  a malformed `from`/`to` is an error; `gather` and `sleep` report `paused` when the model did not read.
- `bench/audited.ts`: judged LoCoMo answers without the questions whose golden answer an independent audit
  ([locomo-audit](https://github.com/dial481/locomo-audit)) found wrong, dropped for every system alike. As shipped,
  84.8 against Mem0's 79.9 (+109/−49).
- `gather`: a memory a small model keeps by quoting its whole line (`[m3] (2023-06-03) …`) now counts; it read as nothing
  kept before.

- Security: the secret filter catches common key formats it missed (GITHUB_TOKEN=, AWS_SECRET_ACCESS_KEY=, JSON-style
  secrets, GitLab, Hugging Face, npm, SendGrid, Telegram, the whole password of a URL); crafted e-mail and HTML can no
  longer make the importers or the filter take quadratic time; downloads are verified on disk in a temp folder of their
  own under a lock, with a stall timeout, and the Qdrant binary runs only while it is this user's alone; Qdrant gets a
  minimal environment and must prove it is ours before it is given the key; an OpenAI key is sent to OpenAI only; the
  memory folder is private; Slack channel folders cannot lead out of the export.
- One process at a time in a memory folder (a lock); consolidate, sleep, rebuild and remove run one at a time; `close()`
  stops the running job and waits for it; an open that fails stops what it started; `remove()` takes the record's kept
  episode out of the long-term log too; `MEMTRAILS_EMBED_URL` and `MEMTRAILS_RERANK_URL` are used.
- The release is built and tested in a job without publishing rights; actions are pinned to commits.

## [0.3.0] — 2026-11-02

- `gather(q, { notes: true })`: the same model call also writes what the kept memories say that answers the question,
  every instance with its date and only what they state, reading all 80 candidates at once. Shown before the
  memories, LoCoMo 79.5 → 80.9 (Mem0 76.5);
  `bench/locomo.ts --gather 40 --notes` reproduces it.

## [0.2.0] — 2026-11-01

- Consolidation reads "last week" as the seven days before it was said, and the calendar line names the last weekend
  and each of the six days before, so "on Friday" is dated without counting. Dates had landed a week early.
- Observations carry 2 to 4 cues (questions they answer, in the asker's words); search reads cues as their own channel
  at a tenth of the dense weight. LoCoMo, all 1,307 held-out questions: multi-hop 50.6 → 55.2, all 77.2 → 77.4.
- `gather` reads as many as `limit` asks for (it always read 200). At 40, for any question, the rooms it builds score
  79.5 on LoCoMo against Mem0's 76.5 (+110/−71 per question); `bench/locomo.ts --gather 40` reproduces it.

## [0.1.0] — 2026-10-31

First public release.

- `openMemory`: records of any kind and conversations, searchable at once with no server and no language model.
- Local embedding (EmbeddingGemma-300m, 4-bit ONNX, fetched and verified on first use) or any OpenAI-compatible
  embeddings server; a Qdrant process of its own (pinned, verified) or your Qdrant server.
- Search: dense keys, chunks and cues plus BM25 keywords, fused by score at the item level; dates first when asked;
  the best 30 reordered word by word (MaxSim on one-bit token vectors), or by any rerank server.
- `room`: the best sentences of what was found, for a small context.
- `open`, `explore` (concept graph, Personalized PageRank), `remove`, `rebuild`, `stats`.
- Optional, with any language model: `consolidate` (typed items, dated observations, episodes, cues, concepts),
  `sleep` (merge, supersede, link, archive, profiles), `gather` (sets for counting and listing questions).
- Agent tools: `memory_search`, `memory_open`, `memory_explore` with zod and JSON schemas.
- Importers: mbox and .eml, Slack exports, Git history, folders of Markdown, text and HTML, JSONL.
- Benchmarks and research notes: docs/BENCHMARKS.md, docs/RESEARCH.md.
