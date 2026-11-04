# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
