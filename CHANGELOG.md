# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- `bench/audited.ts`: judged LoCoMo answers without the questions whose golden answer an independent audit
  ([locomo-audit](https://github.com/dial481/locomo-audit)) found wrong, dropped for every system alike. As shipped,
  84.8 against Mem0's 79.9 (+109/−49).

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
