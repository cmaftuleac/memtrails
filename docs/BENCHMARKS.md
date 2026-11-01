# Benchmarks

What memtrails was measured on, how, and what the numbers do and do not show. Every number here comes from the
project's own research logs (2026-10-27 to 2026-10-31). Nothing is copied from a vendor's paper unless it says so.

Author: Corneliu Maftuleac. Licence: MIT.

## In short

- **Judged answers, 28,000 characters of context, held-out data.** A single search ties Mem0 on LoCoMo (80.3 vs 80.5
  over 1,388 questions). With an agent using each system's own tools, memtrails is ahead (79.0 vs 77.1 on 1,307,
  judged once). On 42 LongMemEval-S questions it is ahead (92.9 vs 86.5). Graphiti trails both by 30+ points.
- **Judged answers, 6,000 characters, all 1,307 held-out LoCoMo questions.** memtrails as published, alone on a CPU
  with no server: 77.4 strict, 89.3 lenient; Mem0 76.5 and 87.8. A tie (+97/−85 per question); ahead on multi-hop
  questions since cues on observations (55.2 vs 53.6), level on dates.
- **Retrieval.** On LongMemEval-S, strict sessions all@5 with the official answer sessions: 96.15, against
  gbrain-evals' published 95.96. A tie at the top, with a 300M local embedder and no reranker.
- **Word-by-word reordering** (MaxSim on the embedder's own token vectors) wins clearly over plain search
  (+80/−24 questions at 5) and gets most of a cross-encoder reranker's gain, in 2.6 ms on one CPU thread.

## Datasets

| Dataset | What it is | Used here | Licence |
|---|---|---|---|
| [LoCoMo](https://github.com/snap-research/locomo) ([paper](https://arxiv.org/abs/2402.17753)) | 10 very long two-person conversations with questions and gold evidence turns | 1,540 questions without the adversarial category (1,535 in later rounds) | **CC BY-NC 4.0.** Not redistributed with memtrails; download it yourself |
| [LongMemEval-S](https://arxiv.org/abs/2410.10813) | 500 questions, each with its own ~50-session chat history (~115k tokens) and gold answer sessions | 470–476 non-abstention questions for retrieval; a seeded sample of 60 (10 of each of 6 types) for judged answers | MIT |
| Private real-work set | A private set of 241 requests from one developer's real work, labelled by a strict model judge (not released). 208 have a needed item | Real-work retrieval checks only | Not released |

Abstention questions in LongMemEval-S have no evidence, so they are left out of every retrieval score.

## What is measured, and how

Embedder for memtrails throughout: EmbeddingGemma-300m. Mem0 and Graphiti use text-embedding-3-small.

### 1. Gold-evidence retrieval (no model asked)

Each LoCoMo question names the dialogue turns that hold its evidence; each LongMemEval-S question names its answer
sessions. A memtrails "request" is a user turn and the replies to it, so a gold turn maps to the request holding it.

| Metric | Meaning |
|---|---|
| any@k | at least one gold request in the top k |
| all@k | every gold request in the top k |
| nDCG@10 | graded order of the gold requests in the top 10 |
| Strict sessions all@5 | every gold answer session among the first 5 *distinct sessions* of the list. This is how gbrain-evals counts it, using LongMemEval's official `answer_session_ids` |

### 2. The answer's words shown (no model asked)

Gold evidence says whether the right memory was found. It does not say whether the answer reached the reader. So the
second measure checks the text an agent would actually see:

- Build the exact text memtrails shows for each hit (the same line, cut and matched passage as the search tool
  shows; a stored memory is the first 150 words of the request **and** its reply).
- Take the gold answer's words; write dates the same way on both sides (months and days written alike).
- **Answer fully shown @k** = every word of the gold answer appears in the text of the top k. In a fixed room (6,000
  characters) the measure is "the answer's words all in the room".

It mirrors what the agent is shown, not what is stored. It needs no reader and no judge, so it can score thousands of
questions for nothing. Its limits are in [What these numbers do not show](#what-these-numbers-do-not-show).

A correction applies: until round 23 the scorer showed only the first 500 characters of the request, not its reply.
All answer-shown numbers below use the corrected display.

### 3. Judged answers (models as reader and judge)

| Part | Setting |
|---|---|
| Reader | GPT-6 Luna, medium effort — the same for every system |
| Strict judge | GPT-5.5 with LongMemEval's official yes/no prompts, on both benchmarks |
| Lenient judge | GPT-5.5 with Mem0's own LoCoMo judge: one gold item of several, dates within 14 days count |
| Context budget | 28,000 characters (~7k tokens, Mem0's published protocol) or 6,000 characters (a small room, closer to how an agent reads) |
| Ingest | Each system ingests the histories its own way and writes the retrieved context for each question |
| Mem0 | `mem0ai` 2.2.1 open source, hybrid search, top 200 (its memory-benchmarks harness); Luna, text-embedding-3-small |
| Graphiti | `graphiti-core` 0.31.2 on Neo4j, Zep's recipe (20 facts + 20 entities, reranked); Luna, text-embedding-3-small |
| memtrails | Capture → consolidation (Luna) → sleep, then search; context filled up to the budget |

### Statistics

- **Per-question sign test.** Two arms on the same questions: `+a/−b` counts questions the first gets right and the
  second wrong, and the reverse. The p-value is a two-sided binomial test on `a + b`. Ties per question are dropped.
- **Noise of one judged reading:** about ±1.2 points at 1,307 questions, ±3–4 points at 166, ±3–6 at 60 and 233.
- **Split.** Ideas were tried on a dev part and confirmed once on held-out data (see [RESEARCH.md](RESEARCH.md)).

## Results

### (a) Judged answers against Mem0 and Graphiti — 28,000 characters

**Held-out confirmation.** Data no change was tried on: 8 LoCoMo conversations (1,307 questions), conv-30 (81 more),
and the 42 LongMemEval-S questions of the 60-question sample not used for tuning. Read and judged once.

| | LoCoMo (1,307) | multi-hop | open-domain | single-hop | temporal | lenient | conv-30 (81) | LongMemEval-S (42) |
|---|---|---|---|---|---|---|---|---|
| Single search: memtrails | 79.9 | 57.7 | 42.2 | 90.2 | **83.3** | **91.0** | **86.4** | **92.9** |
| Single search: Mem0 | **80.6** | **61.9** | 42.2 | **90.4** | 82.6 | 90.8 | 79.0 | 86.5 |
| Single search: Graphiti | 42.1 | 34.3 | 38.6 | 42.8 | 48.4 | 61.7 | 49.4 | 47.6 |
| Agent + own tools: memtrails | **79.0** | 51.5 | **50.6** | **90.2** | 82.2 | **90.7** | | |
| Agent + own tools: Mem0 | 77.1 | **52.7** | 47.0 | 86.7 | **82.6** | 89.9 | | |
| Agent + own tools: Graphiti | 47.0 | 31.8 | 47.0 | 47.5 | 59.7 | 68.2 | | |

- Over all 1,388 held-out LoCoMo questions, a single search is a **tie**: 80.3 vs Mem0's 80.5.
- With an agent, memtrails is ahead by 1.9 points on 1,307 questions. Judged once; no per-question test was run.
- On LongMemEval-S memtrails is ahead by 6.4 points, on 42 questions (each question is 2.4 points). With every
  observation line shown it scored 95.2.
- Where Mem0 leads: multi-hop "list" questions in a single search (−4.2, about 10 of 239 questions).
- Single search, memtrails: the "mixed room" — dated observation lines oldest first, then the best raw requests and
  items opened whole by rank — at limit 200.
- Agent: the same agent (GPT-6 Luna, medium) for every system, with that system's own read tools: memtrails' search,
  open and explore; Mem0's OpenMemory MCP read tools; Graphiti's MCP server read tools. At most 8 calls, 8,000
  characters a result, 40,000 read.

**Later builds, on parts of the held-out set** (one judged pass each; these builds add *gathering*, a model call that
reads a wide candidate set and writes follow-up searches, and the dating steps described in RESEARCH.md):

| Set | memtrails | Mem0 | Reading |
|---|---|---|---|
| LongMemEval-S, 39 of the 42 (3 failed on embedding timeouts) | 97.4 | 82.1–89.7 over 3 passes | ahead, small sample |
| 3 held-out LoCoMo conversations (conv-41, 44, 47), 425 questions | 82.8 | 83.3 | tie |
| Their 166 multi-hop + temporal questions, with start/end dates for what lasts | 78.3 (multi-hop 77.8, temporal 78.8) | 75.9 (multi-hop 74.1, temporal 77.6) | ahead by one pass; not tested per question |

**Reference points from the first, smaller comparison** (60 LongMemEval-S + 233 LoCoMo questions, an earlier build,
2026-10-27; ±3–6 points):

| | LongMemEval-S | LoCoMo strict | LoCoMo lenient | Ingest cost |
|---|---|---|---|---|
| Whole history, unlimited room | 93.3 | 84.5 | 94.8 | — |
| Raw rounds, EmbeddingGemma + BM25, no memory model | 88.3 | 83.3 | 94.4 | — |
| Mem0 | 88.3 | 81.1 | 93.1 | $6.08 |
| memtrails (that build) | 81.7 | 75.1 | 91.8 | $5.98 + embeddings on own GPU |
| Graphiti | 48.3 | 48.1 | 69.1 | $16.64 |
| Whole history cut to the room | 8.3 | 37.3 | 53.6 | — |

Plain retrieval over raw rounds was already level with or above Mem0 at this budget. Graphiti's LongMemEval-S number is
understated: it was ingested in 4,000-character runs of messages, not one message an episode as Zep does.

### (b) The small room — 6,000 characters, 166 hard held-out questions

The 166 multi-hop and temporal questions of three held-out LoCoMo conversations (conv-41, 44, 47). Same reader and
judges. Read once. Mem0's room: its stored memory searched again by its own code, the lowest-scored dropped to fit.

| 166 questions, read once | strict | multi-hop | temporal | lenient | answer's words all in the room |
|---|---|---|---|---|---|
| **Library alone, no server** (word-by-word order on the CPU; round 28) | **69.9** | 59.3 | 80.0 | **92.8** | 65.1 |
| **Library, built from nothing** (consolidation, sleep, dates, observations, profiles; round 30) | 69.3 | 56.8 | 81.2 | 89.8 | **68.7** |
| With a reranker server, one line a hit (round 27) | 69.3 | 56.8 | 81.2 | 89.2 | 63.3 |
| With a reranker server, items opened whole (round 27) | 67.5 | 56.8 | 77.6 | 88.0 | 36.7 |
| No reranker, items opened whole (round 27) | 60.2 | 51.9 | 68.2 | 83.1 | — |
| **Mem0, 6,000** | 68.7 | 59.3 | 77.6 | 88.0 | 65.1 |
| Mem0, 28,000 (reference) | 75.9 | 74.1 | 77.6 | 92.2 | — |

Per question (strict, then lenient):

| Comparison | Strict | Lenient | Reading |
|---|---|---|---|
| Library alone vs Mem0 | +17/−15 (p 0.86) | +13/−5 (p 0.10) | tie, leaning memtrails on the lenient judge |
| Library alone vs reranker server | +11/−10 | | tie: removing the server costs nothing measurable |
| Built from nothing vs Mem0 | +17/−16 | | tie |
| Built from nothing vs library on the older memory | +14/−15 | | tie |
| Reranker, lines vs Mem0 | +18/−17 | +10/−8 | tie |
| Reranker, lines vs opened whole | +10/−7 | | tie |
| Reranker, opened whole vs Mem0 | +17/−19 (p 0.87) | +11/−11 | tie |
| No reranker, opened whole vs Mem0 | +14/−28 (p 0.04) | | **loss** |

Answer's words all in the room, built from nothing: +7/−1 vs the older memory, +15/−9 vs Mem0. More of the answers'
words reach the room; the judged answers do not move.

How the room is spent: opened whole, memtrails fits 12 items and repeats itself (title, summary, raw source); Mem0 fits
31 short facts. Switching to one line a hit (up to 20 hits, best kept, shown oldest first) raised the answer's words in
the room from 36.7 to 63.3 (60 hits: 62.7).

Build and run figures for the library rows:

| | Figure |
|---|---|
| Filling 2,357 memories (word vectors) | 229 s on 4 threads of a 2014 six-core desktop CPU (i7-5820K), about 10 a second |
| Size | 5.3 MB for one conversation of 769 memories: 6.9 KB a memory at 768 signs a word |
| Build from nothing, 3 conversations | $0.22 of model calls; about 10 minutes a conversation (capture 74 s, consolidate and sleep 200–250 s, word vectors 130 s) |
| Search p50 | library 353 ms, reranker server 458 ms, Mem0 562 ms — **not like for like**: measured at different times on a busy machine |

**Every build and layout tried on these 166 questions lands at 68–70 strict.** One reading moves a run of 166 by ±3–4
points, so this set cannot tell them apart. The measurement that can is the full held-out set (±1.2), below.

### (c) Full held-out set, small room

All 8 held-out LoCoMo conversations, all 1,307 questions (adversarial left out), 6,000 characters, read and judged
once. memtrails is the library as published: built from nothing by `openMemory` (records added, consolidated and slept
with GPT-6 Luna, $0.56 for the 8 conversations), embedding and word-by-word ordering on the CPU, no reranker server, the
context as an agent reads `searchText`. Mem0 is its stored memory searched again by its own code, the lowest-scored
memories dropped to fit the same 6,000 characters.

| 1,307 questions | strict | multi-hop (239) | open-domain (83) | single-hop (727) | temporal (258) | lenient (Mem0's judge) |
|---|---|---|---|---|---|---|
| memtrails, no server | **77.4** | **55.2** | **42.2** | **87.9** | 79.8 | **89.3** |
| memtrails 0.1.0 | 77.2 | 50.6 | 45.8 | 89.1 | 78.3 | 89.3 |
| Mem0 | 76.5 | 53.6 | 37.3 | 87.1 | **80.6** | 87.8 |

Per question against Mem0: strict +97/−85 (p 0.42), lenient +62/−43 (p 0.08); multi-hop +31/−27, temporal +16/−18.
memtrails 0.1.0 against Mem0: strict +102/−93 (p 0.57), lenient +65/−46 (p 0.09); multi-hop +28/−35, open-domain
+10/−3, single-hop +44/−29 (p 0.10), temporal +20/−26.

**A tie.** With little to read, memtrails running alone on a CPU answers as well as Mem0 with its model-written facts —
not better. Since 0.1.0, two changes closed the categories Mem0 led: "last week" read as the seven days before it was
said (dates had landed a week early), and cues on every observation, searched as a quieter channel (answers worded
apart from the question were stored but not found). Multi-hop moved 50.6 → 55.2 against 0.1.0 (+24/−13, p 0.10) and
single-hop gave back about as much, so the total barely moves. None of the differences with Mem0 is significant.
Judging cost about $9.4 a system.

### (d) Retrieval against gbrain-evals — LongMemEval-S

Strict: every official answer session among the first 5 distinct sessions. 467 questions with official answer
sessions. No reranker in any memtrails row.

| | Strict sessions all@5 | vs search |
|---|---|---|
| Search (dense + keyword fused) | 94.22 (440) | |
| **Search + word-by-word order** | **96.15 (449)** | +11/−2 (p 0.02) |
| Session scored as one pile of words | 95.72 (447) | |
| GATED (see RESEARCH.md) | 94.00 (439) | |
| gbrain-evals, published, 470 questions | 95.96 (451) | not re-run here |

A tie at the top, with no reranker and a 300M local embedder. gbrain-evals' figure is their own count on 3 more
questions. The 467 include the dev half of the research split, so this is not a held-out confirmation. An earlier count in the logs, by the turns holding the evidence rather than the official answer sessions,
is looser (search 95.6) and is not comparable.

### (e) Word-by-word reordering

A memory's single vector averages it, so one exact detail in a long memory is drowned out. EmbeddingGemma already
computes a vector per token before it averages them. memtrails keeps those token vectors, scores each of the search's
top 30 by `Σ_i max_j q_i·d_j` (MaxSim, as ColBERT does), and fuses that order with the search by rank. No second model,
no training, no word lists.

**All 2,011 questions** (LoCoMo 1,535 + LongMemEval-S 476), requests any@k / answer fully shown @k:

| | @3 | @5 | @10 | @20 | answer per question @5, @10 vs search |
|---|---|---|---|---|---|
| Search | 59.1 / 45.5 | 68.0 / 50.3 | 78.0 / 57.1 | 85.0 / 64.2 | |
| **Word by word, top 30** | 63.5 / 47.5 | 73.1 / 53.1 | 81.2 / 58.4 | 87.2 / 65.2 | **+80/−24 (p < 0.001), +67/−40 (p 0.012)** |
| Word by word, the search's 40 | 63.5 / 47.5 | 73.2 / 53.1 | 81.8 / 58.7 | 88.0 / 65.6 | +82/−26, +74/−41 |

- LoCoMo +68/−18 and +57/−34; LongMemEval-S +12/−6 and +10/−6.
- LongMemEval-S requests @10 86.1 → 92.2, multi-session 72.0 → 82.4. LoCoMo multi-hop answers @10 28.7 → 34.8
  (+19/−2).
- Scoring every memory of a conversation instead of the search's list adds nothing (@10 59.3 vs 59.7): it is an
  ordering gain, not a reach gain.

**Against a cross-encoder reranker** (bge-reranker-v2-m3 over the top 30). 557 questions leaning hard: all 282
multi-hop and 125 multi-session questions, plus 150 others. Compare rows with each other, not with benchmark scores.

| 557 questions | @3 | @5 | @10 | @20 | answer per question @5, @10 vs reranker |
|---|---|---|---|---|---|
| Search | 30.3 / 25.7 | 39.9 / 30.5 | 52.4 / 37.0 | 66.1 / 47.6 | +8/−30, +8/−36 |
| Word by word | 35.0 / 27.1 | 47.6 / 34.3 | 60.1 / 40.4 | 70.9 / 49.9 | +15/−16, +13/−22 (p 0.18) |
| Reranker (fused with search by rank) | 39.3 / 29.8 | 49.6 / 34.5 | 61.2 / 42.0 | 70.6 / 50.4 | |
| Reranker + word by word | 37.3 / 27.6 | 49.4 / 33.8 | 61.9 / 42.4 | 70.7 / 50.1 | +3/−7, +11/−9 |

Word by word gets most of the reranker's gain over search: answers @5 +3.8 against +4.0, @10 +3.4 against +5.0. It is
weakest at the very top (@3 +1.4 against +4.1); that gap is unproven either way. Together they add nothing, so in the
library word by word replaces the reranker; it does not sit on top of it.

**On real work** (the private set: 208 requests with a needed item; code, file names, two languages mixed). Each
request's 60 first-stage candidates; best 30 reordered.

| 208 requests | hit@1 | hit@3 | hit@5 | precision@5 | share of needed @10 | hit@5 vs first stage, vs reranker |
|---|---|---|---|---|---|---|
| First stage | 29.3 | 45.2 | 51.4 | 19.5 | 26.0 | , +15/−36 |
| Word by word | 28.8 | 48.1 | 56.7 | 22.0 | 32.4 | +28/−17 (p 0.14), +22/−32 |
| Reranker | 26.9 | 51.9 | 61.5 | 21.3 | 33.0 | +36/−15 (p 0.005), |

Untrained, word by word earns about half of the reranker's hit@5 gain on real work. A small table taught on the same
person's requests closes the gap; it is not in the library (see RESEARCH.md, "learned tables").

**Questions in other languages** (400 LoCoMo questions rewritten by a model; memories stay English; first list by
dense vectors alone, since keywords cannot cross languages). Answer shown @5 (per question vs dense), @10:

| | Dense | Word by word |
|---|---|---|
| English | 51.0, 58.0 | 52.5 (+16/−10), 59.2 |
| Romanian | 43.0, 51.0 | 48.2 (+28/−7), 53.5 |
| Russian | 49.8, 55.8 | 51.0 (+17/−12), 57.2 |
| Chinese | 47.8, 55.5 | 53.0 (+27/−6), 58.8 |
| Romanian + English, typed fast | 41.8, 50.2 | 46.2 (+26/−8), 52.5 |

It helps most where the vectors are weakest. Memories written in other languages, and code, are not measured.

**Storage and speed.** Same top 30, less kept per token (all 2,011 questions):

| Kept per token | Size a memory | Requests / answer @5 | @10 | Answer per question @5, @10 vs search |
|---|---|---|---|---|
| Everything (768 numbers, 2 bytes each) | 339 KB | 73.1 / 53.1 | 81.2 / 58.4 | +80/−24, +67/−40 |
| 128 random projections | 56 KB | 72.3 / 52.5 | 81.0 / 59.0 | +79/−34, +73/−35 |
| First 128 numbers | 56 KB | 72.0 / 52.7 | 80.9 / 58.7 | +84/−36, +68/−36 |
| **768 signs (one bit each)** | **21 KB** | 72.0 / 52.6 | 81.7 / 58.4 | +77/−30, +61/−34 |
| 128 random signs | 3.5 KB | 71.3 / 52.3 | 79.7 / 57.7 | +73/−32, +64/−51 (p 0.26) |
| Everything, first 128 tokens only | 197 KB | 72.7 / 53.2 | 81.0 / 58.5 | +87/−28, +68/−39 |
| Everything, first 64 tokens only | 98 KB | 66.7 / 50.4 | 76.1 / 56.9 | +74/−71, +58/−61 |

- One bit a number keeps nearly all of it. Differences between the first five rows are noise (8 arms, one data set).
- 128 tokens are enough; 64 are too few.
- Sizes above are for a memory of 220 words on average. In the library build of round 28 a memory took 6.9 KB.
- Scoring 30 memories: **2.6 ms p50, 7.8 ms p95** on one CPU thread (disk not counted). The reranker over the same 30
  took 680 ms p50 on a GTX 1080 Ti.

### (f) The sentence room

`memory.room` (src/core/room.ts) fills a fixed room with sentences rather than one line a memory: each found memory's
text is cut at any script's sentence marks, each sentence is scored word by word against the question, its memory's
rank counts, and each memory's date line is shown once, oldest first.

**On raw memory** (no model at ingest), held-out LoCoMo, the answer's words all in 6,000 characters:

| Test, 668 questions Mem0 has | Words all in the room | Mean share | vs Mem0 |
|---|---|---|---|
| Mem0, its model's facts | 57.2 | 76.7 | |
| Search, memories as shown | 61.5 | 78.6 | +99/−70 (p 0.03) |
| Word-scored list, memories as shown | 63.3 | 79.0 | +105/−64 (p 0.002) |
| **Word-scored list, chosen sentences** | **66.0** | **80.2** | **+111/−52 (p < 0.001)** |

By kind (sentences / Mem0): single-hop 84.9 / 66.7 (+89/−20), multi-hop 39.2 / 30.8 (+16/−6), temporal 51.8 / 66.4
(+6/−26), open-domain 6.1 / 6.1.

**On consolidated memory** (round 29), the same 166 hard questions as (b):

| | All | Multi-hop | Temporal | vs Mem0 |
|---|---|---|---|---|
| One line a memory | 65.1 | 51.9 | 77.6 | +12/−12 |
| Sentences, the date only | 63.9 | 55.6 | 71.8 | +10/−12 |
| Sentences, the date line as search shows it | 65.1 | 55.6 | 74.1 | +12/−12 |
| Mem0 | 65.1 | 54.3 | 75.3 | |

Caveats:

- The 66.0 vs 57.2 result is by word coverage only. It was never judged by a reader.
- The measure leans toward raw text: LoCoMo's answers are spans of the dialogue, and a fact that says the same thing in
  other words scores nothing.
- Temporal goes the other way: Mem0's model resolved "yesterday" into dates while writing, which a reader could also
  do from the date shown, but the word test cannot see.
- On the 166 hard questions the raw-memory sentence room does not lead: 58.4 words all in the room vs Mem0's 65.1.
- On consolidated memory sentences are level with lines and Mem0: they help multi-hop and cost dates. Not judged,
  since they did not pass 65.1. Kept as an option; lines stay the default. Sentences double the stored size (11 MB
  for a conversation of 769).

## What these numbers do not show

- **Ties are ties.** Against Mem0: the single search on 1,388 held-out questions (80.3 vs 80.5), every 6,000-character
  build on 166 questions, the reranker against word by word, and gbrain-evals' 95.96 against 96.15 are ties. Nothing
  here says memtrails answers better than Mem0 at a small budget.
- **The 166-question set cannot separate builds.** One reading moves a run of 166 by ±3–4 points; every build lands at
  68–70 strict. The full held-out small-room run is the one that can.
- **Word coverage favours raw text.** "Answer's words shown" rewards the exact words of the dialogue, not a correct
  paraphrase, and it cannot see a date a reader would work out. On LoCoMo the needed memories hold all the answer's
  words for only 58.0% of questions (LongMemEval-S 56.5%); the rest are inferences, paraphrases or never said.
- **Temporal questions.** Coverage understates raw memory on temporal questions, by an unknown amount. Judged temporal
  scores move by several questions between passes; one flip was the same "about two months" judged both ways.
- **Sample sizes.** 42 and 39 LongMemEval-S questions; 166 hard LoCoMo questions; 557 hard-leaning questions for the
  reranker comparison; 208 real-work requests. Read the per-question tests, not the percentages.
- **A single judge pass.** Held-out runs were read and judged once. On the dev batch the same memory read twice moved by
  ±4 points on 152 questions.
- **Judges are models.** GPT-5.5 decides what counts as right. The literature reports a gpt-4o-mini judge accepting
  62.8% of wrong but on-topic LoCoMo answers, and 6.4% of LoCoMo's answer key being itself wrong.
- **The two splits overlap.** The 166 hard questions (conv-41, 44, 47) are held out from the judged-answer tuning, but
  conv-41 and conv-47 were in the dev half of the split on which the word-by-word scoring and the sentence room were
  designed. The raw-memory sentence-room test in (f) was opened once on that split's own held-out half; the retrieval
  test in (d) covers both halves.
- **Vendors' own published numbers are not comparable.** Every system here was re-run with the same reader, judges and
  budget. Published figures use other readers, judges, budgets and question subsets; several have been recalculated
  downward by third parties.
- **Real-work labels are a model's.** On the private set, a single strict vote was confirmed on a blind repeat 58% of
  the time.
- **Search times are not like for like.** They were measured on different machines and at different loads.
- **Many ideas lost.** Most of what was tried did not hold. They are listed with numbers in
  [RESEARCH.md](RESEARCH.md); among them the graph walk (a wash by the answer's words, removed), learned exploration,
  associative maps, set selection, MMR, list-quality gates and learning from simulated use.

## Reproduce

The harness is in [`bench/`](../bench/README.md): `bench/locomo.ts` builds each conversation and writes each
question's context, `bench/score.ts` scores the answer's words with no model, `bench/judge.ts` reads and judges. The
protocol, in words:

1. **Get the data.** Download LoCoMo yourself (CC BY-NC 4.0; memtrails does not ship it) and LongMemEval-S.
2. **Map turns to requests.** A request is a user turn and the replies to it. In LoCoMo, whoever opens a session makes
   its requests, so a session opened by the second speaker does not lose its first turn.
3. **Ingest.** Capture every conversation with EmbeddingGemma-300m on the CPU; for consolidated runs, consolidate and
   sleep with a language model. Wait until every conversation file is captured before reading, or fail the unit.
4. **Retrieval scores.** For each question, run the search, map hits to gold turns or sessions, and compute any@k,
   all@k, nDCG@10 and strict sessions all@5 (first 5 distinct sessions, official answer sessions). Leave abstention
   questions out.
5. **Coverage scores.** Render exactly the text the search tool shows (or fill the 6,000-character room), normalise
   dates, and check whether every word of the gold answer is present.
6. **Judged answers.** Give each system's context, up to the same budget, to the same reader (GPT-6 Luna, medium
   effort). Judge with GPT-5.5: LongMemEval's official yes/no prompts (strict) and Mem0's LoCoMo judge (lenient).
7. **Compare per question.** Report `+a/−b` and a two-sided sign test for every pair of arms, not only the means.
8. **Keep the split.** Tune on the dev part (LoCoMo conv-26 and a few LongMemEval-S questions of each type); open the
   held-out part once, for arms named beforehand.
