# Research: what was tried and why

How memtrails' retrieval, exploration, consolidation and selection were chosen: every idea tried, how it was tested,
and whether it held. Measured 2026-10-27 to 2026-10-31. For the headline numbers and the full protocol, see
[BENCHMARKS.md](BENCHMARKS.md).

Author: Corneliu Maftuleac. Licence: MIT.

## Method

**Two fixed splits, set before any idea.**

| Split | Dev (ideas designed here) | Held out (opened once) |
|---|---|---|
| Judged answers | LoCoMo conv-26 (152 questions); 3 LongMemEval-S questions of each of 6 types | 8 LoCoMo conversations (1,307 questions) + conv-30 (81); the other 42 LongMemEval-S questions of the 60-question sample |
| Retrieval invention (no model) | LoCoMo conv-26, 41, 43, 47, 49; half of LongMemEval-S; one half of the private real-work set (94 requests) | LoCoMo conv-30, 42, 44, 48, 50; the other half of LongMemEval-S; the other half of the private set (114 requests) |

The private real-work set is a private set of 241 requests from one developer's real work, labelled by a strict model
judge (not released). 208 of them have a needed item.

**Rules.**

- Every arm tried is counted. The retrieval-invention pass tried 110 arms on dev, then opened its test half once, for
  arms named beforehand.
- Two arms on the same questions are compared per question: `+a/−b` questions gained and lost, with a two-sided sign
  test. A mean alone is not a result.
- **A gain found by looking is a lead, not a result.** When a setting was chosen after seeing the data, or a p-value is
  one of many comparisons, it is reported as a lead.
- Every list change is scored two ways: by gold evidence and by whether the answer's words reach the text shown.
  Provenance-based scores credit a consolidated fact with the request it came from, so they overstate long-term items
  (see source de-duplication below).
- Most rounds ask no model and cost nothing: they run on stored vectors on one CPU. Judged runs use the same reader
  (GPT-6 Luna) and judges (GPT-5.5) for every system.

**The measures** (defined in [BENCHMARKS.md](BENCHMARKS.md)): requests any@k / all@k against gold evidence; *answer
fully shown @k*, every word of the gold answer in the text the search tool shows; *words all in the room* for a fixed
6,000-character room; on the private set hit@k, precision@5 and the share of needed items in the top 10; judged
answers (strict and lenient).

**A correction (round 23).** Rounds 14–22 showed only a request's first 500 characters to the word scorer, not its
reply, so answer-shown numbers understated every list. All were rescored. No failed idea turned into a win; the
reranker's gain turned out larger. Numbers below use the corrected display unless marked *(before correction)*.

## Notation and formulas

`q_i` is the i-th token vector of the question, `d_j` the j-th token vector of a memory `m`; all are the embedder's
own (EmbeddingGemma-300m's last hidden state, before pooling). `c` is a memory's conversation, `N` the number of
memories, `df_t` the number holding token `t`.

```
Similarity          s_ij = q_i·d_j
Word by word        s(q, m) = Σ_i max_{j∈m} s_ij                          (MaxSim)
Per memory, per c   M_im = max_{j∈m} s_ij                 T_ic = max_{j∈c} s_ij
Rarity              w_i = log(1 + N/df_t) / log(1 + N)
Two scales          β_c = .25 · min(1, 50/|c|)            S(m) = Σ_i w_i (M_im + β_c T_i,c(m))
Own forgetting      R = 1 − 2 · mean(u)                   (u: how far back a memory's nearest earlier memory lies)
Recency             ρ = max(0, R − .33)                   H(m) = S(m) · (1 + ρ / (1 + a_m))   (a_m: age in months)
GATED               H(m) with β_c used only where ρ < .1; fused 0.7 / 0.3 with the first stage over its best 30
Session pile        P(c) = Σ_i max_{j∈c} s_ij
Sentence score      Σ_i w_i M_is × (1 + 2 / (1 + r/10))   (M_is over sentence s; r: its memory's rank)
Facility location   Σ_i max_{s∈S} M_is                    (chosen set S, by gain per character)
Residual hop        r_i = 1 − max_{s∈seeds} M_is          n_j = (1 − max_i s_ij)(1 − ρ_j)
                    S(m) = Σ_i r_i max_k q_i·d_k + Σ_{b∈B} n_b max_k b·d_k
Phrase vectors      d'_j ∝ d_j + a (d_{j−1} + d_{j+1})
Soft AND            s_p = (mean_i (max_j s_ij)^p)^(1/p)
Whitening           d' ∝ Λ_k^(−α/2) U_kᵀ (d − μ)
Associative map     W = Σ y xᵀ (Σ x xᵀ + λI)⁻¹             (second query vector Wq)
Rank fusion (RRF)   Σ_c w_c / (k + rank_c(m)),  k = 10, dense 0.7, keywords 0.3
CombSUM             Σ_c w_c · minmax_c(score_c(m)),  normalised per question, dense 0.7
```

## Every idea tried

Verdicts: **adopted**, **tie** (no difference that a test supports), **lead** (a gain not yet confirmed), **lost**.
Rounds refer to the research log.

### Retrieval

| Idea | Verdict | Key number |
|---|---|---|
| Fuse item rankings after grouping chunks by item, not fuse raw points | adopted | LoCoMo requests all@10 64.7 → 72.6, multi-hop all@10 11.7 → 21.3 (1,540 questions) |
| Ask each channel as its own grouped query (Qdrant 1.20 ordered a dense prefetch under RRF by its 1-bit scores: top-10 overlap with exact search 0.42 / 0.43) | adopted | dense requests any@5 65.7 → 78.4; wide search p50 604 → 136 ms on 1.1 M points |
| Dense weight 0.7 in RRF, k = 10 | adopted | LoCoMo requests all@10 72.6 (0.5) → 74.0; k = 60 lost at every weight |
| Dense alone instead of hybrid | lost on LoCoMo | LoCoMo requests all@10 71.0 vs 74.0; on LongMemEval-S level |
| CombSUM 0.7 (scores, not ranks) | adopted | LoCoMo any@3 71.6 → 74.5; LongMemEval-S any@1 63.4 → 64.5; no metric down |
| Cross-encoder reranker (bge-reranker-v2-m3) over the top 30 | adopted as an option | LoCoMo any@1 49.6 → 64.4, multi-hop all@10 23.8 → 39.0; answer shown +30/−8 @5, +36/−8 @10 (557 hard-leaning questions); 680 ms p50 on a GPU |
| Query decomposition: 3 model-written queries, fused | lost | LoCoMo multi-hop all@10 25.5 vs the walk's 27.0; together 27.3. A model call per search for the walk's gain |
| **Word by word** (MaxSim on the embedder's token vectors, top 30, fused by rank) | **adopted** | answer shown @5 50.3 → 53.1 (+80/−24, p < 0.001), @10 57.1 → 58.4 (+67/−40, p 0.012), 2,011 questions; 2.6 ms p50 for 30 memories |
| Word by word on top of the reranker | tie | +3/−7 @5, +11/−9 @10 vs the reranker |
| Word by word over every memory of a conversation | tie | @10 59.3 vs 59.7 (an ordering gain, not reach) |
| Drop the fixed prompts; phrase vectors; sharpened sums (`^3`, threshold) | lost | phrases a = .5: dev chat @10 +20/−6 (p .009), but real hit@5 59.6 → 55.3 |
| Soft AND (power mean, p = −6); sum of logs; trimmed sums | lost | top 5 54.7 → 48.0 at p = −6 |
| Identity bonus for the same token | lost | adds nothing |
| Rarity weights `w_i` | lead | dev requests @3 64.4 → 66.5; test answers +15/−13 @5, +15/−12 @10 |
| Two scales (conversation pile in the score) | lead on chat, lost on real work | dev answers @10 +25/−6 (p .001); real hit@1 30.9 → 26.6 |
| Recency by own forgetting `H(m)` | lost | dev real hit@5 60.6 → 68.1 was noise: nothing on test |
| GATED (rarity + gated two scales + recency) | tie | test answers +20/−14 @5 (p .39), +21/−14 @10 (p .31) vs word by word; real hit@5 53.5 vs reranker 63.2 |
| Spectral scores; scaling by how much a conversation fills the list | lost | nothing / worse |
| Refinements of word by word: half and half with search; both directions; one-to-one (Hungarian); best 32-word passage; standardised per word; top 3 of a word | lost or tie | none beats the plain sum (answer @5 53.1); half and half 52.5 with the fewest losses |
| Choose by covered words (facility location) | lost | requests @5 73.1 → 69.0; on the reranker +14/−12, then +14/−22 |
| Character 3–5-grams (TF-IDF) at 0.3 | tie | +29/−28 @5 |
| Learned map 768 → 128 + word weights, taught on other memories | lead | answer @5 54.0 (+109/−35 vs search); on 557 ties the reranker at @5 and @10 (+17/−17, +17/−15), ahead on requests |
| The same, taught on the other benchmark | lost | +90/−45 vs search: no better than a random projection; learns the data's layout |
| Label-free table from a memory's own question–answer pairs | lost | LoCoMo top 5 56.1 → 53.1 (+16/−40) |
| 769-number query-word gate (SmallReason-style) | tie | +79/−23 vs word by word's +80/−24 |
| Table taught per person (each LoCoMo conversation's first half → second half, 2 s) | lead | @10 +16/−7 (p 0.09) vs untrained |
| Table taught on half of the private set | lead | hit@5 60.6 vs reranker 61.5 (+30/−32); precision@5 24.4 vs 21.3; vs first stage +37/−18 (p 0.014); taught on chat benchmarks instead: no gain |
| Fine-tune the reranker's last 2 layers on a CPU | tie | 6 passes at 3e-5: requests @5 49.6 → 53.1, answers +10/−6 (p 0.45); a light run +5/−2 |
| Pruning token vectors: Voronoi, random, first tokens | lost | keep 50%: answer @5 51.1 / 50.1 / 49.4 vs 53.1; first 64 tokens: requests @5 66.7 vs 73.1 |
| Keep only the first 128 tokens | tie | answer @5 53.2, @10 58.5 |
| **One bit a number (768 signs a token)** | **adopted** | 21 KB a memory vs 339 KB; answer @5 52.6 vs 53.1 (noise) |
| 128 random signs | not adopted (weaker) | 3.5 KB; @10 57.7, +64/−51 (p 0.26) vs search |
| GATED stored as 128 PCA bits, no labels | lead | test answer @10 +28/−17 (p .14) vs word by word |
| Whitening for order | lost | compression only; as bits it slips on real work (hit@5 56.4–57.4) |

### Exploration

| Idea | Verdict | Key number |
|---|---|---|
| Rocchio feedback (query moved toward the top hits) | lost | LoCoMo requests all@10 72.6 → 71.6; lost everywhere |
| The top hits' keys as extra queries | tie | LoCoMo multi-hop all@10 21.3 → 23.8, all@10 +0.9 |
| Personalised PageRank over the kNN graph (restart 0.5, 10 neighbours) from the top 5 | adopted, then removed | requests: LongMemEval-S multi-session all@10 75.2 → 80.2; answer shown: LoCoMo +27/−22, LongMemEval-S +6/−3 (a wash), about 300 ms a search |
| Conversation neighbours of each hit | mixed | LoCoMo requests all@10 74.0 → 77.0; LongMemEval-S 87.2 → 82.4 |
| Walk + gated neighbours (neighbour kept if as close as the 10th hit) | lead by requests | LoCoMo 74.0 → 75.7, LongMemEval-S 87.2 → 89.1; not rescored by answers |
| A second walk after the reranker | tie | saturated: LongMemEval-S all@10 91.9 → 92.5, LoCoMo 79.6 → 79.2 |
| Concept graph as a ranker | tie | its lead at the top (any@5 86.8 vs 84.6) came from one fused list, not the graph; by answer words below the plain list (@10 58.4 vs 59.1, *before correction*). Kept as an agent's explore tool |
| Concept-graph expansion of search (HippoRAG-style) | lost | no gain on judged answers |
| Temporal-context keys; residual hops in embedding space; session gists; event-boundary prior | lost | LoCoMo requests all@10 74.4 → 73.0 / 72.2 / 73.7 / 72.5 |
| Cap on hits from one conversation | lost | at most 3: 74.4 → 72.4; at most 1: 51.9. A question's evidence sits together |
| Residual question words as a hop | lead (dev only) | answer @10 +12/−5 (p .14); bridge words add nothing |
| Linear associative map per memory (near turns + rare shared terms) | lost | LongMemEval-S @10 +15/−5 (p 0.04) but LoCoMo +27/−41, gone at @20 |
| The same map learned on the other benchmark | lost | LoCoMo +24/−46 |
| LightGBM ranker over language-free association features | lost | LoCoMo @10 +52/−97 |
| Learned exploration step (classifier over items one step from the top 5) | lost | +3/−2; only 2–7% of what it adds is needed |
| Manifold diffusion; hub correction (CSLS); neighbour blend; reciprocal neighbours | lost | answer @5 50.2 / 49.7 / 46.7 / 39.9 vs search 50.3; at weight 0.1, letters, reciprocal and blend gain half a point @10 (3 of 36 low-weight comparisons) |

### Consolidation

| Idea | Verdict | Key number |
|---|---|---|
| Consolidation into typed long-term items, interleaved with raw requests | adopted | LoCoMo gold requests all@10 71.3 → 77.7, multi-hop 21.6 → 32.6, temporal 77.5 → 86.3 (1,535; by provenance) |
| Atomic dated observations: one event or fact a line, add-only, never merged across dates (from Mem0, Hindsight, Mastra) | adopted | part of the change that took the dev single search 72.4 → 81.4 (not separated) |
| When it happened apart from when it was told; plans apart from what was done | adopted | same change |
| Relative time kept beside the date it means ("last week (the week of 19 June 2023)") | adopted | same change |
| Start and end dates for what lasts | adopted | held-out 166, same search: multi-hop 74.1 → 77.8, temporal 76.5 → 78.8 (one pass) |
| A calendar of "yesterday / last week" ranges in the encoding prompt | adopted | temporal gold fully in context 68 → 69 of 85 |
| "Last week" as the seven days before it was said, with the last weekend and the six days before by name | adopted | the Monday-to-Sunday week put events a week early; held-out temporal (258) judged 78.3 → 80.2 (+20/−15), level with Mem0 |
| Naming what other conversations refer to (`resolve`) | tie, off | +2/−2 questions on held-out; one more lookup a window |
| Profiles: a dated list per concept that ≥ 3 items share, written in sleep, shown whole | adopted | answer @10 59.1 → 61.7 (+56/−16, p < 0.001), temporal 22/1; multi-hop no change (6/6). *Raw memories in this comparison were scored before correction* |
| Cues (3 questions a memory should answer) as a search voice at weight 0.1 | lead | LoCoMo answer @5 +40/−17 (p 0.003), one of 12 arms, weight chosen after equal weight lost |
| Cues on every observation (2 to 4), searched as their own channel at 0.1 of the dense weight | adopted | all 1,307 held-out, judged: multi-hop 50.6 → 55.2 (+24/−13, p 0.10), single-hop 89.1 → 87.9, all 77.2 → 77.4 (with the calendar fix above) |
| Raw passages first in the room (the 6 or 8 best requests as said, before what was learned) | lost | held-out answer words 70.2 → 69.5 / 67.3 (+13/−51); they crowd out dated observations |
| `gather` for every question, at 40 of each (a model keeps what the answer needs, shown first) | adopted, opt-in | held-out answer words 70.2 → 72.0 (+36/−12); judged 77.4 → 79.5, against Mem0 +110/−71 (p 0.005); at 200 no better on the hardest conversation, five times the cost |
| `gather` notes: what the kept memories say, every instance with its date, before the memories | adopted, opt-in | judged 79.5 → 80.6 (+61/−46), against Mem0 +117/−63; notes that also said what memories lack lost (78.7: the reader believed "they don't say") |
| One line a memory in the room; a 200-wide read for every-instance questions; kept memories shown first and apart | lost | answer words 72.0 → 71.7 / 72.2 (ties); judged 79.5 → 79.1 |
| Broad first cue ("What instruments does Priya play?" for her cello) | lost | dev answer words 69.5 → 72.5 (+10/−3); held-out 70.2 → 70.1, multi-hop 41.0 → 38.9: not judged |
| Sleep self-test: store only the cues search fails on | lost | top 3 46.1 → 42.7; saves storage, not cost |
| A per-memory learned model over its own cues | lost | from failed cues alone: +77/−592 @5 |
| Scenes: observations clustered and rewritten once (EverMemOS) | lost | dev multi-hop 68.8 → 64.6 |
| Source de-duplication: move an item whose sources are already shown to the end | lost | by provenance a win; by answer words @10 59.1 → 52.8, temporal −10 (*before correction*) |
| Focused per-turn observation calls; 12k-character windows | tie | no gain |
| A cheap yes/no filter before sleep's relate step | not needed | 0 of 41 related pairs were "different"; the 0.80 cosine gate already removes them |

### Selection: what goes into the room

| Idea | Verdict | Key number |
|---|---|---|
| One line a hit, oldest first, instead of items opened whole (6,000 characters) | adopted | words all in the room 36.7 → 63.3; judged 67.5 → 69.3 (+10/−7) |
| One fused list across kinds for the top, interleave for gathering | adopted | gold any@5 84.6 → 86.8 |
| Deeper lists for "list everything" questions (up to 50, then 200) | adopted | LoCoMo strict 73.4 → 75.1 (233 questions) |
| Two of the top 5 kept for raw requests | tie | any@5 84.6 → 84.2 |
| Choosing a set with the reranker (`set`, `bridge`, `synergy`) | tie | `synergy` +12/−4 (p 0.08), one of about 100 comparisons, gone at @10 |
| MMR 0.5 (variety by vectors) | lost | +14/−6 (p 0.12), requests @10 61.2 → 59.2; over all 2,011 at @5 +13/−17 |
| Coverage of question terms | tie | +11/−5 (p 0.21) |
| Gate: predict when a list is not enough, then show more | lost | AUC 0.49–0.66 (a model), 0.60–0.68 (the reranker's scores); oracle would lift 50.3 → 57.1 |
| Sentence room on raw memory, `score × rank prior` | lead | held-out words all in the room 66.0 vs Mem0 57.2 (+111/−52, p < 0.001); on the 166 hard questions 58.4 vs 65.1 |
| Sentences per character; best 3 of each memory | lost | 62.7 vs 65.6 (dev) |
| Search by sentence over the whole memory | tie | 67.5 vs 67.1 (dev) |
| Sentence room on consolidated memory | tie | 65.1 = lines 65.1 = Mem0 65.1; option, not default |
| Gathering: a model reads a wide set and writes follow-up searches | lead (dev) | LongMemEval-S multi-session 85.0 vs Mem0 82.5 (40 questions, 3 passes) |
| Question's dates as a filter | lost | gold whole in context 54 → 44 of 64 |
| What lies within the dates first, then the rest | adopted | 53 of 64; keeps what the filter lost |
| "Citation lock": answer only what a found memory states | lost | dev LoCoMo 84.2 → 82.2, multi-hop 62.5 → 50.0 |

### Learning from use (simulated)

LoCoMo replayed in order, each learner updated only from earlier questions; opens simulated as every needed request
(`ideal`) or only the needed ones the list showed (`seen`). LongMemEval-S has no history per memory, so it is left out.

| Idea | Verdict | Key number (answer @10 vs search) |
|---|---|---|
| Hebbian co-use links, decaying 3% a question (`seen`) | tie | +11/−9 |
| What past similar questions opened (`ideal`) | tie | +35/−28 (p 0.45) |
| Ridge map from question vectors to opened vectors | lost | +28/−36 (*before correction*) |
| Popularity (control) | lost | +28/−37 (*before correction*) |
| LightGBM ranker over all use signals | tie | +20/−24 (*before correction*); leans on search's own rank |

The ceiling: of 377 questions search misses, the missing request was needed **and** shown before for 37%, all of it for
19%. What repeats is mostly what search already finds. Real opens are the test left.

## What the evidence says

- **Look closer, not elsewhere.** Ideas that change *where* search looks (context keys, residual hops, diversity caps,
  relations between memories) lost. Ideas that look *closer* at what was found (a reranker, word-by-word matching)
  won. The relations between a question's words and a memory's words carry the signal.
- **Ordering is mostly done; reach is the gap.** Of 365 questions missing the answer from the top 5, only 52 (14%)
  have it anywhere in the top 12. When the needed memories hold the answer's words, search shows them in 89.1% of
  lists at 10 (word by word 90.6%, 95.3% at 20).
- **Most misses are not search failures.** The needed memories hold all the answer's words for only 58.0% of LoCoMo
  and 56.5% of LongMemEval-S questions. The rest are inferences, paraphrases, or never said.
- **Learned components learn the layout of their data.** Tables, rankers and exploration steps taught on one
  benchmark did not transfer to the other. Taught on the same person's data, a small table did help, on real work too.
- **A list that lacks the answer looks like one that has it.** No feature of a list predicted its misses well.
- **Consolidation earns its place on time.** Dated observations, start and end dates, and profiles moved temporal
  questions; they did not fix multi-hop lists.

## Open problems

- **Implicit associations.** A memory relevant only through outside knowledge (a tree-nut allergy and a macaron) is
  not found by similarity. Keep It InMind ([2607.24368](https://arxiv.org/abs/2607.24368)) reports 84.0% answered with
  the memory in context, at most 14.4% when six vector, graph and agent systems retrieve it, and a far larger
  embedding does not close the gap. Their open problem is routing: which facts stay visible whatever the query. The
  same shape shows here: most misses are not search failures.
- **Learning from real use.** Simulated use did not beat search. Whether real opens, which are noisier, can teach a
  per-person table (Navigation-Informed Embeddings reports gains with about 26% wrong labels) is untested.
- **Multi-hop lists in a single search.** Mem0 still leads there (−4.2 on held-out); gathering closed it on dev with a
  model call.
- **A measure without the raw-text lean.** Word coverage cannot credit a paraphrase or a date a reader would work out;
  judged runs at 166 questions cannot separate builds. The full held-out small-room run is the next measurement.
- **Other languages and code in memories.** Questions in four languages were measured against English memories;
  memories written in other languages, and code, were not.
- **Cheaper storage.** 128 tokens and one bit a number together (about 12 KB a memory) were not tested as one arm;
  Voronoi pruning lost on short memories.

## Literature

Read in the paper or abstract; numbers are the authors' and were not re-measured here unless a row above says so.

| Work | Why it matters here |
|---|---|
| Keep It InMind, [2607.24368](https://arxiv.org/abs/2607.24368) | The implicit-association gap above |
| SmallReason-ColBERT, [2609.29652](https://arxiv.org/abs/2609.29652) | A 129-parameter query-word gate; its idea tied here |
| Navigation-Informed Embeddings, [2608.15956](https://www.alphaxiv.org/abs/2608.15956) | Retriever taught from agent traces, labels with no judge |
| Voronoi pruning, [2603.09933](https://arxiv.org/abs/2603.09933) | Training-free token pruning; lost here on short memories |
| Token importance in multi-vector retrieval, [2511.16106](https://arxiv.org/abs/2511.16106) | IDF-weighted query tokens; same direction as the rarity weights |
| Complementary curation, [2610.39790](https://arxiv.org/abs/2610.39790) | Episodes plus dated relation tuples |
| REALM, [2609.16053](https://arxiv.org/abs/2609.16053) | A model edits a memory graph after each retrieval |
| LongMemEval, [2410.10813](https://arxiv.org/abs/2410.10813) | The benchmark; index-time fact keys |
| EverMemOS, [2601.02163](https://arxiv.org/abs/2601.02163) | MemScenes; the scenes trial above |
| RippleMem, [2608.13334](https://arxiv.org/abs/2608.13334) | Anchors, then associative expansion |
| SYNAPSE, [2601.02744](https://arxiv.org/abs/2601.02744) | Spreading activation with inhibition and decay |
| HyperAddress, [2610.35412](https://arxiv.org/abs/2610.35412) | Hypergraph PPR for multi-hop |
| AgentIR, [2603.04384](https://arxiv.org/abs/2603.04384) | Embedding an agent's reasoning with its query |
| Reasoning-augmented queries, [2407.12883](https://arxiv.org/abs/2407.12883) | Model query expansion |
| Agent-ModernColBERT, [model card](https://huggingface.co/lightonai/Agent-ModernColBERT) | Small late-interaction retriever for agentic search |
| Mem0 re-evaluated, [2603.04814](https://arxiv.org/html/2603.04814) | Mem0 OSS against full context with the same reader |
| ConvoMem, [2511.10523](https://arxiv.org/abs/2511.10523) | Full context beats retrieval on small histories |
| MemPalace audit, [2604.21284](https://arxiv.org/abs/2604.21284) | recall_any@5 is not recall_all@5 |
| Hindsight, [2512.12818](https://arxiv.org/abs/2512.12818) | Temporal channel, source chunks |
| HaluMem, [2511.03506](https://arxiv.org/abs/2511.03506) | Extraction recall per stage |
| LongMemEval-V2, [2605.12493](https://arxiv.org/pdf/2605.12493) | Experience from web-agent runs |
| Zep's LoCoMo recount, [zep-papers #5](https://github.com/getzep/zep-papers/issues/5) | A vendor figure recalculated (84 → 58.44) |
