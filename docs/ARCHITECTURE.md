# Architecture

memtrails is built like a memory, not like a search engine: what comes in is kept as it came (short-term), searchable
at once; a slower pass turns it into knowledge (long-term); an offline pass reorganises it (sleep); and recall finds,
opens and explores.

```
records, conversations, importers
        │
        ▼
  ┌───────────────┐   writeStm: key vector + every chunk + BM25    ┌──────────────────┐
  │ records.jsonl │ ─────────────────────────────────────────────▶ │  Qdrant index    │
  │ (short-term)  │                                                │  (derived)       │
  └──────┬────────┘                                                └────────▲─────────┘
         │ consolidate (language model): windows of a thread                │
         ▼                                                                  │ writeLtm: key, chunks, cues, concepts
  ┌─────────────────┐  sleep: merge · supersede · link · archive · profiles │
  │ knowledge.jsonl │ ───────────────────────────────────────────────────────┘
  │ (long-term)     │
  └─────────────────┘
        search · searchText · room · open · explore · gather · agent tools
```

## Storage

A memory is one directory (`home`):

| File | What it holds | Truth or derived |
|---|---|---|
| `records.jsonl` | every record and conversation exchange, as added; also consolidation's queue | truth |
| `knowledge.jsonl` | what consolidation and sleep kept: typed items, observations, episodes, profiles, with provenance and status | truth |
| `qdrant/` | the index (when memtrails runs its own Qdrant) | derived |
| `words.bin` | every key and sentence's word vectors, one bit a number | derived |
| `usage.json`, `recall.jsonl` | how often items were shown and opened; what each search showed | record of use |

Both logs are append-only JSONL, read back by byte offset: only each record's place and a small projection stay in RAM.
A later line for the same id replaces the earlier one; a forgotten record is a tombstone line; a log is rewritten
without dead lines once it is twice what is live. The index is rebuilt from the logs, with a resumable checkpoint, when
it is empty or its layout changed.

## Indexing

Each record becomes points in one Qdrant collection:

- **key** — EmbeddingGemma's vector of its title and its first 150 words, with every payload field filters need;
- **children** — its whole text in ~96-word chunks (small-to-big: chunks are searched, the item is returned);
- **keywords** — a BM25 sparse vector over its title and whole text: Unicode word segmentation (`Intl.Segmenter`,
  so Chinese, Japanese and Thai work), compatibility forms unified, accents folded for Latin, Greek and Cyrillic,
  identifiers kept whole and split;
- **cues** (long-term items) — questions someone would ask when the item is needed, written at consolidation;
- **concepts** (long-term items) — the specific things it is about, the nodes of the concept graph.

Vectors are stored on disk, memory-mapped, with a one-bit copy for candidate search and full vectors to rescore.

## Search

1. The query is embedded (EmbeddingGemma's query prompt) and every channel searched at once — dense vectors of keys,
   chunks and cues, and keywords — each grouped by item.
2. Scores are min-max normalised per channel and summed, 0.7 to dense and 0.3 to keywords (CombSUM at the item level:
   it beat rank fusion at the top of both benchmarks).
3. With dates, what happened within them first: an observation by when it happened, anything else by when it was written.
4. The best 30 are reordered word by word and fused with the search's own order by rank (0.7 / 0.3).
5. Each hit carries its key text, its dates, the record it is, and the chunk that matched when that was not its start.

### Word by word

A record's single vector averages it, and one exact detail in a long record is drowned out. The embedder has a vector
per token before it averages them; memtrails keeps them (signs only: 96 bytes a token, at most 128 tokens) and scores a
record by `Σ_i max_j q_i · d_j` over the question's tokens `q_i` and the record's `d_j` (MaxSim, as ColBERT). No second
model and no training. On 2,011 benchmark questions it moved the answer into the top 5 for 80 questions and out for 24.

### The room

`room()` fills a character budget with sentences, not records: each found record is cut at any script's sentence
marks, each sentence scored word by word against the question, its record's rank counting
(`score × (1 + 2 / (1 + rank / 10))`), and the best fitted in, each record's date line once, oldest first.

## Consolidation (optional)

Consolidation reads each settled thread in windows of whole records with one structured language-model call, which
returns typed items (fact, decision, procedure, pitfall, preference, architecture, reference), dated observations
(with when it happened, separate from when it was told, and start and end dates for what lasted), and the exchanges
worth keeping as episodes — each with cues and concepts. A gate in code drops what was inferred without proof,
contested, ambiguous or unimportant. What survives is reconciled with what is known: cosine ≥ 0.93 is the same item (its
provenance grows), 0.80–0.93 asks the model whether it is the same, an update, a contradiction or related, below that
it is new. Each window is read with a calendar of what "yesterday", "last week" and "last month" meant on its days.

## Sleep (optional)

1. **Relate** what was learned since the last sleep to every near neighbour: merge, supersede, link or leave.
2. **Prioritise**: importance × recency (half-lives by kind) × use (how often it was shown and opened).
3. **Archive** what stays low and unused for half a year: out of the index, kept in the log.
4. **Profiles**: for each concept enough items share, the facts they hold gathered into one dated list, rewritten
   when those items change.

## Exploring

The concept graph lives in the index (concept keys on items' key points): `explore` reads the neighbourhood of its
seeds — items found for the topic, or items you name — and walks item ↔ concept and item ↔ item edges with Personalized
PageRank, returning what search did not find, how it was reached, and the concepts around it.

## Gathering (optional)

For questions that count, list or total, `gather` searches wide (up to 200), lets the language model read every
candidate and keep each instance the answer needs, runs the searches it asks for, and puts the kept set first.

## What runs where

| Piece | Where | Needs |
|---|---|---|
| Embedding, word vectors | this process (onnxruntime-node, CPU) or an embeddings server | the model (fetched once) |
| Index | a Qdrant process memtrails starts, or your Qdrant server | the binary (fetched once) |
| Consolidation, sleep, gathering | your language model | an API key or endpoint |
| Everything else | this process | nothing |
