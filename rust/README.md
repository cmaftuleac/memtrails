# memtrails (Rust)

A native Rust port of [memtrails](https://github.com/cmaftuleac/memtrails), the TypeScript library: retrieval and
exploration for any text, on your own machine. The TypeScript code is the reference; this port follows it — the same
algorithms, constants, prompts and files — so a memory folder written by either opens and searches in the other.

This port is being written in pieces. What is in so far: the text pipeline (folding, keywords, children, BM25 terms),
the secret filter, records and conversations, the logs (`records.jsonl`, `knowledge.jsonl`, `usage.json`), word
vectors (`words.bin`, MaxSim), sentence spans, dates, and the text a reader is shown of a search — each checked against
the shared conformance fixtures (`../conformance/fixtures`).

## Known differences

- **Word segmentation of Japanese and folded Thai.** ICU4X's dictionaries split a few words differently from the ICU4C
  that Node.js carries (`がいっぱい`; Thai after NFKD splits sara am). The conformance tests list these cases; every
  other script matches.
- **Characters outside the Basic Multilingual Plane** cannot be cut in half in a Rust string: where the reference would
  slice between the two halves of a surrogate pair, the cut falls before the character.

MIT © Corneliu Maftuleac.
