# Conformance: one memtrails, three languages

memtrails is written in TypeScript (the reference), Python (`python/`) and Rust (`rust/`). The three must behave the
same: the same records give the same keys, words, vectors, prompts and results, and a memory folder written by one is
read by the others. This folder holds what keeps them honest.

- `generate.ts` runs the TypeScript library on fixed inputs and writes what it gives to `fixtures/*.json`.
- Each port's test suite reads the same fixtures and must give the same answers (floating-point values to 1e-6).
- CI regenerates the fixtures and fails when they differ from what is committed: a change in the reference is a
  change every port must follow.

Every input is invented; no benchmark data is used.

## Rules a port must follow

- **Strings are measured in UTF-16 code units**, as JavaScript counts them: `cut(s, n)`, the 60-unit token cap, the
  2-unit minimum word, `firstWords`, `sentenceSpans` offsets. `cut` never ends on the first half of a surrogate pair.
- **`termId` is 32-bit FNV-1a over UTF-16 code units** (not bytes, not code points).
- **`fold`** is NFKD, then combining marks removed after Latin, Greek and Cyrillic letters only, then NFKC, then
  lower case (JavaScript's `toLowerCase`, locale-independent). Thai and Devanagari marks stay — and because NFKD
  splits Thai sara am (U+0E33) and NFKC does not join it again, Thai words come out split as the fixtures show.
- **Words in scripts without spaces** (Han, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar) are split by the ICU word
  segmenter (dictionary-based), keeping only word-like segments. A port uses ICU (PyICU, or ICU4X `icu_segmenter`);
  where it cannot match the fixtures exactly, it says so in its README and its tests mark those cases.
- **Dates**: `timeOf` reads numbers (ms), ISO 8601 (a date alone is UTC midnight; a time without an offset is local
  time, so stored records always carry `Z` or an offset) and RFC 2822 dates; anything else is no time. `dateSpan`
  accepts only `YYYY`, `YYYY-MM`, `YYYY-MM-DD` with a month and a day that exist.
- **Hashes**: `fingerprint` is SHA-1 over the parts, each followed by NUL (a list joined by U+0001), first 16 hex;
  `recordUid` is SHA-256 of `source NUL id`, first 32 hex; `pointId` is SHA-1 of `item#n` laid out as a UUID version 5.
- **Files are shared**: `records.jsonl` / `knowledge.jsonl` (one JSON object a line, the last line of a uid winning,
  `{"uid":…,"removed":true}` removing it), `words.bin`, `usage.json`, `recall.jsonl`, `index.json`, `memory.lock`, and
  the Qdrant collection's layout (`INDEX_LAYOUT`, point ids, payload fields, named vectors `dense`, `bm25`). A port may
  not change any of them without the reference changing first.
- **Prompts are the same text, byte for byte**, and the same JSON Schemas. The prompt fixtures capture each one built
  from fixed inputs.
