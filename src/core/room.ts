/**
 * A small room filled with sentences, not memories. An agent reads a few thousand characters of what memory found; a
 * memory's first 500 of them are rarely where its answer is. Each found memory's text is cut into sentences (at any
 * script's end marks, no word list), each sentence scored word by word against the question, and the room filled with
 * the best sentences, a memory's place in the search counting too (docs/RESEARCH.md: held out, the answer's words in
 * 6,000 characters 66.0 against 57.2 for Mem0's room on raw memory).
 *
 * Sentences are kept packed beside the memories' own words (words.ts WordStore), by their text, so each is encoded once.
 */
import { maxSim, packWords, type WordEncoder, type WordStore } from './words.js';

/** Sentences shorter than this say too little to be worth a place. */
const MIN_SENTENCE = 8;
/** A sentence longer than this is cut into pieces of it. */
const MAX_SENTENCE = 400;
/** A memory's date and kind, once, before its sentences. */
const HEAD_CHARS = 20;

/** [start, end) of each sentence: after `.`, `!`, `?`, their full-width forms, `。` or a line break, then a space. */
export function sentenceSpans(text: string): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  const push = (a: number, b: number): void => {
    while (a < b && /\s/.test(text[a]!)) a++;
    while (b > a && /\s/.test(text[b - 1]!)) b--;
    for (let s = a; s < b; s += MAX_SENTENCE) out.push([s, Math.min(b, s + MAX_SENTENCE)]);
  };
  const end = /[.!?。！？](?=\s|$)|\n/gu;
  let from = 0;
  for (let m = end.exec(text); m; m = end.exec(text)) {
    push(from, m.index + m[0].length);
    from = m.index + m[0].length;
  }
  push(from, text.length);
  return out.filter(([a, b]) => b - a >= MIN_SENTENCE);
}

export interface RoomItem { id: string; head: string; text: string; ts: number }

/** The room: the best sentences of the first `top` memories, fitted in `chars`, each memory's head once, oldest first. */
export async function sentenceRoom(query: string, items: RoomItem[], encode: WordEncoder, store: WordStore, chars: number, top = 30): Promise<string> {
  const listed = items.slice(0, top).map((it, rank) => ({ it, rank, spans: sentenceSpans(it.text) }));
  const texts = [...new Set(listed.flatMap((l) => l.spans.map(([a, b]) => l.it.text.slice(a, b))))];
  const missing = texts.filter((t) => !store.get(t));
  if (missing.length) {
    const encoded = await encode(missing, 'document');
    await store.put(missing.map((t, i) => [t, packWords(encoded[i]!)]));
  }
  const [q] = await encode([query], 'query');
  const candidates = listed.flatMap((l) => l.spans.map(([a, b]) => ({
    l, a, b, score: maxSim(q!, store.get(l.it.text.slice(a, b))!) * (1 + 2 / (1 + l.rank / 10)),
  }))).sort((x, y) => y.score - x.score);
  const picked = new Map<RoomItem, Array<[number, number]>>();
  let used = 0;
  for (const c of candidates) {
    const extra = c.b - c.a + 3 + (picked.has(c.l.it) ? 0 : HEAD_CHARS + c.l.it.head.length);
    if (used + extra > chars) continue;
    picked.set(c.l.it, [...(picked.get(c.l.it) ?? []), [c.a, c.b]]);
    used += extra;
  }
  return [...picked.entries()]
    .sort(([x], [y]) => x.ts - y.ts)
    .map(([it, spans]) => `${it.head} ${spans.sort((x, y) => x[0] - y[0]).map(([a, b]) => it.text.slice(a, b)).join(' … ')}`)
    .join('\n');
}
