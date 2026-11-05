/**
 * What gets indexed from an item's text, besides its content key: the keyword vector (BM25 as a Qdrant sparse vector —
 * term frequencies here, the IDF on Qdrant's side with `modifier: idf`) and the small-to-big children.
 *
 * Measured on a private history of real work: the keyword channel beside the dense one put a needed
 * item in the top 5 for +5–8 points; ~96-word children beat 192-word ones; adding raw text to the key itself hurt.
 */
import type { SparseVector } from './qdrant-client.js';

const K1 = 1.2;
const B = 0.75;
/** Longer than this is not a word a person would type (a hash, a base64 blob). */
const MAX_TOKEN = 60;

/** Scripts written without spaces between words: there the ICU word segmenter finds the words. */
const UNSPACED = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]/u;
const WORDS = new Intl.Segmenter(undefined, { granularity: 'word' });

/** Lower case, compatibility forms unified, accents dropped from Latin, Greek and Cyrillic letters — "bază" and
 *  "baza" are the same word to a person typing either. Marks that make the word elsewhere (Devanagari, Thai) stay. */
export function fold(text: string): string {
  return text.normalize('NFKD').replace(/([\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}])\p{M}+/gu, '$1').normalize('NFKC').toLowerCase();
}

/** Words for the keyword channel, in any language: folded (fold()); identifiers kept whole AND split at `_ . - /`; in
 *  scripts without spaces (Chinese, Japanese, Thai…) the segmenter's words. */
export function tokens(text: string): string[] {
  const out: string[] = [];
  for (const raw of fold(text).split(/[^\p{L}\p{N}\p{M}_.\-/]+/u)) {
    const w = raw.replace(/^[._\-/]+|[._\-/]+$/g, '');
    if (!w) continue;
    if (UNSPACED.test(w)) {
      for (const seg of WORDS.segment(w)) if (seg.isWordLike) out.push(seg.segment);
      continue;
    }
    if (w.length < 2) continue;
    // The length cap is per word: a path too long to be one keyword still gives its parts.
    if (w.length <= MAX_TOKEN) out.push(w);
    if (/[_.\-/]/.test(w)) for (const p of w.split(/[_.\-/]+/)) if (p.length >= 2 && p.length <= MAX_TOKEN && p !== w) out.push(p);
  }
  return out;
}

/** Where each word of a text starts and ends — the segmenter's words, so a text without spaces counts right. */
function wordSpans(text: string): Array<[number, number]> {
  const at: Array<[number, number]> = [];
  for (const seg of WORDS.segment(text)) if (seg.isWordLike) at.push([seg.index, seg.index + seg.segment.length]);
  return at;
}

/** Whitespace runs as one space, the ends trimmed. */
export const flat = (s: string): string => s.replace(/\s+/g, ' ').trim();

/** A text's first `n` words, as written (whitespace flattened). */
export function firstWords(text: string, n: number): string {
  const at = wordSpans(text);
  return flat(at.length > n ? text.slice(0, at[n - 1]![1]) : text);
}

/** 32-bit FNV-1a: a term's index in the sparse vector. */
export function termId(term: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < term.length; i++) {
    h ^= term.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** A document's BM25 term weights (Qdrant applies the IDF). `avgLen`: the typical length of what is indexed. */
export function docSparse(text: string, avgLen = 220): SparseVector {
  const ts = tokens(text);
  const tf = new Map<number, number>();
  for (const t of ts) { const id = termId(t); tf.set(id, (tf.get(id) ?? 0) + 1); }
  const norm = K1 * (1 - B + (B * ts.length) / avgLen);
  const indices = [...tf.keys()].sort((a, b) => a - b);
  return { indices, values: indices.map((i) => { const f = tf.get(i)!; return (f * (K1 + 1)) / (f + norm); }) };
}

/** A query's terms, each once. */
export function querySparse(texts: readonly string[]): SparseVector {
  const ids = [...new Set(texts.flatMap((t) => tokens(t)).map(termId))].sort((a, b) => a - b);
  return { indices: ids, values: ids.map(() => 1) };
}

export const CHILD_WORDS = 96;
export const CHILD_STRIDE = 72;
/** ~96-word windows of a body, a quarter overlapping, all of it — searched, never shown: the item is what comes back.
 *  Words as the segmenter finds them, so a body in a script without spaces is windowed too. */
export function children(text: string, words = CHILD_WORDS, stride = CHILD_STRIDE): string[] {
  const at = wordSpans(text);
  if (!at.length) return flat(text) ? [flat(text)] : [];
  if (at.length <= words) return [flat(text)];
  const out: string[] = [];
  let end = 0;
  for (let i = 0; i < at.length - Math.floor(words / 2); i += stride) {
    end = Math.min(i + words, at.length);
    out.push(flat(text.slice(at[i]![0], at[end - 1]![1])));
  }
  // The last words, when the stride stopped short of them: a window ending at the text's end.
  if (end < at.length) out.push(flat(text.slice(at[Math.max(0, at.length - words)]![0], at[at.length - 1]![1])));
  return out;
}

/** The first `n` UTF-16 units and an ellipsis — never ending on half a surrogate pair (an emoji cut in two is not valid
 *  UTF-8, and Qdrant refuses the whole write over it). */
export function cut(s: string, n: number): string {
  if (s.length <= n) return s;
  const high = n > 0 && s.charCodeAt(n - 1) >= 0xd800 && s.charCodeAt(n - 1) <= 0xdbff;
  const end = high ? n - 1 : n;
  return `${s.slice(0, end)}…`;
}

/** One spelling per concept, the graph's key: folded like a word (case, accents, compatibility forms), inner spaces
 *  collapsed, stray punctuation off the ends — "Proxy.conf " and "proxy.conf" are one node. */
export function conceptKey(name: string): string {
  return fold(name).replace(/\s+/g, ' ').replace(/^[\s.,:;"'`()[\]]+|[\s.,:;"'`()[\]]+$/g, '');
}
