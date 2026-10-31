/**
 * Word by word: memory_search's order with no server. A memory's one vector averages it, so one exact detail in a long
 * memory is drowned out. The embedder has a vector per word before it averages them; a memory scores the sum, over the
 * question's words, of its closest word (MaxSim, as ColBERT does). Over the search's best 30, fused by rank as the
 * reranker is (reranker.ts `rerankedOrder`). No training, no word list, any language.
 *
 * Measured (docs/BENCHMARKS.md): all 2,011 questions, answer shown at 5 50.3 → 53.1 (+80/−24), at 10
 * 57.1 → 58.4 (+67/−40); no refinement of the score beat this plain sum; other languages against English memories gain
 * more (Romanian at 5 43.0 → 48.2). One bit a number keeps nearly all of it, and 128 words are enough where 64 are not.
 *
 * A memory's words are kept small, the question's whole: its cosine with a word stored as signs is the sum of its
 * numbers with those signs, over √dims. Kept by the text's hash, so an unchanged text is never encoded twice.
 */
import { createHash } from 'node:crypto';
import { appendFile, mkdir, readFile } from 'node:fs/promises';
import { dirname } from 'node:path';

/** Words kept of a memory (64 lost most of the gain). */
export const WORD_LIMIT = 128;
const KEY_BYTES = 8;

/** Unit vectors, one a word, flat: `words × dims` numbers. */
export interface WordVectors { words: number; dims: number; data: Float32Array }
export type WordEncoder = (texts: string[], kind: 'query' | 'document') => Promise<WordVectors[]>;
/** A memory's words as signs, one bit a number, a word to a whole number of bytes. */
export interface PackedWords { words: number; dims: number; bits: Uint8Array }

export function packWords(v: WordVectors, limit = WORD_LIMIT): PackedWords {
  const words = Math.min(v.words, limit);
  const stride = Math.ceil(v.dims / 8);
  const bits = new Uint8Array(words * stride);
  for (let w = 0; w < words; w++) {
    for (let d = 0; d < v.dims; d++) if (v.data[w * v.dims + d]! >= 0) bits[w * stride + (d >> 3)]! |= 1 << (d & 7);
  }
  return { words, dims: v.dims, bits };
}

/** The sum, over the question's words, of the closest word of the memory. */
export function maxSim(query: WordVectors, memory: PackedWords): number {
  if (!query.words || !memory.words || query.dims !== memory.dims) return Number.NEGATIVE_INFINITY;
  const { dims } = query;
  const stride = Math.ceil(dims / 8);
  const norm = Math.sqrt(dims);
  let total = 0;
  for (let q = 0; q < query.words; q++) {
    const at = q * dims;
    let best = Number.NEGATIVE_INFINITY;
    for (let w = 0; w < memory.words; w++) {
      const row = w * stride;
      let sum = 0;
      for (let d = 0; d < dims; d++) {
        const x = query.data[at + d]!;
        sum += memory.bits[row + (d >> 3)]! & (1 << (d & 7)) ? x : -x;
      }
      if (sum > best) best = sum;
    }
    total += best / norm;
  }
  return total;
}

const keyOf = (text: string): string => createHash('sha256').update(text).digest('hex').slice(0, KEY_BYTES * 2);

/** The packed words by text, in memory and appended to a file: key, word count, numbers a word, the bits. Derived, like
 *  the index: losing it only means encoding again. */
export class WordStore {
  private readonly byKey = new Map<string, PackedWords>();

  constructor(private readonly file?: string) {}

  get size(): number { return this.byKey.size; }
  get(text: string): PackedWords | undefined { return this.byKey.get(keyOf(text)); }

  async load(): Promise<void> {
    if (!this.file) return;
    const b = await readFile(this.file).catch(() => null);
    if (!b) return;
    for (let at = 0; at + KEY_BYTES + 4 <= b.length;) {
      const key = b.subarray(at, at + KEY_BYTES).toString('hex');
      const words = b.readUInt16LE(at + KEY_BYTES);
      const dims = b.readUInt16LE(at + KEY_BYTES + 2);
      const size = words * Math.ceil(dims / 8);
      const start = at + KEY_BYTES + 4;
      if (start + size > b.length) break; // cut short by a crash: what came before is whole
      this.byKey.set(key, { words, dims, bits: new Uint8Array(b.subarray(start, start + size)) });
      at = start + size;
    }
  }

  async put(entries: Array<[string, PackedWords]>): Promise<void> {
    const parts: Buffer[] = [];
    for (const [text, p] of entries) {
      const key = keyOf(text);
      this.byKey.set(key, p);
      const head = Buffer.alloc(KEY_BYTES + 4);
      Buffer.from(key, 'hex').copy(head);
      head.writeUInt16LE(p.words, KEY_BYTES);
      head.writeUInt16LE(p.dims, KEY_BYTES + 2);
      parts.push(head, Buffer.from(p.bits));
    }
    if (!this.file || !parts.length) return;
    await mkdir(dirname(this.file), { recursive: true });
    await appendFile(this.file, Buffer.concat(parts));
  }
}

/** A reranker for memory_search (RecallDeps.rerank) that needs no server: what is not stored yet is encoded here, once.
 *  Null — the search keeps its own order — when the encoder fails or takes longer than it was given. */
export function wordReranker(encode: WordEncoder, store: WordStore): (query: string, documents: string[], timeoutMs: number) => Promise<number[] | null> {
  return async (query, documents, timeoutMs) => {
    let timer: NodeJS.Timeout | undefined;
    const late = new Promise<null>((resolve) => { timer = setTimeout(() => resolve(null), timeoutMs); });
    const work = (async (): Promise<number[]> => {
      const missing = [...new Set(documents.filter((t) => !store.get(t)))];
      if (missing.length) {
        const encoded = await encode(missing, 'document');
        await store.put(missing.map((t, i) => [t, packWords(encoded[i]!)]));
      }
      const [q] = await encode([query], 'query');
      return documents.map((t) => maxSim(q!, store.get(t)!));
    })();
    try {
      return await Promise.race([work, late]);
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  };
}
