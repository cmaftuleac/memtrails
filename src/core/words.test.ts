import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { rerankedOrder } from './reranker.js';
import { WORD_LIMIT, WordStore, maxSim, packWords, wordReranker, type WordEncoder, type WordVectors } from './words.js';

const DIMS = 16;

/** One orthogonal-ish unit vector per distinct word of a text, the same in any text. */
function wordVector(word: string): Float32Array {
  const v = new Float32Array(DIMS);
  let h = 2166136261;
  for (const c of word) h = Math.imul(h ^ c.codePointAt(0)!, 16777619) >>> 0;
  for (let d = 0; d < DIMS; d++) {
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    v[d] = h & 1 ? 1 : -1;
  }
  return v.map((x) => x / Math.sqrt(DIMS));
}

function vectors(text: string): WordVectors {
  const words = text.split(/\s+/).filter(Boolean);
  const data = new Float32Array(words.length * DIMS);
  words.forEach((w, i) => data.set(wordVector(w), i * DIMS));
  return { words: words.length, dims: DIMS, data };
}

describe('word by word', () => {
  let dir: string;
  beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'words-')); });
  afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

  it('scores a memory by its closest word to each word of the question, not by an average', () => {
    const long = packWords(vectors('deploy failed because the ssl certificate on staging expired last week after renewal stopped'));
    const other = packWords(vectors('we talked about the weather and lunch and a long walk in the park on sunday'));
    expect(maxSim(vectors('certificate expired'), long)).toBeGreaterThan(maxSim(vectors('certificate expired'), other));
  });

  it('gives an exact word its full score through the stored signs', () => {
    const q = vectors('certificate');
    expect(maxSim(q, packWords(q))).toBeCloseTo(1, 5);
  });

  it('keeps at most the first words, a whole number of bytes each', () => {
    const v = vectors(Array.from({ length: WORD_LIMIT + 20 }, (_, i) => `w${i}`).join(' '));
    const p = packWords(v);
    expect(p.words).toBe(WORD_LIMIT);
    expect(p.bits.length).toBe(WORD_LIMIT * Math.ceil(DIMS / 8));
    expect(packWords(vectors('a b'), 1).words).toBe(1);
  });

  it('says a memory cannot be scored rather than comparing vectors of different sizes', () => {
    expect(maxSim(vectors('a'), { words: 1, dims: 8, bits: new Uint8Array(1) })).toBe(Number.NEGATIVE_INFINITY);
    expect(maxSim(vectors('a'), { words: 0, dims: DIMS, bits: new Uint8Array(0) })).toBe(Number.NEGATIVE_INFINITY);
  });

  it('keeps packed words across restarts and survives a file cut short', async () => {
    const file = join(dir, 'nested', 'words.bin');
    const first = new WordStore(file);
    await first.put([['alpha beta', packWords(vectors('alpha beta'))], ['gamma', packWords(vectors('gamma'))]]);
    const again = new WordStore(file);
    await again.load();
    expect(again.size).toBe(2);
    expect(again.get('gamma')).toEqual(first.get('gamma'));
    const { size } = await stat(file);
    const cut = join(dir, 'cut.bin');
    await writeFile(cut, (await readFile(file)).subarray(0, size - 1));
    const part = new WordStore(cut);
    await part.load();
    expect(part.size).toBe(1);
    expect(part.get('alpha beta')).toBeDefined();
  });

  it('orders a search with no server, encoding each text once', async () => {
    const asked: string[][] = [];
    const encode: WordEncoder = async (texts) => {
      asked.push(texts);
      return texts.map(vectors);
    };
    const rerank = wordReranker(encode, new WordStore(join(dir, 'w.bin')));
    const docs = ['lunch in the park', 'the ssl certificate on staging expired', 'lunch in the park'];
    const scores = await rerank('certificate expired', docs, 1000);
    expect(scores).toHaveLength(3);
    expect(rerankedOrder(docs, scores)[0]).toBe(docs[1]);
    expect(asked[0]).toEqual(['lunch in the park', 'the ssl certificate on staging expired']);
    asked.length = 0;
    await rerank('another question', docs, 1000);
    expect(asked).toEqual([['another question']]);
  });

  it('leaves the search its own order when the encoder fails or is late', async () => {
    const failing = wordReranker(async () => { throw new Error('no model'); }, new WordStore());
    expect(await failing('q', ['a'], 1000)).toBeNull();
    const slow = wordReranker(() => new Promise(() => {}), new WordStore());
    expect(await slow('q', ['a'], 10)).toBeNull();
  });
});
