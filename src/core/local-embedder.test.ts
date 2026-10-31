import { describe, expect, it } from 'vitest';
import { loadLocalEmbedder, type OrtModule, type Tokens } from './local-embedder.js';

const DIMS = 4;

/** Each token id becomes a known vector, so what comes out shows which tokens were kept. */
function fakeOrt(seen: Array<{ ids: bigint[]; dims: number[] }>): OrtModule {
  return {
    Tensor: class { constructor(public type: string, public data: BigInt64Array, public dims: number[]) {} } as OrtModule['Tensor'],
    InferenceSession: {
      create: async () => ({
        outputNames: ['last_hidden_state', 'sentence_embedding'],
        run: async (feeds: Record<string, unknown>) => {
          const { data, dims } = feeds['input_ids'] as { data: BigInt64Array; dims: number[] };
          seen.push({ ids: [...data], dims });
          const [n, width] = dims as [number, number];
          const h = new Float32Array(n * width * DIMS);
          for (let i = 0; i < data.length; i++) h[i * DIMS] = 3 * Number(data[i]) || 1;
          const p = new Float32Array(n * DIMS);
          for (let b = 0; b < n; b++) p[b * DIMS + 1] = 5;
          return { last_hidden_state: { dims: [n, width, DIMS], data: h }, sentence_embedding: { dims: [n, DIMS], data: p } };
        },
      }),
    },
  };
}

const tokenizer: Tokens = { encode: (t) => ({ ids: [1, ...t.split(' ').map((w) => w.length + 10), 2] }) };

describe('the local embedder', () => {
  it('gives a unit vector per text, in order, batching like lengths', async () => {
    const seen: Array<{ ids: bigint[]; dims: number[] }> = [];
    const e = await loadLocalEmbedder('/nowhere', { ort: fakeOrt(seen), tokenizer, batch: 2 });
    const out = await e.embed(['one two three', 'a', 'b c']);
    expect(out).toHaveLength(3);
    for (const v of out) expect(Math.hypot(...v)).toBeCloseTo(1);
    expect(seen.map((s) => s.dims)).toEqual([[2, 4], [1, 5]]);
  });

  it('keeps a vector per word without the start and end tokens, prompted as a query or a document', async () => {
    const seen: Array<{ ids: bigint[]; dims: number[] }> = [];
    const encoded: string[] = [];
    const e = await loadLocalEmbedder('/nowhere', { ort: fakeOrt(seen), tokenizer: { encode: (t) => { encoded.push(t); return tokenizer.encode(t); } } });
    const [q] = await e.words(['hello there'], 'query');
    expect(encoded[0]).toBe('task: search result | query: hello there');
    expect(q!.words).toBe(encoded[0]!.split(' ').length);
    expect(q!.dims).toBe(DIMS);
    expect(q!.data.length).toBe(q!.words * DIMS);
    expect(Math.hypot(...q!.data.subarray(0, DIMS))).toBeCloseTo(1);
    await e.words(['hello'], 'document');
    expect(encoded[1]).toBe('title: none | text: hello');
  });

  it('keeps the last word of a text cut short: it is a word, not the end token', async () => {
    const long = Array.from({ length: 200 }, () => 'word').join(' ');
    const e = await loadLocalEmbedder('/nowhere', { ort: fakeOrt([]), tokenizer });
    const [d] = await e.words([long], 'document');
    expect(d!.words).toBe(129);
  });

  it('says what is missing when an optional package is not installed', async () => {
    await expect(loadLocalEmbedder('/nowhere', { tokenizer })).rejects.toThrow();
  });
});
