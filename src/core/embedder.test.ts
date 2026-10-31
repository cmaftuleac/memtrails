import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { docText, EmbeddingUnavailable, MEMORY_EMBEDDER, queryText, remoteEmbedder, resolveEmbedRoute, type EmbedRoute } from './embedder.js';

const DIMS = MEMORY_EMBEDDER.dims;
const route = (over: Partial<EmbedRoute> = {}) => (): EmbedRoute => ({ url: 'http://embed.test/v1', apiKey: 'k', ...over });

function server(reply: (input: string[], call: number) => Response | Promise<Response>) {
  const calls: Array<{ url: string; headers: Record<string, string>; body: { model: string; input: string[] } }> = [];
  const fetchImpl = (async (url: string, init: { headers: Record<string, string>; body: Uint8Array }) => {
    const raw = Buffer.from(init.body);
    const body = JSON.parse((init.headers['content-encoding'] === 'gzip' ? gunzipSync(raw) : raw).toString('utf8'));
    calls.push({ url, headers: init.headers, body });
    return reply(body.input, calls.length);
  }) as unknown as typeof fetch;
  return { fetchImpl, calls };
}
const vec = (k: number): number[] => Array.from({ length: DIMS }, (_, i) => (i === k % DIMS ? 3 : 0));
const ok = (input: string[], encode?: 'base64') => new Response(JSON.stringify({
  data: input.map((_, i) => ({ index: i, embedding: encode ? Buffer.from(new Float32Array(vec(i)).buffer).toString('base64') : vec(i) })),
}), { status: 200 });

describe('the embedding server', () => {
  it("uses EmbeddingGemma's prompts", () => {
    expect(queryText('why')).toBe('task: search result | query: why');
    expect(docText('', 'text')).toBe('title: none | text: text');
    expect(docText('T', 'text')).toBe('title: T | text: text');
  });

  it('posts { model, input } to {url}/embeddings with the key, and returns unit vectors in order', async () => {
    const s = server((input) => ok(input));
    const out = await remoteEmbedder({ route: route(), fetchImpl: s.fetchImpl })(['a', 'b', 'c']);
    expect(s.calls[0]!.url).toBe('http://embed.test/v1/embeddings');
    expect(s.calls[0]!.headers.authorization).toBe('Bearer k');
    expect(s.calls[0]!.body).toEqual({ model: 'embeddinggemma-300m', input: ['a', 'b', 'c'] });
    expect(out.map((v) => v.indexOf(1))).toEqual([0, 1, 2]);
    for (const v of out) expect(Math.hypot(...v)).toBeCloseTo(1);
  });

  it('accepts base64 float32 vectors, a model name of its own, gzip when asked, and no key', async () => {
    const s = server((input) => ok(input, 'base64'));
    const [v] = await remoteEmbedder({ route: route({ model: 'embeddinggemma', gzip: true, apiKey: undefined, url: 'http://e.test/v1/embeddings' }), fetchImpl: s.fetchImpl })(['a']);
    expect(v![0]).toBe(1);
    expect(s.calls[0]!.url).toBe('http://e.test/v1/embeddings');
    expect(s.calls[0]!.headers['content-encoding']).toBe('gzip');
    expect(s.calls[0]!.headers.authorization).toBeUndefined();
    expect(s.calls[0]!.body.model).toBe('embeddinggemma');
  });

  it('splits a large batch and keeps the order', async () => {
    const s = server((input) => ok(input));
    const texts = Array.from({ length: 600 }, (_, i) => `text ${i}`);
    const out = await remoteEmbedder({ route: route(), fetchImpl: s.fetchImpl })(texts);
    expect(s.calls.length).toBeGreaterThan(1);
    expect(out).toHaveLength(600);
    expect(s.calls.flatMap((c) => c.body.input)).toEqual(texts);
  });

  it('retries a busy server, and refuses a wrong width, a zero vector or a missing one before anything is indexed', async () => {
    let n = 0;
    const busy = server((input) => (++n === 1 ? new Response('busy', { status: 503 }) : ok(input)));
    expect(await remoteEmbedder({ route: route(), fetchImpl: busy.fetchImpl })(['a'])).toHaveLength(1);
    const bad = (body: unknown) => server(() => new Response(JSON.stringify(body), { status: 200 })).fetchImpl;
    await expect(remoteEmbedder({ route: route(), fetchImpl: bad({ data: [{ index: 0, embedding: [1, 2] }] }) })(['a'])).rejects.toThrow('dimensions');
    await expect(remoteEmbedder({ route: route(), fetchImpl: bad({ data: [{ index: 0, embedding: vec(0).map(() => 0) }] }) })(['a'])).rejects.toThrow('zero');
    await expect(remoteEmbedder({ route: route(), fetchImpl: bad({ data: [] }) })(['a'])).rejects.toThrow('0 vectors for 1');
  });

  it('says what to configure when there is no server, and reads one from the environment', async () => {
    await expect(remoteEmbedder({ route: () => null })(['a'])).rejects.toBeInstanceOf(EmbeddingUnavailable);
    expect(await remoteEmbedder({ route: () => null })([])).toEqual([]);
    expect(resolveEmbedRoute({})).toBeNull();
    expect(resolveEmbedRoute({ MEMTRAILS_EMBED_URL: 'http://x/v1', MEMTRAILS_EMBED_KEY: 'k', MEMTRAILS_EMBED_MODEL: 'm' })).toEqual({ url: 'http://x/v1', apiKey: 'k', model: 'm' });
  });
});
