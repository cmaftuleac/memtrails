import { describe, expect, it } from 'vitest';
import { remoteReranker, rerankedOrder, resolveRerankRoute, type RerankRoute } from './reranker.js';

const route = (over: Partial<RerankRoute> = {}) => (): RerankRoute => ({ url: 'http://rerank.test/v1', apiKey: 'k', ...over });
const answer = (status: number, body: unknown) => (async () => new Response(JSON.stringify(body), { status })) as unknown as typeof fetch;
const ok = { results: [{ index: 1, relevance_score: 0.9 }, { index: 0, relevance_score: -2 }] };

describe('a rerank server', () => {
  it('posts { model, query, documents } to {url}/rerank and returns each score in the documents\' order', async () => {
    const seen: Array<{ url: string; body: { model: string; query: string; documents: string[] }; headers: Record<string, string> }> = [];
    const fetchImpl = (async (url: string, init: { body: Uint8Array; headers: Record<string, string> }) => {
      seen.push({ url, body: JSON.parse(Buffer.from(init.body).toString('utf8')), headers: init.headers });
      return new Response(JSON.stringify(ok), { status: 200 });
    }) as unknown as typeof fetch;
    expect(await remoteReranker({ route: route(), fetchImpl })('deploy the api', ['invoices', 'buildbot deploys it'], 1000)).toEqual([-2, 0.9]);
    expect(seen[0]).toMatchObject({ url: 'http://rerank.test/v1/rerank', body: { model: 'bge-reranker-v2-m3', query: 'deploy the api', documents: ['invoices', 'buildbot deploys it'] } });
    expect(seen[0]!.headers.authorization).toBe('Bearer k');
  });

  it('is nothing, so the search keeps its order, when refused, failing, incomplete, late or not configured', async () => {
    expect(await remoteReranker({ route: route(), fetchImpl: answer(503, {}) })('q', ['a', 'b'], 1000)).toBeNull();
    expect(await remoteReranker({ route: route(), fetchImpl: answer(200, { results: ok.results.slice(0, 1) }) })('q', ['a', 'b'], 1000)).toBeNull();
    expect(await remoteReranker({ route: route(), fetchImpl: (async () => { throw new Error('down'); }) as unknown as typeof fetch })('q', ['a', 'b'], 1000)).toBeNull();
    expect(await remoteReranker({ route: () => null, fetchImpl: answer(200, ok) })('q', ['a', 'b'], 1000)).toBeNull();
    expect(await remoteReranker({ route: route(), fetchImpl: answer(200, ok) })('q', ['a', 'b'], 50)).toBeNull();
  });

  it('fuses its order with the search\'s own, and leaves the order alone with no scores', () => {
    expect(rerankedOrder(['a', 'b', 'c'], [0.1, 0.2, 0.9])).toEqual(['c', 'b', 'a']);
    expect(rerankedOrder(['a', 'b', 'c'], null)).toEqual(['a', 'b', 'c']);
  });

  it('reads its server from the environment, or none', () => {
    expect(resolveRerankRoute({})).toBeNull();
    expect(resolveRerankRoute({ MEMTRAILS_RERANK_URL: 'http://r/v1' })).toEqual({ url: 'http://r/v1', apiKey: undefined, model: undefined });
  });
});
