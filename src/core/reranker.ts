/**
 * An optional cross-encoder reranker on a server, for search: any service that speaks the common rerank API
 * (`POST {url}/rerank`, `{ model, query, documents }` → `results[{ index, relevance_score }]`, as text-embeddings-
 * inference, Infinity, Jina and Cohere do). Measured with BAAI/bge-reranker-v2-m3 over the best 30 (docs/BENCHMARKS.md):
 * on little text it matched word by word (words.ts), which needs no server — so it is off unless you give it one.
 *
 * Optional by design: no route, a refusal, an error or a late answer — and the search keeps its own order.
 */
import { promisify } from 'node:util';
import { gzip } from 'node:zlib';

export const MEMORY_RERANKER = { model: 'bge-reranker-v2-m3' } as const;
/** How many of the best candidates it reads (20 lost 4.6 points of LoCoMo multi-hop against 30). */
export const RERANK_TOP = 30;
/** What it reads of each: the title and summary, cut here. */
export const RERANK_CHARS = 1500;

export interface RerankRoute {
  /** The API's base, e.g. `http://localhost:8081/v1` (`/rerank` is added). */
  url: string;
  apiKey?: string;
  /** Default `bge-reranker-v2-m3`. */
  model?: string;
  gzip?: boolean;
}

/** The route the environment names: MEMTRAILS_RERANK_URL, MEMTRAILS_RERANK_KEY, MEMTRAILS_RERANK_MODEL; null without one. */
export function resolveRerankRoute(env: NodeJS.ProcessEnv = process.env): RerankRoute | null {
  const url = env.MEMTRAILS_RERANK_URL?.trim();
  if (!url) return null;
  return { url, apiKey: env.MEMTRAILS_RERANK_KEY?.trim() || undefined, model: env.MEMTRAILS_RERANK_MODEL?.trim() || undefined };
}

const compress = promisify(gzip);

export interface RerankerDeps {
  route?: () => RerankRoute | null;
  fetchImpl?: typeof fetch;
}

/** Each document's relevance to the query, in the documents' order — or null (no route, refused, failed, late). */
export function remoteReranker(d: RerankerDeps = {}): (query: string, documents: string[], timeoutMs: number) => Promise<number[] | null> {
  const route = d.route ?? (() => resolveRerankRoute());
  return async (query, documents, timeoutMs) => {
    const r = route();
    if (!r || !documents.length || timeoutMs < 100) return null;
    try {
      const json = Buffer.from(JSON.stringify({ model: r.model ?? MEMORY_RERANKER.model, query, documents: documents.map((t) => t.slice(0, RERANK_CHARS) || ' ') }));
      const body = r.gzip ? await compress(json) : json;
      const res = await (d.fetchImpl ?? fetch)(`${r.url.replace(/\/+$/, '').replace(/\/rerank$/, '')}/rerank`, {
        method: 'POST',
        signal: AbortSignal.timeout(timeoutMs),
        headers: { 'content-type': 'application/json', ...(r.gzip ? { 'content-encoding': 'gzip' } : {}), ...(r.apiKey ? { authorization: `Bearer ${r.apiKey}` } : {}) },
        body: new Uint8Array(body),
      });
      if (!res.ok) { await res.text().catch(() => ''); return null; }
      const reply = (await res.json()) as { model?: string; results?: Array<{ index?: number; relevance_score?: number }> };
      const scores = new Array<number>(documents.length).fill(Number.NaN);
      for (const x of reply.results ?? []) {
        if (typeof x.index === 'number' && x.index >= 0 && x.index < documents.length && typeof x.relevance_score === 'number') scores[x.index] = x.relevance_score;
      }
      return scores.some(Number.isNaN) ? null : scores;
    } catch {
      return null;
    }
  };
}

/** The candidates in the reranker's order, fused with their own (by rank, k = 10: 0.7 to the reranker, 0.3 to the
 *  search's own order — the measured mix); null scores leave the order as it was. */
export function rerankedOrder<T>(items: T[], scores: number[] | null): T[] {
  if (!scores || scores.length !== items.length) return items;
  const byScore = items.map((_, i) => i).sort((a, b) => scores[b]! - scores[a]! || a - b);
  const fused = new Map<number, number>();
  byScore.forEach((i, r) => fused.set(i, (fused.get(i) ?? 0) + 0.7 / (10 + r + 1)));
  items.forEach((_, i) => fused.set(i, (fused.get(i) ?? 0) + 0.3 / (10 + i + 1)));
  return items.map((_, i) => i).sort((a, b) => fused.get(b)! - fused.get(a)! || a - b).map((i) => items[i]!);
}
