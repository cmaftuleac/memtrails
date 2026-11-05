/**
 * The few Qdrant REST calls the memory index makes — plain fetch, no client library. One collection, `memory`, two
 * tiers (payload `tier`):
 *
 *   - `stm` — short-term memory: one point per captured request (its raw text as the key), there until consolidation;
 *   - `ltm` — long-term memory: per item a KEY point (`part: key` — its content title + ~150-word summary as the dense
 *     vector, a BM25 sparse vector over its title, summary, body and files, the item itself as payload) and CHILD
 *     points (`part: child`, ~96-word windows of its body, dense only) pointing at it.
 *
 * A query ranks items once per query vector and once by keywords (each hit grouped by item, so an item found through
 * several children counts once), then fuses those item rankings by rank (RRF). Only `active` items are ever returned
 * to recall; superseded and archived ones stay in the log, out of the index.
 */
import { createHash } from 'node:crypto';
import { MEMORY_EMBEDDER } from './embedder.js';
import type { QdrantEndpoint } from './qdrant-process.js';

export const COLLECTION = 'memory';
export const DIMS = MEMORY_EMBEDDER.dims;
/** What the index is made of; a change empties it and it is written in again from the logs (rebuild.ts).
 *  2: binary quantisation with rescoring (½ the RAM of int8, recall 0.98–1.00 measured at 1–3M points), keyword terms
 *  folded and segmented for every language (text.ts); 3: on-disk binary copy, HNSW and payload indexes;
 *  4: remote EmbeddingGemma GPU vectors (768 dimensions, explicit runtime/pooling tag in embedder.ts); 5: nothing cut —
 *  every chunk of a body (no 12-chunk limit), short-term entries chunked too, cues numbered below the key, full payloads;
 *  6: every captured request indexed for good — however old, consolidated or not (the raw record is the truth);
 *  7: a record's kind (`engine`) on every point of it and indexed, so a kinds filter finds a record by its chunks. */
export const INDEX_LAYOUT = 7;
/** The memory-mapped 1-bit copy finds candidates; full vectors on disk re-score 3× as many as asked. */
const RESCORE = { quantization: { rescore: true, oversampling: 3.0 } };
/** Two segments: Qdrant's default is one a core, which costs a laptop RAM and per-query work for nothing at memory's
 *  size. The write-ahead log keeps its 32 MB: a 4 MB one returned wrong
 *  results in qdrant.integration.test.ts on 1.20, for page cache the system can take back anyway. */
const SMALL = { optimizers_config: { default_segment_number: 2 } } as const;
/** Item-level fusion: each channel (a query vector, the keywords) is its own grouped query, its items' scores are
 *  min-max normalised over its own list, and an item's score is the weighted sum (CombSUM) — the dense channels share
 *  DENSE_SHARE of the weight, keywords the rest. Each channel is its own query, never a prefetch under a fusion: Qdrant
 *  1.20 orders a prefetch by the 1-bit scores whatever its `rescore` (42% of the exact top 10 on LoCoMo, any@1 halved).
 *  Fusing items rather than points keeps one item's many chunk hits from crowding the list; fusing scores rather than
 *  ranks keeps the gap of a hit that is much closer than the rest. Measured against gold evidence (docs/RESEARCH.md): dense 0.7
 *  was the best share on both benchmarks, and CombSUM beat rank fusion at the top of both — LoCoMo requests any@3
 *  71.6 → 74.5, nDCG@10 64.6 → 65.7; LongMemEval-S any@1 63.4 → 64.5, sessions any@1 92.6 → 93.2. */
const DENSE_SHARE = 0.7;
/** Cues — the questions an item answers, in the asker's words — search as their own channel at this share of the dense
 *  weight rather than among the item's points. Measured on LoCoMo's 1,307 held-out questions, together with cues on
 *  every observation and the seven-days-before calendar: multi-hop 50.6 → 55.2, overall 77.2 → 77.4 (docs/RESEARCH.md). */
const CUE_SHARE = 0.1;

export interface SparseVector { indices: number[]; values: number[] }

export interface Filter {
  must?: unknown[];
  must_not?: unknown[];
  should?: unknown[];
}

export interface PointStruct {
  id: string;
  vector: Record<string, number[] | SparseVector>;
  payload: Record<string, unknown>;
}

/** A point's id from its item and its place (0 = the key, n = the n-th child): the same item writes the same ids. */
export function pointId(item: string, n: number): string {
  const h = createHash('sha1').update(`${item}#${n}`).digest('hex');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-5${h.slice(13, 16)}-${((parseInt(h[16]!, 16) & 0x3) | 0x8).toString(16)}${h.slice(17, 20)}-${h.slice(20, 32)}`;
}

export class QdrantError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

export interface QdrantClientOptions {
  /** The collection (COLLECTION when not given). */
  collection?: string;
}

export class QdrantClient {
  /** `/collections/<name>`, the name encoded: a name you give is a path segment, never more of the path. */
  private readonly at: string;

  /** `endpoint` may start Qdrant (the service runs it only while memory has work): every call waits for it. */
  constructor(private readonly endpoint: () => QdrantEndpoint | null | Promise<QdrantEndpoint | null>, private readonly timeoutMs = 10_000,
    opts: QdrantClientOptions = {}) {
    this.at = `/collections/${encodeURIComponent(opts.collection ?? COLLECTION)}`;
  }

  private async call<T>(method: string, path: string, body?: unknown, timeoutMs = this.timeoutMs): Promise<T> {
    const e = await this.endpoint();
    if (!e) throw new QdrantError('memory index is not running', 503);
    let res: Response | null = null;
    const payload = body !== undefined ? JSON.stringify(body) : undefined;
    // Every call here can be sent again (upserts by id, deletes by filter, reads): a connection the server closed while
    // it sat idle (keep-alive) fails before the request reaches it, so it is tried again — twice at most.
    for (let attempt = 0; !res; attempt++) {
      try {
        res = await fetch(`${e.url}${path}`, {
          method,
          headers: { 'api-key': e.apiKey, ...(payload !== undefined ? { 'content-type': 'application/json' } : {}) },
          ...(payload !== undefined ? { body: payload } : {}),
          signal: AbortSignal.timeout(timeoutMs),
        });
      } catch (err) {
        // fetch says only "fetch failed": the reason is in its cause (refused, reset, closed, timed out).
        const cause = (err as { cause?: { code?: string; message?: string } }).cause;
        const timedOut = (err as Error).name === 'TimeoutError';
        if (attempt < 2 && !timedOut) { await new Promise((r) => setTimeout(r, 150 * (attempt + 1))); continue; }
        throw new QdrantError(`${method} ${path}: ${(err as Error).message}${cause ? ` (${cause.code ?? cause.message})` : ''}`, 0);
      }
    }
    const text = await res.text();
    if (!res.ok) throw new QdrantError(`${method} ${path}: ${res.status} ${text.slice(0, 300)}`, res.status);
    const json = text ? (JSON.parse(text) as { result?: T }) : {};
    return json.result as T;
  }

  /** Drop the whole collection: the index is derived, written again from the logs (rebuild.ts). */
  async dropCollection(): Promise<void> {
    await this.call('DELETE', this.at).catch(() => undefined);
  }

  /** The collection and its payload indexes, made if missing — an index added since a collection was made is added to
   *  it in place. All of it on disk, memory-mapped — the vectors, their 1-bit copy, the graph, the payloads and their
   *  indexes: RAM holds what queries touch, and the system may take it back. Measured at 30k / 100k items (0.33M /
   *  1.1M points): 205 → 68 MB and 591 → 144 MB that cannot be given back,
   *  for 12–16 ms more of Qdrant's CPU a turn (26 → 38, 81 → 97). */
  async ensureCollection(): Promise<void> {
    let info: { payload_schema?: Record<string, unknown>; config?: { optimizer_config?: { default_segment_number?: number } } } | null = null;
    try {
      info = (await this.call<typeof info>('GET', this.at)) ?? {};
    } catch (err) {
      if (!(err instanceof QdrantError) || err.status !== 404) throw err;
    }
    if (!info) {
      await this.call('PUT', this.at, {
        vectors: { dense: { size: DIMS, distance: 'Cosine', on_disk: true } },
        sparse_vectors: { bm25: { modifier: 'idf', index: { on_disk: true } } },
        quantization_config: { binary: { always_ram: false } },
        hnsw_config: { m: 16, ef_construct: 100, on_disk: true },
        on_disk_payload: true,
        ...SMALL,
      });
      info = { payload_schema: {}, config: { optimizer_config: { default_segment_number: SMALL.optimizers_config.default_segment_number } } };
    } else if (info.config?.optimizer_config?.default_segment_number !== SMALL.optimizers_config.default_segment_number) {
      // Made before: two segments from now on (the optimiser merges them in the background).
      await this.call('PATCH', this.at, { optimizers_config: SMALL.optimizers_config });
    }
    const have = info.payload_schema ?? {};
    for (const [field_name, field_schema] of [['item', 'keyword'], ['part', 'keyword'], ['tier', 'keyword'], ['status', 'keyword'],
      ['kind', 'keyword'], ['engine', 'keyword'], ['scope', 'keyword'], ['project', 'keyword'], ['session', 'keyword'], ['ts', 'integer'],
      ['synthetic', 'bool'], ['concepts', 'keyword'], ['eventFrom', 'integer'], ['eventTo', 'integer']] as const) {
      const schema = field_schema === 'keyword' || field_schema === 'integer' ? { type: field_schema, on_disk: true } : field_schema;
      if (!(field_name in have)) await this.call('PUT', `${this.at}/index?wait=true`, { field_name, field_schema: schema });
    }
  }

  async upsert(points: PointStruct[], wait = false): Promise<void> {
    if (!points.length) return;
    await this.call('PUT', `${this.at}/points?wait=${wait}`, { points }, 60_000);
  }

  /** Every point of these items — before an item is written again (its children may be fewer now). */
  async deleteItems(items: string[]): Promise<void> {
    if (!items.length) return;
    await this.call('POST', `${this.at}/points/delete?wait=true`, { filter: { must: [{ key: 'item', match: { any: items } }] } }, 60_000);
  }

  /** One kind of point of these items (`cue`), before that part is written again. */
  async deleteParts(items: string[], part: string): Promise<void> {
    if (!items.length) return;
    await this.call('POST', `${this.at}/points/delete?wait=true`, {
      filter: { must: [{ key: 'item', match: { any: items } }, { key: 'part', match: { value: part } }] },
    }, 60_000);
  }

  /** Change some payload fields of these items' points (status, a merged key's display fields). */
  async setPayload(items: string[], payload: Record<string, unknown>, keyOnly = false): Promise<void> {
    if (!items.length) return;
    const must: unknown[] = [{ key: 'item', match: { any: items } }];
    if (keyOnly) must.push({ key: 'part', match: { value: 'key' } });
    await this.call('POST', `${this.at}/points/payload?wait=true`, { payload, filter: { must } }, 60_000);
  }

  /** The nearest key points to one vector (consolidation's "do I know this already?"), with their cosine. */
  async neighbors(vector: number[], filter: Filter, limit = 5): Promise<Array<{ item: string; score: number; payload: Record<string, unknown> }>> {
    const must = [...(filter.must ?? []), { key: 'part', match: { value: 'key' } }];
    const r = await this.call<{ points: Array<{ score: number; payload?: Record<string, unknown> }> }>('POST', `${this.at}/points/query`, {
      query: vector, using: 'dense', limit, params: RESCORE, filter: { ...filter, must }, with_payload: ['item', 'title', 'summary', 'kind', 'scope', 'project', 'status'],
    });
    return (r?.points ?? []).map((p) => ({ item: String(p.payload?.item ?? ''), score: p.score, payload: p.payload ?? {} }));
  }

  /** `nearest` for several vectors in one call. */
  /** `fields`: only these payload fields (and the item), for a caller that needs no more — a walk over
   *  many neighbours reads ids, not the bodies of long requests. */
  async nearestMany(vectors: number[][], filter: Filter, limit = 10, timeoutMs = 5000, fields?: string[]): Promise<Array<Array<{ item: string; score: number; payload: Record<string, unknown> }>>> {
    if (!vectors.length) return [];
    const withPayload = fields ? [...new Set([...fields, 'item'])] : true;
    const r = await this.call<Array<{ points: Array<{ score: number; payload?: Record<string, unknown> }> }>>('POST', `${this.at}/points/query/batch`, {
      searches: vectors.map((v) => ({ query: v, using: 'dense', limit, params: RESCORE, filter, with_payload: withPayload })),
    }, timeoutMs);
    return (r ?? []).map((x) => (x?.points ?? []).map((p) => ({ item: String(p.payload?.item ?? ''), score: p.score, payload: p.payload ?? {} })));
  }

  /** Points within a filter, with their payload (and dense vector when asked) — at most `limit`. */
  async scroll(filter: Filter, limit = 256, withVector = false, timeoutMs = 5000): Promise<Array<{ id: string; payload: Record<string, unknown>; vector?: number[] }>> {
    const r = await this.call<{ points: Array<{ id: string; payload?: Record<string, unknown>; vector?: { dense?: number[] } }> }>('POST', `${this.at}/points/scroll`, {
      filter, limit, with_payload: true, with_vector: withVector ? ['dense'] : false,
    }, timeoutMs);
    return (r?.points ?? []).map((p) => ({ id: String(p.id), payload: p.payload ?? {}, ...(p.vector?.dense ? { vector: p.vector.dense } : {}) }));
  }

  async count(filter?: Filter): Promise<number> {
    const r = await this.call<{ count: number }>('POST', `${this.at}/points/count`, { exact: false, ...(filter ? { filter } : {}) });
    return r?.count ?? 0;
  }

  /** The items behind `ids` (key points), with their payload — and their dense vector when asked. */
  async retrieve(ids: string[], withVector = false): Promise<Array<{ id: string; payload: Record<string, unknown>; vector?: number[] }>> {
    if (!ids.length) return [];
    const r = await this.call<Array<{ id: string; payload: Record<string, unknown>; vector?: { dense?: number[] } }>>('POST', `${this.at}/points`, {
      ids, with_payload: true, with_vector: withVector ? ['dense'] : false,
    });
    return (r ?? []).map((p) => ({ id: String(p.id), payload: p.payload ?? {}, ...(p.vector?.dense ? { vector: p.vector.dense } : {}) }));
  }

  /** The best items for the query, fused by rank — each with the point that matched best (`point`: its key, a chunk of
   *  its text, or a cue), so a caller can show the passage that matched. Each channel (a query vector, the keywords)
   *  ranks up to `per` items (default: `limit`, at least 50); their scores are then fused (CombSUM, above). */
  async fusedItems(q: { dense: number[][]; sparse?: SparseVector; filter?: Filter; per?: number; limit?: number }, timeoutMs = 3000): Promise<Array<{ item: string; score: number; point?: string }>> {
    const limit = q.limit ?? 40;
    const per = q.per ?? Math.max(limit, 50);
    const filter = q.filter;
    const channels: Array<{ query: unknown; using: string; params?: unknown; weight: number; filter?: Filter }> = [];
    const sparse = q.sparse?.indices.length ? q.sparse : null;
    const denseWeight = (sparse ? DENSE_SHARE : 1) / Math.max(1, q.dense.length);
    const cueOff: Filter = { ...(filter ?? {}), must_not: [...((filter as { must_not?: unknown[] } | undefined)?.must_not ?? []), { key: 'part', match: { value: 'cue' } }] } as Filter;
    const cueOnly: Filter = { ...(filter ?? {}), must: [...((filter as { must?: unknown[] } | undefined)?.must ?? []), { key: 'part', match: { value: 'cue' } }] } as Filter;
    for (const v of q.dense) {
      channels.push({ query: v, using: 'dense', params: RESCORE, weight: denseWeight * (1 - CUE_SHARE), filter: cueOff });
      channels.push({ query: v, using: 'dense', params: RESCORE, weight: denseWeight * CUE_SHARE, filter: cueOnly });
    }
    if (sparse) channels.push({ query: sparse, using: 'bm25', weight: 1 - DENSE_SHARE });
    if (!channels.length) return [];
    const lists = await Promise.all(channels.map((c) => this.call<{ groups: Array<{ id: string; hits: Array<{ id?: string; score: number }> }> }>('POST', `${this.at}/points/query/groups`, {
      query: c.query, using: c.using, ...(c.params ? { params: c.params } : {}),
      ...(c.filter ?? filter ? { filter: c.filter ?? filter } : {}),
      group_by: 'item',
      group_size: 1,
      limit: per,
      with_payload: false,
    }, timeoutMs)));
    const fused = new Map<string, { score: number; rank: number; point?: string }>();
    lists.forEach((r, c) => {
      const groups = r?.groups ?? [];
      const scores = groups.map((g) => g.hits[0]?.score ?? 0);
      const lo = Math.min(...scores);
      const hi = Math.max(...scores);
      groups.forEach((g, rank) => {
        const item = String(g.id);
        const raw = g.hits[0]?.id;
        const at = fused.get(item) ?? { score: 0, rank: Infinity };
        // A channel whose items all score alike (or that found one) counts each of them fully.
        at.score += channels[c]!.weight * (hi > lo ? (scores[rank]! - lo) / (hi - lo) : 1);
        // The matched passage: the point from the channel that ranked the item highest.
        if (rank < at.rank && raw) { at.rank = rank; at.point = String(raw); }
        fused.set(item, at);
      });
    });
    return [...fused.entries()].sort((a, b) => b[1].score - a[1].score).slice(0, limit)
      .map(([item, f]) => ({ item, score: f.score, ...(f.point ? { point: f.point } : {}) }));
  }
}
