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
 *
 * MANY USERS (a server holding a memory per user): one collection for all of them, its `user` field a
 * tenant index (`is_tenant`). A client made with `tenant` adds that user to every query, count, scroll, update and
 * delete, stamps it on every point it writes, and keys each point by user and item, so two users' identical items are
 * two points. The user comes from the caller's context (whose work this is), never from a query's words; a call made
 * with no user is refused before it reaches Qdrant.
 */
import { createHash } from 'node:crypto';
import { MEMORY_EMBEDDER } from './embedder.js';
import type { QdrantEndpoint } from './qdrant-process.js';

export const COLLECTION = 'memory';
/** The payload field a tenant client keys every point by, and the original point id it keeps beside the scoped one. */
export const TENANT_FIELD = 'user';
const TENANT_PID = 'pid';
export const DIMS = MEMORY_EMBEDDER.dims;
/** What the index is made of; a change empties it and it is written in again from the logs (rebuild.ts).
 *  2: binary quantisation with rescoring (½ the RAM of int8, recall 0.98–1.00 measured at 1–3M points), keyword terms
 *  folded and segmented for every language (text.ts); 3: on-disk binary copy, HNSW and payload indexes;
 *  4: remote EmbeddingGemma GPU vectors (768 dimensions, explicit runtime/pooling tag in embedder.ts); 5: nothing cut —
 *  every chunk of a body (no 12-chunk limit), short-term entries chunked too, cues numbered below the key, full payloads;
 *  6: every captured request indexed for good — however old, consolidated or not (the raw record is the truth). */
export const INDEX_LAYOUT = 6;
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
  /** Many users in one collection: the user every call is for — read on each call from the caller's context; null or ''
   *  refuses the call. */
  tenant?: () => string | null | undefined;
}

export class QdrantClient {
  private readonly collection: string;

  /** `endpoint` may start Qdrant (the service runs it only while memory has work): every call waits for it. */
  constructor(private readonly endpoint: () => QdrantEndpoint | null | Promise<QdrantEndpoint | null>, private readonly timeoutMs = 10_000,
    private readonly opts: QdrantClientOptions = {}) {
    this.collection = opts.collection ?? COLLECTION;
  }

  /** The user this call is for (a tenant client), or null (one person's own index). Throws when a tenant client has none. */
  private user(): string | null {
    if (!this.opts.tenant) return null;
    const u = this.opts.tenant();
    if (!u) throw new QdrantError('memory index: this call names no user, so it is refused', 403);
    return u;
  }

  /** A filter with the user added — unchanged for one person's own index. */
  private scoped(filter: Filter | undefined, user: string | null): Filter | undefined {
    if (user === null) return filter;
    return { ...filter, must: [{ key: TENANT_FIELD, match: { value: user } }, ...(filter?.must ?? [])] };
  }

  /** A point's id in a shared collection: the user's and the item's (pointId of both), so users never share a point. */
  private scopedId(id: string, user: string | null): string {
    return user === null ? id : pointId(`${user}\u0000${id}`, 0);
  }

  /** What a read hands back: the point's own id (as the writer named it) and only this user's points. */
  private own<P extends { id: string; payload: Record<string, unknown> }>(points: P[], user: string | null): P[] {
    if (user === null) return points;
    return points.filter((p) => p.payload[TENANT_FIELD] === user).map((p) => {
      const { [TENANT_FIELD]: _u, [TENANT_PID]: pid, ...payload } = p.payload;
      return { ...p, id: String(pid ?? p.id), payload };
    });
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
    await this.call('DELETE', `/collections/${this.collection}`).catch(() => undefined);
  }

  /** The collection and its payload indexes, made if missing — an index added since a collection was made is added to
   *  it in place. All of it on disk, memory-mapped — the vectors, their 1-bit copy, the graph, the payloads and their
   *  indexes: RAM holds what queries touch, and the system may take it back. Measured at 30k / 100k items (0.33M /
   *  1.1M points): 205 → 68 MB and 591 → 144 MB that cannot be given back,
   *  for 12–16 ms more of Qdrant's CPU a turn (26 → 38, 81 → 97). */
  async ensureCollection(): Promise<void> {
    let info: { payload_schema?: Record<string, unknown>; config?: { optimizer_config?: { default_segment_number?: number } } } | null = null;
    try {
      info = (await this.call<typeof info>('GET', `/collections/${this.collection}`)) ?? {};
    } catch (err) {
      if (!(err instanceof QdrantError) || err.status !== 404) throw err;
    }
    if (!info) {
      await this.call('PUT', `/collections/${this.collection}`, {
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
      await this.call('PATCH', `/collections/${this.collection}`, { optimizers_config: SMALL.optimizers_config });
    }
    const have = info.payload_schema ?? {};
    // Many users: the user field first, as Qdrant's tenant index (its points kept together on disk, searched per user).
    if (this.opts.tenant && !(TENANT_FIELD in have)) {
      await this.call('PUT', `/collections/${this.collection}/index?wait=true`, { field_name: TENANT_FIELD, field_schema: { type: 'keyword', is_tenant: true, on_disk: true } });
    }
    for (const [field_name, field_schema] of [['item', 'keyword'], ['part', 'keyword'], ['tier', 'keyword'], ['status', 'keyword'],
      ['kind', 'keyword'], ['scope', 'keyword'], ['project', 'keyword'], ['session', 'keyword'], ['ts', 'integer'],
      ['synthetic', 'bool'], ['concepts', 'keyword'], ['eventFrom', 'integer'], ['eventTo', 'integer']] as const) {
      const schema = field_schema === 'keyword' || field_schema === 'integer' ? { type: field_schema, on_disk: true } : field_schema;
      if (!(field_name in have)) await this.call('PUT', `/collections/${this.collection}/index?wait=true`, { field_name, field_schema: schema });
    }
  }

  async upsert(points: PointStruct[], wait = false): Promise<void> {
    if (!points.length) return;
    const user = this.user();
    const out = user === null ? points
      : points.map((p) => ({ ...p, id: this.scopedId(p.id, user), payload: { ...p.payload, [TENANT_FIELD]: user, [TENANT_PID]: p.id } }));
    await this.call('PUT', `/collections/${this.collection}/points?wait=${wait}`, { points: out }, 60_000);
  }

  /** Every point of these items — before an item is written again (its children may be fewer now). */
  async deleteItems(items: string[]): Promise<void> {
    if (!items.length) return;
    await this.call('POST', `/collections/${this.collection}/points/delete?wait=true`, { filter: this.scoped({ must: [{ key: 'item', match: { any: items } }] }, this.user()) }, 60_000);
  }

  /** One kind of point of these items (`cue`), before that part is written again. */
  async deleteParts(items: string[], part: string): Promise<void> {
    if (!items.length) return;
    await this.call('POST', `/collections/${this.collection}/points/delete?wait=true`, {
      filter: this.scoped({ must: [{ key: 'item', match: { any: items } }, { key: 'part', match: { value: part } }] }, this.user()),
    }, 60_000);
  }

  /** Change some payload fields of these items' points (status, a merged key's display fields). */
  async setPayload(items: string[], payload: Record<string, unknown>, keyOnly = false): Promise<void> {
    if (!items.length) return;
    const must: unknown[] = [{ key: 'item', match: { any: items } }];
    if (keyOnly) must.push({ key: 'part', match: { value: 'key' } });
    await this.call('POST', `/collections/${this.collection}/points/payload?wait=true`, { payload, filter: this.scoped({ must }, this.user()) }, 60_000);
  }

  /** The nearest key points to one vector (consolidation's "do I know this already?"), with their cosine. */
  async neighbors(vector: number[], filter: Filter, limit = 5): Promise<Array<{ item: string; score: number; payload: Record<string, unknown> }>> {
    const must = [...(filter.must ?? []), { key: 'part', match: { value: 'key' } }];
    const r = await this.call<{ points: Array<{ score: number; payload?: Record<string, unknown> }> }>('POST', `/collections/${this.collection}/points/query`, {
      query: vector, using: 'dense', limit, params: RESCORE, filter: this.scoped({ ...filter, must }, this.user()), with_payload: ['item', 'title', 'summary', 'kind', 'scope', 'project', 'status'],
    });
    return (r?.points ?? []).map((p) => ({ item: String(p.payload?.item ?? ''), score: p.score, payload: p.payload ?? {} }));
  }

  /** Each of these items' best cosine to one vector among its points of one part (key, child, cue) — the ranker's
   *  per-family similarities. An item with no such point is not in the map. */
  async bestPerItem(vector: number[], items: string[], part: string, timeoutMs = 2000): Promise<Map<string, number>> {
    if (!items.length) return new Map();
    const r = await this.call<{ groups: Array<{ id: string; hits: Array<{ id?: string; score: number }> }> }>('POST', `/collections/${this.collection}/points/query/groups`, {
      query: vector, using: 'dense', params: RESCORE,
      filter: this.scoped({ must: [{ key: 'item', match: { any: items } }, { key: 'part', match: { value: part } }] }, this.user()),
      group_by: 'item', group_size: 1, limit: items.length, with_payload: false,
    }, timeoutMs);
    return new Map((r?.groups ?? []).map((g) => [String(g.id), g.hits[0]?.score ?? 0]));
  }

  /** `nearest` for several vectors in one call. */
  /** `fields`: only these payload fields (and the item and its user), for a caller that needs no more — a walk over
   *  many neighbours reads ids, not the bodies of long requests. */
  async nearestMany(vectors: number[][], filter: Filter, limit = 10, timeoutMs = 5000, fields?: string[]): Promise<Array<Array<{ item: string; score: number; payload: Record<string, unknown> }>>> {
    if (!vectors.length) return [];
    const user = this.user();
    const withPayload = fields ? [...new Set([...fields, 'item', TENANT_FIELD])] : true;
    const r = await this.call<Array<{ points: Array<{ score: number; payload?: Record<string, unknown> }> }>>('POST', `/collections/${this.collection}/points/query/batch`, {
      searches: vectors.map((v) => ({ query: v, using: 'dense', limit, params: RESCORE, filter: this.scoped(filter, user), with_payload: withPayload })),
    }, timeoutMs);
    return (r ?? []).map((x) => this.own((x?.points ?? []).map((p) => ({ id: '', payload: p.payload ?? {}, score: p.score })), user)
      .map((p) => ({ item: String(p.payload.item ?? ''), score: p.score, payload: p.payload })));
  }

  /** Points within a filter, with their payload (and dense vector when asked) — at most `limit`. */
  async scroll(filter: Filter, limit = 256, withVector = false, timeoutMs = 5000): Promise<Array<{ id: string; payload: Record<string, unknown>; vector?: number[] }>> {
    const user = this.user();
    const r = await this.call<{ points: Array<{ id: string; payload?: Record<string, unknown>; vector?: { dense?: number[] } }> }>('POST', `/collections/${this.collection}/points/scroll`, {
      filter: this.scoped(filter, user), limit, with_payload: true, with_vector: withVector ? ['dense'] : false,
    }, timeoutMs);
    return this.own((r?.points ?? []).map((p) => ({ id: String(p.id), payload: p.payload ?? {}, ...(p.vector?.dense ? { vector: p.vector.dense } : {}) })), user);
  }

  async count(filter?: Filter): Promise<number> {
    const scoped = this.scoped(filter, this.user());
    const r = await this.call<{ count: number }>('POST', `/collections/${this.collection}/points/count`, { exact: false, ...(scoped ? { filter: scoped } : {}) });
    return r?.count ?? 0;
  }

  /** The items behind `ids` (key points), with their payload — and their dense vector when asked. */
  async retrieve(ids: string[], withVector = false): Promise<Array<{ id: string; payload: Record<string, unknown>; vector?: number[] }>> {
    if (!ids.length) return [];
    const user = this.user();
    const r = await this.call<Array<{ id: string; payload: Record<string, unknown>; vector?: { dense?: number[] } }>>('POST', `/collections/${this.collection}/points`, {
      ids: ids.map((id) => this.scopedId(id, user)), with_payload: true, with_vector: withVector ? ['dense'] : false,
    });
    // A point is fetched by its id alone; another user's (a hash of their id and the item) is dropped all the same.
    return this.own((r ?? []).map((p) => ({ id: String(p.id), payload: p.payload ?? {}, ...(p.vector?.dense ? { vector: p.vector.dense } : {}) })), user);
  }

  /** The best items for the query, fused by rank — each with the point that matched best (`point`: its key, a chunk of
   *  its text, or a cue), so a caller can show the passage that matched. Each channel (a query vector, the keywords)
   *  ranks up to `per` items (default: `limit`, at least 50); their scores are then fused (CombSUM, above). */
  async fusedItems(q: { dense: number[][]; sparse?: SparseVector; filter?: Filter; per?: number; limit?: number }, timeoutMs = 3000): Promise<Array<{ item: string; score: number; point?: string }>> {
    const limit = q.limit ?? 40;
    const per = q.per ?? Math.max(limit, 50);
    const user = this.user();
    const filter = this.scoped(q.filter, user);
    const channels: Array<{ query: unknown; using: string; params?: unknown; weight: number }> = [];
    const sparse = q.sparse?.indices.length ? q.sparse : null;
    const denseWeight = (sparse ? DENSE_SHARE : 1) / Math.max(1, q.dense.length);
    for (const v of q.dense) channels.push({ query: v, using: 'dense', params: RESCORE, weight: denseWeight });
    if (sparse) channels.push({ query: sparse, using: 'bm25', weight: 1 - DENSE_SHARE });
    if (!channels.length) return [];
    const lists = await Promise.all(channels.map((c) => this.call<{ groups: Array<{ id: string; hits: Array<{ id?: string; score: number; payload?: Record<string, unknown> }> }> }>('POST', `/collections/${this.collection}/points/query/groups`, {
      query: c.query, using: c.using, ...(c.params ? { params: c.params } : {}),
      ...(filter ? { filter } : {}),
      group_by: 'item',
      group_size: 1,
      limit: per,
      // A shared collection's point ids are scoped: the writer's own id is in the payload.
      with_payload: user === null ? false : [TENANT_FIELD, TENANT_PID],
    }, timeoutMs)));
    const fused = new Map<string, { score: number; rank: number; point?: string }>();
    lists.forEach((r, c) => {
      const groups = (r?.groups ?? []).filter((g) => user === null || g.hits[0]?.payload?.[TENANT_FIELD] === user);
      const scores = groups.map((g) => g.hits[0]?.score ?? 0);
      const lo = Math.min(...scores);
      const hi = Math.max(...scores);
      groups.forEach((g, rank) => {
        const item = String(g.id);
        const raw = user === null ? g.hits[0]?.id : g.hits[0]?.payload?.[TENANT_PID];
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
