/**
 * Recall: what a search returns, an item opened whole, and exploring around a topic.
 *
 * `memorySearch` embeds the query and searches dense vectors (each item's key, its chunks and its cues) and a keyword
 * vector at once, fused by rank and grouped by item; with dates, what happened within them first. The best 30 can be
 * reordered (`rerank`: word by word on this machine, words.ts, or any cross-encoder) and fused with the search's own order.
 * `memoryOpen` reads one item whole. `memoryExplore` walks the concept graph (graph.ts) from a topic or from items to
 * what is related but not found directly.
 */
import { pointId, type Filter, type QdrantClient } from './qdrant-client.js';
import { queryText } from './embedder.js';
import { explore } from './graph.js';
import type { RecallEvent } from './recall-log.js';
import { RERANK_TOP, rerankedOrder } from './reranker.js';
import { children, cut, querySparse } from './text.js';
import { dateSpan, LONG_SUMMARY_CHUNKS } from './writer.js';
import { collectionKey } from '../records.js';

export interface RecallDeps {
  client: QdrantClient;
  embed: (texts: string[], timeoutMs?: number) => Promise<Float32Array[]>;
  /** What was shown / opened: the sleep pass's "use" (store.ts UsageBook). */
  used?: (items: string[], what: 'shown' | 'opened') => void;
  /** What each search showed, for what, and what was opened (recall-log.ts). */
  record?: (e: RecallEvent) => void;
  now?: () => number;
  /** Each document's relevance to the query, or null — the search then keeps its own order. None: no reordering. */
  rerank?: (query: string, documents: string[], timeoutMs: number) => Promise<number[] | null>;
}

/** Where a found item came from, when it is a record you added. */
export interface RecordRef { id: string; source?: string; author?: string; url?: string; metadata?: Record<string, string | number | boolean | null> }

export interface RecalledItem {
  id: string;
  title: string;
  summary: string;
  where: string;
  project: string;
  engine: string;
  ts: number;
  /** For a record you added: its kind (`email`, `message`, …, `record` when none was given) and `record` says which.
   *  For what consolidation learned: decision, procedure, pitfall, fact, preference, …, observation (one dated thing a
   *  thread told), profile (what is known about one subject), episode (a kept summary of earlier work). */
  kind: string;
  /** The record it is, for a record you added. */
  record?: RecordRef;
  /** For an observation: when it happened (YYYY-MM-DD, YYYY-MM or YYYY) — `ts` is when it was said. */
  eventAt?: string;
  /** For an observation of something that lasted: when it began and ended. */
  startAt?: string;
  endAt?: string;
  /** The passage of its text that matched the search, when that was not its key (memory_search): a fact said deep in
   *  a long request is shown, not only the request's start. */
  match?: string;
  /** The search's fused rank score. */
  score: number;
}

/** What search may show: active items, never a throwaway run, never superseded or archived knowledge. */
function baseFilter(kinds?: string[]): Filter {
  return {
    must: [{ key: 'synthetic', match: { value: false } }, { key: 'status', match: { value: 'active' } },
      ...(kinds?.length ? [{ key: 'engine', match: { any: kinds } }] : [])],
  };
}

async function items(d: RecallDeps, uids: string[]): Promise<RecalledItem[]> {
  const got = await d.client.retrieve(uids.map((u) => pointId(u, 0)));
  const byItem = new Map(got.map((p) => [String(p.payload.item), p.payload]));
  return uids.flatMap((uid) => {
    const p = byItem.get(uid);
    if (!p) return [];
    return [{
      id: uid, title: String(p.title ?? ''), summary: String(p.summary ?? ''), where: String(p.where ?? ''),
      project: String(p.project ?? ''), engine: String(p.engine ?? ''), ts: Number(p.ts ?? 0), kind: String(p.kind ?? ''), score: 0,
      ...(p.eventAt ? { eventAt: String(p.eventAt) } : {}), ...(p.startAt ? { startAt: String(p.startAt) } : {}),
      ...(p.endAt ? { endAt: String(p.endAt) } : {}), ...(p.record ? { record: p.record as RecordRef } : {}),
      ...(p.tier === 'stm' ? { kind: String(p.engine ?? 'record') } : {}),
    }];
  });
}

export const MAX_SEARCH = 200;
/** A captured request's title shorter than this says nothing on its own. */
const SHORT_TITLE = 24;

/** What happened within the dates (YYYY-MM-DD, YYYY-MM or YYYY; either end may be open): an observation by when it
 *  happened — its time overlapping the range — anything else by when it was written. The caller reads the dates out of
 *  a question; nothing here parses words. */
function withinDates(f: Filter, from: string | undefined, to: string | undefined): Filter {
  const lo = dateSpan(from)?.from ?? 0;
  const hi = dateSpan(to)?.to ?? Number.MAX_SAFE_INTEGER;
  return {
    ...f,
    must: [...(f.must ?? []), {
      should: [
        { must: [{ key: 'eventFrom', range: { lte: hi } }, { key: 'eventTo', range: { gte: lo } }] },
        { must: [{ is_empty: { key: 'eventFrom' } }, { key: 'ts', range: { gte: lo, lte: hi } }] },
      ],
    }],
  };
}

/** How long a search waits for its reranker before keeping its own order. */
const RERANK_WAIT_MS = 3000;

/** The best RERANK_TOP in the reranker's order fused with their own; with dates, the hits within them stay first — each
 *  block is reordered on its own. */
async function reranked(d: RecallDeps, query: string, list: RecalledItem[], dated: Set<string>): Promise<RecalledItem[]> {
  if (!d.rerank || list.length < 2) return list;
  const top = list.slice(0, RERANK_TOP);
  const scores = await d.rerank(query, top.map((it) => `${it.title}\n${it.summary}`), RERANK_WAIT_MS).catch(() => null);
  if (!scores) return list;
  const score = new Map(top.map((it, i) => [it.id, scores[i]!]));
  const order = (xs: RecalledItem[]): RecalledItem[] => rerankedOrder(xs, xs.map((it) => score.get(it.id)!));
  return [...order(top.filter((it) => dated.has(it.id))), ...order(top.filter((it) => !dated.has(it.id))), ...list.slice(RERANK_TOP)];
}

/** Up to this many asked for, memory_search ranks one fused list of twice as many (the room the two kinds had); past
 *  it — a question that gathers (memoryGather asks for MAX_SEARCH, then 30 a follow-up) — the dated observations and
 *  everything else are two searches of that many each, interleaved, so neither crowds the other out of the room. */
const SINGLE_LIST_UP_TO = 20;

/** A search: best first, in every collection or the ones named — and with dates, what happened within them first, then
 *  the rest. Never only what lies within them: on 64 held-out questions that name a time, the answer reached the reader
 *  whole for 54 searched without dates, 53 with what lies within them first, 44 with only that. */
export interface SearchQuery {
  query: string;
  /** Only these collections (records' `collection`). Default: all. */
  collections?: string[];
  /** Only these kinds (records' `kind`, or what consolidation learned: `observation`, `fact`, …). */
  kinds?: string[];
  limit?: number;
  from?: string;
  to?: string;
}

export async function memorySearch(d: RecallDeps, q: SearchQuery): Promise<RecalledItem[]> {
  const projects = q.collections?.length ? q.collections.map(collectionKey) : undefined;
  // Up to 200: a question that gathers ("everything she did with her family") needs the facts of many threads at once,
  // a dated observation a line.
  const limit = Math.max(1, Math.min(MAX_SEARCH, q.limit ?? 8));
  const vecs = await d.embed([queryText(q.query)]);
  const base = baseFilter(q.kinds);
  const scoped = projects ? { ...base, must: [...(base.must ?? []), { key: 'project', match: { any: projects } }] } : base;
  // Two searches at once: the dated observations, and everything else — up to `limit` of each. In one search an
  // observation (one point) competed with every chunk of every long request for the same places: 78 of 200 hits on a
  // list question, while the store held the facts it needed (measured on LoCoMo: 22 observation lines in the room
  // against Mem0's 145).
  const dense = vecs.map((v) => Array.from(v));
  const sparse = querySparse([q.query]);
  const observations = { key: 'kind', match: { value: 'observation' } };
  const ranked = async (filter: Filter) => {
    // A search for a few: one fused list over every kind, twice as deep as asked. Interleaving the kinds by rank cost
    // the top of the list (all of LoCoMo, consolidated, gold evidence any@5 84.6 → 86.8 with one list); one list only as
    // deep as asked lost the bottom (multi-hop all@40 57.8 → 44.7; twice as deep: 59.6). The interleave is for gathering.
    if (limit <= SINGLE_LIST_UP_TO) return d.client.fusedItems({ dense, sparse, filter, limit: 2 * limit });
    const [obs, rest] = await Promise.all([
      d.client.fusedItems({ dense, sparse, filter: { ...filter, must: [...(filter.must ?? []), observations] }, limit }),
      d.client.fusedItems({ dense, sparse, filter: { ...filter, must_not: [...(filter.must_not ?? []), observations] }, limit }),
    ]);
    // Interleaved by rank, so each kind keeps its order and neither crowds out the other.
    return Array.from({ length: Math.max(obs.length, rest.length) }, (_, i) => [rest[i], obs[i]]).flat().filter((h): h is NonNullable<typeof h> => !!h);
  };
  const [inDates, open] = await Promise.all([q.from || q.to ? ranked(withinDates(scoped, q.from, q.to)) : [], ranked(scoped)]);
  const dated = new Set(inDates.map((h) => h.item));
  const merged = [...inDates, ...open.filter((h) => !dated.has(h.item))].slice(0, Math.max(open.length, inDates.length));
  // A kept episode and its captured request hold the same text: the one found first stands for both.
  const seen = new Set<string>();
  const hits = merged.filter((h) => {
    const request = h.item.startsWith('ep:') ? h.item.slice(3) : h.item.startsWith('stm:') ? h.item.slice(4) : null;
    if (request === null) return true;
    if (seen.has(request)) return false;
    seen.add(request);
    return true;
  });
  const list = await reranked(d, q.query, await items(d, hits.map((h) => h.item)), dated);
  const score = new Map(hits.map((h) => [h.item, h.score]));
  const matched = await passages(d, hits);
  // A record titled by its first line, when that line is as short as "continue" or "ok", says nothing of what was found:
  // it shows the start of its key instead, which is what matched.
  const titled = (it: RecalledItem): RecalledItem => (it.id.startsWith('stm:') && it.title.length < SHORT_TITLE ? { ...it, title: cut(it.summary.replace(/\s+/g, ' '), 100) } : it);
  const out = list.map((it) => ({ ...titled(it), score: score.get(it.id) ?? 0, ...(matched.has(it.id) ? { match: matched.get(it.id)! } : {}) }));
  d.record?.({ ev: 'search', project: projects?.join(','), query: q.query, shown: out.map((it) => it.id) });
  return out;
}

/** For each hit whose best point was a chunk of its text, that chunk — found again from the text (writer.ts chunks a
 *  request's whole body; an item's body, and a summary too long for one key vector). */
async function passages(d: RecallDeps, hits: Array<{ item: string; point?: string }>): Promise<Map<string, string>> {
  const chunked = hits.filter((h) => h.point && h.point !== pointId(h.item, 0));
  if (!chunked.length) return new Map();
  const got = await d.client.retrieve(chunked.map((h) => pointId(h.item, 0)));
  const byItem = new Map(got.map((p) => [String(p.payload.item), p.payload]));
  const out = new Map<string, string>();
  for (const h of chunked) {
    const p = byItem.get(h.item);
    if (!p) continue;
    const body = String(p.body ?? '');
    const summary = String(p.summary ?? '');
    const kids = p.tier === 'stm' ? children(body) : [...children(body), ...(children(summary).length >= LONG_SUMMARY_CHUNKS ? children(summary) : [])];
    const k = kids.findIndex((_, i) => pointId(h.item, i + 1) === h.point);
    if (k >= 0) out.set(h.item, kids[k]!);
  }
  return out;
}

export interface ExploreQuery { topic?: string; from?: string[]; collections?: string[]; limit?: number }
export interface Explored {
  /** What search found for the topic, best first. */
  found: RecalledItem[];
  /** Reached through the graph and not found by search, best first, with the way it was reached. */
  related: Array<RecalledItem & { via: string }>;
  /** The concepts around it: where to look next. */
  concepts: string[];
}

/** An agent exploring: from a topic (search finds the first items; the topic's concepts start the walk too) or from
 *  items, what the concept graph reaches that search did not, and the concepts around it. */
export async function memoryExplore(d: RecallDeps, q: ExploreQuery): Promise<Explored> {
  const projects = q.collections?.length ? q.collections.map(collectionKey) : undefined;
  const base = baseFilter();
  const filter = projects ? { ...base, must: [...(base.must ?? []), { key: 'project', match: { any: projects } }] } : base;
  const r = await explore(d, { topic: q.topic, from: q.from, filter, projects, limit: Math.max(1, Math.min(20, q.limit ?? 10)) });
  const [found, related] = await Promise.all([items(d, r.found.slice(0, 8)), items(d, r.related.map((x) => x.id))]);
  const how = new Map(r.related.map((x) => [x.id, x]));
  return { found, related: related.map((it) => ({ ...it, score: how.get(it.id)!.score, via: how.get(it.id)!.via })), concepts: r.concepts.map((c) => c.name) };
}

/** One item whole, or null. */
export async function memoryOpen(d: RecallDeps, id: string): Promise<Record<string, unknown> | null> {
  const got = await d.client.retrieve([pointId(id, 0)]);
  const p = got[0]?.payload ?? null;
  if (p) {
    d.used?.([id], 'opened');
    d.record?.({ ev: 'open', item: id });
  }
  return p;
}
