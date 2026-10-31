/**
 * The concept graph — memory's associative index (HippoRAG's hippocampal index, 2405.14831), for an agent EXPLORING a
 * topic or what it found, rather than for a search's top few (as a ranker feature it added ~1 point; for reaching
 * what search does not, it is where the gain is).
 *
 * Nodes are long-term items and the concepts they are about (consolidation writes 3–12 per item: files, services,
 * commands, named things), a concept being its spelling key (text.ts conceptKey). Edges:
 *   - item ↔ concept, weighted by how specific the concept is (1 / log(2 + the items that share it));
 *   - item ↔ item: `related` (the same subject, different facts — consolidation and sleep link them) and each seed's
 *     nearest items by key vector, two hops out.
 * Concepts are NOT embedded: matched and linked by embedding instead of spelling, the walk found the same needed items
 * (195 either way) and ~0.4 more items a topic, for a point and six embeddings an item more — not worth a laptop's RAM
 * and CPU (measured 2026-10-25). It lives in Qdrant — the concept keys on items' key points, indexed — so it grows with
 * memory without a copy in RAM: `explore` reads the neighbourhood of its seeds and walks it with Personalized PageRank
 * (restart 0.5, the eval's), within recall's walls (active, real, the scope asked).
 */
import { pointId, type Filter, type QdrantClient } from './qdrant-client.js';
import { queryText } from './embedder.js';
import { conceptKey, fold, querySparse } from './text.js';

/** A seed's nearest items by key vector (from cos 0.5: a small memory's "nearest" can be anything), the walk's search
 *  hits used as seeds, the items read through concepts. */
const KNN = 10;
const KNN_NEAR = 0.5;
const SEEDS = 5;
const FOUND = 20;
const POSTINGS = 400;
/** A `related` link's weight, beside item ↔ concept's 1/log(2 + df). */
const RELATED = 1.0;
const RESTART = 0.5;
const STEPS = 20;

export interface GraphDeps {
  client: QdrantClient;
  embed: (texts: string[]) => Promise<Float32Array[]>;
}

const strings = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []);

export interface ExploreQuery {
  /** What to explore, in plain words: search finds the first items, and the topic's concepts start the walk too. */
  topic?: string;
  /** Items to explore from (memory ids). */
  from?: string[];
  /** Recall's walls: what may be shown (active, real), and the collections when the search names some. */
  filter: Filter;
  projects?: string[];
  limit?: number;
}

export interface Explored {
  /** What search found for the topic, best first (the walk's other seeds). */
  found: string[];
  /** Reached through the graph and not among `found` or the seeds, best first, with the way it was reached. */
  related: Array<{ id: string; score: number; via: string }>;
  /** The concepts the walk spent most time on: where to look next. */
  concepts: Array<{ name: string; score: number }>;
}

/** A small weighted undirected graph, walked by Personalized PageRank. */
class Walk {
  private readonly ids = new Map<string, number>();
  private readonly adj: Array<Map<number, number>> = [];

  node(name: string): number {
    let i = this.ids.get(name);
    if (i === undefined) { i = this.adj.length; this.ids.set(name, i); this.adj.push(new Map()); }
    return i;
  }

  link(a: string, b: string, w: number): void {
    if (a === b || !(w > 0)) return;
    const i = this.node(a);
    const j = this.node(b);
    this.adj[i]!.set(j, Math.max(this.adj[i]!.get(j) ?? 0, w));
    this.adj[j]!.set(i, Math.max(this.adj[j]!.get(i) ?? 0, w));
  }

  /** Each node's share of a walk that starts again at `seeds` (by weight) half the time. */
  ppr(seeds: Map<string, number>): Map<string, number> {
    const n = this.adj.length;
    const e = new Float64Array(n);
    let total = 0;
    for (const [name, w] of seeds) { const i = this.ids.get(name); if (i !== undefined) { e[i] += w; total += w; } }
    if (!total) return new Map();
    for (let i = 0; i < n; i++) e[i]! /= total;
    const deg = this.adj.map((m) => [...m.values()].reduce((s, w) => s + w, 0));
    let x = Float64Array.from(e);
    for (let step = 0; step < STEPS; step++) {
      const next = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (!x[i] || !deg[i]) continue;
        for (const [j, w] of this.adj[i]!) next[j]! += (1 - RESTART) * x[i]! * (w / deg[i]!);
      }
      for (let i = 0; i < n; i++) next[i]! += RESTART * e[i]!;
      x = next;
    }
    return new Map([...this.ids].map(([name, i]) => [name, x[i]!]));
  }
}

/** Walk the graph from a topic and/or items: what is related but not found directly, and the concepts around it. */
export async function explore(d: GraphDeps, q: ExploreQuery): Promise<Explored> {
  const keyOnly = (extra: unknown[] = []): Filter => ({
    ...q.filter, must: [...(q.filter.must ?? []), { key: 'tier', match: { value: 'ltm' } }, { key: 'part', match: { value: 'key' } }, ...extra],
  });
  const shown = (p: Record<string, unknown>): boolean => p.tier === 'ltm' && p.status === 'active' && p.synthetic === false && (!q.projects?.length || q.projects.includes(String(p.project)));
  const seedItems = new Map<string, number>();
  const seedConcepts = new Map<string, number>();
  let found: string[] = [];
  if (q.topic?.trim()) {
    const [qv] = await d.embed([queryText(q.topic)]);
    found = (await d.client.fusedItems({ dense: [Array.from(qv!)], sparse: querySparse([q.topic]), filter: q.filter, limit: FOUND })).map((h) => h.item);
    for (const u of found.slice(0, SEEDS)) seedItems.set(u, 1);
    // The topic's own concepts: the topic named as one, and those of what search found that the topic names.
    const topic = conceptKey(q.topic);
    if (topic) seedConcepts.set(topic, 1);
    const said = fold(q.topic);
    for (const p of await d.client.retrieve(found.slice(0, SEEDS).map((u) => pointId(u, 0)))) {
      for (const k of strings(p.payload.concepts)) if (k.length > 2 && said.includes(k)) seedConcepts.set(k, 1);
    }
  }
  for (const u of q.from ?? []) seedItems.set(u, 1);
  if (!seedItems.size && !seedConcepts.size) return { found, related: [], concepts: [] };

  const g = new Walk();
  const items = new Map<string, Record<string, unknown>>();
  // Hop 1: the seed items, their concepts and links, and their nearest items by key vector.
  const seeds = (await d.client.retrieve([...seedItems.keys()].map((u) => pointId(u, 0)), true)).filter((p) => p.vector);
  for (const p of seeds) items.set(String(p.payload.item), p.payload);
  const near = await d.client.nearestMany(seeds.map((p) => p.vector!), keyOnly(), KNN + 1);
  const hop1 = new Map<string, number>();
  seeds.forEach((p, i) => {
    for (const n of near[i] ?? []) {
      if (n.item === p.payload.item || n.score < KNN_NEAR) continue;
      if (!items.has(n.item)) items.set(n.item, n.payload);
      if (!seedItems.has(n.item)) hop1.set(n.item, Math.max(hop1.get(n.item) ?? 0, n.score));
      g.link(`i:${String(p.payload.item)}`, `i:${n.item}`, n.score);
    }
  });
  // A second hop from the nearest of those (measured: the multi-hop item walk found more than one hop).
  const next = (await d.client.retrieve([...hop1].sort((a, b) => b[1] - a[1]).slice(0, KNN).map(([u]) => pointId(u, 0)), true)).filter((p) => p.vector);
  const near2 = await d.client.nearestMany(next.map((p) => p.vector!), keyOnly(), KNN + 1);
  next.forEach((p, i) => {
    for (const n of near2[i] ?? []) {
      if (n.item === p.payload.item || n.score < KNN_NEAR) continue;
      if (!items.has(n.item)) items.set(n.item, n.payload);
      g.link(`i:${String(p.payload.item)}`, `i:${n.item}`, n.score);
    }
  });
  // The concepts of what was reached and of the topic.
  const concepts = new Set<string>(seedConcepts.keys());
  for (const p of items.values()) for (const k of strings(p.concepts)) concepts.add(k);
  // Hop 2: the items holding those concepts — a concept many of them share links each one less — and the items
  // linked to anything reached.
  const asked = [...concepts].slice(0, 300);
  if (asked.length) {
    for (const p of await d.client.scroll(keyOnly([{ key: 'concepts', match: { any: asked } }]), POSTINGS)) {
      if (!items.has(String(p.payload.item))) items.set(String(p.payload.item), p.payload);
    }
  }
  const linkedOut = [...new Set([...items.values()].flatMap((p) => strings(p.related)))].filter((u) => !items.has(u));
  for (const p of await d.client.retrieve(linkedOut.map((u) => pointId(u, 0)))) items.set(String(p.payload.item), p.payload);
  const df = new Map<string, number>();
  for (const p of items.values()) for (const k of strings(p.concepts)) if (concepts.has(k)) df.set(k, (df.get(k) ?? 0) + 1);
  for (const [u, p] of items) {
    if (!shown(p)) continue;
    for (const k of strings(p.concepts)) if (concepts.has(k)) g.link(`i:${u}`, `c:${k}`, 1 / Math.log(2 + (df.get(k) ?? 1)));
    for (const r of strings(p.related)) if (items.has(r) && shown(items.get(r)!)) g.link(`i:${u}`, `i:${r}`, RELATED);
  }

  // The walk starts again half at the seed items and half at the topic's concepts (either alone when only one is).
  const start = new Map<string, number>();
  for (const u of seedItems.keys()) start.set(`i:${u}`, (seedConcepts.size ? 0.5 : 1) / seedItems.size);
  const cTotal = [...seedConcepts.values()].reduce((s, w) => s + w, 0);
  for (const [k, w] of seedConcepts) start.set(`c:${k}`, (seedItems.size ? 0.5 : 1) * (w / cTotal));
  const x = g.ppr(start);

  const skip = new Set([...found, ...seedItems.keys()]);
  const seedTitles = new Map([...seedItems.keys()].map((u) => [u, String(items.get(u)?.title ?? u)]));
  const seedKeys = new Set([...seedConcepts.keys(), ...[...seedItems.keys()].flatMap((u) => strings(items.get(u)?.concepts))]);
  const related = [...items].filter(([u, p]) => !skip.has(u) && shown(p) && (x.get(`i:${u}`) ?? 0) > 0)
    .map(([u, p]) => ({ id: u, score: x.get(`i:${u}`)!, via: via(p) }))
    .sort((a, b) => b.score - a.score).slice(0, q.limit ?? 10);

  /** How an item was reached: the concepts it shares with where the walk started, a link, or nearness to a seed. */
  function via(p: Record<string, unknown>): string {
    const shared = strings(p.concepts).filter((k) => seedKeys.has(k)).slice(0, 3).map((k) => `"${k}"`);
    if (shared.length) return `shares ${shared.join(', ')}`;
    const linked = strings(p.related).find((r) => seedTitles.has(r));
    if (linked) return `linked to "${seedTitles.get(linked)}"`;
    const through = strings(p.concepts).find((k) => (x.get(`c:${k}`) ?? 0) > 0);
    if (through) return `through "${through}"`;
    return 'near what was found';
  }

  const conceptScores = [...x].filter(([n, s]) => n.startsWith('c:') && s > 0).sort((a, b) => b[1] - a[1]).slice(0, 12)
    .map(([n, score]) => ({ name: n.slice(2), score }));
  return { found, related, concepts: conceptScores };
}
