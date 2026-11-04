/**
 * Search by kind and by date: a record's kind is on every point of it (a match deep in its text counts), a learned
 * item's kind is its own; a date that is not one is refused rather than read as an open end.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { QdrantClient, type Filter, type PointStruct } from './qdrant-client.js';
import { memorySearch, type RecallDeps } from './recall.js';
import type { MemoryItem, StmEntry } from './store.js';
import { termId, tokens } from './text.js';
import { dateSpan, writeLtm, writeStm } from './writer.js';

const DIMS = 512;
async function embed(texts: string[]): Promise<Float32Array[]> {
  return texts.map((t) => {
    const v = new Float32Array(DIMS);
    for (const w of new Set(tokens(t.replace(/^title: | \| text: |^task: search result \| query: /g, ' ')))) v[termId(w) % DIMS] = 1;
    const n = Math.hypot(...v) || 1;
    return v.map((x) => x / n);
  });
}
const cos = (a: number[], b: number[]): number => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0);

type Cond = { key?: string; match?: { value?: unknown; any?: unknown[] }; range?: { lte?: number; gte?: number }; is_empty?: { key: string } } & Filter;
function matches(p: Record<string, unknown>, f?: Filter): boolean {
  const one = (c: Cond): boolean => {
    if (c.must || c.should || c.must_not) return matches(p, c);
    if (c.is_empty) return p[c.is_empty.key] === undefined || p[c.is_empty.key] === null;
    const v = p[c.key!];
    if (c.match && 'value' in c.match) return Array.isArray(v) ? v.includes(c.match.value) : v === c.match.value;
    if (c.match?.any) return Array.isArray(v) ? v.some((x) => c.match!.any!.includes(x)) : c.match.any.includes(v);
    if (c.range) return typeof v === 'number' && (c.range.lte === undefined || v <= c.range.lte) && (c.range.gte === undefined || v >= c.range.gte);
    return true;
  };
  return ((f?.must ?? []) as Cond[]).every(one) && !((f?.must_not ?? []) as Cond[]).some(one) && (!f?.should?.length || (f.should as Cond[]).some(one));
}

/** Qdrant as far as search uses it: each item ranked by its best point within the filter. */
function fakeQdrant() {
  const points = new Map<string, PointStruct>();
  const client = {
    async upsert(ps: PointStruct[]) { for (const p of ps) points.set(p.id, p); },
    async deleteItems(items: string[]) { for (const [id, p] of points) if (items.includes(String(p.payload.item))) points.delete(id); },
    async retrieve(ids: string[]) { return ids.flatMap((id) => (points.has(id) ? [{ id, payload: points.get(id)!.payload }] : [])); },
    async fusedItems(q: { dense: number[][]; filter?: Filter; limit?: number }) {
      const best = new Map<string, { s: number; point: string }>();
      for (const p of points.values()) {
        if (!matches(p.payload, q.filter)) continue;
        const s = cos(q.dense[0]!, p.vector.dense as number[]);
        const item = String(p.payload.item);
        if (s > (best.get(item)?.s ?? -1)) best.set(item, { s, point: p.id });
      }
      return [...best.entries()].sort((a, b) => b[1].s - a[1].s).slice(0, q.limit ?? 40).map(([item, b]) => ({ item, score: b.s, point: b.point }));
    },
  };
  return { client: client as unknown as QdrantClient, points };
}

const entry = (uid: string, engine: string, reply: string): StmEntry => ({
  uid, fingerprint: uid, engine, session: uid, turn: 0, cwd: null, project: '-p', title: 'Weekly notes', ts: Date.parse('2026-04-02T00:00:00Z'),
  synthetic: false, typed: 'human', request: 'Weekly notes', reply, said: reply, steps: [], files: [], done: false,
});
const fact = (uid: string, summary: string): MemoryItem => ({
  uid, kind: 'fact', scope: 'project', project: '-p', engine: 'email', session: 's', cwd: null, where: 'w', ts: 1, firstTs: 1,
  title: summary, summary, body: summary, files: [], appliesWhen: '', confidence: 'stated', importance: 3, status: 'active',
  provenance: [], seen: 1, fingerprint: uid,
});

describe('search by kind', () => {
  const filler = Array.from({ length: 300 }, (_, i) => `agenda${i % 41} minutes${i % 29}`).join(' ');

  async function seeded() {
    const q = fakeQdrant();
    const d: RecallDeps = { client: q.client, embed };
    // The fact the search is for sits deep in each record's text: only a chunk of it matches, never its key.
    const said = `${filler} the backup server moved to rack seven ${filler}`;
    await writeStm(d, [entry('mail:1', 'email', said), entry('chat:1', 'message', said)]);
    await writeLtm(d, [fact('mem:rack', 'the backup server moved to rack seven')]);
    return { q, d };
  }

  it('finds a record of that kind by a chunk of its text, and shows the chunk', async () => {
    const { d } = await seeded();
    const hits = await memorySearch(d, { query: 'backup server rack seven', kinds: ['email'] });
    expect(hits.map((h) => h.id)).toEqual(['stm:mail:1']);
    expect(hits[0]!.kind).toBe('email');
    expect(hits[0]!.match).toContain('backup server moved to rack seven');
  });

  it('finds learned items by their own kind, not by their source record\'s', async () => {
    const { d } = await seeded();
    expect((await memorySearch(d, { query: 'backup server rack seven', kinds: ['fact'] })).map((h) => h.id)).toEqual(['mem:rack']);
    // The fact came from an e-mail; asking for e-mails gives records, not the fact.
    expect((await memorySearch(d, { query: 'backup server rack seven', kinds: ['email'] })).map((h) => h.id)).not.toContain('mem:rack');
    expect((await memorySearch(d, { query: 'backup server rack seven', kinds: ['email', 'fact'] })).map((h) => h.id).sort()).toEqual(['mem:rack', 'stm:mail:1']);
  });

  it('puts the record\'s kind on every point of it', async () => {
    const { q } = await seeded();
    const mail = [...q.points.values()].filter((p) => p.payload.item === 'stm:mail:1');
    expect(mail.length).toBeGreaterThan(1);
    expect(mail.every((p) => p.payload.engine === 'email')).toBe(true);
  });
});

describe('search by date', () => {
  it('refuses a date that is not one instead of leaving that end open', async () => {
    const { client } = fakeQdrant();
    const d: RecallDeps = { client, embed };
    await expect(memorySearch(d, { query: 'x', from: 'April 2026' })).rejects.toThrow(/from "April 2026" is not a date/);
    await expect(memorySearch(d, { query: 'x', to: '2026-13' })).rejects.toThrow(/to "2026-13"/);
    await expect(memorySearch(d, { query: 'x', from: '2026-02-30' })).rejects.toThrow(/is not a date/);
    await expect(memorySearch(d, { query: 'x', from: '2026-04', to: '2026-04-30' })).resolves.toEqual([]);
  });

  it('reads only days and months that exist', () => {
    expect(dateSpan('2024-02-30')).toBeNull();
    expect(dateSpan('2024-02-00')).toBeNull();
    expect(dateSpan('2024-13')).toBeNull();
    expect(dateSpan('2024-00')).toBeNull();
    expect(dateSpan('2024-02-29')).toEqual({ from: Date.UTC(2024, 1, 29), to: Date.UTC(2024, 2, 1) - 1 });
    expect(dateSpan('2023-02-29')).toBeNull();
    expect(dateSpan('2024-12')).toEqual({ from: Date.UTC(2024, 11, 1), to: Date.UTC(2025, 0, 1) - 1 });
  });
});

describe('the index', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('indexes the record kind, and keeps a collection name inside its path segment', async () => {
    const calls: Array<{ url: string; body?: { field_name?: string } }> = [];
    vi.stubGlobal('fetch', async (url: string, init: { method: string; body?: string }) => {
      calls.push({ url, ...(init.body ? { body: JSON.parse(init.body) } : {}) });
      if (init.method === 'GET') return new Response('{"status":{"error":"not found"}}', { status: 404 });
      return new Response(JSON.stringify({ result: true }));
    });
    await new QdrantClient(() => ({ url: 'http://q', apiKey: 'k' }), 1000, { collection: 'a/b?c' }).ensureCollection();
    expect(calls.every((c) => c.url.startsWith('http://q/collections/a%2Fb%3Fc'))).toBe(true);
    expect(calls.map((c) => c.body?.field_name)).toContain('engine');
  });
});
