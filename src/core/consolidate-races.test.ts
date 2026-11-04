/**
 * Consolidation and sleep against what changes around them: records removed or changed while the model reads them, a
 * thread that never goes quiet, a model that quotes its labels, a subject linked without end, history imported after a
 * sleep, and a model that is not there.
 */
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { LlmAnswer, LlmRequest } from '../llm.js';
import { Consolidator, labelNo, MAX_LINKS } from './consolidate.js';
import { memoryGather } from './gather.js';
import type { Filter, PointStruct, QdrantClient } from './qdrant-client.js';
import { Sleep } from './sleep.js';
import { JsonlLog, ltmMetaOf, stmLog, UsageBook, type LtmMeta, type MemoryItem, type StmEntry } from './store.js';
import { termId, tokens } from './text.js';
import { writeLtm } from './writer.js';

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

type Cond = { key?: string; match?: { value?: unknown; any?: unknown[] } } & Filter;
function matches(p: Record<string, unknown>, f?: Filter): boolean {
  const one = (c: Cond): boolean => {
    if (c.must || c.should || c.must_not) return matches(p, c);
    const v = p[c.key!];
    if (c.match && 'value' in c.match) return Array.isArray(v) ? v.includes(c.match.value) : v === c.match.value;
    if (c.match?.any) return Array.isArray(v) ? v.some((x) => c.match!.any!.includes(x)) : c.match.any.includes(v);
    return true;
  };
  return ((f?.must ?? []) as Cond[]).every(one) && !((f?.must_not ?? []) as Cond[]).some(one) && (!f?.should?.length || (f.should as Cond[]).some(one));
}

/** Qdrant as far as consolidation, sleep and search use it. */
function fakeQdrant() {
  const points = new Map<string, PointStruct>();
  const near = (vector: number[], filter: Filter, limit: number) => [...points.values()].filter((p) => matches(p.payload, filter))
    .map((p) => ({ item: String(p.payload.item), score: cos(vector, p.vector.dense as number[]), payload: p.payload }))
    .sort((a, b) => b.score - a.score).slice(0, limit);
  const client = {
    async upsert(ps: PointStruct[]) { for (const p of ps) points.set(p.id, p); },
    async deleteItems(items: string[]) { for (const [id, p] of points) if (items.includes(String(p.payload.item))) points.delete(id); },
    async deleteParts(items: string[], part: string) { for (const [id, p] of points) if (items.includes(String(p.payload.item)) && p.payload.part === part) points.delete(id); },
    async setPayload(items: string[], payload: Record<string, unknown>) { for (const p of points.values()) if (items.includes(String(p.payload.item))) Object.assign(p.payload, payload); },
    async retrieve(ids: string[], withVector = false) {
      return ids.flatMap((id) => (points.has(id) ? [{ id, payload: points.get(id)!.payload, ...(withVector ? { vector: points.get(id)!.vector.dense as number[] } : {}) }] : []));
    },
    async neighbors(vector: number[], filter: Filter, limit = 5) { return near(vector, { ...filter, must: [...(filter.must ?? []), { key: 'part', match: { value: 'key' } }] }, limit); },
    async nearestMany(vectors: number[][], filter: Filter, limit = 10) { return vectors.map((v) => near(v, filter, limit)); },
    async fusedItems(q: { dense: number[][]; filter?: Filter; limit?: number }) {
      const best = new Map<string, number>();
      for (const h of near(q.dense[0]!, q.filter ?? {}, points.size)) if (!best.has(h.item)) best.set(h.item, h.score);
      return [...best.entries()].slice(0, q.limit ?? 40).map(([item, score]) => ({ item, score }));
    },
  };
  return { client: client as unknown as QdrantClient, points };
}

const DAY = 86_400_000;
const OLD = Date.parse('2026-01-01T00:00:00Z');
const NOW = Date.parse('2026-10-23T12:00:00Z');

function item(over: Partial<MemoryItem> & Pick<MemoryItem, 'uid' | 'title' | 'summary'>): MemoryItem {
  return {
    kind: 'fact', scope: 'project', project: '-p', engine: 'conversation', session: 's0', cwd: null, where: 'conv', ts: OLD, firstTs: OLD,
    body: over.summary, files: [], appliesWhen: '', confidence: 'verified', importance: 3, status: 'active', provenance: [], seen: 1,
    fingerprint: over.uid, ...over,
  };
}

function entry(uid: string, request: string, over: Partial<StmEntry> = {}): StmEntry {
  return {
    uid, fingerprint: uid, engine: 'conversation', session: 'sess-1', turn: 0, cwd: null, project: '-p', title: 'Deploy work', ts: OLD,
    synthetic: false, typed: 'human', request, reply: 'ok', said: 'ok', steps: [], files: [], done: false, ...over,
  };
}

/** `learnedAt` projected into the in-memory meta, as store.ts's ltmMetaOf does once the field is declared there. */
type Meta = LtmMeta & { learnedAt?: number };
const knowledgeLog = (path: string) => new JsonlLog<MemoryItem, Meta>(path, (r) => ({ ...ltmMetaOf(r), learnedAt: (r as { learnedAt?: number }).learnedAt }));

function setup() {
  const dir = mkdtempSync(join(tmpdir(), 'memory-races-'));
  const { client, points } = fakeQdrant();
  const stm = stmLog(join(dir, 'stm.jsonl'));
  const ltm = knowledgeLog(join(dir, 'ltm.jsonl'));
  return { dir, client, points, stm, ltm, base: { client, embed, stm, ltm, log: () => {}, aborted: () => false, now: () => NOW } };
}

const nothing = { answer: { segment_summary: 'x', episodes: [], items: [], observations: [] } };
const learnedOf = (it: MemoryItem | null): number | undefined => (it as { learnedAt?: number } | null)?.learnedAt;

describe('consolidation marks done only what it read', () => {
  it('leaves a record removed while the model read it removed, and one changed meanwhile to be read again', async () => {
    const s = setup();
    s.stm.put([entry('c:1', 'rotate the staging certificate', { turn: 0 }), entry('c:2', 'renew the token', { turn: 1 })]);
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      if (a.name === 'memory_consolidation') {
        // Meanwhile: one record removed, the other changed (added again with new text).
        s.stm.remove(['c:1']);
        s.stm.put([{ ...s.stm.get('c:2')!, fingerprint: 'c:2-new', request: 'renew the token, and the key too' }]);
      }
      return nothing;
    };
    const r = await new Consolidator({ ...s.base, ask }).run();
    expect(r.windows).toBe(1);
    expect(s.stm.get('c:1')).toBeNull();
    expect(s.stm.get('c:2')).toMatchObject({ request: 'renew the token, and the key too', done: false });
  });

  it('marks done a record left as it was', async () => {
    const s = setup();
    s.stm.put([entry('c:1', 'rotate the staging certificate')]);
    await new Consolidator({ ...s.base, ask: async () => nothing }).run();
    expect(s.stm.get('c:1')?.done).toBe(true);
  });
});

describe('a thread that never goes quiet', () => {
  it('is read once its oldest waiting entry is a day old', () => {
    const s = setup();
    // A message every half hour for 25 hours: never an hour of quiet.
    s.stm.put(Array.from({ length: 51 }, (_, n) => entry(`busy:${n}`, `update ${n}`, { session: 'busy', turn: n, ts: NOW - 25 * 3600_000 + n * 30 * 60_000 })));
    s.stm.put([entry('fresh:0', 'just now', { session: 'fresh', ts: NOW - 10 * 60_000 })]);
    const pending = new Consolidator(s.base).pending(NOW).map((p) => p.session);
    expect(pending).toEqual(['busy']);
  });
});

describe('labels a model quotes', () => {
  it('reads p and i labels whether given alone or with their line', () => {
    expect(labelNo('p3', 'p')).toBe(3);
    expect(labelNo('[p12]\nNEW (fact): x', 'p')).toBe(12);
    expect(labelNo('[i2] (fact) deploy the api', 'i')).toBe(2);
    expect(labelNo('item', 'i')).toBe(0);
  });

  it('takes a relation for a pair the model labelled with its whole line', async () => {
    const s = setup();
    const it = item({ uid: 'mem:new', title: 'new', summary: 'deploy through jenkins' });
    const known = item({ uid: 'mem:old', title: 'old', summary: 'deploy by hand' });
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => ({ answer: { pairs: [
      { id: a.prompt.split('\n\n')[0]!.split('\n').slice(0, 2).join('\n'), relation: 'related', title: '', summary: '', body: '' },
    ] } });
    const got = await new Consolidator({ ...s.base, ask }).relate([{ it, known }]);
    expect(got instanceof Map && got.get('p1')?.relation).toBe('related');
  });

  it('says why when the model does not answer a relate ask', async () => {
    const s = setup();
    const got = await new Consolidator({ ...s.base, ask: async () => ({ unavailable: 'no model is set' }) })
      .relate([{ it: item({ uid: 'a', title: 'a', summary: 'a' }), known: item({ uid: 'b', title: 'b', summary: 'b' }) }]);
    expect(got).toEqual({ paused: 'no model is set' });
  });

  it('gives an item its cues when the model quotes its label line', async () => {
    const s = setup();
    const it = item({ uid: 'mem:api', title: 'deploy the api', summary: 'push main and jenkins deploys' });
    await writeLtm(s.base, [it]);
    s.ltm.put([it]);
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => ({ answer: { items: [
      { id: a.prompt.split('\n')[0]!, cues: ['how do I ship the api'], concepts: ['jenkins'] },
    ] } });
    expect(await new Consolidator({ ...s.base, ask }).fillCues()).toEqual({ cued: 1 });
    expect(s.ltm.get('mem:api')).toMatchObject({ cues: ['how do I ship the api'], concepts: ['jenkins'] });
  });
});

describe('links', () => {
  it('keeps the newest MAX_LINKS', async () => {
    const s = setup();
    const busy = item({ uid: 'mem:busy', title: 'busy', summary: 'busy', related: Array.from({ length: MAX_LINKS }, (_, n) => `mem:r${n}`) });
    const other = item({ uid: 'mem:new', title: 'new', summary: 'new' });
    await writeLtm(s.base, [busy, other]);
    s.ltm.put([busy, other]);
    await new Consolidator(s.base).link(busy, other);
    const related = s.ltm.get('mem:busy')!.related!;
    expect(related).toHaveLength(MAX_LINKS);
    expect(related.at(-1)).toBe('mem:new');
    expect(related).not.toContain('mem:r0');
    // Linked again: not listed twice.
    await new Consolidator(s.base).link(s.ltm.get('mem:busy')!, other);
    expect(s.ltm.get('mem:busy')!.related!.filter((u) => u === 'mem:new')).toHaveLength(1);
  });
});

describe('history imported after a sleep', () => {
  it('stamps what consolidation writes with when it was learned', async () => {
    const s = setup();
    s.stm.put([entry('c:1', 'deploy the api through jenkins', { ts: OLD })]);
    const ask = async (): Promise<LlmAnswer> => ({ answer: { segment_summary: 'x', episodes: [], observations: [], items: [{
      kind: 'procedure', title: 'deploy through jenkins', summary: 'push main and jenkins deploys', body: 'b', applies_when: '', scope: 'project',
      requests: [1], confidence: 'verified', importance: 4, contested: false, ambiguous: false, cues: [], concepts: [] }] } });
    await new Consolidator({ ...s.base, ask }).run();
    const made = [...s.ltm.all()].find((x) => x.title === 'deploy through jenkins')!;
    expect(made.ts).toBe(OLD);
    expect(learnedOf(made)).toBe(NOW);
  });

  it('is related and merged the night after, and not archived as old and unused', async () => {
    const s = setup();
    const learned = NOW - 3600_000;
    // Two copies of one fact from old records, both learned an hour ago; the last sleep was two hours ago.
    const a = { ...item({ uid: 'mem:a', title: 'ports', summary: 'the gateway listens on port 8080 behind nginx', ts: OLD - 30 * DAY }), learnedAt: learned } as MemoryItem;
    const b = { ...item({ uid: 'mem:b', title: 'ports', summary: 'the gateway listens on port 8080 behind nginx', ts: OLD }), learnedAt: learned } as MemoryItem;
    const old = { ...item({ uid: 'ep:old', kind: 'episode', title: 'one-off fix', summary: 'fixed a typo in a readme long ago', importance: 2, ts: NOW - 400 * DAY }), learnedAt: learned } as MemoryItem;
    await writeLtm(s.base, [a, b, old]);
    s.ltm.put([a, b, old]);
    const statePath = join(s.dir, 'sleep.json');
    writeFileSync(statePath, JSON.stringify({ last: NOW - 2 * 3600_000 }));
    const r = await new Sleep({ ...s.base, usage: new UsageBook(join(s.dir, 'usage.json')), statePath }).run();
    expect(r.merged).toBe(1);
    expect(r.archived).toBe(0);
    expect(s.ltm.get('ep:old')?.status).toBe('active');
  });

  it('still archives an item without learnedAt by its own time (logs written before the field)', async () => {
    const s = setup();
    const old = item({ uid: 'ep:old', kind: 'episode', title: 'one-off fix', summary: 'fixed a typo in a readme long ago', importance: 2, ts: NOW - 400 * DAY });
    await writeLtm(s.base, [old]);
    s.ltm.put([old]);
    const r = await new Sleep({ ...s.base, usage: new UsageBook(join(s.dir, 'usage.json')), statePath: join(s.dir, 'sleep.json') }).run();
    expect(r.archived).toBe(1);
  });
});

describe('a model that is not there', () => {
  it('shows in what sleep returns', async () => {
    const s = setup();
    // Nine words of eleven shared: near enough to ask about (0.82), not the same.
    const words = 'deploy jenkins pipeline main branch build docker image registry';
    const items = [item({ uid: 'mem:alpha', title: 'alpha', summary: `${words} one` }), item({ uid: 'mem:bravo', title: 'bravo', summary: `${words} three`, ts: OLD + DAY })];
    await writeLtm(s.base, items);
    s.ltm.put(items);
    const r = await new Sleep({ ...s.base, ask: async () => ({ unavailable: 'no model is set' }), usage: new UsageBook(join(s.dir, 'usage.json')), statePath: join(s.dir, 'sleep.json') }).run();
    expect(r.paused).toBe('no model is set');
  });

  it('shows in what gathering returns — every candidate kept, by rank', async () => {
    const s = setup();
    const obs = ['bought an evening gown', 'went camping'].map((t, n) => item({ uid: `obs:${n}`, kind: 'observation', title: `Nadia ${t}`, summary: `Nadia ${t}`, scope: 'person' }));
    await writeLtm(s.base, obs);
    const deps = { client: s.client, embed };
    const g = await memoryGather(deps, { query: 'what did Nadia buy' }, { ask: async () => ({ unavailable: 'no model is set' }) });
    expect(g.paused).toBe('no model is set');
    expect(g.kept.sort()).toEqual(['obs:0', 'obs:1']);
    const read = await memoryGather(deps, { query: 'what did Nadia buy' }, { ask: async () => ({ answer: { keep: ['m1'], searches: [] } }) });
    expect(read.paused).toBeUndefined();
  });
});
