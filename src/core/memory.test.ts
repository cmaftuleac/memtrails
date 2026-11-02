/**
 * Memory end to end — records (short-term), consolidation (long-term), sleep and recall — over an
 * in-memory stand-in for Qdrant and an embedder whose vectors are the words of the text (so "the same" and "nearly
 * the same" are under the test's control), with a stand-in for the fast model. No model, no Qdrant, no network.
 *
 * Each case is a way memory would turn into chaos or mislead an agent: a request missing from short-term memory,
 * the same knowledge stored twice, a changed decision kept beside the new one, junk kept, an old unused item never
 * leaving, a superseded or throwaway item shown, a turn held up by a slow or missing judge.
 */
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { LlmAnswer, LlmRequest } from '../llm.js';
import { Consolidator, relateBatches } from './consolidate.js';
import { memoryGather } from './gather.js';
import { pointId, type Filter, type PointStruct, type QdrantClient } from './qdrant-client.js';
import { RECALL_QUERY_CHARS, RecallBook, type RecallEvent } from './recall-log.js';
import { memoryExplore, memoryOpen, memorySearch, type RecallDeps } from './recall.js';
import { Sleep } from './sleep.js';
import { ltmLog, stmLog, UsageBook, type MemoryItem, type StmEntry } from './store.js';
import { termId, tokens } from './text.js';
import { cuePoint, writeCues, writeLtm, writeStm } from './writer.js';
import { children } from './text.js';

const DIMS = 512;

/** The words of a text as a unit vector (the embedding prompt's wrapper left out). */
async function embed(texts: string[]): Promise<Float32Array[]> {
  return texts.map((t) => {
    const v = new Float32Array(DIMS);
    for (const w of new Set(tokens(t.replace(/^title: | \| text: |^task: search result \| query: /g, ' ')))) v[termId(w) % DIMS] = 1;
    const n = Math.hypot(...v) || 1;
    return v.map((x) => x / n);
  });
}

const cos = (a: number[], b: number[]): number => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0);

type Cond = { key?: string; match?: { value?: unknown; any?: unknown[] }; range?: { lt?: number; lte?: number; gte?: number }; is_empty?: { key: string } } & Filter;

function matches(p: Record<string, unknown>, f?: Filter): boolean {
  const one = (c: Cond): boolean => {
    // A nested filter is a condition (Qdrant's rule).
    if (c.must || c.should || c.must_not) return matches(p, c);
    if (c.is_empty) return p[c.is_empty.key] === undefined || p[c.is_empty.key] === null;
    const v = p[c.key!];
    // An array field matches when any of its values does (Qdrant's rule).
    if (c.match && 'value' in c.match) return Array.isArray(v) ? v.includes(c.match.value) : v === c.match.value;
    if (c.match?.any) return Array.isArray(v) ? v.some((x) => c.match!.any!.includes(x)) : c.match.any.includes(v);
    if (c.range) {
      if (typeof v !== 'number') return false;
      const r = c.range;
      return (r.lt === undefined || v < r.lt) && (r.lte === undefined || v <= r.lte) && (r.gte === undefined || v >= r.gte);
    }
    return true;
  };
  return ((f?.must ?? []) as Cond[]).every(one) && !((f?.must_not ?? []) as Cond[]).some(one)
    && (!f?.should?.length || (f.should as Cond[]).some(one));
}

/** Qdrant, as far as memory uses it. */
function fakeQdrant() {
  const points = new Map<string, PointStruct>();
  const client = {
    async ensureCollection() {},
    async upsert(ps: PointStruct[]) { for (const p of ps) points.set(p.id, p); },
    async deleteItems(items: string[]) { for (const [id, p] of points) if (items.includes(String(p.payload.item))) points.delete(id); },
    async deleteParts(items: string[], part: string) { for (const [id, p] of points) if (items.includes(String(p.payload.item)) && p.payload.part === part) points.delete(id); },
    async setPayload(items: string[], payload: Record<string, unknown>) { for (const p of points.values()) if (items.includes(String(p.payload.item))) Object.assign(p.payload, payload); },
    async count() { return points.size; },
    async retrieve(ids: string[], withVector = false) {
      return ids.flatMap((id) => (points.has(id) ? [{ id, payload: points.get(id)!.payload, ...(withVector ? { vector: points.get(id)!.vector.dense as number[] } : {}) }] : []));
    },
    async nearest(vector: number[], filter: Filter, limit = 10) {
      return [...points.values()].filter((p) => matches(p.payload, filter))
        .map((p) => ({ item: String(p.payload.item), score: cos(vector, p.vector.dense as number[]), payload: p.payload }))
        .sort((a, b) => b.score - a.score).slice(0, limit);
    },
    async nearestMany(vectors: number[][], filter: Filter, limit = 10) { return Promise.all(vectors.map((v) => client.nearest(v, filter, limit))); },
    async scroll(filter: Filter, limit = 256, withVector = false) {
      return [...points.values()].filter((p) => matches(p.payload, filter)).slice(0, limit)
        .map((p) => ({ id: p.id, payload: p.payload, ...(withVector ? { vector: p.vector.dense as number[] } : {}) }));
    },
    async neighbors(vector: number[], filter: Filter, limit = 5) {
      const must = [...(filter.must ?? []), { key: 'part', match: { value: 'key' } }];
      return [...points.values()].filter((p) => matches(p.payload, { ...filter, must }))
        .map((p) => ({ item: String(p.payload.item), score: cos(vector, p.vector.dense as number[]), payload: p.payload }))
        .sort((a, b) => b.score - a.score).slice(0, limit);
    },
    async bestPerItem(vector: number[], items: string[], part: string) {
      const best = new Map<string, number>();
      for (const p of points.values()) {
        const item = String(p.payload.item);
        if (!items.includes(item) || p.payload.part !== part) continue;
        best.set(item, Math.max(best.get(item) ?? -1, cos(vector, p.vector.dense as number[])));
      }
      return best;
    },
    async fusedItems(q: { dense: number[][]; filter?: Filter; per?: number; limit?: number }) {
      const score = new Map<string, number>();
      const point = new Map<string, string>();
      for (const v of q.dense) {
        const ranked = [...points.values()].filter((p) => matches(p.payload, q.filter))
          .map((p) => ({ item: String(p.payload.item), id: p.id, s: cos(v, p.vector.dense as number[]) })).sort((a, b) => b.s - a.s);
        const best = new Map<string, number>();
        for (const r of ranked) {
          if (best.has(r.item)) continue;
          best.set(r.item, best.size);
          if (!point.has(r.item)) point.set(r.item, r.id);
        }
        for (const [item, rank] of best) if (rank < (q.per ?? 50)) score.set(item, (score.get(item) ?? 0) + 1 / (60 + rank + 1));
      }
      return [...score.entries()].sort((a, b) => b[1] - a[1]).slice(0, q.limit ?? 40).map(([item, s]) => ({ item, score: s, point: point.get(item) }));
    },
  };
  return { client: client as unknown as QdrantClient, points };
}

const OLD = Date.parse('2026-01-01T00:00:00Z');
const NOW = Date.parse('2026-10-23T12:00:00Z');

function item(over: Partial<MemoryItem> & Pick<MemoryItem, 'uid' | 'title' | 'summary'>): MemoryItem {
  return {
    kind: 'fact', scope: 'project', project: '-p', engine: 'conversation', session: 's0', cwd: '/p', where: 'conv', ts: OLD, firstTs: OLD,
    body: over.summary, files: [], appliesWhen: '', confidence: 'verified', importance: 3, status: 'active', provenance: [], seen: 1,
    fingerprint: over.uid, ...over,
  };
}

function entry(uid: string, request: string, over: Partial<StmEntry> = {}): StmEntry {
  return {
    uid, fingerprint: uid, engine: 'conversation', session: 'sess-1', turn: 0, cwd: '/p', project: '-p', title: 'Deploy work', ts: OLD,
    synthetic: false, typed: 'human', request, reply: 'ok', said: 'ok', steps: [], files: [], done: false, ...over,
  };
}

function setup() {
  const dir = mkdtempSync(join(tmpdir(), 'memory-e2e-'));
  const { client, points } = fakeQdrant();
  const stm = stmLog(join(dir, 'stm.jsonl'));
  const ltm = ltmLog(join(dir, 'ltm.jsonl'));
  return { dir, client, points, stm, ltm, base: { client, embed, stm, ltm, log: () => {}, aborted: () => false, now: () => NOW } };
}


describe('nothing lost', () => {


  it('gives a typed memory the files its requests changed, read from their steps, not from the model', async () => {
    const s = setup();
    s.stm.put([entry('conv:f:0', 'wire the deploy script', { session: 'f', files: ['/p/deploy.sh', '/p/Jenkinsfile'] })]);
    const ask = async (): Promise<LlmAnswer> => ({ answer: { segment_summary: 'x', episodes: [{ request: 1, outcome: 'success', keep: false, title: '', summary: '', cues: [], concepts: [] }],
      items: [{ kind: 'procedure', title: 'deploy through the script', summary: 'run deploy sh after pushing', body: 'b', applies_when: '', scope: 'project', requests: [1], confidence: 'verified', importance: 4, contested: false, ambiguous: false, cues: [], concepts: [] }] } });
    await new Consolidator({ ...s.base, ask }).run();
    const made = [...s.ltm.all()].find((x) => x.title === 'deploy through the script')!;
    expect(made.files).toEqual(['/p/deploy.sh', '/p/Jenkinsfile']);
  });

  it('a merge keeps both texts: the new item, and the known one before its rewrite, as records the item lists', async () => {
    const s = setup();
    const known = item({ uid: 'mem:k', title: 'deploy api jenkins', summary: 'deploy the api by pushing main jenkins builds and deploys it', body: 'the first text' });
    await writeLtm(s.base, [known]);
    s.ltm.put([known]);
    const w = new Consolidator(s.base);
    const fresh = item({ uid: 'mem:n', title: 'deploy api jenkins', summary: 'deploy the api with jenkins', body: 'the second text, with a detail' });
    await w.merge(s.ltm.get('mem:k')!, fresh, { title: 'Deploy the API through Jenkins', summary: 'merged', body: 'merged text' });
    const merged = s.ltm.get('mem:k')!;
    expect(merged.body).toBe('merged text');
    expect(merged.absorbed?.sort()).toEqual([`mem:k@${known.fingerprint}`, 'mem:n'].sort());
    expect(s.ltm.get('mem:n')).toMatchObject({ status: 'merged', mergedInto: 'mem:k', body: 'the second text, with a detail' });
    expect(s.ltm.get(`mem:k@${known.fingerprint}`)).toMatchObject({ status: 'merged', body: 'the first text' });
    // Out of recall: only the merged item is in the index.
    expect([...s.points.values()].map((p) => p.payload.item)).not.toContain('mem:n');
  });

  it('indexes every chunk of a long body, its cues apart from them, and a short-term entry whole', async () => {
    const s = setup();
    const body = Array.from({ length: 3000 }, (_, i) => `detail${i}`).join(' ');
    const it = item({ uid: 'mem:long', title: 'long', summary: 'a long one', body, cues: Array.from({ length: 10 }, (_, i) => `cue question ${i}`) });
    await writeLtm(s.base, [it]);
    const parts = (part: string) => [...s.points.values()].filter((p) => p.payload.item === 'mem:long' && p.payload.part === part);
    expect(parts('child')).toHaveLength(children(body).length);
    expect(parts('cue')).toHaveLength(10);
    expect(s.points.has(pointId('mem:long', cuePoint(9)))).toBe(true);
    expect(parts('key')[0]!.payload.body).toBe(body);
    const e = entry('conv:x:0', Array.from({ length: 1500 }, (_, i) => `ask${i}`).join(' '), { ts: Date.now() });
    await writeStm(s.base, [e]);
    const stm = [...s.points.values()].filter((p) => p.payload.item === 'stm:conv:x:0');
    expect(stm.filter((p) => p.payload.part === 'child').length).toBeGreaterThan(10);
  });

  it('gives a long batch time to embed: more chunks, more time, never less than the 30 s floor', async () => {
    const s = setup();
    const budgets: Array<[number, number | undefined]> = [];
    const timed = { ...s.base, embed: async (texts: string[], ms?: number) => { budgets.push([texts.length, ms]); return embed(texts); } };
    const e = entry('conv:big:0', Array.from({ length: 30_000 }, (_, i) => `step${i}`).join(' '), { ts: Date.now() });
    await writeStm(timed, [e]);
    const [n, ms] = budgets[0]!;
    expect(n).toBeGreaterThan(300);
    expect(ms).toBe(Math.max(30_000, n * 150));
    await writeStm(timed, [entry('conv:small:0', 'short', { ts: Date.now() })]);
    expect(budgets[1]![1]).toBe(30_000);
  });

  it('asks about pairs in batches of at most 8 and at most a size, a long pair on its own', () => {
    const pair = (n: number) => ({ it: item({ uid: `a${n}`, title: 't', summary: 's', body: 'x'.repeat(n) }), known: item({ uid: `b${n}`, title: 't', summary: 's', body: '' }) });
    expect(relateBatches(Array.from({ length: 20 }, () => pair(10))).map((b) => b.length)).toEqual([8, 8, 4]);
    expect(relateBatches([pair(10), pair(70_000), pair(10)]).map((b) => b.length)).toEqual([1, 1, 1]);
  });
});

describe('remote embedding failure', () => {
  it('keeps the old item and cues searchable until replacement vectors are available', async () => {
    const s = setup();
    const known = item({ uid: 'mem:known', title: 'deploy', summary: 'push main', cues: ['deploy the api'] });
    await writeLtm(s.base, [known]);
    const before = [...s.points.entries()];
    const unavailable = { ...s.base, embed: async () => { throw new Error('GPU unavailable'); } };
    await expect(writeLtm(unavailable, [{ ...known, summary: 'new deployment procedure' }])).rejects.toThrow('GPU unavailable');
    expect([...s.points.entries()]).toEqual(before);
    await expect(writeCues(unavailable, [{ ...known, cues: ['new deployment question'] }])).rejects.toThrow('GPU unavailable');
    expect([...s.points.entries()]).toEqual(before);
    await writeCues(s.base, [{ ...known, cues: [] }]);
    expect([...s.points.values()].some((p) => p.payload.part === 'cue')).toBe(false);
    await writeLtm(s.base, [{ ...known, status: 'archived' }]);
    expect(s.points.size).toBe(0);
  });
});

describe('consolidation — long-term memory', () => {
  it('keeps what matters once: merges the known, supersedes the changed, adds the new, drops the junk, empties the queue', async () => {
    const s = setup();
    const known = item({ uid: 'mem:known', title: 'deploy api jenkins', summary: 'deploy the api by pushing main jenkins builds and deploys it' });
    const old = item({ uid: 'mem:old', kind: 'decision', title: 'database choice postgres', summary: 'alpha beta gamma delta epsilon zeta eta theta' });
    await writeLtm(s.base, [known, old]);
    s.ltm.put([known, old]);
    const pending = [entry('conv:sess-1:0', 'how do we deploy'), entry('conv:sess-1:1', 'switch the database', { turn: 1 })];
    await writeStm(s.base, pending);
    s.stm.put(pending);

    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      if (a.name === 'memory_consolidation') {
        const base = { body: 'b', applies_when: '', scope: 'project', requests: [1], confidence: 'verified', importance: 4, contested: false, ambiguous: false };
        return { answer: {
          segment_summary: 'Deploy and database work.',
          episodes: [{ request: 1, keep: false, title: '', summary: '', cues: [] }, { request: 2, keep: true, title: 'database switched', summary: 'the database was switched to sqlite after load tests', cues: ['de ce am trecut pe sqlite', 'which database do we use'] }],
          items: [
            { ...base, kind: 'fact', title: 'deploy api jenkins', summary: 'deploy the api by pushing main jenkins builds and deploys it' },
            { ...base, kind: 'decision', title: 'database choice postgres', summary: 'alpha beta gamma delta epsilon zeta eta iota', requests: [1] },
            { ...base, kind: 'pitfall', title: 'flaky e2e on wifi', summary: 'the e2e suite times out on hotel wifi run it wired', cues: ['the e2e tests keep timing out', 'testele e2e cad pe wifi', 'run the suite wired'] },
            { ...base, kind: 'fact', title: 'a guess', summary: 'probably redis', confidence: 'inferred' },
          ],
        } };
      }
      // reconcile: the database decision changed.
      return { answer: { pairs: [{ id: 'p1', relation: 'contradicts', title: '', summary: '', body: '' }] } };
    };
    const r = await new Consolidator({ ...s.base, ask }).run();
    expect(r).toMatchObject({ windows: 1, merged: 1, superseded: 1, dropped: 1, episodes: 1 });
    // Known knowledge learned again: one item, seen twice, with the new provenance.
    expect(s.ltm.get('mem:known')).toMatchObject({ seen: 2, status: 'active' });
    expect(s.ltm.get('mem:known')!.provenance).toContain('conv:sess-1:0');
    // The changed decision: the old one superseded (kept, out of the index), the new one active.
    expect(s.ltm.get('mem:old')).toMatchObject({ status: 'superseded' });
    expect([...s.points.values()].some((p) => p.payload.item === 'mem:old')).toBe(false);
    const fresh = [...s.ltm.all()].filter((x) => x.status === 'active').map((x) => x.title);
    expect(fresh).toEqual(expect.arrayContaining(['database choice postgres', 'flaky e2e on wifi', 'database switched']));
    expect(fresh).not.toContain('a guess');
    // Each new item and kept episode is also found by its cues — points of its own, in every language it came with.
    const cuesOf = (title: string) => {
      const uid = [...s.ltm.all()].find((x) => x.title === title)!.uid;
      return [...s.points.values()].filter((p) => p.payload.item === uid && p.payload.part === 'cue').length;
    };
    expect(cuesOf('flaky e2e on wifi')).toBe(3);
    expect(cuesOf('database switched')).toBe(2);
    // The queue: read — and the requests themselves still searchable: the raw record stays, the memories are keys to it.
    expect([...s.stm.entries()].every(([, m]) => m.done)).toBe(true);
    expect(pending.every((e) => s.points.has(pointId(`stm:${e.uid}`, 0)))).toBe(true);
  });

  it('does not keep a request as an episode when the window\'s typed memories already came from it', async () => {
    const s = setup();
    s.stm.put([entry('conv:sess-1:0', 'fix the flaky e2e'), entry('conv:sess-1:1', 'rename the job', { turn: 1 })]);
    const base = { body: 'b', applies_when: '', scope: 'project', confidence: 'verified', importance: 4, contested: false, ambiguous: false };
    const ask = async (): Promise<LlmAnswer> => ({ answer: {
      segment_summary: 'e2e and job work.',
      episodes: [{ request: 1, keep: true, title: 'e2e fixed', summary: 'the e2e suite was fixed by running it wired', cues: [] },
        { request: 2, keep: true, title: 'job renamed', summary: 'the deploy job was renamed to ship', cues: [] }],
      items: [{ ...base, kind: 'pitfall', title: 'flaky e2e on wifi', summary: 'the e2e suite times out on hotel wifi run it wired', requests: [1] }],
    } });
    const r = await new Consolidator({ ...s.base, ask }).run();
    expect(r.episodes).toBe(1);
    const titles = [...s.ltm.all()].filter((x) => x.status === 'active').map((x) => x.title).sort();
    expect(titles).toEqual(['flaky e2e on wifi', 'job renamed']);
    expect(s.ltm.get('ep:conv:sess-1:0')).toBeNull();
    expect([...s.stm.entries()].every(([, m]) => m.done)).toBe(true);
  });

  it('waits for a conversation to settle before reading it', () => {
    const s = setup();
    s.stm.put([entry('e:busy:0', 'still going', { session: 'busy', ts: NOW - 60_000 }), entry('e:calm:0', 'done', { session: 'calm', ts: NOW - 2 * 3600_000 })]);
    const w = new Consolidator(s.base);
    expect(w.pending().map((p) => p.session)).toEqual(['calm']);
  });

  it('extracts several windows at once and still reconciles them in order, newest conversation first', async () => {
    const s = setup();
    const words = ['kangaroo', 'volcano', 'trumpet', 'glacier', 'saffron'];
    s.stm.put(words.map((w, i) => entry(`e:s${i}:0`, `ask about ${w}`, { session: `s${i}`, ts: NOW - (10 - i) * 3600_000 })));
    let inFlight = 0;
    let most = 0;
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      inFlight++;
      most = Math.max(most, inFlight);
      await new Promise((r) => setTimeout(r, 20));
      inFlight--;
      const w = words.find((x) => a.prompt.includes(x))!;
      const base = { body: 'b', applies_when: '', scope: 'project', requests: [1], confidence: 'verified', importance: 4, contested: false, ambiguous: false };
      return { answer: { segment_summary: w, episodes: [{ request: 1, keep: false, title: '', summary: '' }],
        items: [{ ...base, kind: 'fact', title: w, summary: `${w} ${w} ${w}` }] } };
    };
    const r = await new Consolidator({ ...s.base, ask, concurrency: 3 }).run();
    expect(r.windows).toBe(5);
    expect(most).toBe(3);
    // Newest conversation first, as the queue orders them — whatever order the extractions finished in.
    expect([...s.ltm.all()].map((x) => x.title)).toEqual([...words].reverse());
    expect([...s.stm.entries()].every(([, m]) => m.done)).toBe(true);
  });

  it('gives a saved note its cues and concepts, and keeps both sides\' cues when a memory is learned again', async () => {
    const s = setup();
    const note = item({ uid: 'note:/m/deploy.md', kind: 'note', title: 'deploy rule', summary: 'deploys go through ci never rsync' });
    const known = item({ uid: 'mem:k', title: 'deploy api jenkins', summary: 'deploy the api by pushing main jenkins builds and deploys it', cues: ['how is the api deployed'] });
    await writeLtm(s.base, [note, known]);
    s.ltm.put([note, known]);
    s.stm.put([entry('conv:s9:0', 'how do we deploy', { session: 's9' })]);
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      if (a.name === 'memory_cues') {
        const ids = [...a.prompt.matchAll(/\[(i\d+)\]/g)].map((m) => m[1]!);
        return { answer: { items: ids.map((id) => ({ id, cues: ['how do deploys run here', 'cum facem deploy'], concepts: ['Jenkins', 'deploy.sh', ' jenkins '] })) } };
      }
      const base = { body: 'b', applies_when: '', scope: 'project', requests: [1], confidence: 'verified', importance: 4, contested: false, ambiguous: false };
      return { answer: { segment_summary: 'deploy', episodes: [{ request: 1, keep: false, title: '', summary: '', cues: [] }],
        items: [{ ...base, kind: 'fact', title: 'deploy api jenkins', summary: 'deploy the api by pushing main jenkins builds and deploys it', cues: ['push to main to ship the api'] }] } };
    };
    const r = await new Consolidator({ ...s.base, ask }).run();
    expect(r.merged).toBe(1);
    expect(s.ltm.get('mem:k')!.cues).toEqual(['how is the api deployed', 'push to main to ship the api']);
    expect(s.ltm.get('note:/m/deploy.md')!.cues).toEqual(['how do deploys run here', 'cum facem deploy']);
    expect(s.ltm.meta('note:/m/deploy.md')?.cued).toBe(true);
    const cuePoints = (uid: string) => [...s.points.values()].filter((p) => p.payload.item === uid && p.payload.part === 'cue').length;
    expect(cuePoints('mem:k')).toBe(2);
    expect(cuePoints('note:/m/deploy.md')).toBe(2);
    expect(r.cued).toBeGreaterThanOrEqual(1);
    // Concepts: one per key however it was spelled, on the item's key point.
    expect(s.ltm.get('note:/m/deploy.md')!.concepts).toEqual(['Jenkins', 'deploy.sh']);
    expect(s.ltm.meta('note:/m/deploy.md')?.mapped).toBe(true);
    expect(s.points.get(pointId('note:/m/deploy.md', 0))!.payload.concepts).toEqual(['jenkins', 'deploy.sh']);
    expect([...s.points.values()].filter((p) => p.payload.part === 'concept')).toHaveLength(0); // concepts are not embedded
  });

  it('reads a window that keeps failing in halves, and never holds up the rest or marks a request done unread', async () => {
    const s = setup();
    // One conversation of two requests (one window) whose answer is always cut off while whole, and an older one.
    s.stm.put([entry('e:long:0', 'first long request', { session: 'long' }), entry('e:long:1', 'second long request', { session: 'long', turn: 1 }),
      entry('e:old:0', 'older request', { session: 'old', ts: OLD - 3600_000 })]);
    const answered: string[][] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      if (a.name !== 'memory_consolidation') return { answer: { items: [] } };
      const asked = ['first long request', 'second long request', 'older request'].filter((t) => a.prompt.includes(t));
      if (asked.length > 1) return { unavailable: 'Unterminated string in JSON' };
      answered.push(asked);
      return { answer: { segment_summary: 'x', episodes: [], items: [] } };
    };
    const w = new Consolidator({ ...s.base, ask });
    // First failure: a pause, as when the model is away — nothing marked done.
    expect((await w.run()).paused).toContain('Unterminated');
    expect([...s.stm.entries()].filter(([, m]) => m.done)).toHaveLength(0);
    // Second failure of the same window: it is narrowed, and the older conversation is read anyway.
    const second = await w.run();
    expect(second.paused).toBeUndefined();
    expect(s.stm.meta('e:old:0')?.done).toBe(true);
    expect(s.stm.meta('e:long:0')?.done).toBe(false);
    // Then each half on its own: every request read, none skipped.
    await w.run();
    expect([...s.stm.entries()].every(([, m]) => m.done)).toBe(true);
    expect(new Set(answered.map((x) => x.join()))).toEqual(new Set(['older request', 'first long request', 'second long request']));
  });

  it('goes on past a window that fails once after others were read: it waits for the next run, the rest are kept', async () => {
    const s = setup();
    s.stm.put([entry('e:a:0', 'newest request', { session: 'a' }), entry('e:b:0', 'middle request', { session: 'b', ts: OLD - 3600_000 }),
      entry('e:c:0', 'oldest request', { session: 'c', ts: OLD - 7200_000 })]);
    let failures = 0;
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      if (a.name !== 'memory_consolidation') return { answer: { items: [] } };
      if (a.prompt.includes('middle request') && !failures++) return { unavailable: 'Unterminated string in JSON' };
      return { answer: { segment_summary: 'x', episodes: [], items: [] } };
    };
    const w = new Consolidator({ ...s.base, ask });
    const first = await w.run();
    expect(first).toMatchObject({ windows: 2 });
    expect(first.paused).toBeUndefined();
    expect([s.stm.meta('e:a:0')?.done, s.stm.meta('e:b:0')?.done, s.stm.meta('e:c:0')?.done]).toEqual([true, false, true]);
    await w.run();
    expect(s.stm.meta('e:b:0')?.done).toBe(true);
  });

  it('pauses, leaving the queue as it was, when the fast model cannot be asked', async () => {
    const s = setup();
    s.stm.put([entry('e:s:0', 'x')]);
    const r = await new Consolidator({ ...s.base, ask: async () => ({ unavailable: 'no route' }) }).run();
    expect(r.paused).toBe('no route');
    expect(s.stm.meta('e:s:0')?.done).toBe(false);
  });
});

describe('sleep — reorganising', () => {
  it('merges a fresh near-duplicate into the older item and archives what is old and unused', async () => {
    const s = setup();
    const usage = new UsageBook(join(s.dir, 'usage.json'));
    const a = item({ uid: 'mem:a', title: 'ports', summary: 'the gateway listens on port 8080 behind nginx', ts: NOW - 10 * 86_400_000 });
    const b = item({ uid: 'mem:b', title: 'ports', summary: 'the gateway listens on port 8080 behind nginx', ts: NOW - 3600_000 });
    const stale = item({ uid: 'ep:old', kind: 'episode', title: 'one-off fix', summary: 'fixed a typo in a readme long ago', importance: 2, ts: NOW - 400 * 86_400_000 });
    const lasting = item({ uid: 'mem:arch', kind: 'architecture', title: 'how sync works', summary: 'sync runs in the background worker not the client', ts: NOW - 400 * 86_400_000 });
    await writeLtm(s.base, [a, b, stale, lasting]);
    s.ltm.put([a, b, stale, lasting]);
    const r = await new Sleep({ ...s.base, usage, statePath: join(s.dir, 'sleep.json') }).run();
    expect(r.merged).toBe(1);
    const statuses = Object.fromEntries([...s.ltm.entries()].map(([u, m]) => [u, m.status]));
    expect([statuses['mem:a'], statuses['mem:b']].sort()).toEqual(['active', 'superseded']);
    expect(statuses['ep:old']).toBe('archived');
    expect(statuses['mem:arch']).toBe('active');
  });

  it('relates a fresh item to every neighbour from 0.80, not only the nearest: links, retires the outdated, merges', async () => {
    const s = setup();
    const usage = new UsageBook(join(s.dir, 'usage.json'));
    // A one-word title and nine shared words of eleven: cosine 9/11 ≈ 0.82 inside a cluster — below the 0.90 the pass
    // used to start at; the clusters share nothing.
    const deploy = 'deploy jenkins pipeline main branch build docker image registry';
    const db = 'database postgres cluster replica failover backup nightly snapshot restore';
    const pdf = 'invoice pdf render service template header footer locale font';
    const day = 86_400_000;
    const items = [
      item({ uid: 'mem:alpha', title: 'alpha', summary: `${deploy} one`, ts: NOW - 5 * day }),
      item({ uid: 'mem:golf', title: 'golf', summary: `${deploy} two`, ts: NOW - 4 * day }),
      item({ uid: 'mem:bravo', title: 'bravo', summary: `${deploy} three`, ts: NOW - day }),
      item({ uid: 'mem:delta', kind: 'decision', title: 'delta', summary: `${db} four`, ts: NOW - 5 * day }),
      item({ uid: 'mem:echo', kind: 'decision', title: 'echo', summary: `${db} five`, ts: NOW - day }),
      item({ uid: 'mem:foxtrot', title: 'foxtrot', summary: `${pdf} six`, ts: NOW - 5 * day }),
      item({ uid: 'mem:hotel', title: 'hotel', summary: `${pdf} seven`, ts: NOW - day }),
    ];
    await writeLtm(s.base, items);
    s.ltm.put(items);
    const said: Record<string, string> = { 'bravo alpha': 'related', 'bravo golf': 'related', 'golf alpha': 'different', 'echo delta': 'contradicts', 'hotel foxtrot': 'same' };
    const asked: string[] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      const pairs = [...a.prompt.matchAll(/\[(p\d+)\]\nNEW \([^)]*\): (\S+)[\s\S]*?KNOWN \([^)]*\): (\S+)/g)].map(([, id, nw, known]) => {
        asked.push(`${nw} ${known}`);
        const relation = said[`${nw} ${known}`] ?? 'different';
        return { id, relation, title: relation === 'same' ? 'invoice pdf rendering' : '', summary: relation === 'same' ? `${pdf} six seven` : '', body: '' };
      });
      return { answer: { pairs } };
    };
    const r = await new Sleep({ ...s.base, ask, usage, statePath: join(s.dir, 'sleep.json') }).run();
    expect(asked.sort()).toEqual(Object.keys(said).sort());
    expect(r).toMatchObject({ linked: 2, superseded: 1, merged: 1 });
    expect(s.ltm.get('mem:bravo')?.related?.sort()).toEqual(['mem:alpha', 'mem:golf']);
    expect(s.ltm.get('mem:alpha')?.related).toEqual(['mem:bravo']);
    expect(s.ltm.get('mem:delta')).toMatchObject({ status: 'superseded', supersededBy: 'mem:echo' });
    expect(s.ltm.get('mem:foxtrot')).toMatchObject({ status: 'active', title: 'invoice pdf rendering' });
    expect(s.ltm.get('mem:hotel')?.status).toBe('superseded');
  });
});

describe('sleep — profiles', () => {
  // A project of 15 items: three about the brigade (on 3 of 15, under a fifth), John on all of them (too broad), and a
  // profile of the brigade asked for once — then again only when one of its items changed.
  function project(): MemoryItem[] {
    const day = 86_400_000;
    return [
      item({ uid: 'obs:join', kind: 'observation', title: 'John joined the brigade', summary: 'John joined the fire-fighting brigade', eventAt: '2023-05-08', provenance: ['c:s1:0'], concepts: ['John', 'fire-fighting brigade'], ts: NOW - 30 * day }),
      item({ uid: 'mem:drill', title: 'brigade drill', summary: 'the brigade drills every Tuesday at 7pm', provenance: ['c:s2:3'], concepts: ['John', 'Fire-fighting brigade '], ts: NOW - 20 * day }),
      item({ uid: 'obs:rescue', kind: 'observation', title: 'first rescue', summary: 'John helped rescue a family from a flood', eventAt: '2023-06-02', provenance: ['c:s3:1', 'c:s3:2'], concepts: ['John', 'fire-fighting brigade', 'flood'], ts: NOW - 10 * day }),
      // Twelve unlike each other (no words shared), so sleep merges none of them.
      ...Array.from({ length: 12 }, (_, k) => item({ uid: `mem:other${k}`, title: `topic${k}`, summary: `w${k}a w${k}b w${k}c w${k}d`, concepts: ['John'], ts: NOW - (40 + k) * day })),
    ];
  }

  it('gathers a subject enough items share into one dated profile, written again only when they change', async () => {
    const s = setup();
    const items = project();
    await writeLtm(s.base, items);
    s.ltm.put(items);
    const asked: string[] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      if (a.name !== 'memory_profile') return { answer: { pairs: [] } };
      asked.push(a.prompt);
      return { answer: { title: 'John and the fire-fighting brigade', facts: ['2023-05-08: John joined the fire-fighting brigade', '2023-06-02: John helped rescue a family from a flood'] } };
    };
    const sleep = () => new Sleep({ ...s.base, ask, usage: new UsageBook(join(s.dir, 'usage.json')), statePath: join(s.dir, 'sleep.json') }).run();
    const r = await sleep();
    expect(r).toMatchObject({ profiled: 1, unprofiled: 0 });
    // One ask, for the brigade — as most items spell it, its items oldest first with their dates — not John.
    expect(asked).toHaveLength(1);
    expect(asked[0]!.startsWith('SUBJECT: fire-fighting brigade\n\nMEMORIES:\n- (2023-05-08) John joined the brigade: John joined the fire-fighting brigade\n')).toBe(true);
    expect(asked[0]).toContain('- (2023-06-02) first rescue: John helped rescue a family from a flood');
    const profiles = [...s.ltm.entries()].filter(([, m]) => m.kind === 'profile');
    expect(profiles).toHaveLength(1);
    const profile = s.ltm.get(profiles[0]![0])!;
    expect(profile).toMatchObject({ kind: 'profile', status: 'active', title: 'John and the fire-fighting brigade', concepts: ['fire-fighting brigade'], project: '-p' });
    expect(profile.summary).toBe('2023-05-08: John joined the fire-fighting brigade\n2023-06-02: John helped rescue a family from a flood');
    expect(profile.provenance.sort()).toEqual(['c:s1:0', 'c:s2:3', 'c:s3:1', 'c:s3:2']);
    // Searchable like any item.
    const hits = await memorySearch({ client: s.client, embed }, { query: 'fire-fighting brigade flood rescue', limit: 5 });
    expect(hits.map((h) => h.id)).toContain(profile.uid);

    // Nothing changed: nothing asked. An item changed: asked again, from the items — never from the profile.
    expect(await sleep()).toMatchObject({ profiled: 0 });
    expect(asked).toHaveLength(1);
    const drill = { ...s.ltm.get('mem:drill')!, summary: 'the brigade drills every Wednesday at 7pm now' };
    await writeLtm(s.base, [drill]);
    s.ltm.put([drill]);
    expect(await sleep()).toMatchObject({ profiled: 1 });
    expect(asked).toHaveLength(2);
    expect(asked[1]).toContain('drills every Wednesday at 7pm now');
    expect(asked[1]).not.toContain('2023-06-02: John helped');

    // The subject no longer qualifies (two of its items retired): its profile is archived.
    const retired = ['obs:rescue', 'mem:drill'].map((u) => ({ ...s.ltm.get(u)!, status: 'superseded' as const }));
    await writeLtm(s.base, retired);
    s.ltm.put(retired);
    expect(await sleep()).toMatchObject({ profiled: 0, unprofiled: 1 });
    expect(s.ltm.get(profile.uid)?.status).toBe('archived');
    expect(asked).toHaveLength(2);
  });

  it('a night the fast model does not answer writes nothing, and the next night asks again', async () => {
    const s = setup();
    const items = project();
    await writeLtm(s.base, items);
    s.ltm.put(items);
    let up = false;
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      if (a.name !== 'memory_profile') return { answer: { pairs: [] } };
      return up ? { answer: { title: 'the brigade', facts: ['2023-05-08: John joined'] } } : { unavailable: 'no route' };
    };
    const sleep = () => new Sleep({ ...s.base, ask, usage: new UsageBook(join(s.dir, 'usage.json')), statePath: join(s.dir, 'sleep.json') }).run();
    expect(await sleep()).toMatchObject({ profiled: 0 });
    up = true;
    expect(await sleep()).toMatchObject({ profiled: 1 });
  });
});

describe('concept graph — exploring', () => {
  async function seeded() {
    const s = setup();
    const items = [
      item({ uid: 'mem:image', title: 'gateway config baked into the image', summary: 'the gateway config is baked into the image rebuild it after edits', concepts: ['gateway.conf', 'fg-gateway:dev'] }),
      item({ uid: 'mem:reload', title: 'nginx reload ignores host edits', summary: 'reloading inside the container keeps the old file', concepts: ['Gateway.conf', 'nginx'] }),
      item({ uid: 'mem:recreate', title: 'recreate after env changes', summary: 'restart keeps stale env use force recreate', concepts: ['docker compose'], related: ['mem:image'] }),
      item({ uid: 'mem:invoice', title: 'invoice pdf', summary: 'invoices render with the pdf service', concepts: ['pdf service'] }),
      item({ uid: 'mem:alias', title: 'gateway file location', summary: 'it lives in the project folder', concepts: ['gateway.conf file'] }),
    ];
    items[0]!.related = ['mem:recreate'];
    await writeLtm(s.base, items);
    s.ltm.put(items);
    return s;
  }

  it('reaches from what was found to what shares its concepts or is linked to it — not the unrelated', async () => {
    const s = await seeded();
    const r = await memoryExplore({ client: s.client, embed, now: () => NOW }, { from: ['mem:image'] });
    const ids = r.related.map((x) => x.id);
    expect(ids).toEqual(expect.arrayContaining(['mem:reload', 'mem:recreate']));
    expect(ids).not.toContain('mem:image');
    expect(ids).not.toContain('mem:invoice');
    expect(ids).not.toContain('mem:alias'); // another spelling, and not near: concepts match by spelling only
    expect(r.related.find((x) => x.id === 'mem:reload')?.via).toBe('shares "gateway.conf"');
    expect(r.related.find((x) => x.id === 'mem:recreate')?.via).toBe('linked to "gateway config baked into the image"');
    expect(r.concepts).toContain('gateway.conf');
  });
});

describe('recall', () => {
  function recallDeps(s: ReturnType<typeof setup>): RecallDeps {
    return { client: s.client, embed, now: () => NOW };
  }

  async function seeded() {
    const s = setup();
    const items = [
      item({ uid: 'mem:deploy', title: 'deploy through jenkins', summary: 'deploy the api by pushing main jenkins deploys it' }),
      item({ uid: 'mem:deploy2', title: 'deploy rollback', summary: 'roll back a bad api deploy from the jenkins job history' }),
      item({ uid: 'mem:old-deploy', title: 'deploy by rsync', summary: 'deploy the api by rsync to the server', status: 'superseded' }),
      item({ uid: 'mem:other', title: 'invoice pdf', summary: 'invoices render with the pdf service' }),
    ];
    await writeLtm(s.base, items);
    await writeStm(s.base, [entry('conv:here:0', 'deploy the api now', { session: 'here', ts: NOW - 3600_000 })]);
    return s;
  }







  it('records what a search showed, for what, and what was opened after — a line each, in order', async () => {
    const s = await seeded();
    const path = join(s.dir, 'recall.jsonl');
    const book = new RecallBook(path, () => NOW);
    const deps = { ...recallDeps(s), record: (e: RecallEvent) => book.record(e) };
    const found = await memorySearch(deps, { query: 'invoice pdf' });
    await memoryOpen(deps, 'mem:other');
    await memoryOpen(deps, 'mem:nothing-here');
    await book.settled();
    const lines = readFileSync(path, 'utf8').trim().split('\n').map((l) => JSON.parse(l) as Record<string, unknown>);
    expect(lines.map((l) => l.ev)).toEqual(['search', 'open']);
    expect(lines[0]).toMatchObject({ t: NOW, query: 'invoice pdf', shown: found.map((h) => h.id) });
    expect(lines[1]).toEqual({ t: NOW, ev: 'open', item: 'mem:other' });
    expect(statSync(path).mode & 0o777).toBe(0o600);
    // A long message is kept up to its start.
    book.record({ ev: 'search', query: 'x'.repeat(RECALL_QUERY_CHARS + 100), shown: [] });
    await book.settled();
    expect((JSON.parse(readFileSync(path, 'utf8').trim().split('\n').at(-1)!) as { query: string }).query).toHaveLength(RECALL_QUERY_CHARS);
  });

  it('trims the record to its last days and its newest bytes, and never loses what is appended meanwhile', async () => {
    const s = setup();
    const path = join(s.dir, 'recall.jsonl');
    let now = NOW - 200 * 86_400_000;
    const book = new RecallBook(path, () => now, { ms: 180 * 86_400_000, bytes: 400 });
    book.record({ ev: 'open', item: 'mem:ancient' });
    now = NOW;
    for (let i = 0; i < 6; i++) book.record({ ev: 'search', query: `q${i} ${'x'.repeat(60)}`, shown: [`mem:${i}`] });
    const trimming = book.trim();
    book.record({ ev: 'open', item: 'mem:after' });
    expect(await trimming).toBeGreaterThanOrEqual(2);
    await book.settled();
    const lines = readFileSync(path, 'utf8').trim().split('\n').map((l) => JSON.parse(l) as { item?: string; query?: string });
    expect(lines.some((l) => l.item === 'mem:ancient')).toBe(false);
    expect(lines.at(-1)).toMatchObject({ item: 'mem:after' });
    expect(lines.at(-2)?.query).toMatch(/^q5 /);
    expect(readFileSync(path).length).toBeLessThanOrEqual(400 + 200);
    expect(statSync(path).mode & 0o777).toBe(0o600);
    // What came in after a trim is the next trim's: then it fits, and a trim with nothing to drop leaves it be.
    await book.trim();
    expect(readFileSync(path).length).toBeLessThanOrEqual(400);
    expect(await book.trim()).toBe(0);
  });

  it('lets an agent search in its own words, this project or all', async () => {
    const s = await seeded();
    const hits = await memorySearch(recallDeps(s), { query: 'invoice pdf' });
    expect(hits[0]?.id).toBe('mem:other');
  });

  it('reorders its best 30 by the reranker, and keeps its own order when the reranker gives nothing', async () => {
    const s = await seeded();
    const plain = await memorySearch(recallDeps(s), { query: 'deploy the api' });
    // A reranker that puts the invoice first: it moves up (fused with the search's own order, 0.7 to 0.3).
    const rerank = async (_q: string, docs: string[]) => docs.map((t) => (t.includes('invoice') ? 10 : 0));
    const reranked = await memorySearch({ ...recallDeps(s), rerank }, { query: 'deploy the api' });
    expect(reranked.findIndex((h) => h.id === 'mem:other')).toBeLessThan(plain.findIndex((h) => h.id === 'mem:other'));
    expect([...reranked.map((h) => h.id)].sort()).toEqual([...plain.map((h) => h.id)].sort());
    const none = await memorySearch({ ...recallDeps(s), rerank: async () => null }, { query: 'deploy the api' });
    expect(none.map((h) => h.id)).toEqual(plain.map((h) => h.id));
  });

  it('titles a captured request too short to say anything by what it asked and what was done', async () => {
    const s = setup();
    await writeStm(s.base, [entry('conv:x:0', 'continue', { reply: 'Signed the Windows binaries before publishing them.', ts: NOW - 3600_000 })]);
    const [hit] = await memorySearch(recallDeps(s), { query: 'sign the windows binaries' });
    expect(hit?.id).toBe('stm:conv:x:0');
    expect(hit?.title).toBe('continue Signed the Windows binaries before publishing them.');
  });
});

describe('observations — every dated thing a conversation told', () => {
  const obs = (text: string, event_date: string, over: Record<string, unknown> = {}) => ({ text, about: 'Melanie', event_date, phase: 'done', requests: [1], concepts: ['Melanie', 'beach'], ...over });
  const answer = (observations: unknown[], items: unknown[] = []) => ({ via: 'endpoint' as const, answer: { segment_summary: 's', episodes: [], items, observations } });
  const deps = (s: ReturnType<typeof setup>): RecallDeps => ({ client: s.client, embed, now: () => NOW });

  it('keeps each one, dated — the same event on two days twice, the same told again once — and never merges a fact or a night into them', async () => {
    const s = setup();
    const first = [entry('conv:sess-1:0', 'we went camping')];
    s.stm.put(first);
    const camping = 'Melanie went camping with her family at the beach and roasted marshmallows';
    const fact = { kind: 'fact', title: 'camping at the beach', summary: camping, body: camping, applies_when: '', scope: 'person', requests: [1], confidence: 'stated', importance: 3, contested: false, ambiguous: false, cues: [], concepts: [] };
    let ask = async (): Promise<LlmAnswer> => answer([
      obs(camping, '2023-06-20'), obs(camping, '2023-07-10'), obs('Melanie painted a sunset over the lake', 'last summer'),
      obs('', '2023-06-20'), obs('Melanie read a book', '2023-06-01', { requests: [9] }),
    ], [fact]);
    const r = await new Consolidator({ ...s.base, ask }).run();
    expect(r.observed).toBe(3);
    const active = () => [...s.ltm.all()].filter((x) => x.status === 'active');
    const observations = () => active().filter((x) => x.kind === 'observation');
    expect(observations().map((o) => o.eventAt ?? '').sort()).toEqual(['', '2023-06-20', '2023-07-10']);
    expect(observations().every((o) => o.about === 'Melanie' && o.provenance[0] === 'conv:sess-1:0')).toBe(true);
    // The fact says the same words, and is still its own item: knowledge is never folded into a dated event.
    expect(active().filter((x) => x.kind === 'fact')).toHaveLength(1);

    // Told again on the same date: the known one, told twice. On another date it would be a new one.
    const second = [entry('conv:sess-2:0', 'remember the camping', { session: 'sess-2' })];
    s.stm.put(second);
    ask = async (): Promise<LlmAnswer> => answer([obs(camping, '2023-06-20')]);
    const again = await new Consolidator({ ...s.base, ask }).run();
    expect(again).toMatchObject({ observed: 0, merged: 1 });
    const told = observations().find((o) => o.eventAt === '2023-06-20')!;
    expect(told.seen).toBe(2);
    expect(told.provenance).toEqual(['conv:sess-1:0', 'conv:sess-2:0']);

    // A night's sleep relates and merges knowledge, never observations: the two campings stay two.
    const sleepAsk = async (): Promise<LlmAnswer> => ({ answer: { pairs: [] } });
    await new Sleep({ ...s.base, ask: sleepAsk, usage: new UsageBook(join(s.dir, 'usage.json')), statePath: join(s.dir, 'sleep.json') }).run();
    expect(observations()).toHaveLength(3);

    // memory_search finds them with the date they happened, apart from when they were told.
    const hits = await memorySearch(deps(s), { query: 'where did Melanie camp', limit: 50 });
    expect(hits.filter((h) => h.kind === 'observation').map((h) => h.eventAt ?? '').sort()).toEqual(['', '2023-06-20', '2023-07-10']);
  });

  it('keeps when what lasts began and ended, and a search by dates finds it while it went on', async () => {
    const s = setup();
    s.stm.put([entry('conv:sess-1:0', 'I have been playing drums again for about a month')]);
    const drums = 'John has been playing drums again for about a month';
    const ask = async (): Promise<LlmAnswer> => answer([
      obs(drums, '2022-03-27', { about: 'John', phase: 'ongoing', start_date: '2022-02', end_date: '' }),
      obs('John played drums in a recording studio', '2022-03-20', { about: 'John', start_date: 'a while ago', end_date: '' }),
    ]);
    await new Consolidator({ ...s.base, ask }).run();
    const kept = [...s.ltm.all()].filter((x) => x.kind === 'observation');
    expect(kept.find((o) => o.summary === drums)).toMatchObject({ eventAt: '2022-03-27', startAt: '2022-02' });
    expect(kept.find((o) => o.summary !== drums)?.startAt).toBeUndefined();
    // Asked about February: the one that was going on then first — and the other still after it (a date can be off).
    const hits = await memorySearch(deps(s), { query: 'John drums', from: '2022-02-10', to: '2022-02-20', limit: 10 });
    expect(hits.filter((h) => h.kind === 'observation').map((h) => [h.summary, h.startAt ?? ''])).toEqual([[drums, '2022-02'], ['John played drums in a recording studio', '']]);
  });

  it('tells one twice when a window says it twice (the window\'s observations are written together)', async () => {
    const s = setup();
    s.stm.put([entry('conv:sess-1:0', 'we went camping'), entry('conv:sess-1:1', 'the camping again', { turn: 1 })]);
    const camping = 'Melanie went camping with her family at the beach and roasted marshmallows';
    const ask = async (): Promise<LlmAnswer> => answer([obs(camping, '2023-06-20'), obs(camping, '2023-06-20', { requests: [2] }), obs(camping, '2023-07-10')]);
    const r = await new Consolidator({ ...s.base, ask }).run();
    expect(r).toMatchObject({ observed: 2, merged: 1 });
    const observations = [...s.ltm.all()].filter((x) => x.status === 'active' && x.kind === 'observation');
    const told = observations.find((o) => o.eventAt === '2023-06-20')!;
    expect(told.seen).toBe(2);
    expect(told.provenance).toEqual(['conv:sess-1:0', 'conv:sess-1:1']);
    expect(await s.client.count()).toBeGreaterThan(0);
  });

  it('reconciles windows at once and loses nothing of an item two of them tell again at the same time', async () => {
    const s = setup();
    const camping = 'Melanie went camping with her family at the beach and roasted marshmallows';
    const known = item({ uid: 'obs:known', kind: 'observation', title: camping, summary: camping, eventAt: '2023-06-20', scope: 'person' });
    await writeLtm(s.base, [known]);
    s.ltm.put([known]);
    s.stm.put([entry('conv:sess-1:0', 'we went camping', { session: 'sess-1' }), entry('conv:sess-2:0', 'the camping again', { session: 'sess-2' })]);
    const ask = async (): Promise<LlmAnswer> => answer([obs(camping, '2023-06-20')]);
    const r = await new Consolidator({ ...s.base, ask, reconcileConcurrency: 2 }).run();
    expect(r).toMatchObject({ windows: 2, merged: 2, observed: 0 });
    const told = s.ltm.get('obs:known')!;
    expect(told.seen).toBe(known.seen + 2);
    expect(told.provenance).toEqual(expect.arrayContaining(['conv:sess-1:0', 'conv:sess-2:0']));
  });

  it('reads a window with its calendar and with what other conversations say it refers to — never its own', async () => {
    const s = setup();
    const older = entry('conv:sess-0:0', 'I just finished Becoming Nicole by Amy Ellis Nutt, I recommend it to you', { session: 'sess-0', ts: Date.parse('2023-07-06T10:00:00Z') });
    const later = entry('conv:sess-1:0', 'I am reading the book you recommended about Nicole', { ts: Date.parse('2023-10-13T10:00:00Z') });
    s.stm.put([older, later]);
    await writeStm(s.base, [older, later]);
    const prompts: string[] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => { prompts.push(a.prompt); return answer([]); };
    await new Consolidator({ ...s.base, ask, resolve: true }).run();
    const read = prompts.find((p) => p.includes('the book you recommended'))!;
    expect(read).toContain('- Fri 2023-10-13: yesterday Thu 2023-10-12 · the days before Wed 2023-10-11, Tue 2023-10-10, Mon 2023-10-09, Sun 2023-10-08, Sat 2023-10-07, Fri 2023-10-06 · last week (the week before) 2023-10-06 to 2023-10-12 · last weekend 2023-10-07 to 2023-10-08 · this week 2023-10-09 to 2023-10-15 · last month 2023-09 · last year 2022');
    const known = read.slice(read.indexOf('KNOWN FROM OTHER CONVERSATIONS'), read.indexOf('## Request 1'));
    expect(known).toContain('(2023-07-06) I just finished Becoming Nicole');
    expect(known).not.toContain('reading the book you recommended');
  });

  it('searches within dates first — an observation by when it happened, a request by when it was made — then the rest', async () => {
    const s = setup();
    const at = (uid: string, eventAt: string, summary: string) => item({ uid, kind: 'observation', title: summary, summary, eventAt, scope: 'person', ts: Date.parse('2023-08-01T00:00:00Z') });
    const june = at('obs:june', '2023-06-19', 'Melanie went camping with her family in the mountains');
    const july = at('obs:july', '2023-07', 'Melanie went camping with her family at the beach');
    const year = at('obs:year', '2022', 'Melanie went camping with her family in the forest');
    await writeLtm(s.base, [june, july, year]);
    const req = entry('conv:sess-1:0', 'Melanie went camping with her family again', { ts: Date.parse('2023-06-28T10:00:00Z') });
    await writeStm(s.base, [req]);
    const search = (from?: string, to?: string) => memorySearch(deps(s), { query: 'Melanie camping family', limit: 20, from, to });
    // The first `n` found are the ones within the dates; every one is still found.
    const lead = async (n: number, from?: string, to?: string) => {
      const hits = await search(from, to);
      expect(hits).toHaveLength(4);
      return hits.slice(0, n).map((h) => h.id).sort();
    };
    expect(await lead(2, '2023-06', '2023-06')).toEqual(['obs:june', `stm:${req.uid}`]);
    expect(await lead(1, '2023-07-15')).toEqual(['obs:july']);
    expect(await lead(1, undefined, '2022-12-31')).toEqual(['obs:year']);
    expect(await search()).toHaveLength(4);
  });

  it('with dates, reranks within the dates and after them — the hits within them stay first', async () => {
    const s = setup();
    const june = Date.parse('2023-06-15T10:00:00Z');
    const rows = ['deploy the api now', 'deploy the api today', 'deploy the api again'].map((t, i) => entry(`conv:j${i}:0`, t, { session: `j${i}`, ts: june }));
    const older = entry('conv:o:0', 'deploy the api through jenkins safely', { session: 'o', ts: Date.parse('2022-03-01T00:00:00Z') });
    await writeStm(s.base, [...rows, older]);
    // The reranker prefers the older one above all; it still comes after every hit within June.
    const rerank = async (_q: string, docs: string[]) => docs.map((t) => (t.includes('jenkins') ? 10 : 0));
    const hits = (await memorySearch({ ...deps(s), rerank }, { query: 'deploy the api', limit: 8, from: '2023-06', to: '2023-06' })).map((h) => h.id);
    expect(hits.indexOf('stm:conv:o:0')).toBe(3);
  });

  it('shows the passage of a long request that matched — a fact said deep in it — not only its start', async () => {
    const s = setup();
    const filler = Array.from({ length: 400 }, (_, i) => `plan${i % 37} schedule${i % 23}`).join(' ');
    const said = `${filler} we celebrated my daughter birthday last night with a concert under the stars ${filler}`;
    const e = entry('conv:sess-1:0', `what should we do this weekend ${filler}`, { reply: said, said });
    await writeStm(s.base, [e]);
    const [hit] = await memorySearch(deps(s), { query: 'daughter birthday concert stars', limit: 5 });
    expect(hit?.id).toBe(`stm:${e.uid}`);
    expect(hit?.match).toContain('daughter birthday last night with a concert');
  });

  it('finds observations beside long requests in a gathering search: their chunks never take the observations\' places', async () => {
    const s = setup();
    const filler = Array.from({ length: 300 }, (_, i) => `hiking${i % 7} camping trail${i % 11}`).join(' ');
    const long = Array.from({ length: 4 }, (_, n) => entry(`conv:sess-1:${n}`, `camping hiking trail plans ${filler}`, { turn: n, said: filler }));
    await writeStm(s.base, long);
    const obs = ['went camping in the forest', 'went hiking on a mountain trail', 'went camping at the beach'].map((t, n) =>
      item({ uid: `obs:${n}`, kind: 'observation', title: `Melanie ${t}`, summary: `Melanie ${t}`, eventAt: `2023-0${n + 6}`, scope: 'person' }));
    await writeLtm(s.base, obs);
    // A search for many (past SINGLE_LIST_UP_TO) interleaves the two kinds; a search for a few ranks one fused list.
    const hits = await memorySearch(deps(s), { query: 'Melanie camping hiking trail', limit: 30 });
    expect(hits.filter((h) => h.kind === 'observation').map((h) => h.id).sort()).toEqual(['obs:0', 'obs:1', 'obs:2']);
    expect(hits.some((h) => h.id.startsWith('stm:'))).toBe(true);
  });

  it('gathers a set: what the model kept first, and what its own searches found that the first one did not', async () => {
    const s = setup();
    const said = ['bought an evening gown for $800', 'went camping at the beach', 'bought a Gucci handbag for $1,200', 'bought leather boots for $500'];
    await writeLtm(s.base, said.map((t, n) => item({ uid: `obs:${n}`, kind: 'observation', title: `Melanie ${t}`, summary: `Melanie ${t}`, eventAt: '2023-05', scope: 'person' })));
    const asked: string[] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      asked.push(a.prompt);
      const lines = a.prompt.split('\n').filter((l) => l.startsWith('[m'));
      return { answer: { keep: lines.filter((l) => l.includes('bought')).map((l) => l.slice(1, l.indexOf(']'))), searches: asked.length === 1 ? ['Melanie leather boots'] : [] } };
    };
    // A first room of two: one asked for, which a search for a few answers twice as deep in one list.
    const g = await memoryGather(deps(s), { query: 'how much did Melanie spend on luxury items' }, { ask, limit: 1 });
    expect(asked).toHaveLength(2);
    expect(g.searches).toEqual(['Melanie leather boots']);
    expect(g.kept).toContain('obs:3');
    expect(g.found.slice(0, g.kept.length).every((h) => h.summary.includes('bought'))).toBe(true);
    expect(g.found.map((h) => h.id)).toContain('obs:1');
  });

  it('writes notes on what it kept only when asked, scrubbed, and asks for none otherwise', async () => {
    const s = setup();
    await writeLtm(s.base, [item({ uid: 'obs:0', kind: 'observation', title: 'Tim read The Hobbit', summary: 'Tim read The Hobbit', eventAt: '2023-12', scope: 'person' })]);
    const asked: LlmRequest[] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      asked.push(a);
      return { answer: { keep: ['m1'], searches: [], notes: ['Books Tim has read: The Hobbit (2023-12)', ' ', 'key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD'] } };
    };
    const g = await memoryGather(deps(s), { query: 'what books has Tim read' }, { ask, notes: true });
    expect(g.notes[0]).toBe('Books Tim has read: The Hobbit (2023-12)');
    expect(g.notes).toHaveLength(2);
    expect(g.notes.join(' ')).not.toContain('sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD');
    expect(asked[0]!.system).toContain('never write that something is not said');
    expect(JSON.stringify(asked[0]!.schema)).toContain('"notes"');
    const plain = await memoryGather(deps(s), { query: 'what books has Tim read' }, { ask });
    expect(plain.notes).toEqual([]);
    expect(JSON.stringify(asked.at(-1)!.schema)).not.toContain('"notes"');
  });

  it('gathers what happened in the question\'s dates first, and still what lies outside them', async () => {
    const s = setup();
    const said = [['adopted a puppy named Toby', '2023-07'], ['adopted a second dog named Buddy', '2023-10'], ['adopted Scout, a third dog', '2023-11']];
    await writeLtm(s.base, said.map(([t, d], n) => item({ uid: `obs:${n}`, kind: 'observation', title: `Andrew ${t}`, summary: `Andrew ${t}`, eventAt: d, scope: 'person' })));
    const read: string[][] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      read.push(a.prompt.split('\n').filter((l) => l.startsWith('[m')));
      return { answer: { keep: [], searches: [] } };
    };
    const g = await memoryGather(deps(s), { query: 'how many dogs did Andrew adopt, as of November 2023', from: '2023-11', to: '2023-11' }, { ask, limit: 10 });
    expect(read[0]![0]).toContain('Scout');
    expect(g.found.map((h) => h.id).sort()).toEqual(['obs:0', 'obs:1', 'obs:2']);
  });

  it('shows a kept episode and its captured request once, not twice', async () => {
    const s = setup();
    const e = entry('conv:sess-1:0', 'rotate the gateway certificate on staging');
    await writeStm(s.base, [e]);
    const ep = item({ uid: `ep:${e.uid}`, kind: 'episode', title: 'rotate the gateway certificate on staging', summary: 'rotate the gateway certificate on staging', ts: OLD });
    await writeLtm(s.base, [ep]);
    const hits = await memorySearch(deps(s), { query: 'rotate the gateway certificate', limit: 10 });
    expect(hits.filter((h) => h.id.endsWith(e.uid))).toHaveLength(1);
  });
});
