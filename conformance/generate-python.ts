/**
 * More of the answer key, for the Python port (any port may read it): what the TypeScript library asks a language model
 * for fixed inputs — every prompt, system text and JSON Schema, byte for byte — how the fused search adds up its
 * channels, how Date.parse reads more forms of a date, and, when a model directory is given, the vectors the local
 * embedder gives for fixed texts.
 *
 *   npx tsx conformance/generate-python.ts                                   # python-prompts.json, python-fusion.json, python-dates.json
 *   MEMTRAILS_FIXTURE_MODEL=<model dir> npx tsx conformance/generate-python.ts   # also python-embeddings.json
 *
 * Every input here is invented. No benchmark data (LoCoMo, LongMemEval) is used. Times are read in UTC.
 */
process.env.TZ = 'UTC';

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Consolidator } from '../src/core/consolidate.js';
import { extractWindow, observeWindow } from '../src/core/extract.js';
import { memoryGather } from '../src/core/gather.js';
import { loadLocalEmbedder } from '../src/core/local-embedder.js';
import { writeProfiles } from '../src/core/profiles.js';
import { pointId, QdrantClient } from '../src/core/qdrant-client.js';
import { ltmLog, stmLog, type MemoryItem, type StmEntry } from '../src/core/store.js';
import type { LlmAnswer, LlmRequest } from '../src/llm.js';
import { conversationEntries, entryOf } from '../src/records.js';

const OUT = join(import.meta.dirname, 'fixtures');
const write = (name: string, data: unknown): void => writeFileSync(join(OUT, `${name}.json`), `${JSON.stringify(data, null, 1)}\n`);

/** What a language model is asked, as a port must ask it. */
const asked: Array<Record<string, unknown>> = [];
const capture = (answer: (req: LlmRequest) => LlmAnswer) => async (req: LlmRequest): Promise<LlmAnswer> => {
  asked.push({ purpose: req.purpose, name: req.name ?? null, system: req.system, prompt: req.prompt, schema: req.schema,
    maxTokens: req.maxTokens ?? null, timeoutMs: req.timeoutMs ?? null, consequence: req.consequence ?? null });
  return answer(req);
};
const take = (): Array<Record<string, unknown>> => asked.splice(0);

const NOW = Date.UTC(2026, 3, 2, 12, 0, 0);
const conversation = conversationEntries({
  id: 'conv-7', collection: 'personal', at: '2026-03-27T09:00:00Z',
  turns: [
    { role: 'user', text: 'I have been playing the cello again for about a month, every Tuesday with Priya.' },
    { role: 'assistant', text: 'Lovely! Which piece are you learning?' },
    { role: 'user', text: 'The Swan, by Saint-Saëns. My key is sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD, keep it safe.' },
    { role: 'assistant', text: 'I will not keep keys. The Swan is a good choice.' },
    { role: 'user', text: 'Yesterday I bought a new bow at the music shop on Calea Victoriei.' },
  ],
}, NOW);
const mail = entryOf({ id: 'mail-1', kind: 'email', author: 'Ana', title: 'Proxy certificate expired', at: '2026-04-02T08:15:00Z',
  text: 'It expired on the 1st. Renewal stopped because the cron job was removed in March.\nFiles: deploy/cron/renew.sh', thread: 'cert' }, NOW);
const meta = { cwd: null, engine: 'conversation', title: 'Music lessons' };
const known = ['(2026-02-11) Priya teaches cello at the conservatory', '(2026-01-05) The music shop on Calea Victoriei sells bows'];

const nothing = (): LlmAnswer => ({ answer: { segment_summary: '', episodes: [], items: [], observations: [] } });

// ── Reading a window ───────────────────────────────────────────────────────────────────────────────────────────
await extractWindow(conversation, meta, { ask: capture(nothing), known });
await extractWindow([mail], { cwd: '/srv/ops', engine: 'email', title: 'cert' }, { ask: capture(nothing), observations: false });
await observeWindow(conversation.slice(1), conversation.slice(0, 1), meta, { ask: capture(() => ({ answer: { observations: [] } })), chars: 300, known });
const long: StmEntry = { ...mail, uid: 'long-1', request: 'Migrate every table', reply: '',
  said: Array.from({ length: 7000 }, (_, i) => `table${i} moved`).join(' '), engine: 'conversation' };
let part = 0;
await extractWindow([long], meta, { ask: capture(() => ({ answer: { segment_summary: `part ${++part}`, items: [], observations: [],
  episodes: [{ request: 1, outcome: 'success', keep: true, title: `Migration, part ${part}`, summary: `what part ${part} shows`, cues: [], concepts: [] }] } })) });
const extraction = take();

// ── Reconciling, cues, profiles ────────────────────────────────────────────────────────────────────────────────
const item = (uid: string, over: Partial<MemoryItem>): MemoryItem => ({
  uid, kind: 'fact', scope: 'project', project: 'personal', engine: 'conversation', session: 'conv-7', cwd: null, where: 'Music lessons',
  ts: Date.UTC(2026, 2, 27), firstTs: Date.UTC(2026, 2, 27), title: uid, summary: uid, body: '', files: [], appliesWhen: '',
  confidence: 'stated', importance: 3, status: 'active', provenance: ['c:1'], seen: 1, fingerprint: uid, ...over,
});
const dir = mkdtempSync(join(tmpdir(), 'memtrails-fixtures-'));
const noIndex = {
  async upsert() {}, async deleteItems() {}, async deleteParts() {}, async setPayload() {},
} as unknown as QdrantClient;
const embed = async (texts: string[]): Promise<Float32Array[]> => texts.map(() => Float32Array.from({ length: 768 }, (_, i) => (i === 0 ? 1 : 0)));
const deps = (ltm: ReturnType<typeof ltmLog>, ask: (req: LlmRequest) => Promise<LlmAnswer>) => ({
  client: noIndex, embed, stm: stmLog(join(dir, 'stm.jsonl')), ltm, log: () => {}, aborted: () => false, ask, now: () => NOW,
});

const pairs = [
  { it: item('mem:new', { title: 'Cello lessons on Tuesdays', summary: 'Lessons with Priya are on Tuesdays.', body: 'Since March 2026.', appliesWhen: 'planning the week' }),
    known: item('mem:old', { title: 'Cello lessons', summary: 'Lessons with Priya are on Mondays.', body: '' }) },
  { it: item('mem:bow', { kind: 'preference', title: 'Bow from the shop', summary: 'Buys bows at the shop on Calea Victoriei.', body: 'ü 数据 🎻' }),
    known: item('mem:shop', { title: 'Music shop', summary: 'The shop on Calea Victoriei sells strings and bows.', body: 'Open until 19:00.' }) },
];
const ltmA = ltmLog(join(dir, 'a.jsonl'));
await new Consolidator(deps(ltmA, capture(() => ({ answer: { pairs: [] } })))).relate(pairs);
const cueItems = [item('mem:c1', { title: 'Renewal cron', summary: 'deploy/cron/renew.sh renews the proxy certificate.', appliesWhen: 'the certificate expires' }),
  item('note:n', { kind: 'note', title: 'Bows', summary: 'A bow is rehaired every year.', ts: Date.UTC(2026, 2, 28) })];
const ltmB = ltmLog(join(dir, 'b.jsonl'));
ltmB.put(cueItems);
await new Consolidator(deps(ltmB, capture(() => ({ answer: { items: [] } })))).fillCues();
const ltmC = ltmLog(join(dir, 'c.jsonl'));
ltmC.put([
  item('obs:1', { kind: 'observation', title: 'Started cello', summary: 'Dana started the cello again.', eventAt: '2026-02', concepts: ['Cello', 'Dana'], ts: Date.UTC(2026, 2, 1) }),
  item('mem:2', { title: 'Lessons', summary: 'Lessons are on   Tuesdays\nwith Priya.', concepts: ['cello ', 'Priya'], ts: Date.UTC(2026, 2, 5) }),
  item('mem:3', { title: 'The Swan', summary: 'Dana learns The Swan by Saint-Saëns on the cello.', concepts: ['CELLO', 'The Swan'], ts: Date.UTC(2026, 2, 9) }),
  ...Array.from({ length: 12 }, (_, k) => item(`mem:o${k}`, { title: `topic${k}`, summary: `w${k}a w${k}b`, concepts: ['Dana'], ts: Date.UTC(2026, 1, k + 1) })),
]);
const profileItems = [...ltmC.all()];
await writeProfiles(deps(ltmC, capture(() => ({ unavailable: 'enough' }))));
const knowledge = take();

// ── Gathering ──────────────────────────────────────────────────────────────────────────────────────────────────
const found: Array<Record<string, unknown>> = [
  { item: 'obs:1', tier: 'ltm', kind: 'observation', title: 'Started cello', summary: 'Dana started the cello again.', eventAt: '2026-02', ts: Date.UTC(2026, 2, 1), project: 'personal', engine: 'conversation', where: 'w' },
  { item: 'obs:2', tier: 'ltm', kind: 'observation', title: 'Lessons', summary: 'Lessons with Priya, weekly.', startAt: '2026-02', endAt: '2026-05', ts: Date.UTC(2026, 2, 5), project: 'personal', engine: 'conversation', where: 'w' },
  { item: 'pf:1', tier: 'ltm', kind: 'profile', title: 'Dana and the cello', summary: `2026-02: started again\n${'x'.repeat(1600)}`, ts: Date.UTC(2026, 2, 9), project: 'personal', engine: 'conversation', where: 'w' },
  { item: 'stm:r1', tier: 'stm', kind: 'episode', engine: 'email', title: 'Proxy certificate expired', summary: `Proxy certificate expired ${'y'.repeat(700)}`, ts: Date.UTC(2026, 3, 2), project: 'ops', where: 'cert' },
];
const gatherClient = {
  async fusedItems() { return found.map((p, i) => ({ item: String(p.item), score: 1 - i / 10 })); },
  async retrieve(ids: string[]) { return found.filter((p) => ids.includes(pointId(String(p.item), 0))).map((p) => ({ id: pointId(String(p.item), 0), payload: p })); },
} as unknown as QdrantClient;
await memoryGather({ client: gatherClient, embed }, { query: 'Since when does Dana play the cello?' }, { ask: capture(() => ({ answer: { keep: [], searches: [] } })) });
await memoryGather({ client: gatherClient, embed }, { query: 'Since when does Dana play the cello?' }, { ask: capture(() => ({ answer: { keep: [], searches: [], notes: [] } })), notes: true });
const gathering = take();
rmSync(dir, { recursive: true, force: true });

write('python-prompts', {
  note: 'Each request the TypeScript library makes of a language model for the inputs written here, in order: the same text, byte for byte.',
  inputs: { now: NOW, conversation, mail, meta, known, long, pairs, cueItems, profileItems, found },
  extraction, knowledge, gathering,
});

// ── The fused search's arithmetic ──────────────────────────────────────────────────────────────────────────────
const groups: Record<string, Array<{ id: string; point: string; score: number }>> = {
  'dense-0': [{ id: 'a', point: 'a-1', score: 0.91 }, { id: 'b', point: 'b-0', score: 0.84 }, { id: 'c', point: 'c-3', score: 0.5 }],
  'cue-0': [{ id: 'c', point: 'c--1', score: 0.88 }, { id: 'a', point: 'a--2', score: 0.2 }],
  'dense-1': [{ id: 'b', point: 'b-2', score: 0.7 }, { id: 'd', point: 'd-0', score: 0.7 }],
  'cue-1': [],
  bm25: [{ id: 'd', point: 'd-0', score: 12.5 }, { id: 'a', point: 'a-0', score: 3.25 }, { id: 'e', point: 'e-0', score: 1 }],
};
const bodies: unknown[] = [];
const realFetch = globalThis.fetch;
globalThis.fetch = (async (_url: string, init: { body: string }) => {
  const b = JSON.parse(init.body) as { using: string; query: number[]; filter?: { must?: unknown[] } };
  bodies.push(b);
  const cue = JSON.stringify(b.filter?.must ?? []).includes('"cue"');
  const key = b.using === 'bm25' ? 'bm25' : `${cue ? 'cue' : 'dense'}-${b.query[0] === 1 ? 0 : 1}`;
  return new Response(JSON.stringify({ result: { groups: groups[key]!.map((g) => ({ id: g.id, hits: [{ id: g.point, score: g.score }] })) } }));
}) as unknown as typeof fetch;
const client = new QdrantClient(() => ({ url: 'http://qdrant.test', apiKey: 'k' }));
const query = { dense: [[1, 0], [0, 1]], sparse: { indices: [5, 9], values: [1, 1] }, filter: { must: [{ key: 'project', match: { any: ['ops'] } }] }, limit: 4 };
const fused = await client.fusedItems(query);
const denseOnly = await client.fusedItems({ dense: [[1, 0]], limit: 10 });
globalThis.fetch = realFetch;
write('python-fusion', { note: 'QdrantClient.fusedItems over the groups each channel answers: the bodies sent, in order, and the fused list.',
  groups, query, bodies, fused, denseOnly });

// ── Date.parse, more of its forms (UTC) ────────────────────────────────────────────────────────────────────────
const DATES = ['2024-02-30', 'April 2026', '2026/04/02', 'Apr 2 2026', '4/2', '4/2/26', '2026-04-02 10:30', '2026-04-02T10:30',
  'Thu, 02 Apr 2026 10:30 PM EST', '12 August 2025 3:15 pm', '2026-04-02T24:00:00Z', '2026-04-02T24:30:00Z', '+002026-04-02', '-000001-01-01',
  '2026-4-2', '  2026-04-02', '01.02.2024', 'Sat Jan 06 2024 12:00:00 GMT+0200 (EET)', '2026-04-02T10:30:00.1234567Z', '2026-04-02T10:30Z',
  '2026-04-02T10:30:00+0530', '2026-04-02T10:30:00-08:00', 'Tue, 2 Apr 2024 10:30:00 +0000 (UTC)', 'Mon Jan  1 10:00:00 2024',
  'December 17, 1995 03:24:00', '1995-12-17T03:24:00', 'Wed, 09 Aug 1995 00:00:00', '2024-01-01T00:00:00.000+00:00', '1 Jan 99',
  '1 Jan 49', '1 Jan 50', '12:30', 'noon', '2026-13-01', '2026-02-29', '2024-02-29T12:00:00Z', 'Jan 32 2024', '31 June 2024',
  '2024-06-31', '10:30 2026-04-02', '2026 April 2', 'April 2, 2026 1:02:03.4 am', '20260402', '1712050200', 'T10:30', '2026-04-02T',
  '2026-04-02T10', '2026-04-02T10:30:60Z', 'Z', 'GMT', '(comment) 2 Apr 2026', '2 Apr 2026 (comment', '2 Apr 2026 +05', '2 Apr 2026 UT',
];
write('python-dates', { note: 'Date.parse as V8 reads each, in UTC (TZ=UTC); null for NaN.', cases: DATES.map((s) => ({ s, ms: Number.isNaN(Date.parse(s)) ? null : Date.parse(s) })) });

// ── The local model's vectors (only with a model directory) ───────────────────────────────────────────────────
const modelDir = process.env.MEMTRAILS_FIXTURE_MODEL?.trim();
if (modelDir) {
  const e = await loadLocalEmbedder(modelDir, { threads: 1 });
  const texts = ['title: Proxy certificate expired | text: It expired on the 1st.', 'task: search result | query: why did the renewal stop?',
    'title: none | text: Baza de date a căzut azi-noapte — 数据库备份失败了 🚀'];
  const vectors = (await e.embed(texts)).map((v) => Array.from(v));
  const words = (await e.words(['why did the renewal stop?', 'Baza de date a căzut — 数据库'], 'query'))
    .map((w) => ({ words: w.words, dims: w.dims, head: Array.from({ length: w.words }, (_, k) => Array.from(w.data.subarray(k * w.dims, k * w.dims + 32))) }));
  await e.close();
  write('python-embeddings', { note: 'EmbeddingGemma-300m q4 through the local embedder: unit vectors, and a vector a word (its first 32 numbers each).', texts, vectors,
    words: { texts: ['why did the renewal stop?', 'Baza de date a căzut — 数据库'], kind: 'query', vectors: words } });
}

console.log(`fixtures written to ${OUT}`);
