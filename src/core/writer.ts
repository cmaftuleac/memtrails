/**
 * How memory reaches the index: points built, embedded in batches and written to Qdrant — both tiers.
 *
 *   - a record (a short-term entry): ONE key point, its start as the key, and its whole text in chunks: no model,
 *     searchable the moment it is added;
 *   - a long-term item: a key point (content title + ~150-word summary, BM25 over title, summary, body and files) and
 *     its ~96-word children (small-to-big: searched, never shown — the item is what comes back), its cues, and the keys
 *     of its concepts on the key point (the concept graph's links: graph.ts).
 */
import { basename } from 'node:path';
import { docText } from './embedder.js';
import { episodeBody } from './episodes.js';
import { pointId, type PointStruct, type QdrantClient } from './qdrant-client.js';
import type { MemoryItem, StmEntry } from './store.js';
import { children, conceptKey, cut, docSparse, firstWords } from './text.js';

export interface WriterDeps {
  client: QdrantClient;
  embed: (texts: string[], timeoutMs?: number) => Promise<Float32Array[]>;
}

/** Time for embedding this many texts: a 30 s floor, and more for a long batch (a long record is many chunks). */
export const embedBudget = (n: number): number => Math.max(30_000, n * 150);

/** A short-term entry's item id in the index (its long-term episode, if consolidation keeps it, is `ep:<uid>`). */
export const stmItem = (uid: string): string => `stm:${uid}`;

/** The raw key of a captured request: what it asked and the start of what was done — ~150 words. */
export function rawKey(e: Pick<StmEntry, 'request' | 'reply'>): string {
  return firstWords(`${e.request}\n${e.reply}`, 150);
}

/** A captured request's own title, its first line: the conversation's title says where it was, not what it is (one
 *  long conversation's requests would all read alike, and match alike). */
export function stmTitle(e: Pick<StmEntry, 'request' | 'title'>): string {
  const first = e.request.split('\n').map((l) => l.trim()).find(Boolean) ?? '';
  return cut(first, 100) || e.title;
}

export async function writeStm(d: WriterDeps, entries: StmEntry[]): Promise<void> {
  if (!entries.length) return;
  // The raw key (its start), and the whole body in chunks: every word of a captured request can be found.
  const plan = entries.map((e) => { const body = episodeBody(e); return { e, body, kids: children(body) }; });
  const texts = plan.flatMap(({ e, kids }) => [docText(stmTitle(e), rawKey(e)), ...kids.map((k) => docText(stmTitle(e), k))]);
  const vecs = await d.embed(texts, embedBudget(texts.length));
  const points: PointStruct[] = [];
  let at = 0;
  for (const { e, body, kids } of plan) {
    // The record's kind (`engine`) on every point: a kinds filter must find a record by any of its chunks.
    const common = { item: stmItem(e.uid), tier: 'stm', status: 'active', kind: 'episode', engine: e.engine, scope: 'project', project: e.project, session: e.session, ts: e.ts, synthetic: e.synthetic };
    points.push({
      id: pointId(stmItem(e.uid), 0),
      vector: { dense: Array.from(vecs[at]!), bm25: docSparse(`${stmTitle(e)}\n${body}`) },
      payload: { ...common, part: 'key', cwd: e.cwd ?? '', where: e.title, title: stmTitle(e), summary: rawKey(e), body, files: e.files, importance: 0, ...(e.record ? { record: e.record } : {}) },
    });
    kids.forEach((_, k) => points.push({ id: pointId(stmItem(e.uid), k + 1), vector: { dense: Array.from(vecs[at + 1 + k]!) }, payload: { ...common, part: 'child' } }));
    at += 1 + kids.length;
  }
  await d.client.deleteItems(entries.map((e) => stmItem(e.uid)));
  for (let i = 0; i < points.length; i += 256) await d.client.upsert(points.slice(i, i + 256), true);
}

/** A cue's point number: the k-th cue is -(k + 1) — below the key (0) and the children (1, 2, …), however many of
 *  either there are, so the two never meet. */
export const cuePoint = (k: number): number => -(k + 1);

/** The time a date stands for, as the first and last millisecond: a day, a month or a year (an observation's
 *  `eventAt`, or a date an agent searches by) — null for anything else, a month or day that does not exist too
 *  (2024-02-30 is not 1 March). */
export function dateSpan(d: string | undefined): { from: number; to: number } | null {
  const m = /^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$/.exec(d ?? '');
  if (!m) return null;
  const [y, mo, dd] = [Number(m[1]), m[2] ? Number(m[2]) - 1 : null, m[3] ? Number(m[3]) : null];
  if (mo !== null && (mo < 0 || mo > 11)) return null;
  if (mo !== null && dd !== null && (dd < 1 || new Date(Date.UTC(y, mo, dd)).getUTCMonth() !== mo)) return null;
  if (mo === null) return { from: Date.UTC(y, 0, 1), to: Date.UTC(y + 1, 0, 1) - 1 };
  if (dd === null) return { from: Date.UTC(y, mo, 1), to: Date.UTC(y, mo + 1, 1) - 1 };
  return { from: Date.UTC(y, mo, dd), to: Date.UTC(y, mo, dd + 1) - 1 };
}

/** What every point of a long-term item carries: what recall and the filters select on — for an observation, the time
 *  it happened too (memory_search's `from`/`to`): from its date, or from when what it tells began to when it ended. */
function ltmPayload(it: MemoryItem): Record<string, unknown> {
  const spans = [it.eventAt, it.startAt, it.endAt].map(dateSpan).filter((s): s is NonNullable<typeof s> => !!s);
  const span = spans.length ? { from: Math.min(...spans.map((s) => s.from)), to: Math.max(...spans.map((s) => s.to)) } : null;
  return {
    item: it.uid, tier: 'ltm', status: it.status, kind: it.kind, scope: it.scope, project: it.project,
    session: it.session ?? '', ts: it.ts, synthetic: false, ...(span ? { eventFrom: span.from, eventTo: span.to } : {}),
  };
}

/** Only the cues of these items, written again (the key and the children stay as they are). */
export async function writeCues(d: WriterDeps, items: MemoryItem[]): Promise<void> {
  const live = items.filter((it) => it.status === 'active');
  const texts = live.flatMap((it) => (it.cues ?? []).map((c) => docText(it.title, c)));
  if (!texts.length) { await d.client.deleteParts(live.map((it) => it.uid), 'cue'); return; }
  const vecs = await d.embed(texts, embedBudget(texts.length));
  const points: PointStruct[] = [];
  let n = 0;
  for (const it of live) {
    (it.cues ?? []).forEach((_, k) => points.push({ id: pointId(it.uid, cuePoint(k)), vector: { dense: Array.from(vecs[n++]!) }, payload: { ...ltmPayload(it), part: 'cue' } }));
  }
  await d.client.deleteParts(live.map((it) => it.uid), 'cue');
  for (let i = 0; i < points.length; i += 256) await d.client.upsert(points.slice(i, i + 256), true);
}

/** Embed and write long-term items: their old points out, the key point, the children and the cues in. Cues are
 *  embedded as documents titled by the item (how they were measured); the item, not the cue, is what comes back. */
/** A summary this many chunks long (~380 words) is also chunked: its key vector reads only its start. */
export const LONG_SUMMARY_CHUNKS = 5;

export async function writeLtm(d: WriterDeps, items: MemoryItem[]): Promise<void> {
  const live = items.filter((it) => it.status === 'active');
  if (!live.length) { await d.client.deleteItems(items.map((it) => it.uid)); return; }
  const texts: string[] = [];
  const plan: Array<{ it: MemoryItem; kids: string[]; cues: string[]; at: number }> = [];
  for (const it of live) {
    // Every chunk of the body — and of a summary too long for one key vector (the embedder reads its start).
    const kids = [...children(it.body), ...(children(it.summary).length >= LONG_SUMMARY_CHUNKS ? children(it.summary) : [])];
    const cues = it.cues ?? [];
    plan.push({ it, kids, cues, at: texts.length });
    texts.push(docText(it.title, it.summary), ...kids.map((k) => docText(it.title, k)), ...cues.map((c) => docText(it.title, c)));
  }
  const vecs = await d.embed(texts, embedBudget(texts.length));
  const points: PointStruct[] = [];
  for (const { it, kids, cues, at } of plan) {
    const common = ltmPayload(it);
    points.push({
      id: pointId(it.uid, 0),
      vector: { dense: Array.from(vecs[at]!), bm25: docSparse(`${it.title}\n${it.summary}\n${it.body}\n${it.files.map((f) => basename(f)).join(' ')}`) },
      payload: {
        ...common, part: 'key', engine: it.engine, cwd: it.cwd ?? '', where: it.where, title: it.title, summary: it.summary,
        body: it.body, files: it.files, importance: it.importance, appliesWhen: it.appliesWhen,
        related: it.related ?? [], concepts: conceptKeys(it.concepts),
        ...(it.eventAt ? { eventAt: it.eventAt } : {}), ...(it.startAt ? { startAt: it.startAt } : {}), ...(it.endAt ? { endAt: it.endAt } : {}),
        ...(it.phase ? { phase: it.phase } : {}), ...(it.about ? { about: it.about } : {}),
      },
    });
    kids.forEach((_, k) => points.push({ id: pointId(it.uid, k + 1), vector: { dense: Array.from(vecs[at + 1 + k]!) }, payload: { ...common, part: 'child' } }));
    cues.forEach((_, k) => points.push({ id: pointId(it.uid, cuePoint(k)), vector: { dense: Array.from(vecs[at + 1 + kids.length + k]!) }, payload: { ...common, part: 'cue' } }));
  }
  // Remote embedding/validation can fail: keep the old searchable item until every replacement vector exists.
  await d.client.deleteItems(items.map((it) => it.uid));
  for (let i = 0; i < points.length; i += 256) await d.client.upsert(points.slice(i, i + 256), true);
}

/** The keys of an item's concepts, one per concept (text.ts conceptKey). */
export function conceptKeys(names: readonly string[] | undefined): string[] {
  return [...new Set((names ?? []).map(conceptKey).filter(Boolean))];
}

/** Only the concepts of these items, written again (their keys on the key point). */
export async function writeConcepts(d: WriterDeps, items: MemoryItem[]): Promise<void> {
  for (const it of items.filter((x) => x.status === 'active')) await d.client.setPayload([it.uid], { concepts: conceptKeys(it.concepts) }, true);
}
