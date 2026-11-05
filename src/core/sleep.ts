/**
 * Sleep — the daily pass that keeps long-term memory organised (LightMem's sleep-time update; MemoryOS's heat):
 *
 *   1. RELATE what was learned since the last sleep to every near neighbour (cosine >= 0.80, any knowledge kind, the
 *      same scope) — not only the nearest one consolidation looked at: from 0.93 they merge; below, the fast model
 *      merges the two, says one replaces the other (superseded), links them (the same subject, different facts) or
 *      says they differ. Measured: 0.80–0.90 held 62% duplicates, outdated items and
 *      contradictions, 38% that belong together — the 0.90 floor this pass had left them all;
 *   2. PRIORITY for every active item = importance × recency (a half-life by kind: preferences, architecture,
 *      references and notes do not fade; facts and pitfalls fade in a year; episodes in three months) × use (how
 *      often recall showed it and an agent opened it — MemRL-style utility);
 *   3. ARCHIVE what stays low and unused for half a year: out of the index, kept in the log. Nothing is forgotten
 *      outright: everything is remembered, efficiently;
 *   4. PROFILES (profiles.ts): for each subject enough items share, what they say gathered into one dated list —
 *      written again when those items changed, archived when the subject no longer qualifies.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { Consolidator, learnedAt, RELATED, relateBatches, SAME, type ConsolidateDeps, type Decision } from './consolidate.js';
import { docText } from './embedder.js';
import { allOrAbort } from '../llm.js';
import { pointId, type Filter } from './qdrant-client.js';
import { writeProfiles } from './profiles.js';
import type { MemoryItem, MemoryKind, UsageLog } from './store.js';
import { writeLtm } from './writer.js';

const DAY = 24 * 60 * 60_000;
/** A fresh item's neighbours looked at, and the pairs a night asks about (a large first sleep goes on the next nights). */
const NEIGHBOURS = 8;
export const PAIRS_PER_SLEEP = 1600;
const RELATE_IN_FLIGHT = 4;
export const ARCHIVE_BELOW = 0.02;
export const ARCHIVE_UNUSED_MS = 180 * DAY;

const HALF_LIFE: Partial<Record<MemoryKind, number>> = {
  episode: 90 * DAY, fact: 365 * DAY, pitfall: 365 * DAY,
  decision: 540 * DAY, procedure: 540 * DAY, snippet: 540 * DAY, template: 540 * DAY,
};

export function priority(it: Pick<MemoryItem, 'kind' | 'importance' | 'ts'>, use: { shown: number; opened: number }, now: number): number {
  const half = HALF_LIFE[it.kind];
  const recency = half ? 0.5 ** (Math.max(0, now - it.ts) / half) : 1;
  return (it.importance / 5) * recency * (1 + Math.log1p(use.opened) + 0.3 * Math.log1p(use.shown));
}

export interface SleepDeps extends ConsolidateDeps {
  usage: UsageLog;
  statePath: string;
  /** Where the last sleep is kept instead of `statePath` (a server running many memories: a row per memory). */
  sleepState?: { read(): Promise<{ last?: number } | null>; write(state: { last: number } & SleepReport): Promise<void> };
  /** A backfill's night: fresh items looked at this many at once (1), and relate asks in flight (RELATE_IN_FLIGHT). */
  scanAtOnce?: number;
  relateInFlight?: number;
  /** Answers applied at once (1, in order). */
  applyAtOnce?: number;
}

export interface SleepReport {
  looked: number; merged: number; superseded: number; linked: number; archived: number; profiled: number; unprofiled: number;
  /** Set when the language model did not answer (none configured, unreachable, busy): why. What it was to read waits for
   *  the next night. */
  paused?: string;
}

export class Sleep {
  constructor(private readonly d: SleepDeps) {}

  private async lastSleep(): Promise<number> {
    if (this.d.sleepState) return (await this.d.sleepState.read())?.last ?? 0;
    try { return (JSON.parse(readFileSync(this.d.statePath, 'utf8')) as { last?: number }).last ?? 0; } catch { return 0; }
  }

  async run(): Promise<SleepReport> {
    const now = (this.d.now ?? Date.now)();
    const since = await this.lastSleep();
    const r: SleepReport = { looked: 0, merged: 0, superseded: 0, linked: 0, archived: 0, profiled: 0, unprofiled: 0 };
    // 1. What was learned since the last sleep, against its near neighbours, oldest first.
    const worker = new Consolidator(this.d);
    // Observations are add-only (consolidate.ts observe): two events are never one, and an older one stays, dated. A
    // profile is written again from its items (4), never related to anything.
    const knowledge = (m: { status: string; kind: string }): boolean => m.status === 'active' && m.kind !== 'episode' && m.kind !== 'note' && m.kind !== 'observation' && m.kind !== 'profile';
    // Learned since: by when it was written, not by its source's time — history imported after a sleep is new to memory.
    const fresh = [...this.d.ltm.entries()].filter(([, m]) => knowledge(m) && learnedAt(m) >= since)
      .sort((a, b) => learnedAt(a[1]) - learnedAt(b[1])).map(([uid]) => uid);
    /** `at`: when the fresh item of the pair was learned — where the next night starts when this pair is not asked. */
    const pairs: Array<{ it: MemoryItem; known: MemoryItem; score: number; at: number }> = [];
    const seen = new Set<string>();
    let resumeAt: number | null = null;
    const look = async (it: MemoryItem): Promise<Array<{ uid: string; item: string; score: number }>> => {
      // The key vector the index already holds; embedded again only when it is not there.
      const held = (await this.d.client.retrieve([pointId(it.uid, 0)], true))[0]?.vector;
      const v = held ?? Array.from((await this.d.embed([docText(it.title, it.summary)]))[0]!);
      const must: unknown[] = [{ key: 'tier', match: { value: 'ltm' } }, { key: 'status', match: { value: 'active' } }];
      must.push(it.scope === 'project' ? { key: 'project', match: { value: it.project } } : { key: 'scope', match: { any: ['global', 'person'] } });
      const filter = { must, must_not: [{ key: 'kind', match: { any: ['episode', 'note', 'observation', 'profile'] } }] } as Filter;
      return (await this.d.client.neighbors(Array.from(v), filter, NEIGHBOURS)).map((n) => ({ uid: it.uid, item: n.item, score: n.score }));
    };
    // Oldest first, a few at once: each group's neighbours become pairs in order, and a full night stops at the next
    // group, which the next night starts from.
    const atOnce = Math.max(1, this.d.scanAtOnce ?? 1);
    for (let i = 0; i < fresh.length; i += atOnce) {
      if (this.d.aborted()) return r;
      const group = fresh.slice(i, i + atOnce).map((uid) => this.d.ltm.get(uid)).filter((it): it is MemoryItem => !!it && it.status === 'active');
      if (!group.length) continue;
      if (pairs.length >= PAIRS_PER_SLEEP) { resumeAt = learnedAt(group[0]!); break; }
      r.looked += group.length;
      for (const [k, near] of (await Promise.all(group.map(look))).entries()) {
        const it = group[k]!;
        for (const n of near) {
          const known = n.item !== it.uid && n.score >= RELATED ? this.d.ltm.get(n.item) : null;
          const key = [it.uid, n.item].sort().join(' ');
          if (!known || !knowledge(known) || seen.has(key)) continue;
          seen.add(key);
          // The newer of the two is NEW, the older KNOWN: a contradiction retires the older.
          const at = learnedAt(it);
          pairs.push(it.ts >= known.ts ? { it, known, score: n.score, at } : { it: known, known: it, score: n.score, at });
        }
      }
    }
    // The fast model reads the pairs below 0.93, several asks at once; the answers are applied in order, each to the
    // items as they are by then (one an earlier answer retired is left alone).
    const asked = pairs.filter((p) => p.score < SAME);
    const answers = new Map<(typeof pairs)[number], Decision>();
    const batches = relateBatches(asked);
    // As many asks in flight as allowed, the next one starting as soon as any is answered; one the model did not answer
    // stops the asking, and the next night starts from its pairs.
    const inFlight = Math.max(1, this.d.relateInFlight ?? RELATE_IN_FLIGHT);
    let next = 0;
    let unanswered = false;
    // A lane that throws stops the others, their asks too (allOrAbort).
    await allOrAbort(Array.from({ length: Math.min(inFlight, batches.length) }, () => async (signal: AbortSignal) => {
      while (!unanswered && next < batches.length && !this.d.aborted() && !signal.aborted) {
        const b = batches[next++]!;
        const got = await worker.relate(b, signal);
        if (!(got instanceof Map)) { unanswered = true; r.paused ??= got.paused; resumeAt = Math.min(resumeAt ?? Infinity, ...b.map((p) => p.at)); return; }
        b.forEach((p, n) => { const d = got.get(`p${n + 1}`); if (d) answers.set(p, d); });
      }
    }));
    // The answers applied — in order, one at a time, unless a backfill applies several at once — each to the two items as
    // they are by then, under their lock (one an earlier answer retired is left alone).
    const apply = async (p: (typeof pairs)[number]): Promise<void> => worker.locked([p.it.uid, p.known.uid], async () => {
      const it = this.d.ltm.get(p.it.uid);
      const known = this.d.ltm.get(p.known.uid);
      if (!it || !known || it.status !== 'active' || known.status !== 'active') return;
      const d = p.score >= SAME ? { relation: 'same' as const, title: '', summary: '', body: '' } : answers.get(p);
      if (!d || d.relation === 'different') return;
      if (d.relation === 'related') { await worker.link(it, known); r.linked++; return; }
      if (d.relation === 'contradicts') { await worker.retire(known, it.uid); r.superseded++; return; }
      await worker.merge(known, it, d.title && d.summary ? { title: d.title, summary: d.summary, body: d.body || known.body } : null);
      await worker.retire(it, known.uid);
      r.merged++;
    });
    let at = 0;
    // An answer that fails to apply stops the other lanes at their next pair.
    await allOrAbort(Array.from({ length: Math.max(1, this.d.applyAtOnce ?? 1) }, () => async (signal: AbortSignal) => {
      while (at < pairs.length && !this.d.aborted() && !signal.aborted) await apply(pairs[at++]!);
    }));
    if (this.d.aborted()) return r;
    // 2–3. Priority, and the archive.
    const retire: MemoryItem[] = [];
    for (const [uid, m] of this.d.ltm.entries()) {
      if (m.status !== 'active' || m.kind === 'note' || !HALF_LIFE[m.kind]) continue;
      const use = this.d.usage.of(uid);
      // Unused for half a year since it was learned, not since its source was written: imported history is not archived
      // the night it arrives.
      if (priority(m, use, now) >= ARCHIVE_BELOW || now - Math.max(m.ts, learnedAt(m), use.lastUsed) < ARCHIVE_UNUSED_MS) continue;
      const it = this.d.ltm.get(uid);
      if (it && now - learnedAt(it) >= ARCHIVE_UNUSED_MS) retire.push({ ...it, status: 'archived' });
    }
    for (let i = 0; i < retire.length && !this.d.aborted(); i += 50) {
      const part = retire.slice(i, i + 50);
      await writeLtm(this.d, part);
      this.d.ltm.put(part);
      r.archived += part.length;
    }
    // 4. Profiles of what is known now; one the fast model did not answer waits for the next night.
    if (this.d.aborted()) return r;
    const p = await writeProfiles(this.d);
    r.profiled = p.profiled;
    r.unprofiled = p.unprofiled;
    if (p.paused) {
      r.paused ??= p.paused;
      this.d.log(`memory sleep: profiles wait for the next night (${p.paused})`);
    }
    this.d.ltm.compact();
    // A night that stopped early (the pair cap, the fast model away) starts the next one where it stopped.
    if (this.d.sleepState) {
      await this.d.ltm.flush();
      await this.d.sleepState.write({ last: resumeAt ?? now, ...r });
      return r;
    }
    try {
      mkdirSync(dirname(this.d.statePath), { recursive: true });
      writeFileSync(this.d.statePath, JSON.stringify({ last: resumeAt ?? now, ...r }), { mode: 0o600 });
    } catch { /* the next sleep looks at a little more */ }
    return r;
  }
}
