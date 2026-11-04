/**
 * Profiles — what memory knows about one subject, gathered in one place (sleep's last step).
 *
 * A question about a subject whose facts were told over many conversations ("when did Dana join the choir, and what
 * did she sing there?") needs the dated facts of all of them, and a top 10 holds a few. For every concept enough long-term
 * items share — at least PROFILE_MIN_ITEMS of a project's items, and on at most PROFILE_BROAD of them (not the person
 * themselves, not the project's own name) — the fast model gathers what they say into one dated list of facts, word for
 * word: a `profile` item, keyed and searched like any other, whose provenance is every source it read.
 *
 * Measured on all of LoCoMo (docs/BENCHMARKS.md): the answer fully shown in the search's top 10
 * 59.1% → 61.7% (56 questions gained, 16 lost), dates 66.6% → 73.1%, 709 profiles for $0.11 of the fast model — shown
 * whole (cut at 500 characters like any item, the gain was not sure). Questions over many conversations did not move.
 *
 *   - written again only when what it read changed (a hash of those items is its fingerprint), never from a profile;
 *   - a subject that no longer qualifies has its profile archived;
 *   - never related, merged or superseded by sleep or consolidation: it is written again from its items.
 */
import { createHash } from 'node:crypto';
import { redactString } from './redact.js';
import { defaultAsk } from '../llm.js';
import { stamped, type ConsolidateDeps } from './consolidate.js';
import type { MemoryItem, MemoryScope } from './store.js';
import { conceptKey, cut } from './text.js';
import { writeLtm } from './writer.js';

export const PROFILE_MIN_ITEMS = 3;
export const PROFILE_BROAD = 0.2;
/** The items a profile reads: the newest this many, oldest first, each line cut to PROFILE_LINE. */
export const PROFILE_READ = 40;
const PROFILE_LINE = 300;
/** Profiles a night writes (a large first sleep goes on the next nights), and asks in flight. */
export const PROFILES_PER_SLEEP = 400;
const PROFILE_IN_FLIGHT = 8;
/** What memory_search and gathering show of a profile: all of it, up to this (measured p90 1,082 characters). */
export const PROFILE_SHOWN = 1500;

const PROFILE_SYSTEM = `You write a profile: everything a person's memories say about one subject, gathered in one place, so that a
later question about the subject finds all of it at once. You get the subject and the memories that mention it, each
with its date. Write:
- title: the subject and what the profile covers, at most 12 words
- facts: one line per distinct fact about the subject, oldest first, each starting with the date it happened or was
  said, as given. Keep every name, number, place, title and quote exactly as written; say who did what. Merge
  repeats into one line, keep facts that differ apart. Nothing that is not in the memories. Never include secret
  values. Never translate.`;

const PROFILE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['title', 'facts'],
  properties: { title: { type: 'string' }, facts: { type: 'array', items: { type: 'string' } } },
} as const;

export interface ProfileSubject { uid: string; concept: string; scope: MemoryScope; project: string; read: MemoryItem[]; fingerprint: string }
export interface ProfileReport { profiled: number; unprofiled: number; paused?: string }

const day = (ts: number): string => new Date(ts).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
const when = (it: MemoryItem): string => it.eventAt || day(it.ts);
const hash = (s: string): string => createHash('sha256').update(s).digest('hex');

/** The subjects worth a profile, per project: each concept (as conceptKey folds it, named as most items spell it) on at
 *  least PROFILE_MIN_ITEMS items and on at most PROFILE_BROAD of the project's `sizes` (its active items). */
export function profileSubjects(items: MemoryItem[], sizes: Map<string, number>): ProfileSubject[] {
  const on = new Map<string, { project: string; key: string; names: Map<string, number>; items: MemoryItem[] }>();
  for (const it of items) {
    if (it.status !== 'active' || it.kind === 'profile') continue;
    const named = new Map<string, string>();
    for (const c of it.concepts ?? []) { const key = conceptKey(c); if (key && !named.has(key)) named.set(key, c.trim()); }
    for (const [key, name] of named) {
      const id = `${it.project}\n${key}`;
      const s = on.get(id) ?? { project: it.project, key, names: new Map<string, number>(), items: [] };
      s.names.set(name, (s.names.get(name) ?? 0) + 1);
      s.items.push(it);
      on.set(id, s);
    }
  }
  const out: ProfileSubject[] = [];
  for (const [id, s] of on) {
    if (s.items.length < PROFILE_MIN_ITEMS || s.items.length > PROFILE_BROAD * (sizes.get(s.project) ?? 0)) continue;
    const read = [...s.items].sort((a, b) => b.ts - a.ts || a.uid.localeCompare(b.uid)).slice(0, PROFILE_READ).reverse();
    const scopes = new Map<MemoryScope, number>();
    for (const it of read) scopes.set(it.scope, (scopes.get(it.scope) ?? 0) + 1);
    out.push({
      uid: `pf:${hash(id).slice(0, 24)}`,
      concept: [...s.names].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]![0],
      scope: [...scopes].sort((a, b) => b[1] - a[1])[0]![0],
      project: s.project,
      read,
      fingerprint: hash([s.key, ...read.map((it) => `${it.uid}\n${when(it)}\n${it.title}\n${it.summary}`)].join('\n\n')),
    });
  }
  return out;
}

const promptOf = (s: ProfileSubject): string =>
  `SUBJECT: ${s.concept}\n\nMEMORIES:\n${s.read.map((it) => `- (${when(it)}) ${it.title}: ${cut(it.summary.replace(/\s+/g, ' '), PROFILE_LINE)}`).join('\n')}`;

function profileItem(s: ProfileSubject, title: string, facts: string[], now: number): MemoryItem {
  const newest = s.read[s.read.length - 1]!;
  return stamped({
    uid: s.uid, kind: 'profile', scope: s.scope, project: s.project, engine: newest.engine, session: null, cwd: newest.cwd,
    where: s.concept, ts: Math.max(...s.read.map((it) => it.ts)), firstTs: Math.min(...s.read.map((it) => it.firstTs || it.ts)),
    title: cut(title.trim() || s.concept, 120), summary: facts.join('\n'), body: '', files: [], appliesWhen: '',
    confidence: 'curated', importance: 3, status: 'active', provenance: [...new Set(s.read.flatMap((it) => it.provenance))],
    seen: 1, fingerprint: s.fingerprint, concepts: [s.concept],
  }, now);
}

/** Profiles written for the subjects whose items changed (at most `max`, several asks at once), and archived for the
 *  subjects that no longer qualify. An ask the model did not answer stops the night's profiles; the next night asks
 *  again, as their fingerprints still differ. */
export async function writeProfiles(d: ConsolidateDeps, max = PROFILES_PER_SLEEP): Promise<ProfileReport> {
  const r: ProfileReport = { profiled: 0, unprofiled: 0 };
  const sizes = new Map<string, number>();
  const mapped: MemoryItem[] = [];
  for (const [uid, m] of d.ltm.entries()) {
    if (m.status !== 'active' || m.kind === 'profile') continue;
    sizes.set(m.project, (sizes.get(m.project) ?? 0) + 1);
    const it = m.mapped ? d.ltm.get(uid) : undefined;
    if (it) mapped.push(it);
  }
  const subjects = profileSubjects(mapped, sizes);
  const wanted = new Set(subjects.map((s) => s.uid));
  const gone = [...d.ltm.entries()].filter(([uid, m]) => m.kind === 'profile' && m.status === 'active' && !wanted.has(uid))
    .map(([uid]) => d.ltm.get(uid)).filter((it): it is MemoryItem => !!it).map((it): MemoryItem => ({ ...it, status: 'archived' }));
  if (gone.length) {
    await writeLtm(d, gone);
    d.ltm.put(gone);
    r.unprofiled = gone.length;
  }
  const todo = subjects.filter((s) => { const m = d.ltm.meta(s.uid); return m?.status !== 'active' || m.fp !== s.fingerprint; }).slice(0, max);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(PROFILE_IN_FLIGHT, todo.length) }, async () => {
    while (!r.paused && next < todo.length && !d.aborted()) {
      const s = todo[next++]!;
      const a = await (d.ask ?? defaultAsk)({
        purpose: 'memory', system: PROFILE_SYSTEM, prompt: redactString(promptOf(s)), schema: PROFILE_SCHEMA as unknown as Record<string, unknown>,
        name: 'memory_profile', maxTokens: 3000, timeoutMs: 180_000, consequence: "so this subject's profile waits for the next night",
      });
      if (!('answer' in a)) { r.paused = 'unavailable' in a ? a.unavailable : 'busy'; return; }
      const answer = a.answer as { title?: unknown; facts?: unknown };
      const facts = (Array.isArray(answer.facts) ? answer.facts : []).filter((f): f is string => typeof f === 'string' && !!f.trim()).map((f) => f.trim());
      if (!facts.length) continue;
      const it = profileItem(s, typeof answer.title === 'string' ? answer.title : '', facts, (d.now ?? Date.now)());
      await writeLtm(d, [it]);
      d.ltm.put([it]);
      r.profiled++;
    }
  }));
  return r;
}
