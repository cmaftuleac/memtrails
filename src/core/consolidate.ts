/**
 * Consolidation — LONG-TERM memory. Like the brain: short-term memory keeps everything as it came; long-term memory
 * filters, stores, reorganises and reprioritises.
 *
 * It reads the queue (the records log) a thread at a time — one that has settled (quiet for an hour, or a day old),
 * newest first — in windows of whole records (extract.ts: one language-model ask a window). What survives the gate is
 * RECONCILED with what long-term memory already holds before it is written, so nothing is stored twice:
 *
 *   - cosine >= 0.93 to a known item (0.97 for episodes) → the same memory: its provenance and `seen` grow, it is
 *     refreshed — no new item;
 *   - 0.80–0.93 → one fast-model decision a batch: same / updates / contradicts / related / different → merge,
 *     rewrite, supersede (the old one stays in the log, out of recall), add and link the two, or add;
 *   - below → added.
 *
 * Then the window's entries are marked done and leave the short-term index: what mattered of them is long-term now.
 */
import { redactString } from './redact.js';
import { defaultAsk } from '../llm.js';
import { episodeBody } from './episodes.js';
import { docText } from './embedder.js';
import { cleanConcepts, cleanCues, extractWindow, observeWindow, windowsOf, type Ask, type Candidate, type Extraction, type Observation } from './extract.js';
import type { Filter } from './qdrant-client.js';
import { memorySearch } from './recall.js';
import { fingerprint, type LtmStore, type MemoryItem, type StmEntry, type StmStore } from './store.js';
import { conceptKey, cut } from './text.js';
import { embedBudget, writeConcepts, writeCues, writeLtm, type WriterDeps } from './writer.js';

export const SAME = 0.93;
export const SAME_EPISODE = 0.97;
/** An observation is the same only when it says the same thing (this close) about the same date: two dinners on two
 *  days are two observations, and a list question needs both. */
export const SAME_OBSERVATION = 0.97;
export const RELATED = 0.80;
/** A conversation is consolidated once it has been quiet this long — or its oldest waiting entry is a day old. */
export const SETTLED_MS = 60 * 60_000;
/** Extraction calls in flight: they are independent, and one at a time made a backfill take days (~86 s a window). */
export const EXTRACT_CONCURRENCY = 8;
const DAY_MS = 24 * 60 * 60_000;

export interface ConsolidateDeps extends WriterDeps {
  stm: StmStore;
  ltm: LtmStore;
  log: (line: string) => void;
  aborted: () => boolean;
  ask?: Ask;
  now?: () => number;
  /** Windows the fast model extracts at once (EXTRACT_CONCURRENCY when not given). */
  concurrency?: number;
  /** Windows reconciled at once (1, in order, when not given). A backfill sets more: a window may then not see what one
   *  reconciled beside it wrote — the duplicates that leaves are the sleep pass's to merge — and an item two windows
   *  touch at once is changed under its lock, from what it is by then. */
  reconcileConcurrency?: number;
  /** A window's size in characters (WINDOW_CHARS when not given). */
  windowChars?: number;
  /** Observations read apart, in windows of this many characters (observeWindow); absent: in the window's one ask. */
  observeChars?: number;
  /** Each window read with what other conversations say that it may refer to (knownFor): "the book you recommended"
   *  written as the book's name. */
  resolve?: boolean;
  /** What a window's reading may keep, checked before anything is reconciled or written (e.g. facts only
   *  from the person's own words and results the agent confirmed, each with where it came from). An answer it cannot
   *  give leaves the window for the next run, as an extraction that was not answered does. */
  guard?: (win: StmEntry[], x: Extraction) => Promise<Extraction | { unavailable: string } | { busy: true }>;
}

/** Requests of a window searched for, and the passages of other conversations kept to name what it refers to. */
const RESOLVE_QUERIES = 4;
const RESOLVE_KEPT = 8;

export interface ConsolidationReport { windows: number; added: number; merged: number; updated: number; superseded: number; linked: number; dropped: number; episodes: number; observed: number; cued: number; paused?: string }

/** Items given their cues and concepts after a run's windows (the backlog: saved notes, what was learned before
 *  them), and how many go in one ask. */
export const CUE_BACKLOG = 40;
const CUE_BATCH = 10;

const CUE_SYSTEM = `You index a person's long-term memory of their work with AI agents. For EACH item:
- cues: 4 to 8 requests the person might type to an agent LATER, in other work, when this item would be needed. Vary
  them: short and long, direct and indirect (describe the task, not the memory), never copying the title. Write them in
  the language the item is written in — when that is not English, make one or two of them English.
- concepts: the 3 to 12 most specific things it is about — files, folders, repositories, services, servers, tools,
  commands, APIs, libraries, models, products, settings, error names, people or organisations, project-specific terms
  — named the way the item names them (other terms in the item's language, never translated). No generic words.
Keep code, identifiers, paths and commands exactly as written. Return every id with its cues and concepts.`;

const CUE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['items'],
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['id', 'cues', 'concepts'],
        properties: { id: { type: 'string' }, cues: { type: 'array', items: { type: 'string' } }, concepts: { type: 'array', items: { type: 'string' } } },
      },
    },
  },
} as const;

export type Relation = 'same' | 'updates' | 'contradicts' | 'related' | 'different';
export type Decision = { relation: Relation; title: string; summary: string; body: string };
const RELATIONS: readonly string[] = ['same', 'updates', 'contradicts', 'related', 'different'];
/** Links an item keeps to others about the same subject (exploration walks them), newest last. */
export const MAX_LINKS = 24;

const RELATE_SYSTEM = `You keep a person's long-term memory free of duplicates and stale knowledge without ever losing what it
knows. For EACH pair, compare the NEW memory with the KNOWN one and answer:
- same: they say the same thing (one may be more detailed) — give the merged version
- updates: NEW refines or extends KNOWN without contradicting it — give the merged version
- contradicts: NEW replaces KNOWN — the text shows KNOWN stopped being true (a decision reversed, a value changed, a
  thing removed or renamed)
- related: the same subject or system, different facts that belong together — reading one, you would want the other
- different: they are about different things
Rules:
- Different numbers, dates, versions, names, paths, commands or qualifiers mean they are NOT the same: updates when one
  extends the other, related otherwise.
- contradicts needs explicit evidence in the texts that KNOWN was replaced or ended; being newer is not evidence. When
  unsure between contradicts and related, answer related.
- For same and updates, write title (at most 12 words), summary (about 150 words, standing alone) and body. The merged
  body keeps EVERY fact of both — every value, path, command, error, code, reason and caveat, exactly as written —
  never shortened or dropped to be brief. For contradicts, related and different leave them empty.
- Never include secret values. Never translate.`;

const RELATE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pairs'],
  properties: {
    pairs: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['id', 'relation', 'title', 'summary', 'body'],
        properties: {
          id: { type: 'string' }, relation: { type: 'string', enum: RELATIONS },
          title: { type: 'string' }, summary: { type: 'string' }, body: { type: 'string' },
        },
      },
    },
  },
} as const;

/** Pairs asked about together: at most 8, and at most RELATE_CHARS of their texts — a long pair on its own. */
export const RELATE_CHARS = 60_000;
export function relateBatches<T extends { it: MemoryItem; known: MemoryItem }>(pairs: T[]): T[][] {
  const size = (p: T): number => p.it.summary.length + p.it.body.length + p.known.summary.length + p.known.body.length;
  const out: T[][] = [];
  let cur: T[] = [];
  let chars = 0;
  for (const p of pairs) {
    if (cur.length && (cur.length >= 8 || chars + size(p) > RELATE_CHARS)) { out.push(cur); cur = []; chars = 0; }
    cur.push(p);
    chars += size(p);
  }
  if (cur.length) out.push(cur);
  return out;
}

/** When an item was written to long-term memory. `ts` is its source's own time — for history imported today, long ago —
 *  so "learned since the last sleep" and "unused for half a year" read this. An item written before the field existed
 *  has only `ts`. */
export const learnedAt = (m: { ts: number; learnedAt?: number }): number => m.learnedAt ?? m.ts;
export const stamped = (it: MemoryItem, at: number): MemoryItem => ({ ...it, learnedAt: at });

/** The number of a label the model wrote back: `p3`, or the line it labels quoted whole (`[p3] NEW …`) — 0 for none. */
export function labelNo(id: string, letter: 'p' | 'i'): number {
  return Number(new RegExp(`\\b${letter}(\\d+)`).exec(id)?.[1] ?? 0);
}

/** One remembered episode or candidate as a long-term item (not yet reconciled). */
function itemOf(c: Candidate, src: StmEntry, now: number, sources: StmEntry[] = [src]): MemoryItem {
  const project = c.scope === 'project' ? src.project : c.scope;
  // The files its requests changed, from their recorded steps — by code, not the model: paths are what summaries lose.
  const files = [...new Set(sources.flatMap((e) => e.files))];
  return stamped({
    uid: `mem:${fingerprint(c.kind, c.scope, project, c.title, c.summary)}`, kind: c.kind, scope: c.scope, project: src.project,
    engine: src.engine, session: src.session, cwd: src.cwd, where: src.title, ts: src.ts || now, firstTs: src.ts || now,
    title: c.title, summary: c.summary, body: c.body, files, appliesWhen: c.appliesWhen, confidence: c.confidence,
    importance: c.importance, status: 'active', provenance: c.provenance, seen: 1,
    fingerprint: fingerprint(c.kind, c.title, c.summary, c.body), cues: c.cues, concepts: c.concepts,
    ...(c.origin ? { origin: c.origin } : {}),
  }, now);
}

function episodeItem(e: StmEntry, key: { title: string; summary: string; cues?: string[]; concepts?: string[]; outcome?: MemoryItem['outcome']; origin?: MemoryItem['origin'] }, now: number): MemoryItem {
  return stamped({
    uid: `ep:${e.uid}`, kind: 'episode', scope: 'project', project: e.project, engine: e.engine, session: e.session, cwd: e.cwd,
    where: e.title, ts: e.ts, firstTs: e.ts, title: key.title || e.title, summary: key.summary, body: episodeBody(e), files: e.files,
    appliesWhen: '', confidence: 'observed', importance: e.files.length ? 3 : 2, status: 'active', provenance: [e.uid], seen: 1,
    fingerprint: fingerprint(e.request, e.reply, e.files), cues: key.cues ?? [], concepts: key.concepts ?? [],
    ...(key.outcome ? { outcome: key.outcome } : {}), ...(key.origin ? { origin: key.origin } : {}),
  }, now);
}

/** An observation as a long-term item: its text is its key, its date beside when it was said. */
function observationItem(o: Observation, src: StmEntry, now: number): MemoryItem {
  return stamped({
    uid: `obs:${fingerprint(o.text, o.eventAt, o.provenance[0] ?? src.uid)}`, kind: 'observation', scope: 'person', project: src.project,
    engine: src.engine, session: src.session, cwd: src.cwd, where: src.title, ts: src.ts, firstTs: src.ts,
    title: cut(o.text, 100), summary: o.text, body: '', files: [], appliesWhen: '', confidence: 'stated', importance: 2,
    status: 'active', provenance: o.provenance, seen: 1, fingerprint: fingerprint('observation', o.text, o.eventAt),
    cues: o.cues ?? [], concepts: o.concepts, ...(o.eventAt ? { eventAt: o.eventAt } : {}),
    ...(o.startAt ? { startAt: o.startAt } : {}), ...(o.endAt ? { endAt: o.endAt } : {}), phase: o.phase, ...(o.about ? { about: o.about } : {}),
    ...(o.origin ? { origin: o.origin } : {}),
  }, now);
}

function cosine(a: Float32Array, b: Float32Array): number {
  let ab = 0; let aa = 0; let bb = 0;
  for (let i = 0; i < a.length; i++) { ab += a[i]! * b[i]!; aa += a[i]! * a[i]!; bb += b[i]! * b[i]!; }
  return aa && bb ? ab / Math.sqrt(aa * bb) : 0;
}

/** Where an item's duplicates can be: its project for project knowledge, anywhere for the global and the personal. */
function scopeFilter(it: MemoryItem): Filter {
  const must: unknown[] = [{ key: 'tier', match: { value: 'ltm' } }, { key: 'status', match: { value: 'active' } }];
  if (it.scope === 'project') must.push({ key: 'project', match: { value: it.project } });
  else must.push({ key: 'scope', match: { any: ['global', 'person'] } });
  // Observations are reconciled among themselves (observe): a fact is never merged into a dated event — nor into a
  // profile, which is written again from its items (profiles.ts).
  return { must, must_not: [{ key: 'kind', match: { any: ['observation', 'profile'] } }] };
}

export class Consolidator {
  private readonly vecs = new Map<string, Float32Array>();
  /** Windows whose extraction failed, by their first request: how many times in a row. */
  private readonly failed = new Map<string, number>();
  /** Windows read in smaller parts from now on, by their first request: at most this many requests a part. */
  private readonly narrowed = new Map<string, number>();
  /** An item changed by one window at a time (windows reconciled at once): the last holder's turn, by uid. */
  private readonly locks = new Map<string, Promise<void>>();

  constructor(private readonly d: ConsolidateDeps) {}

  /** Runs with the items locked — taken in uid order, so two windows never wait on each other (sleep's answers too). */
  async locked<T>(uids: string[], run: () => Promise<T>): Promise<T> {
    const release: Array<() => void> = [];
    for (const uid of [...new Set(uids)].sort()) {
      const before = this.locks.get(uid) ?? Promise.resolve();
      let done!: () => void;
      const gate = new Promise<void>((ok) => { done = ok; });
      const mine = before.then(() => gate);
      this.locks.set(uid, mine);
      await before;
      release.push(() => { done(); if (this.locks.get(uid) === mine) this.locks.delete(uid); });
    }
    try { return await run(); } finally { for (const r of release) r(); }
  }

  /** The item as it is now, or null when it stopped being active (merged or superseded by a window beside this one). */
  private latest(it: MemoryItem): MemoryItem | null {
    const now = this.d.ltm.get(it.uid);
    return now && now.status === 'active' ? now : null;
  }

  /** Embed with a per-run cache: the key is embedded once for reconcile and reused when it is written. */
  private readonly embed = async (texts: string[], timeoutMs?: number): Promise<Float32Array[]> => {
    const missing = [...new Set(texts.filter((t) => !this.vecs.has(t)))];
    if (missing.length) (await this.d.embed(missing, timeoutMs)).forEach((v, i) => this.vecs.set(missing[i]!, v));
    return texts.map((t) => this.vecs.get(t)!);
  };

  private get writer(): WriterDeps { return { client: this.d.client, embed: this.embed }; }

  /** Conversations with requests waiting, settled, newest first. */
  pending(now = (this.d.now ?? Date.now)()): Array<{ session: string; uids: string[]; last: number }> {
    const by = new Map<string, { uids: string[]; first: number; last: number }>();
    for (const [uid, m] of this.d.stm.entries()) {
      if (m.done || m.synthetic) continue;
      const s = by.get(m.session) ?? { uids: [], first: Infinity, last: 0 };
      s.uids.push(uid);
      s.first = Math.min(s.first, m.ts);
      s.last = Math.max(s.last, m.ts);
      by.set(m.session, s);
    }
    // Quiet for an hour, or waiting a day since its oldest unread entry: a thread that never goes quiet (a message every
    // half hour) is still read.
    return [...by.entries()].filter(([, s]) => now - s.last >= SETTLED_MS || now - s.first >= DAY_MS)
      .map(([session, { uids, last }]) => ({ session, uids, last }))
      .sort((a, b) => b.last - a.last);
  }

  /** Consolidate up to `maxWindows` windows. Stops early when the model pauses or the worker is stopped.
   *  The fast model extracts several windows at once (they are independent); each is reconciled in order, newest
   *  conversation first, so a later item still sees what an earlier one wrote. */
  async run(maxWindows = 20): Promise<ConsolidationReport> {
    const r: ConsolidationReport = { windows: 0, added: 0, merged: 0, updated: 0, superseded: 0, linked: 0, dropped: 0, episodes: 0, observed: 0, cued: 0 };
    const wins: StmEntry[][] = [];
    for (const conv of this.pending()) {
      const entries = conv.uids.map((u) => this.d.stm.get(u)).filter((e): e is StmEntry => !!e).sort((a, b) => a.turn - b.turn);
      for (const win of windowsOf(entries, this.d.windowChars).flatMap((w) => this.narrow(w))) if (wins.length < maxWindows) wins.push(win);
      if (wins.length >= maxWindows) break;
    }
    const limit = Math.max(1, this.d.concurrency ?? EXTRACT_CONCURRENCY);
    const extracting: Array<ReturnType<Consolidator['read']> | undefined> = [];
    // A pool: the next window starts when ANY one is read, not when the oldest is — one slow answer (measured: an
    // answer that ran on for 19,000 lines) held every other place idle while the windows after it waited their turn.
    // They are still reconciled in order.
    let next = 0;
    let reading = 0;
    let stopped = false;
    const launch = (): void => {
      while (!stopped && reading < limit && next < wins.length && !this.d.aborted()) {
        const win = wins[next]!;
        const head = win[0]!;
        const p = this.read(win, { cwd: head.cwd, engine: head.engine, title: head.title });
        reading++;
        // One left behind by a pause or a stop is not an unhandled rejection; the awaited one still throws.
        p.then(() => { reading--; launch(); }, () => { reading--; });
        extracting[next++] = p;
      }
    };
    launch();
    const together = Math.max(1, Math.min(limit, this.d.reconcileConcurrency ?? 1));
    /** Window i, once read: reconciled, observed, marked done — or 'stop' when the run pauses. */
    const settle = async (i: number): Promise<'stop' | undefined> => {
      const x = await extracting[i]!.catch((err: unknown) => { stopped = true; throw err; });
      const win = wins[i]!;
      const head = win[0]!;
      if ('unavailable' in x || 'busy' in x) {
        // A window that failed before fails for itself (its answer too long to finish, say), not because the model is
        // away: it is read in halves from now on, and the windows after it go on — one window must never hold up the
        // rest, and none is marked done unread. A first failure before any window was read pauses the run, as when the
        // model is unreachable; after others were read it is the window's own, and it waits for the next run — the
        // windows already read after it are not thrown away.
        const times = 'busy' in x ? 0 : (this.failed.get(head.uid) ?? 0) + 1;
        if (times < 2) {
          if (times) this.failed.set(head.uid, times);
          if (r.windows) return undefined;
          stopped = true;
          r.paused = 'busy' in x ? 'the fast model is busy' : x.unavailable;
          return 'stop';
        }
        this.failed.delete(head.uid);
        if (win.length > 1) this.narrowed.set(head.uid, Math.ceil(win.length / 2));
        this.d.log(`[memory] a window of ${win.length} requests failed again (${'unavailable' in x ? x.unavailable : 'busy'}); ${win.length > 1 ? 'reading it in halves' : 'it waits for the next run'}`);
        return undefined;
      }
      this.failed.delete(head.uid);
      const now = (this.d.now ?? Date.now)();
      const byUid = new Map(win.map((e) => [e.uid, e]));
      // A request the window's typed memories already came from is not kept again as an episode: measured on the full
      // backfill, leaving those out took 4,631 records to 2,782 and raised hit@5.
      const learned = new Set(x.candidates.flatMap((c) => c.provenance));
      const kept = x.episodes.filter((k) => byUid.has(k.uid) && !learned.has(k.uid));
      const fresh: MemoryItem[] = [
        ...kept.map((k) => episodeItem(byUid.get(k.uid)!, k, now)),
        ...x.candidates.map((c) => itemOf(c, byUid.get(c.provenance.at(-1) ?? '') ?? head, now, c.provenance.flatMap((u) => byUid.get(u) ?? []))),
      ];
      await this.reconcile(fresh, r);
      await this.observe(x.observations.map((o) => observationItem(o, byUid.get(o.provenance.at(-1) ?? '') ?? head, now)), r);
      r.dropped += x.dropped;
      r.episodes += kept.length;
      // Read: marked done in the queue. The requests themselves stay searchable — the raw record is the truth and the
      // memories learned from them are keys to it; dropping them lost every fact the extraction did not keep (measured on
      // LongMemEval: 47 of 60 misses were facts said in passing inside a request). Each as it is now, not as it was read
      // before the model's long answer: one removed meanwhile stays removed, one changed meanwhile is read again.
      this.d.stm.put(win.flatMap((e) => {
        if (this.d.stm.meta(e.uid)?.fp !== e.fingerprint) return [];
        const cur = this.d.stm.get(e.uid);
        return cur ? [{ ...cur, done: true }] : [];
      }));
      r.windows++;
      // One at a time, each window's keys are dropped after it; windows at once share the cache until the run ends.
      if (together === 1) this.vecs.clear();
      return undefined;
    };
    let at = 0;
    let halt = false;
    const lane = async (): Promise<void> => {
      while (!halt && at < wins.length && extracting[at] && !this.d.aborted()) {
        try { if (await settle(at++) === 'stop') halt = true; } catch (err) { halt = true; throw err; }
      }
    };
    await Promise.all(Array.from({ length: together }, lane));
    if (halt && r.paused !== undefined) return r;
    this.vecs.clear();
    if (!this.d.aborted()) {
      const c = await this.fillCues(CUE_BACKLOG);
      r.cued = c.cued;
      if (c.paused) r.paused = c.paused;
    }
    this.d.ltm.compact();
    this.d.stm.compact();
    return r;
  }

  /** One window's reading, and what the guard (when there is one) lets it keep. */
  private async read(win: StmEntry[], meta: { cwd: string | null; engine: string; title: string }): Promise<Extraction | { unavailable: string } | { busy: true }> {
    const x = await this.extract(win, meta);
    return this.d.guard && 'candidates' in x ? this.d.guard(win, x) : x;
  }

  /** One window's extraction: the one ask — and, when observations are read apart, the window's small windows too, each
   *  with the one before it as context. Any part not answered leaves the whole window for the next run. */
  private async extract(win: StmEntry[], meta: { cwd: string | null; engine: string; title: string }): Promise<Extraction | { unavailable: string } | { busy: true }> {
    const chars = this.d.observeChars;
    const known = this.d.resolve ? await this.knownFor(win) : undefined;
    if (!chars) return extractWindow(win, meta, { ask: this.d.ask, known });
    const subs = windowsOf(win, chars);
    const [x, ...seen] = await Promise.all([
      extractWindow(win, meta, { ask: this.d.ask, observations: false, known }),
      ...subs.map((sub, k) => observeWindow(sub, k ? subs[k - 1]! : [], meta, { ask: this.d.ask, chars, known })),
    ]);
    if (!('candidates' in x!)) return x!;
    const failed = seen.find((o) => !('observations' in o));
    if (failed) return failed as { unavailable: string } | { busy: true };
    return { ...x!, observations: seen.flatMap((o) => ('observations' in o ? o.observations : [])) };
  }

  /** What other conversations say that this window may refer to: a few of its requests searched for (memory_search —
   *  the raw requests of every conversation, captured before any is consolidated, and what was learned), the best
   *  passages outside this conversation, dated. The order conversations are consolidated in does not matter: the raw
   *  record holds them all. Nothing found, or a search that fails, reads the window without it. */
  private async knownFor(win: StmEntry[]): Promise<string[]> {
    const step = Math.max(1, Math.ceil(win.length / RESOLVE_QUERIES));
    const asks = win.filter((_, i) => i % step === 0).slice(0, RESOLVE_QUERIES).map((e) => cut(e.request, 600)).filter(Boolean);
    const session = win[0]!.session;
    const deps = { client: this.d.client, embed: this.d.embed, now: this.d.now };
    try {
      const found = (await Promise.all(asks.map((query) => memorySearch(deps, { query, limit: 6 })))).flat();
      const best = new Map<string, (typeof found)[number]>();
      for (const h of found) {
        const uid = h.id.startsWith('stm:') ? h.id.slice(4) : h.id.startsWith('ep:') ? h.id.slice(3) : null;
        if (uid && this.d.stm.meta(uid)?.session === session) continue;
        if ((best.get(h.id)?.score ?? -1) < h.score) best.set(h.id, h);
      }
      return [...best.values()].sort((a, b) => b.score - a.score).slice(0, RESOLVE_KEPT)
        .map((h) => `(${h.eventAt ?? new Date(h.ts).toISOString().slice(0, 10)}) ${cut(h.match ?? h.summary, 400)}`);
    } catch { return []; }
  }

  /** A window as it is read now: in parts of at most the size its earlier failures narrowed it to, each part narrowed
   *  again by its own failures. */
  private narrow(win: StmEntry[]): StmEntry[][] {
    const n = this.narrowed.get(win[0]!.uid);
    if (!n || n >= win.length) return [win];
    const parts: StmEntry[][] = [];
    for (let i = 0; i < win.length; i += n) parts.push(...this.narrow(win.slice(i, i + n)));
    return parts;
  }

  /** Encoding's backlog: active items without cues or concepts — saved notes (they come in curated, without the
   *  consolidation call) and what was learned before either existed — get them, newest first, a few an ask, several
   *  asks at once. Cues an item already has stay. */
  async fillCues(max = CUE_BACKLOG): Promise<{ cued: number; paused?: string }> {
    // An observation is its own key (a sentence the person's question matches), with the concepts it was extracted with;
    // a profile has its subject as its concept, and no cues (as measured).
    const todo = [...this.d.ltm.entries()].filter(([, m]) => m.status === 'active' && m.kind !== 'observation' && m.kind !== 'profile' && (!m.cued || !m.mapped))
      .sort((a, b) => b[1].ts - a[1].ts).slice(0, max)
      .map(([uid]) => this.d.ltm.get(uid)).filter((x): x is MemoryItem => !!x);
    const batches: MemoryItem[][] = [];
    for (let i = 0; i < todo.length; i += CUE_BATCH) batches.push(todo.slice(i, i + CUE_BATCH));
    const limit = Math.max(1, this.d.concurrency ?? EXTRACT_CONCURRENCY);
    let cued = 0;
    for (let i = 0; i < batches.length; i += limit) {
      if (this.d.aborted()) break;
      const answers = await Promise.all(batches.slice(i, i + limit).map((b) => this.cueBatch(b)));
      const paused = answers.find((a): a is { paused: string } => 'paused' in a);
      const got = answers.flatMap((a) => ('items' in a ? a.items : []));
      if (got.length) {
        const recued = got.filter((it) => !this.d.ltm.meta(it.uid)?.cued);
        if (recued.length) await writeCues(this.writer, recued);
        await writeConcepts(this.writer, got);
        this.d.ltm.put(got);
        cued += got.length;
      }
      if (paused) return { cued, paused: paused.paused };
    }
    return { cued };
  }

  private async cueBatch(items: MemoryItem[]): Promise<{ items: MemoryItem[] } | { paused: string }> {
    const prompt = items.map((it, n) => `[i${n + 1}] (${it.kind}) ${it.title}\n${it.summary}${it.appliesWhen ? `\nNeeded when: ${it.appliesWhen}` : ''}`).join('\n\n');
    const a = await (this.d.ask ?? defaultAsk)({
      purpose: 'memory', system: CUE_SYSTEM, prompt: redactString(prompt), schema: CUE_SCHEMA as unknown as Record<string, unknown>,
      name: 'memory_cues', maxTokens: 550 * items.length + 200, timeoutMs: 120_000, consequence: 'so these memories wait for their cues',
    });
    if ('unavailable' in a) return { paused: a.unavailable };
    if ('busy' in a) return { paused: 'the fast model is busy' };
    const out: MemoryItem[] = [];
    for (const x of (a.answer as { items?: Array<{ id?: string; cues?: unknown; concepts?: unknown }> }).items ?? []) {
      const it = typeof x.id === 'string' ? items[labelNo(x.id, 'i') - 1] : undefined;
      const cues = it?.cues?.length ? it.cues : cleanCues(x.cues);
      const concepts = it?.concepts?.length ? it.concepts : cleanConcepts(x.concepts);
      if (it && (cues.length || concepts.length)) out.push({ ...it, cues, concepts }); // what is missing is asked again next time
    }
    return { items: out };
  }

  /** Each item against what long-term memory holds — written as it goes, so a later item sees an earlier one. */
  private async reconcile(items: MemoryItem[], r: ConsolidationReport): Promise<void> {
    const ask: Array<{ it: MemoryItem; known: MemoryItem }> = [];
    // Every key of the window in one embedding call (the cache serves each below and again when it is written).
    await this.embed(items.map((it) => docText(it.title, it.summary)), embedBudget(items.length));
    for (const it of items) {
      const [v] = await this.embed([docText(it.title, it.summary)]);
      const near = await this.d.client.neighbors(Array.from(v!), scopeFilter(it), 3);
      const top = near.find((n) => n.item !== it.uid);
      const known = top ? this.d.ltm.get(top.item) : null;
      const same = it.kind === 'episode' ? SAME_EPISODE : SAME;
      if (known && top!.score >= same) {
        const merged = await this.locked([known.uid], async () => {
          const now = this.latest(known);
          if (now) await this.merge(now, it, null);
          return !!now;
        });
        if (merged) { r.merged++; continue; }
      } else if (known && top!.score >= RELATED && it.kind !== 'episode' && known.kind !== 'episode') { ask.push({ it, known }); continue; }
      await this.add(it);
      r.added++;
    }
    // The pairs are fixed before any is asked about: every batch asked at once, the answers applied in order.
    const batches = relateBatches(ask);
    const answers = await Promise.all(batches.map((b) => this.relate(b)));
    for (const [b, batch] of batches.entries()) {
      const got = answers[b];
      for (const [n, { it, known }] of batch.entries()) {
        const d = got instanceof Map ? got.get(`p${n + 1}`) : undefined;
        if (!d || d.relation === 'different') { await this.add(it); r.added++; continue; }
        await this.locked([it.uid, known.uid], async () => {
          // Changed by a window beside this one since it was asked about: the answer was about another text — kept
          // beside it, linked, for the sleep pass to read again.
          const now = this.latest(known);
          const moved = !now || now.fingerprint !== known.fingerprint;
          if (d.relation === 'related' || (moved && d.relation !== 'contradicts')) {
            await this.add(it); r.added++;
            if (now) { await this.link(it, now); r.linked++; }
            return;
          }
          if (d.relation === 'contradicts') {
            await this.add(it); r.added++;
            if (now) { await this.retire(now, it.uid); r.superseded++; }
            return;
          }
          await this.merge(now!, it, d.title && d.summary ? { title: d.title, summary: d.summary, body: d.body || now!.body } : null);
          if (d.relation === 'updates') r.updated++; else r.merged++;
        });
      }
    }
  }

  /** Observations are only ever added: one is the same as a known one only when it says the same (SAME_OBSERVATION)
   *  about the same date — then the known one is told again (provenance, `seen`). Never asked about, never merged into
   *  another text, never superseded: an older one stays, dated, beside the newer. */
  private async observe(items: MemoryItem[], r: ConsolidationReport): Promise<void> {
    if (!items.length) return;
    // The window's observations looked up together and written together: one embedding call, one batched query, one
    // write — and each also against the ones before it in this window, as when each was written before the next looked.
    const vecs = await this.embed(items.map((it) => docText(it.title, it.summary)), embedBudget(items.length));
    const filter: Filter = { must: [{ key: 'tier', match: { value: 'ltm' } }, { key: 'status', match: { value: 'active' } }, { key: 'kind', match: { value: 'observation' } }, { key: 'part', match: { value: 'key' } }] };
    const near = await this.d.client.nearestMany(vecs.map((v) => Array.from(v)), filter, 3, 30_000);
    const fresh: Array<{ it: MemoryItem; v: Float32Array }> = [];
    /** Told again in this window: how many times, by whom, the newest time. */
    const told = new Map<string, { times: number; ts: number; provenance: string[] }>();
    for (const [n, it] of items.entries()) {
      const same = (eventAt: string | undefined): boolean => (eventAt ?? '') === (it.eventAt ?? '');
      const fromWindow = fresh.find((f) => f.it.uid !== it.uid && same(f.it.eventAt) && cosine(f.v, vecs[n]!) >= SAME_OBSERVATION);
      if (fromWindow) {
        fromWindow.it = { ...fromWindow.it, seen: fromWindow.it.seen + 1, ts: Math.max(fromWindow.it.ts, it.ts), provenance: [...new Set([...fromWindow.it.provenance, ...it.provenance])] };
        r.merged++;
        continue;
      }
      const hit = (near[n] ?? []).find((h) => h.item !== it.uid);
      const known = hit && hit.score >= SAME_OBSERVATION ? this.d.ltm.get(hit.item) : null;
      if (known && same(known.eventAt)) {
        const t = told.get(known.uid) ?? { times: 0, ts: 0, provenance: [] };
        told.set(known.uid, { times: t.times + 1, ts: Math.max(t.ts, it.ts), provenance: [...new Set([...t.provenance, ...it.provenance])] });
        r.merged++;
        continue;
      }
      fresh.push({ it, v: vecs[n]! });
      r.observed++;
    }
    // Told again: what this window adds (times, sources, the newest time) put onto the item as it is by now — a window
    // beside this one may have told it too.
    for (const [uid, t] of told) {
      await this.locked([uid], async () => {
        const was = this.d.ltm.get(uid);
        if (!was) return;
        const now: MemoryItem = { ...was, ts: Math.max(was.ts, t.ts), seen: was.seen + t.times, provenance: [...new Set([...was.provenance, ...t.provenance])] };
        await this.d.client.setPayload([uid], { ts: now.ts });
        this.d.ltm.put([now]);
      });
    }
    if (fresh.length) {
      await writeLtm(this.writer, fresh.map((f) => f.it));
      this.d.ltm.put(fresh.map((f) => f.it));
    }
  }

  private async add(it: MemoryItem): Promise<void> {
    await writeLtm(this.writer, [it]);
    this.d.ltm.put([it]);
  }

  /** The known item absorbs the new one: provenance, `seen`, the newest time, the higher importance — and, when the
   *  model merged the texts, the merged key (embedded again). Nothing of either is lost: the new item's own text, and the
   *  known one's before a rewrite, stay in the log as `merged` records the item lists (`absorbed`). */
  async merge(known: MemoryItem, it: MemoryItem, text: { title: string; summary: string; body: string } | null): Promise<void> {
    const kept: MemoryItem[] = [{ ...it, status: 'merged', mergedInto: known.uid }];
    if (text) kept.push({ ...known, uid: `${known.uid}@${known.fingerprint}`, status: 'merged', mergedInto: known.uid });
    const merged: MemoryItem = {
      ...known,
      ...(text ? { title: redactString(text.title), summary: redactString(text.summary), body: redactString(text.body) } : {}),
      ts: Math.max(known.ts, it.ts), importance: Math.max(known.importance, it.importance), seen: known.seen + 1,
      provenance: [...new Set([...known.provenance, ...it.provenance])],
      files: [...new Set([...known.files, ...it.files])],
      confidence: known.confidence === 'verified' || it.confidence === 'verified' ? 'verified' : known.confidence,
      cues: [...new Set([...(known.cues ?? []), ...(it.cues ?? [])])],
      related: [...new Set([...(known.related ?? []), ...(it.related ?? [])])].filter((u) => u !== known.uid).slice(-MAX_LINKS),
      concepts: [...new Map([...(known.concepts ?? []), ...(it.concepts ?? [])].map((c) => [conceptKey(c), c])).values()],
      absorbed: [...new Set([...(known.absorbed ?? []), ...kept.map((k) => k.uid)])],
    };
    merged.fingerprint = fingerprint(merged.kind, merged.title, merged.summary, merged.body);
    // Changed now: what sleep relates next.
    merged.learnedAt = Math.max(learnedAt(known), learnedAt(it));
    if (text) await writeLtm(this.writer, [merged]);
    else {
      await this.d.client.setPayload([known.uid], { ts: merged.ts, importance: merged.importance });
      if (merged.related?.join('\n') !== (known.related ?? []).join('\n')) await this.d.client.setPayload([known.uid], { related: merged.related }, true);
      if ((merged.concepts?.length ?? 0) > (known.concepts?.length ?? 0)) await writeConcepts(this.writer, [merged]);
      // Learned again in other words: its new cues are found from now on.
      if ((merged.cues?.length ?? 0) > (known.cues?.length ?? 0)) await writeCues(this.writer, [merged]);
    }
    this.d.ltm.put([...kept, merged]);
  }

  /** Two items about the same subject that say different things: each lists the other (memory_explore walks these) —
   *  the newest MAX_LINKS, so an item on a busy subject does not grow its list, and its payload, without end. */
  async link(a: MemoryItem, b: MemoryItem): Promise<void> {
    const add = (x: MemoryItem, y: MemoryItem): MemoryItem => ({ ...x, related: [...(x.related ?? []).filter((u) => u !== y.uid), y.uid].slice(-MAX_LINKS) });
    const [one, two] = [add(a, b), add(b, a)];
    this.d.ltm.put([one, two]);
    await this.d.client.setPayload([one.uid], { related: one.related }, true);
    await this.d.client.setPayload([two.uid], { related: two.related }, true);
  }

  /** Superseded: out of the index and of recall, kept in the log with what replaced it. */
  async retire(known: MemoryItem, by: string): Promise<void> {
    const old: MemoryItem = { ...known, status: 'superseded', supersededBy: by, supersededAt: (this.d.now ?? Date.now)() };
    await writeLtm(this.writer, [old]);
    this.d.ltm.put([old]);
  }

  /** The model's decision for each pair, by its label (`p1`, …) — or why it did not answer. */
  async relate(batch: Array<{ it: MemoryItem; known: MemoryItem }>): Promise<Map<string, Decision> | { paused: string }> {
    // Whole texts: a merge the model writes from a shortened body loses the rest of it.
    const prompt = batch.map(({ it, known }, n) => `[p${n + 1}]\nNEW (${it.kind}): ${it.title}\n${it.summary}\n${it.body}\n\nKNOWN (${known.kind}): ${known.title}\n${known.summary}\n${known.body}`).join('\n\n');
    // Room for every merged text in full (≈ 2.5 characters a token, the two sides together), with the decisions.
    const room = batch.reduce((n, { it, known }) => n + Math.ceil((it.summary.length + it.body.length + known.summary.length + known.body.length) / 2.5), 0);
    const a = await (this.d.ask ?? defaultAsk)({
      purpose: 'memory', system: RELATE_SYSTEM, prompt: redactString(prompt), schema: RELATE_SCHEMA as unknown as Record<string, unknown>,
      name: 'memory_reconcile', maxTokens: 900 * batch.length + 200 + room, timeoutMs: 300_000, consequence: 'so these memories are kept apart for now',
    });
    if ('unavailable' in a) return { paused: a.unavailable };
    if (!('answer' in a)) return { paused: 'the fast model is busy' };
    const out = new Map<string, Decision>();
    for (const p of (a.answer as { pairs?: Array<{ id?: string; relation?: string; title?: string; summary?: string; body?: string }> }).pairs ?? []) {
      // A small model may quote the label's line rather than the label: the number is looked for.
      const n = typeof p.id === 'string' ? labelNo(p.id, 'p') : 0;
      if (!n || !RELATIONS.includes(String(p.relation))) continue;
      out.set(`p${n}`, { relation: p.relation as Relation, title: (p.title ?? '').trim(), summary: (p.summary ?? '').trim(), body: (p.body ?? '').trim() });
    }
    return out;
  }
}
