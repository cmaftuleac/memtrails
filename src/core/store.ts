/**
 * What a memory holds, on disk, as two append-only logs in its home:
 *
 *   - `records.jsonl` — SHORT-TERM memory: every record as it was added (records.ts), kept for good; also the queue the
 *     consolidation worker reads (`done`);
 *   - `knowledge.jsonl` — LONG-TERM memory: what consolidation kept (consolidate.ts) — typed items, dated observations,
 *     kept episodes, profiles — with their keys, status (active / superseded / archived) and provenance.
 *
 * The logs are the truth; Qdrant is derived from them (rebuilt without asking a model, after a Qdrant upgrade or a
 * lost index). In memory: each record's place in its file and a small projection — never the text. A later line for
 * the same uid replaces the earlier one; a log is rewritten without replaced lines once it is twice what is live.
 *
 * `usage.json`: how often each long-term item was shown and opened (the sleep pass's "use", MemRL-style).
 * `recall.jsonl`: what each search showed, for what, and what was opened after (recall-log.ts).
 *
 * The logs are behind `MemoryLog`: these files, or any store you implement (a database table per user). Reads are from what `load()` brought in; `put` writes behind and `flush()`
 * waits for it (a file write is done when `put` returns).
 */
import { createHash } from 'node:crypto';
import { appendFileSync, closeSync, createReadStream, existsSync, mkdirSync, openSync, readFileSync, readSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { createInterface } from 'node:readline';

export type MemoryKind =
  | 'decision' | 'procedure' | 'snippet' | 'template' | 'pitfall' | 'fact' | 'architecture' | 'preference' | 'reference'
  | 'episode' | 'note' | 'observation'
  /** What memory knows about one subject, gathered from its items (profiles.ts) — written again from them, never merged. */
  | 'profile';
export type MemoryScope = 'project' | 'global' | 'person';
/** `merged`: a text another item absorbed — kept whole, out of recall, so a merge never loses what either side said. */
export type MemoryStatus = 'active' | 'superseded' | 'archived' | 'merged';

/** A request as captured: short-term memory. */
export interface StmEntry {
  uid: string;
  fingerprint: string;
  engine: string;
  session: string;
  turn: number;
  cwd: string | null;
  project: string;
  title: string;
  ts: number;
  synthetic: boolean;
  typed: 'human' | 'unknown';
  request: string;
  reply: string;
  said: string;
  steps: string[];
  files: string[];
  /** Read by the consolidation worker. */
  done: boolean;
  /** The record it is (records.ts): its own id, where it came from, who wrote it, where to open it, and your metadata. */
  record?: { id: string; source?: string; author?: string; url?: string; metadata?: Record<string, string | number | boolean | null> };
}

/** What long-term memory keeps. */
export interface MemoryItem {
  uid: string;
  kind: MemoryKind;
  scope: MemoryScope;
  project: string;
  engine: string;
  session: string | null;
  cwd: string | null;
  /** Where it came from, for a person: the conversation's title, the note's name. */
  where: string;
  /** Last learned or refreshed. */
  ts: number;
  firstTs: number;
  /** The key: a content title and a ~150-word summary (what is embedded). */
  title: string;
  summary: string;
  body: string;
  files: string[];
  appliesWhen: string;
  /** verified / stated (consolidation keeps only these), curated (a saved note), observed (an episode). */
  confidence: 'verified' | 'stated' | 'curated' | 'observed';
  /** 1–5, from consolidation. */
  importance: number;
  status: MemoryStatus;
  supersededBy?: string;
  /** When it was superseded — kept beside when it was learned (`ts`), so what was true when stays answerable. */
  supersededAt?: number;
  /** For an episode: how its request ended (success, partial, fail, uncertain). */
  outcome?: 'success' | 'partial' | 'fail' | 'uncertain';
  /** For a `merged` record: the item that absorbed it. */
  mergedInto?: string;
  /** The `merged` records this item absorbed: the texts it was made from, whole, in the log. */
  absorbed?: string[];
  /** The requests it was learned from (episode uids), newest last, at most 50. */
  provenance: string[];
  /** How many times consolidation learned it again (a duplicate merged into it). */
  seen: number;
  fingerprint: string;
  /** Requests the person would type when it is needed — searched beside the key (absent before round 2: the worker
   *  fills them, see Consolidator.fillCues). */
  cues?: string[];
  /** Where it came from (a guard that keeps only trusted sources): the person's own words, or a result the agent confirmed —
   *  with the words that show it. Absent on a desktop, where every source is the person's own machine. */
  origin?: MemoryOrigin;
  /** Items about the same subject that say different things (consolidation and sleep link them; memory_explore walks
   *  them), newest last. */
  related?: string[];
  /** What it is about — files, services, commands, named things — as the conversation named them: the concept graph's
   *  links (memory_explore walks them; absent before the graph: the worker's backlog fills them). */
  concepts?: string[];
  /** For an observation: when what it tells happened or applies — an absolute date as precise as the conversation made
   *  it (YYYY-MM-DD, YYYY-MM or YYYY), resolved from the request's own date; absent when the conversation gave none.
   *  `ts` stays when it was said. */
  eventAt?: string;
  /** For an observation of something that lasted (a state, a habit, a job, a stay): when it began and when it ended,
   *  as precise as the conversation made them; absent when it gave none. */
  startAt?: string;
  endAt?: string;
  /** For an observation: done (it happened), plan (intended, not yet done), ongoing, or state (how things are). */
  phase?: 'done' | 'plan' | 'ongoing' | 'state';
  /** For an observation: who or what it is about, named as the conversation names them ("the person", "Melanie"). */
  about?: string;
}

/** What a kept thing was learned from, and the exact words that show it (a guard checks they are there). */
export interface MemoryOrigin { by: 'user' | 'result'; evidence: string }

export function fingerprint(...parts: Array<string | readonly string[] | null | undefined>): string {
  const h = createHash('sha1');
  for (const p of parts) h.update(typeof p === 'string' || p == null ? (p ?? '') : p.join('\u0001')).update('\u0000');
  return h.digest('hex').slice(0, 16);
}

interface Slot<M> { at: number; len: number; meta: M }

/** An append-only log of records by uid: a later record for the same uid replaces the earlier one. What consolidation,
 *  sleep and capture read and write, wherever the records are kept. */
export interface MemoryLog<T extends { uid: string }, M> {
  size(): number;
  meta(uid: string): M | undefined;
  entries(): Iterable<[string, M]>;
  load(): Promise<void>;
  get(uid: string): T | null;
  put(recs: T[]): void;
  /** Forget these records: gone from memory, and from the file when it is next compacted. */
  remove(uids: string[]): void;
  /** Every live record, oldest first. */
  all(): Iterable<T>;
  compact(): void;
  /** Resolves once every `put` is kept; rejects with the first write that failed. */
  flush(): Promise<void>;
}

/** An append-only JSONL log, read back by byte offset. `meta` is what stays in memory for each record. */
export class JsonlLog<T extends { uid: string }, M> implements MemoryLog<T, M> {
  private readonly live = new Map<string, Slot<M>>();
  private lines = 0;
  private end = 0;

  constructor(private readonly path: string, private readonly project: (rec: T) => M) {}

  size(): number { return this.live.size; }
  meta(uid: string): M | undefined { return this.live.get(uid)?.meta; }
  *entries(): Generator<[string, M]> { for (const [uid, s] of this.live) yield [uid, s.meta]; }

  async load(): Promise<void> {
    this.live.clear();
    this.lines = 0;
    this.end = 0;
    if (!existsSync(this.path)) return;
    let at = 0;
    for await (const line of createInterface({ input: createReadStream(this.path, { encoding: 'utf8' }), crlfDelay: Infinity })) {
      const len = Buffer.byteLength(line, 'utf8') + 1;
      if (line) {
        try {
          const rec = JSON.parse(line) as T & { removed?: true };
          if (rec.removed) this.live.delete(rec.uid);
          else this.live.set(rec.uid, { at, len, meta: this.project(rec) });
          this.lines++;
        } catch { /* a torn last line */ }
      }
      at += len;
    }
    this.end = at;
  }

  get(uid: string): T | null {
    const s = this.live.get(uid);
    if (!s) return null;
    const fd = openSync(this.path, 'r');
    try {
      const buf = Buffer.alloc(s.len);
      readSync(fd, buf, 0, s.len, s.at);
      return JSON.parse(buf.toString('utf8').trimEnd()) as T;
    } catch {
      return null;
    } finally {
      closeSync(fd);
    }
  }

  put(recs: T[]): void {
    if (!recs.length) return;
    mkdirSync(dirname(this.path), { recursive: true });
    if (!this.end && existsSync(this.path)) this.end = statSync(this.path).size;
    const chunks: Buffer[] = [];
    let at = this.end;
    for (const rec of recs) {
      const b = Buffer.from(`${JSON.stringify(rec)}\n`, 'utf8');
      this.live.set(rec.uid, { at, len: b.length, meta: this.project(rec) });
      chunks.push(b);
      at += b.length;
    }
    appendFileSync(this.path, Buffer.concat(chunks), { mode: 0o600 });
    this.end = at;
    this.lines += recs.length;
  }

  remove(uids: string[]): void {
    const gone = uids.filter((u) => this.live.has(u));
    if (!gone.length) return;
    mkdirSync(dirname(this.path), { recursive: true });
    const b = Buffer.from(gone.map((uid) => `${JSON.stringify({ uid, removed: true })}\n`).join(''), 'utf8');
    appendFileSync(this.path, b, { mode: 0o600 });
    if (!this.end) this.end = statSync(this.path).size - b.length;
    this.end += b.length;
    this.lines += gone.length;
    for (const uid of gone) this.live.delete(uid);
  }

  /** Every live record, in file order. */
  *all(): Generator<T> {
    const order = [...this.live.entries()].sort((a, b) => a[1].at - b[1].at);
    for (const [uid] of order) { const r = this.get(uid); if (r) yield r; }
  }

  /** A file is written when `put` returns. */
  async flush(): Promise<void> {}

  /** Rewrite the log without replaced lines once it is twice what is live. */
  compact(): void {
    if (this.lines < 2 * this.live.size + 1000) return;
    const recs = [...this.all()];
    const tmp = `${this.path}.compact`;
    writeFileSync(tmp, recs.map((r) => `${JSON.stringify(r)}\n`).join(''), { mode: 0o600 });
    renameSync(tmp, this.path);
    // Offsets again, from the file just written.
    this.live.clear();
    let at = 0;
    for (const r of recs) {
      const len = Buffer.byteLength(`${JSON.stringify(r)}\n`, 'utf8');
      this.live.set(r.uid, { at, len, meta: this.project(r) });
      at += len;
    }
    this.end = at;
    this.lines = recs.length;
  }
}

export type StmMeta = { fp: string; done: boolean; ts: number; session: string; project: string; synthetic: boolean };
export type LtmMeta = { fp: string; status: MemoryStatus; kind: MemoryKind; scope: MemoryScope; project: string; ts: number; importance: number; cued: boolean; mapped: boolean };

export const stmMetaOf = (r: StmEntry): StmMeta => ({ fp: r.fingerprint, done: r.done, ts: r.ts, session: r.session, project: r.project, synthetic: r.synthetic });
export const ltmMetaOf = (r: MemoryItem): LtmMeta => ({ fp: r.fingerprint, status: r.status, kind: r.kind, scope: r.scope, project: r.project, ts: r.ts, importance: r.importance, cued: !!r.cues?.length, mapped: !!r.concepts?.length });

export const stmLog = (path: string): JsonlLog<StmEntry, StmMeta> => new JsonlLog<StmEntry, StmMeta>(path, stmMetaOf);
export const ltmLog = (path: string): JsonlLog<MemoryItem, LtmMeta> => new JsonlLog<MemoryItem, LtmMeta>(path, ltmMetaOf);

export type StmStore = MemoryLog<StmEntry, StmMeta>;
export type LtmStore = MemoryLog<MemoryItem, LtmMeta>;

/** How often each long-term item was shown and opened — what the sleep pass reads as "use". */
export interface UsageLog {
  of(uid: string): { shown: number; opened: number; lastUsed: number };
  note(uids: string[], what: 'shown' | 'opened', now?: number): void;
  flush(): void | Promise<void>;
}

/** How often each long-term item was shown with a turn and opened by an agent. Saved at most once a minute. */
export class UsageBook implements UsageLog {
  private data: Record<string, { shown: number; opened: number; lastUsed: number }> = {};
  private dirty = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly path: string) {
    try { this.data = JSON.parse(readFileSync(path, 'utf8')) as typeof this.data; } catch { this.data = {}; }
  }

  of(uid: string): { shown: number; opened: number; lastUsed: number } {
    return this.data[uid] ?? { shown: 0, opened: 0, lastUsed: 0 };
  }

  note(uids: string[], what: 'shown' | 'opened', now = Date.now()): void {
    for (const uid of uids) {
      const u = (this.data[uid] ??= { shown: 0, opened: 0, lastUsed: 0 });
      u[what] += 1;
      u.lastUsed = now;
    }
    this.dirty = true;
    this.timer ??= setTimeout(() => { this.timer = null; this.flush(); }, 60_000);
    this.timer.unref?.();
  }

  flush(): void {
    if (!this.dirty) return;
    try {
      mkdirSync(dirname(this.path), { recursive: true });
      writeFileSync(this.path, JSON.stringify(this.data), { mode: 0o600 });
      this.dirty = false;
    } catch { /* read-only: kept in memory */ }
  }
}
