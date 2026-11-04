/**
 * What recall showed, for what, and what was opened after it — the record a model that learns from USE reads.
 *
 * Learned only from what memory says and how it is laid out, two light models did not beat the first stage on the
 * answers shown (docs/RESEARCH.md: a hetero-associative map over the memory's own pairs, a LightGBM
 * ranker over language-free association features). What works in the literature is learned from use — links that
 * strengthen when memories are recalled together and fade when not (HeLa-Mem, ACL 2026), rankers trained on what was
 * opened (online learning to rank). The usage counts (store.ts UsageBook) say how often, never with what nor for what;
 * this says both.
 *
 * One line per event in `recall.jsonl` beside the logs — on this machine only, cleared with the index, never sent:
 *   {"t", "ev": "search", "project", "query", "shown": [ids]}                                   memory_search
 *   {"t", "ev": "open",   "item"}                                                                memory_open
 * Appended in order, a line at a time; a write that fails is dropped (recall never waits on, or fails for, its record).
 * The ids are the logs' own (`stm:…`, long-term uids): a rebuild from the logs keeps them (only `qdrant/` is made
 * again), so the record outlives a rebuild and stays true to it. Kept: the last RECALL_KEEP_MS, at most
 * RECALL_KEEP_BYTES of the newest (`trim()`: when memory opens, and every RECALL_TRIM_EVERY events).
 */
import { appendFile, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

/** A query kept up to this: the start of a long message is what the search embedded first. */
export const RECALL_QUERY_CHARS = 2000;
export const RECALL_KEEP_MS = 180 * 24 * 60 * 60_000;
export const RECALL_KEEP_BYTES = 32 * 1024 * 1024;

export type RecallEvent =
  | { ev: 'search'; project?: string; query: string; shown: string[] }
  | { ev: 'open'; item: string };

/** Events between trims of a book left open for long (a server): the cap holds without a restart. */
export const RECALL_TRIM_EVERY = 1000;

export class RecallBook {
  private chain: Promise<void> = Promise.resolve();
  private appended = 0;

  constructor(
    private readonly path: string,
    private readonly now: () => number = Date.now,
    private readonly keep: { ms: number; bytes: number } = { ms: RECALL_KEEP_MS, bytes: RECALL_KEEP_BYTES },
  ) {}

  record(e: RecallEvent): void {
    const line = `${JSON.stringify({ t: this.now(), ...e, ...('query' in e ? { query: e.query.slice(0, RECALL_QUERY_CHARS) } : {}) })}\n`;
    this.chain = this.chain
      .then(async () => {
        await mkdir(dirname(this.path), { recursive: true });
        await appendFile(this.path, line, { mode: 0o600 });
      })
      .catch(() => { /* recall never fails for its record */ });
    if (++this.appended % RECALL_TRIM_EVERY === 0) void this.trim();
  }

  /** The events older than `keep.ms` dropped, then the oldest until the rest fit `keep.bytes` — rewritten between
   *  appends (a new file renamed over the old). How many were dropped; 0 on any failure, the record as it was. */
  trim(): Promise<number> {
    const run = this.chain.then(async () => {
      let text: string;
      try { text = await readFile(this.path, 'utf8'); } catch { return 0; }
      const lines = text.split('\n').filter(Boolean);
      const since = this.now() - this.keep.ms;
      const recent = lines.filter((l) => { try { return Number((JSON.parse(l) as { t?: number }).t) >= since; } catch { return false; } });
      let bytes = recent.reduce((n, l) => n + Buffer.byteLength(l, 'utf8') + 1, 0);
      let from = 0;
      while (bytes > this.keep.bytes && from < recent.length) bytes -= Buffer.byteLength(recent[from++]!, 'utf8') + 1;
      const kept = recent.slice(from);
      if (kept.length === lines.length) return 0;
      const next = `${this.path}.trim`;
      await writeFile(next, kept.map((l) => `${l}\n`).join(''), { mode: 0o600 });
      await rename(next, this.path);
      return lines.length - kept.length;
    });
    this.chain = run.then(() => {}, () => {});
    return run.catch(() => 0);
  }

  /** Every event recorded so far written (tests, shutdown). */
  settled(): Promise<void> {
    return this.chain;
  }
}
