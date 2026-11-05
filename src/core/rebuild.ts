/** A derived index rebuild survives unavailable embeddings, a failed batch and a restart. Logs stay truth. */
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import type { LtmStore, MemoryItem, StmEntry, StmStore } from './store.js';

interface Checkpoint { phase: 'ltm' | 'stm'; after: string | null }
interface RebuildDeps {
  path: string;
  ltm: LtmStore;
  stm: StmStore;
  writeLtm: (items: MemoryItem[]) => Promise<void>;
  writeStm: (items: StmEntry[]) => Promise<void>;
  aborted: () => boolean;
  now?: () => number;
  /** Batches written at once (1 when not given). The checkpoint only ever passes batches already written: it moves over
   *  the written ones at the front, so a crash never skips one still in flight. */
  concurrency?: number;
}

export class MemoryRebuild {
  private running: Promise<void> | null = null;
  /** The generation `running` was started for: a run() of the same one joins it, a later one waits for it. */
  private runningFor = -1;
  private generation = 0;
  constructor(private readonly d: RebuildDeps) {}

  cancel(): void { this.generation++; }
  async settled(): Promise<void> { await this.running?.catch(() => {}); }

  pending(): boolean { return existsSync(this.d.path); }

  /** Written BEFORE clearing an old index. A crash at any later point leaves a resumable rebuild. A run in flight is
   *  cancelled: it wrote into the index about to be cleared, and must neither move nor delete the new checkpoint. */
  begin(): void {
    this.generation++;
    this.save({ phase: 'ltm', after: null });
  }

  /** Joins the run in flight for the same begin(); after a newer begin(), waits for the cancelled one to stop first. */
  run(): Promise<void> {
    const generation = this.generation;
    if (this.running && this.runningFor === generation) return this.running;
    const before = this.running?.catch(() => {});
    const work = before ? before.then(() => this.resume(generation)) : this.resume(generation);
    const p: Promise<void> = work.finally(() => { if (this.running === p) this.running = null; });
    this.running = p;
    this.runningFor = generation;
    return p;
  }

  private save(checkpoint: Checkpoint): void {
    mkdirSync(dirname(this.d.path), { recursive: true });
    const tmp = `${this.d.path}.tmp`;
    writeFileSync(tmp, JSON.stringify(checkpoint), { mode: 0o600 });
    renameSync(tmp, this.d.path);
  }

  /** One phase from its checkpoint: batches of `size`, up to `concurrency` written at once, the checkpoint moved over
   *  the written batches at the front. False when cancelled. A failed batch fails the run, the checkpoint before it. */
  private async phase<T extends { uid: string }>(all: Iterable<T>, after: string | null, keep: (it: T) => boolean, size: number,
    write: (items: T[]) => Promise<void>, phase: Checkpoint['phase'], cancelled: () => boolean): Promise<boolean> {
    const limit = Math.max(1, this.d.concurrency ?? 1);
    const queue: Array<{ last: string; done: boolean; p: Promise<void> }> = [];
    const advance = (): void => {
      let last: string | null = null;
      while (queue.length && queue[0]!.done) last = queue.shift()!.last;
      if (last !== null && !cancelled()) this.save({ phase, after: last });
    };
    // On a failure: the batches still in flight settle first, and the checkpoint takes every written one at the front.
    const fail = async (err: unknown): Promise<never> => {
      await Promise.allSettled(queue.map((q) => q.p));
      advance();
      throw err;
    };
    const send = async (items: T[]): Promise<void> => {
      const slot = { last: items.at(-1)!.uid, done: false, p: Promise.resolve() };
      slot.p = write(items).then(() => { slot.done = true; });
      slot.p.catch(() => {}); // awaited below; a batch left in flight by a failure elsewhere is not unhandled
      queue.push(slot);
      for (let busy = queue.filter((q) => !q.done); busy.length >= limit; busy = queue.filter((q) => !q.done)) {
        await Promise.race(busy.map((q) => q.p)).catch(fail);
        advance();
      }
    };
    let skip = after !== null;
    let batch: T[] = [];
    for (const it of all) {
      if (cancelled()) {
        // Stopped (a newer begin, a close): what is being written ends before this returns, not into the next index.
        await Promise.allSettled(queue.map((q) => q.p));
        return false;
      }
      if (skip) { if (it.uid === after) skip = false; continue; }
      if (!keep(it)) continue;
      batch.push(it);
      if (batch.length >= size) { await send(batch); batch = []; }
    }
    if (batch.length) await send(batch);
    await Promise.all(queue.map((q) => q.p)).catch(fail);
    advance();
    return !cancelled();
  }

  private async resume(generation: number): Promise<void> {
    const cancelled = (): boolean => generation !== this.generation || this.d.aborted();
    if (!this.pending() || cancelled()) return;
    let checkpoint: Checkpoint;
    try {
      checkpoint = JSON.parse(readFileSync(this.d.path, 'utf8')) as Checkpoint;
      if (!['ltm', 'stm'].includes(checkpoint.phase) || (checkpoint.after !== null && typeof checkpoint.after !== 'string')) throw new Error('invalid checkpoint');
    } catch { checkpoint = { phase: 'ltm', after: null }; }
    // While rebuilding, nothing else writes to the logs. Live log order therefore stays stable; if a
    // checkpoint's item is absent (a repaired/compacted log), replay that phase instead of silently skipping records.
    if (checkpoint.after && !(checkpoint.phase === 'ltm' ? this.d.ltm : this.d.stm).meta(checkpoint.after)) checkpoint.after = null;
    if (checkpoint.phase === 'ltm') {
      const done = await this.phase(this.d.ltm.all(), checkpoint.after, (it) => it.status === 'active', 20, this.d.writeLtm, 'ltm', cancelled);
      if (!done) return;
      checkpoint = { phase: 'stm', after: null };
      this.save(checkpoint);
    }
    // Every request the person made stays searchable, consolidated or not: the raw record is the truth, a memory a key
    // to it (LongMemEval 2410.10813). Only throwaway runs (synthetic) stay out.
    if (!(await this.phase(this.d.stm.all(), checkpoint.after, (it) => !it.synthetic, 32, this.d.writeStm, 'stm', cancelled))) return;
    if (!cancelled()) rmSync(this.d.path);
  }
}
