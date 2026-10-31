/** Checkpoints survive a failed remote batch and restart, and never turn old STM backfill into live recall. */
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { MemoryRebuild } from './rebuild.js';
import { ltmLog, stmLog, type MemoryItem, type StmEntry } from './store.js';

const now = 1_800_000_000_000;
const recentMs = 7 * 86400_000;
const item = (uid: string, status: MemoryItem['status'] = 'active'): MemoryItem => ({
  uid, kind: 'fact', scope: 'project', project: '-p', engine: 'conversation', session: 's', cwd: '/p', where: 'c',
  ts: 1, firstTs: 1, title: uid, summary: uid, body: uid, files: [], appliesWhen: '', confidence: 'verified',
  importance: 3, status, provenance: [], seen: 1, fingerprint: uid,
});
const entry = (uid: string, over: Partial<StmEntry> = {}): StmEntry => ({
  uid, fingerprint: uid, engine: 'conversation', session: 's', turn: 1, cwd: '/p', project: '-p', title: uid,
  ts: now - 1000, synthetic: false, typed: 'human', request: uid, reply: '', said: '', steps: [], files: [], done: false,
  ...over,
});

function rig() {
  const dir = mkdtempSync(join(tmpdir(), 'memory-rebuild-'));
  const ltm = ltmLog(join(dir, 'ltm.jsonl'));
  const stm = stmLog(join(dir, 'stm.jsonl'));
  const path = join(dir, 'rebuild.json');
  const written: string[][] = [];
  let aborted = false;
  const deps = { path, ltm, stm, now: () => now, aborted: () => aborted,
    writeLtm: async (xs: MemoryItem[]) => { written.push(xs.map((x) => x.uid)); },
    writeStm: async (xs: StmEntry[]) => { written.push(xs.map((x) => x.uid)); },
  };
  return { dir, path, ltm, stm, written, deps, stop: () => { aborted = true; } };
}

describe('durable memory rebuild', () => {
  it('resumes after a failed batch and process restart without re-embedding completed batches', async () => {
    const r = rig();
    r.ltm.put(Array.from({ length: 41 }, (_, i) => item(`mem:${i}`)));
    let calls = 0;
    const first = new MemoryRebuild({ ...r.deps, writeLtm: async (xs) => {
      if (++calls === 2) throw new Error('GPU service temporarily unavailable');
      await r.deps.writeLtm(xs);
    } });
    first.begin();
    await expect(first.run()).rejects.toThrow('unavailable');
    expect(JSON.parse(readFileSync(r.path, 'utf8'))).toEqual({ phase: 'ltm', after: 'mem:19' });
    expect(first.pending()).toBe(true);
    await r.ltm.load(); // What the next process reads from durable logs, not a retained in-memory list.
    const resumed = new MemoryRebuild(r.deps);
    await resumed.run();
    expect(resumed.pending()).toBe(false);
    expect(r.written.flat()).toEqual(Array.from({ length: 41 }, (_, i) => `mem:${i}`));
  });

  it('writes several batches at once, and its checkpoint never passes one that is still in flight or failed', async () => {
    const r = rig();
    r.stm.put(Array.from({ length: 32 * 6 }, (_, i) => entry(`stm:${String(i).padStart(3, '0')}`)));
    // Batch 2 (the third) fails after the later ones were written: the checkpoint stays after batch 1.
    let n = 0;
    const slow = new MemoryRebuild({ ...r.deps, concurrency: 4, writeStm: async (xs) => {
      const k = n++;
      await new Promise((ok) => setTimeout(ok, [30, 10, 40, 1, 1, 1][k]));
      if (k === 2) throw new Error('GPU worker reset');
      await r.deps.writeStm(xs);
    } });
    slow.begin();
    await expect(slow.run()).rejects.toThrow('reset');
    expect(JSON.parse(readFileSync(r.path, 'utf8'))).toEqual({ phase: 'stm', after: 'stm:063' });
    // Resumed: every request written, none of those before the checkpoint again.
    await r.stm.load();
    r.written.length = 0;
    await new MemoryRebuild({ ...r.deps, concurrency: 4 }).run();
    expect(r.written.flat()).toEqual(Array.from({ length: 32 * 4 }, (_, i) => `stm:${String(i + 64).padStart(3, '0')}`));
  });

  it('does no work before sign-in and leaves the marker through cancellation', async () => {
    const r = rig();
    r.ltm.put([item('mem:a')]);
    const job = new MemoryRebuild(r.deps);
    job.begin(); // service can set this while no embedding route exists; a later scheduled job resumes it.
    expect(job.pending()).toBe(true);
    expect(r.written).toEqual([]);
    r.stop();
    await job.run();
    expect(job.pending()).toBe(true);
    expect(r.written).toEqual([]);
  });

  it('indexes active LTM and every nonsynthetic request, consolidated or not, however old', async () => {
    const r = rig();
    r.ltm.put([item('mem:active'), item('mem:archived', 'archived'), item('mem:old', 'superseded')]);
    r.stm.put([entry('recent'), entry('expired', { ts: now - recentMs - 1 }), entry('done', { done: true }), entry('synthetic', { synthetic: true })]);
    const job = new MemoryRebuild(r.deps);
    job.begin();
    await job.run();
    expect(r.written).toEqual([['mem:active'], ['recent', 'expired', 'done']]);
    expect(job.pending()).toBe(false);
  });

  it('serializes simultaneous retries into one rebuild', async () => {
    const r = rig();
    r.ltm.put([item('mem:a')]);
    const job = new MemoryRebuild(r.deps);
    job.begin();
    await Promise.all([job.run(), job.run(), job.run()]);
    expect(r.written).toEqual([['mem:a']]);
  });

  it('replays a phase safely if its checkpoint is damaged or the saved uid no longer exists', async () => {
    for (const checkpoint of ['broken', JSON.stringify({ phase: 'ltm', after: 'not-here' })]) {
      const r = rig();
      r.ltm.put([item('mem:a')]);
      writeFileSync(r.path, checkpoint);
      const job = new MemoryRebuild(r.deps);
      await job.run();
      expect(r.written).toEqual([['mem:a']]);
      expect(job.pending()).toBe(false);
    }
  });

  it('does not advance a checkpoint after stop, even if the route finishes during a rapid restart', async () => {
    const r = rig();
    r.ltm.put([item('mem:a')]);
    let finish!: () => void;
    const blocked = new Promise<void>((resolve) => { finish = resolve; });
    const job = new MemoryRebuild({ ...r.deps, writeLtm: async (xs) => { await blocked; await r.deps.writeLtm(xs); } });
    job.begin();
    const running = job.run();
    job.cancel(); // A generation token survives aborted becoming false again on restart.
    finish();
    await running;
    await job.settled();
    expect(JSON.parse(readFileSync(r.path, 'utf8'))).toEqual({ phase: 'ltm', after: null });
    await new MemoryRebuild(r.deps).run();
    expect(r.written).toEqual([['mem:a'], ['mem:a']]); // The interrupted batch replays idempotently.
    expect(job.pending()).toBe(false);
  });

  it('resumes STM independently after all LTM was written', async () => {
    const r = rig();
    r.ltm.put([item('mem:a')]);
    r.stm.put(Array.from({ length: 33 }, (_, i) => entry(`stm:${i}`)));
    let calls = 0;
    const job = new MemoryRebuild({ ...r.deps, writeStm: async (xs) => {
      if (++calls === 2) throw new Error('retry');
      await r.deps.writeStm(xs);
    } });
    job.begin();
    await expect(job.run()).rejects.toThrow('retry');
    expect(JSON.parse(readFileSync(r.path, 'utf8'))).toEqual({ phase: 'stm', after: 'stm:31' });
    await new MemoryRebuild(r.deps).run();
    expect(r.written.flat()).toEqual(['mem:a', ...Array.from({ length: 33 }, (_, i) => `stm:${i}`)]);
  });
});
