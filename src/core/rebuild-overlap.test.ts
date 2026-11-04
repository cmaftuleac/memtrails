/** A rebuild begun again while one runs (the index cleared under it): the one running stops without touching the new
 *  checkpoint, and the new one writes everything. */
import { existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { MemoryRebuild } from './rebuild.js';
import { ltmLog, stmLog, type StmEntry } from './store.js';

const entry = (uid: string): StmEntry => ({
  uid, fingerprint: uid, engine: 'conversation', session: 's', turn: 1, cwd: '/p', project: '-p', title: uid,
  ts: 1, synthetic: false, typed: 'human', request: uid, reply: '', said: '', steps: [], files: [], done: false,
});

describe('a rebuild begun again while one runs', () => {
  it('stops the one running, then writes everything into the cleared index; the checkpoint goes only at the end', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'memory-rebuild-overlap-'));
    const stm = stmLog(join(dir, 'stm.jsonl'));
    const ltm = ltmLog(join(dir, 'ltm.jsonl'));
    stm.put(Array.from({ length: 32 * 3 }, (_, i) => entry(`r${String(i).padStart(3, '0')}`)));
    const index = new Set<string>();
    const job = new MemoryRebuild({
      path: join(dir, 'rebuild.json'), stm, ltm, aborted: () => false,
      writeLtm: async () => {},
      writeStm: async (xs) => { await new Promise((ok) => setTimeout(ok, 20)); for (const x of xs) index.add(x.uid); },
    });
    job.begin();
    const first = job.run();
    while (!index.size) await new Promise((ok) => setTimeout(ok, 5));
    // memory.rebuild(): the checkpoint first, then the index cleared, then run.
    job.begin();
    index.clear();
    const second = job.run();
    await Promise.all([first, second]);
    expect(index.size).toBe(32 * 3);
    expect(existsSync(join(dir, 'rebuild.json'))).toBe(false);
  });
});
