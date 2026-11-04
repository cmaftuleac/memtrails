/**
 * A log after a crash (store.ts): the last line cut off without its '\n'. What is added after it must stay readable — in
 * the same session, after a reopen and after a compaction — and a compaction that cannot read a record must not drop it.
 */
import { appendFileSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { JsonlLog } from './store.js';

type Rec = { uid: string; text: string; n: number };
const dir = () => mkdtempSync(join(tmpdir(), 'memory-store-recovery-'));
const log = (path: string) => new JsonlLog<Rec, { n: number }>(path, (r) => ({ n: r.n }));

describe('JsonlLog after a crash', () => {
  it('drops a cut-off last line, and keeps what is added after it in this session, after a reopen and after a compaction', async () => {
    const path = join(dir(), 'x.jsonl');
    log(path).put([{ uid: 'a', text: 'ok é', n: 1 }]);
    appendFileSync(path, '{"uid":"b","text":"torn');
    const l = log(path);
    await l.load();
    l.put([{ uid: 'c', text: 'after the crash', n: 2 }, { uid: 'd', text: 'and this', n: 3 }]);
    expect(l.get('c')?.text).toBe('after the crash');
    expect(l.get('d')?.text).toBe('and this');
    expect(readFileSync(path, 'utf8')).not.toContain('torn');

    const r = log(path);
    await r.load();
    expect([...r.all()].map((x) => x.uid)).toEqual(['a', 'c', 'd']);
    // Enough replaced lines to compact: nothing added since the crash is lost.
    for (let i = 0; i < 1100; i++) r.put([{ uid: 'a', text: `v${i}`, n: i }]);
    r.compact();
    const s = log(path);
    await s.load();
    expect([...s.all()].map((x) => [x.uid, x.text])).toEqual([['c', 'after the crash'], ['d', 'and this'], ['a', 'v1099']]);
  });

  it('keeps a last line that is whole but for its \'\\n\'', async () => {
    const path = join(dir(), 'x.jsonl');
    writeFileSync(path, `${JSON.stringify({ uid: 'a', text: 'one', n: 1 })}\n${JSON.stringify({ uid: 'b', text: 'two', n: 2 })}`);
    const l = log(path);
    await l.load();
    l.put([{ uid: 'c', text: 'three', n: 3 }]);
    expect(l.get('b')?.text).toBe('two');
    expect(l.get('c')?.text).toBe('three');
    const r = log(path);
    await r.load();
    expect([...r.all()].map((x) => x.uid)).toEqual(['a', 'b', 'c']);
  });

  it('refuses to compact when a record cannot be read, leaving the file as it was', async () => {
    const path = join(dir(), 'x.jsonl');
    const l = log(path);
    l.put([{ uid: 'keep', text: 'precious', n: 0 }]);
    for (let i = 0; i < 1100; i++) l.put([{ uid: 'u', text: `v${i}`, n: i }]);
    // Damage the first record in place (same length): the index still points at it.
    const before = readFileSync(path, 'utf8').replace('precious', 'prec"ous');
    writeFileSync(path, before);
    expect(l.get('keep')).toBeNull();
    expect(() => l.compact()).toThrow(/keep/);
    expect(readFileSync(path, 'utf8')).toBe(before);
    expect(l.get('u')?.text).toBe('v1099');
  });

  it('compacts a log larger than one write, every offset right after a reopen', async () => {
    const path = join(dir(), 'x.jsonl');
    const l = log(path);
    const big = 'ü'.repeat(3000);
    const live = 800;
    for (let round = 0; round < 4; round++) {
      l.put(Array.from({ length: live }, (_, i) => ({ uid: `u${i}`, text: `${round}:${i}:${big}`, n: round })));
    }
    l.compact();
    expect(readFileSync(path, 'utf8').trim().split('\n')).toHaveLength(live);
    expect(l.get('u799')?.text).toBe(`3:799:${big}`);
    l.put([{ uid: 'new', text: 'after', n: 9 }]);
    const r = log(path);
    await r.load();
    expect(r.size()).toBe(live + 1);
    for (let i = 0; i < live; i++) expect(r.get(`u${i}`)?.text).toBe(`3:${i}:${big}`);
    expect(r.get('new')?.text).toBe('after');
  });
});
