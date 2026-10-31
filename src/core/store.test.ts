/**
 * Memory's logs (store.ts): the truth the Qdrant index is rebuilt from. Each case is a way a rebuild or a sweep would
 * go wrong: a replaced record coming back, a record read at the wrong offset after an append or a rewrite, a text with
 * multi-byte characters shifting every offset after it, usage counts lost.
 */
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { JsonlLog, UsageBook, fingerprint } from './store.js';

type Rec = { uid: string; text: string; n: number };
const dir = () => mkdtempSync(join(tmpdir(), 'memory-store-'));
const log = (path: string) => new JsonlLog<Rec, { n: number }>(path, (r) => ({ n: r.n }));

describe('JsonlLog', () => {
  it('reads each record back by its offset, the latest line of a uid winning, across a reload', async () => {
    const path = join(dir(), 'x.jsonl');
    const a = log(path);
    a.put([{ uid: 'a', text: 'первый — ünïcödé', n: 1 }, { uid: 'b', text: 'second', n: 2 }]);
    a.put([{ uid: 'a', text: 'replaced', n: 3 }]);
    expect(a.get('a')?.text).toBe('replaced');
    expect(a.get('b')?.text).toBe('second');
    expect(a.meta('a')).toEqual({ n: 3 });
    const b = log(path);
    await b.load();
    expect(b.size()).toBe(2);
    expect(b.get('a')?.text).toBe('replaced');
    expect(b.get('b')?.text).toBe('second');
    expect([...b.all()].map((r) => r.uid)).toEqual(['b', 'a']);
  });

  it('rewrites itself without replaced lines once it holds twice what is live — offsets still right', async () => {
    const path = join(dir(), 'x.jsonl');
    const l = log(path);
    for (let i = 0; i < 1200; i++) l.put([{ uid: `u${i % 3}`, text: `v${i} é`, n: i }]);
    l.compact();
    expect(readFileSync(path, 'utf8').trim().split('\n')).toHaveLength(3);
    expect(l.get('u0')?.text).toBe('v1197 é');
    l.put([{ uid: 'u3', text: 'after', n: 0 }]);
    expect(l.get('u3')?.text).toBe('after');
    expect(l.get('u2')?.text).toBe('v1199 é');
  });

  it('skips a torn last line', async () => {
    const path = join(dir(), 'x.jsonl');
    const l = log(path);
    l.put([{ uid: 'a', text: 'ok', n: 1 }]);
    const { appendFileSync } = await import('node:fs');
    appendFileSync(path, '{"uid":"b","te');
    const r = log(path);
    await r.load();
    expect(r.size()).toBe(1);
    expect(r.get('a')?.text).toBe('ok');
  });
});

describe('UsageBook', () => {
  it('counts what was shown and opened, and keeps it', () => {
    const path = join(dir(), 'usage.json');
    const u = new UsageBook(path);
    u.note(['m1', 'm2'], 'shown', 100);
    u.note(['m1'], 'opened', 200);
    u.flush();
    const again = new UsageBook(path);
    expect(again.of('m1')).toEqual({ shown: 1, opened: 1, lastUsed: 200 });
    expect(again.of('m3')).toEqual({ shown: 0, opened: 0, lastUsed: 0 });
  });
});

describe('fingerprint', () => {
  it('changes with any part, and does not confuse a list with its joined text', () => {
    expect(fingerprint('a', ['b', 'c'])).not.toBe(fingerprint('a', 'b,c'));
    expect(fingerprint('a', null)).toBe(fingerprint('a', ''));
    expect(fingerprint('a')).not.toBe(fingerprint('b'));
  });
});
