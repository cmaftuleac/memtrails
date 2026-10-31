import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { MemoryRecord } from '../records.js';
import { batches } from './index.js';
import { fromJsonl } from './jsonl.js';

async function all<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

describe('fromJsonl', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memtrails-jsonl-'));

  it('reads record-shaped lines in any language and skips and counts the bad ones', async () => {
    const path = join(dir, 'records.jsonl');
    writeFileSync(path, '﻿' + [
      JSON.stringify({ id: 'ja', text: '会議は火曜日に延期されました 🎉', title: '会議', at: '2024-05-01T10:00:00Z' }),
      '{ not json',
      JSON.stringify({ text: 'no id here' }),
      '',
      JSON.stringify({ id: 7, text: 'Întâlnirea de mâine e la București, în sala „Ștefan”.' }),
      JSON.stringify({ id: 'ar', text: 'تم تأجيل الاجتماع إلى يوم الثلاثاء' }),
      JSON.stringify({ id: 'bad-time', text: 'x', at: 'whenever' }),
      '[1, 2]',
    ].join('\r\n'));
    const skipped: string[] = [];
    const stats = { lines: 0, records: 0, skipped: 0 };
    const records = await all(fromJsonl(path, { onSkip: (_line, reason) => skipped.push(reason), stats, source: 'notes' }));
    expect(records.map((r) => r.id)).toEqual(['ja', '7', 'ar']);
    expect(records[0]).toMatchObject({ title: '会議', text: '会議は火曜日に延期されました 🎉', source: 'notes' });
    expect(records[1].text).toBe('Întâlnirea de mâine e la București, în sala „Ștefan”.');
    expect(stats).toEqual({ lines: 7, records: 3, skipped: 4 });
    expect(skipped).toHaveLength(4);
    expect(skipped[0]).toMatch(/^not JSON/);
    expect(skipped[1]).toMatch(/needs an id/);
  });

  it('maps any object shape, leaving out what map returns null for', async () => {
    const path = join(dir, 'chat.jsonl');
    writeFileSync(path, [
      { msg_id: 1, body: 'Привет, как дела?', who: 'Олег', ts: 1714557600000 },
      { msg_id: 2, body: '', who: 'bot' },
      { msg_id: 3, body: 'नमस्ते, सब ठीक है।', who: 'प्रिया', ts: 1714557660000 },
    ].map((o) => JSON.stringify(o)).join('\n') + '\n');
    const reasons: string[] = [];
    const map = (o: unknown): MemoryRecord | null => {
      const m = o as { msg_id: number; body: string; who: string; ts?: number };
      return m.body ? { id: String(m.msg_id), text: m.body, author: m.who, at: m.ts, kind: 'message' } : null;
    };
    const records = await all(fromJsonl(path, map, { onSkip: (_l, r) => reasons.push(r) }));
    expect(records.map((r) => [r.id, r.author, r.text])).toEqual([['1', 'Олег', 'Привет, как дела?'], ['3', 'प्रिया', 'नमस्ते, सब ठीक है।']]);
    expect(reasons).toEqual(['left out by map']);
  });

  it('batches records as they come', async () => {
    async function* count(): AsyncIterable<number> {
      for (let i = 0; i < 7; i++) yield i;
    }
    expect(await all(batches(count(), 3))).toEqual([[0, 1, 2], [3, 4, 5], [6]]);
  });
});
