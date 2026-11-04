/** The word-vector cache after a crash (words.ts): what is added after a cut-off entry is read back on the next start. */
import { appendFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { WordStore, type PackedWords } from './words.js';

const packed = (seed: number): PackedWords => ({ words: 2, dims: 16, bits: Uint8Array.from([seed, seed + 1, seed + 2, seed + 3]) });

describe('WordStore after a crash', () => {
  let dir: string;
  beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'words-recovery-')); });
  afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

  it('drops a cut-off entry, so what is added after it loads on the next start', async () => {
    const file = join(dir, 'words.bin');
    await new WordStore(file).put([['first', packed(1)]]);
    // Half an entry: a whole head, half its bits.
    const head = Buffer.alloc(12);
    head.writeUInt16LE(2, 8);
    head.writeUInt16LE(16, 10);
    await appendFile(file, Buffer.concat([head, Buffer.from([9, 9])]));
    const s = new WordStore(file);
    await s.load();
    expect(s.size).toBe(1);
    await s.put([['second', packed(5)]]);
    const again = new WordStore(file);
    await again.load();
    expect(again.size).toBe(2);
    expect(again.get('first')).toEqual(packed(1));
    expect(again.get('second')).toEqual(packed(5));
  });
});
