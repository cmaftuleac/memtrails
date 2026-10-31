import { describe, expect, it } from 'vitest';
import { sentenceRoom, sentenceSpans, type RoomItem } from './room.js';
import { WordStore, type WordEncoder, type WordVectors } from './words.js';

const DIMS = 16;
function wordVector(word: string): Float32Array {
  const v = new Float32Array(DIMS);
  let h = 2166136261;
  for (const c of word.toLowerCase()) h = Math.imul(h ^ c.codePointAt(0)!, 16777619) >>> 0;
  for (let d = 0; d < DIMS; d++) {
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    v[d] = (h & 1 ? 1 : -1) / Math.sqrt(DIMS);
  }
  return v;
}
const encode: WordEncoder = async (texts) => texts.map((t): WordVectors => {
  const words = t.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
  const data = new Float32Array(words.length * DIMS);
  words.forEach((w, i) => data.set(wordVector(w), i * DIMS));
  return { words: words.length, dims: DIMS, data };
});

describe('the sentence room', () => {
  it('cuts at the end marks of any script and at line breaks, never inside a decimal', () => {
    const text = 'Version 2.5 shipped today. Did it work? Yes!\nNext line here。下一句在这里。';
    const got = sentenceSpans(text).map(([a, b]) => text.slice(a, b));
    expect(got).toEqual(['Version 2.5 shipped today.', 'Did it work?', 'Next line here。下一句在这里。']);
  });

  it('cuts a sentence too long to choose into pieces', () => {
    expect(sentenceSpans('word '.repeat(300)).length).toBeGreaterThan(2);
  });

  it('keeps the sentence that answers, from deep in a memory, over the start of every other', async () => {
    const filler = 'We talked about many different and unrelated things for a long while that afternoon. ';
    const items: RoomItem[] = [
      { id: 'a', head: '(1 Mar)', ts: 2, text: `${filler.repeat(6)}The gateway certificate expired because the renewal cron was removed.` },
      { id: 'b', head: '(2 Mar)', ts: 1, text: filler.repeat(8) },
    ];
    const room = await sentenceRoom('why did the gateway certificate expire', items, encode, new WordStore(), 160);
    expect(room).toContain('certificate expired');
    expect(room.length).toBeLessThanOrEqual(160);
    expect(room.startsWith('(1 Mar)')).toBe(true);
  });

  it('shows each memory once, oldest first, its sentences in their own order', async () => {
    const items: RoomItem[] = [
      { id: 'new', head: '(new)', ts: 9, text: 'Ana likes the Italian place. Ana goes there on Fridays.' },
      { id: 'old', head: '(old)', ts: 1, text: 'Ana once mentioned the Italian place near the office.' },
    ];
    const room = await sentenceRoom('where does Ana eat on Fridays', items, encode, new WordStore(), 1000);
    expect(room.split('\n')).toEqual([
      '(old) Ana once mentioned the Italian place near the office.',
      '(new) Ana likes the Italian place. … Ana goes there on Fridays.',
    ]);
  });
});
