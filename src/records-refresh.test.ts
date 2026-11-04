/** A conversation added again (records.ts): unchanged turns keep their fingerprint, whenever they are added. */
import { describe, expect, it } from 'vitest';
import { conversationEntries, type Conversation } from './records.js';

const NOW = Date.parse('2026-05-01T00:00:00Z');
const turns: Conversation['turns'] = [
  { role: 'user', text: 'Deploy failed.' },
  { role: 'assistant', text: 'The certificate expired.' },
  { role: 'user', text: 'Renewed it.' },
  { role: 'assistant', text: 'Good.' },
];

describe('a conversation added again', () => {
  it('is unchanged when it has no times — its turns still a minute apart from when it was added', () => {
    const first = conversationEntries({ id: 'c', turns }, NOW);
    const again = conversationEntries({ id: 'c', turns }, NOW + 3_600_000);
    expect(again.map((e) => e.fingerprint)).toEqual(first.map((e) => e.fingerprint));
    expect(first.map((e) => e.ts)).toEqual([NOW, NOW + 60_000]);
  });

  it('changes when a time it was given changes, and only that turn', () => {
    const at = { id: 'c', at: NOW, turns };
    const a = conversationEntries(at, NOW);
    expect(conversationEntries(at, NOW + 1).map((e) => e.fingerprint)).toEqual(a.map((e) => e.fingerprint));
    expect(a.map((e) => e.ts)).toEqual([NOW, NOW + 60_000]);
    expect(conversationEntries({ ...at, at: NOW + 5 }, NOW)[0]!.fingerprint).not.toBe(a[0]!.fingerprint);
    const moved = conversationEntries({ id: 'c', turns: [turns[0]!, turns[1]!, { ...turns[2]!, at: NOW + 7 }, turns[3]!] }, NOW);
    expect(moved[0]!.fingerprint).toBe(conversationEntries({ id: 'c', turns }, NOW)[0]!.fingerprint);
    expect(moved[1]!.fingerprint).not.toBe(conversationEntries({ id: 'c', turns }, NOW)[1]!.fingerprint);
    expect(moved[1]!.ts).toBe(NOW + 7);
  });
});
