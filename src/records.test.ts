import { describe, expect, it } from 'vitest';
import { episodeBody, renderForExtraction } from './core/episodes.js';
import { collectionKey, conversationEntries, entryOf, invalidRecord, recordUid, timeOf } from './records.js';

const NOW = Date.parse('2026-05-01T00:00:00Z');

describe('records', () => {
  it('keeps a titled record as title and text, its author, kind, collection, thread and reference', () => {
    const e = entryOf({ id: 'm1', source: 'mailbox', kind: 'email', title: 'Gateway certificate', author: 'Ana',
      text: 'It expired on the 1st.\nThe renewal job had been removed.', collection: 'Ops team', thread: 't-9', at: '2026-04-02', url: 'mid:1', metadata: { to: 'ops' } }, NOW);
    expect(e).toMatchObject({ engine: 'email', session: 't-9', project: 'Ops-team', title: 'Gateway certificate', request: 'Gateway certificate',
      reply: 'Ana: It expired on the 1st.\nThe renewal job had been removed.', ts: Date.parse('2026-04-02'), files: ['mid:1'],
      record: { id: 'm1', source: 'mailbox', author: 'Ana', url: 'mid:1', metadata: { to: 'ops' } } });
    expect(e.uid).toBe(recordUid({ id: 'm1', source: 'mailbox' }));
    expect(episodeBody(e)).toBe('Gateway certificate\nAna: It expired on the 1st.\nThe renewal job had been removed.');
    expect(renderForExtraction(e, 3)).toBe('## Record 3 · 2026-04-02 00:00 · email\nGateway certificate\nAna: It expired on the 1st.\nThe renewal job had been removed.');
  });

  it('takes the first line as the title of an untitled record, without saying it twice, in any script', () => {
    const e = entryOf({ id: 'n', text: '会议记录：部署推迟到周五\n原因是证书过期。' }, NOW);
    expect(e.title).toBe('会议记录：部署推迟到周五');
    expect(e.reply).toBe('原因是证书过期。');
    expect(e.ts).toBe(NOW);
    expect(e.engine).toBe('record');
    expect(e.session).toBe(e.uid);
    expect(collectionKey(undefined)).toBe('default');
  });

  it('changes its fingerprint when anything about it changes, and only then', () => {
    const r = { id: 'x', text: 'a\nb', author: 'Ion' };
    expect(entryOf(r, NOW).fingerprint).toBe(entryOf(r, NOW + 1).fingerprint);
    expect(entryOf({ ...r, text: 'a\nc' }, NOW).fingerprint).not.toBe(entryOf(r, NOW).fingerprint);
    expect(entryOf({ ...r, metadata: { k: 1 } }, NOW).fingerprint).not.toBe(entryOf(r, NOW).fingerprint);
  });

  it('says what is wrong with a record', () => {
    expect(invalidRecord({ id: '', text: 'x' })).toMatch(/id/);
    expect(invalidRecord({ id: 'a', text: '  ' })).toMatch(/no text/);
    expect(invalidRecord({ id: 'a', text: 'x', at: 'not a date' })).toMatch(/time/);
    expect(invalidRecord({ id: 'a', text: '', title: 'only a title' })).toBeNull();
    expect(timeOf(new Date(5))).toBe(5);
    expect(timeOf('')).toBeUndefined();
  });

  it('cuts a conversation at each time the person speaks, with what was answered until they spoke again', () => {
    const es = conversationEntries({ id: 'c1', collection: 'p', at: NOW, turns: [
      { role: 'assistant', text: 'Hello, how can I help?' },
      { role: 'user', text: 'Deploy failed.' },
      { role: 'user', text: 'Logs say TLS.' },
      { role: 'assistant', text: 'The certificate expired.' },
      { role: 'assistant', text: 'Renew it with the cron job.' },
      { role: 'user', text: 'Thanks', author: 'Ana' },
    ] }, NOW);
    expect(es.map((e) => [e.request, e.reply])).toEqual([
      ['Hello, how can I help?', 'Hello, how can I help?'],
      ['Deploy failed.\nLogs say TLS.', 'The certificate expired.\nRenew it with the cron job.'],
      ['Ana: Thanks', ''],
    ]);
    expect(es.map((e) => e.engine)).toEqual(['conversation', 'conversation', 'conversation']);
    expect(new Set(es.map((e) => e.session))).toEqual(new Set(['c1']));
    expect(es.map((e) => e.turn)).toEqual([0, 1, 2]);
    expect(es[1]!.said).toBe(es[1]!.reply);
    expect(renderForExtraction(es[1]!, 1)).toContain('Agent said:\nThe certificate expired.');
  });
});
