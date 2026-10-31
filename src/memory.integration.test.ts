import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { openMemory, type Memory, type MemoryOptions } from './memory.js';
import type { MemoryRecord } from './records.js';

/** The whole library on the real embedding model and a real Qdrant, with no language model and no server.
 *  Both are fetched into the cache on first use; MEMTRAILS_MODEL_DIR and QDRANT_BINARY point at copies instead. */
const opts = (home: string): MemoryOptions => ({
  home,
  ...(process.env.MEMTRAILS_MODEL_DIR ? { embedding: { modelDir: process.env.MEMTRAILS_MODEL_DIR, threads: 2 } } : {}),
  ...(process.env.QDRANT_BINARY ? { qdrant: { binary: process.env.QDRANT_BINARY } } : {}),
});

const mail: MemoryRecord[] = [
  { id: '<a1@ops>', source: 'mail', kind: 'email', collection: 'ops', thread: 'cert', author: 'Ana Popescu', title: 'Gateway certificate expired',
    text: 'The TLS certificate of the gateway expired on the 1st. Renewal stopped because the cron job was removed in March.', at: '2026-04-02' },
  { id: '<a2@ops>', source: 'mail', kind: 'email', collection: 'ops', thread: 'cert', author: 'Ion', title: 'Re: Gateway certificate expired',
    text: 'I added the cron job back and renewed it by hand. It is valid until July.', at: '2026-04-03' },
  { id: 'n1', source: 'notes', kind: 'note', collection: 'home', title: 'Router', text: 'Parola routerului este în caietul albastru din sertarul de la birou.', at: '2026-04-04' },
  { id: 'c9f1e2', source: 'repo', kind: 'commit', collection: 'ops', author: 'Ion', title: 'Restore certificate renewal cron', text: 'Files: deploy/cron/renew.sh', at: '2026-04-03' },
];

describe('openMemory', () => {
  it('runs from a relative home, as the README shows', async () => {
    const cwd = process.cwd();
    const base = await mkdtemp(join(tmpdir(), 'memtrails-rel-'));
    process.chdir(base);
    try {
      const m = await openMemory(opts('./memory'));
      await m.add({ id: 'r', text: 'A relative home works.' });
      expect((await m.search({ query: 'relative home', limit: 1 }))[0]?.summary).toContain('relative');
      await m.close();
    } finally {
      process.chdir(cwd);
      await rm(base, { recursive: true, force: true });
    }
  }, 300_000);


  let home: string;
  let memory: Memory;

  beforeAll(async () => {
    home = await mkdtemp(join(tmpdir(), 'memtrails-'));
    memory = await openMemory(opts(home));
    expect(await memory.add(mail)).toEqual({ added: 4, unchanged: 0 });
    await memory.addConversation({ id: 'lunch', collection: 'home', at: '2026-04-05', turns: [
      { role: 'user', text: 'Where did we say we would have lunch on Friday with Ana?' },
      { role: 'assistant', text: 'At the small Italian place near the office, at one o\'clock.' },
    ] });
  }, 600_000);

  afterAll(async () => {
    await memory?.close();
    await rm(home, { recursive: true, force: true });
  });

  it('finds what was added at once, with who wrote it and where it came from', async () => {
    const [hit] = await memory.search({ query: 'why did the certificate stop renewing?', limit: 3 });
    expect(hit?.title).toContain('certificate');
    expect(hit?.kind).toBe('email');
    expect(hit?.record).toMatchObject({ source: 'mail', author: expect.any(String) });
  }, 60_000);

  it('searches within collections and kinds', async () => {
    const commits = await memory.search({ query: 'certificate renewal', kinds: ['commit'], limit: 5 });
    expect(commits.map((h) => h.kind)).toEqual(['commit']);
    const home_ = await memory.search({ query: 'certificate renewal', collections: ['home'], limit: 5 });
    expect(home_.every((h) => h.project === 'home')).toBe(true);
  }, 60_000);

  it('answers in another language than it was asked in, and as an agent reads it', async () => {
    const text = await memory.searchText({ query: 'unde am scris parola de la router?', limit: 3 });
    expect(text.split('\n')[0]).toMatch(/Router/);
  }, 60_000);

  it('fills a small room with the sentence that answers', async () => {
    const room = await memory.room({ query: 'what made the renewal stop?', chars: 300 });
    expect(room.length).toBeLessThanOrEqual(300);
    expect(room).toContain('cron job was removed');
  }, 60_000);

  it('opens a record whole, and gives an agent tools that work', async () => {
    const [hit] = await memory.search({ query: 'lunch with Ana', limit: 1 });
    expect(String((await memory.open(hit!.id))?.body)).toContain('Italian');
    const search = memory.tools().find((t) => t.name === 'memory_search')!;
    expect(search.jsonSchema).toMatchObject({ type: 'object' });
    const out = await search.handler({ query: 'gateway certificate' });
    expect(out.content[0]!.text).toContain('certificate');
  }, 60_000);

  it('adds nothing twice, updates a changed record, forgets on request', async () => {
    expect(await memory.add(mail)).toEqual({ added: 0, unchanged: 4 });
    expect(await memory.add({ ...mail[1]!, text: 'Renewed by hand; valid until August.' })).toEqual({ added: 1, unchanged: 0 });
    expect((await memory.search({ query: 'valid until August', limit: 3 }))[0]?.summary).toContain('August');
    expect(await memory.remove([{ id: '<a2@ops>', source: 'mail' }])).toBe(1);
    expect((await memory.search({ query: 'valid until August', limit: 10 })).some((h) => h.summary.includes('August'))).toBe(false);
  }, 120_000);

  it('streams a source in, and comes back as it was after a restart and after its index is lost', async () => {
    async function* many() { for (let i = 0; i < 40; i++) yield { id: `msg-${i}`, source: 'chat', kind: 'message', text: `Standup ${i}: worked on ticket ${1000 + i}.` }; }
    expect(await memory.import(many(), { batch: 16 })).toEqual({ added: 40, unchanged: 0 });
    const before = memory.stats();
    await memory.close();
    memory = await openMemory(opts(home));
    expect(memory.stats()).toEqual(before);
    expect(await memory.indexWords()).toBe(0);
    await memory.rebuild();
    expect((await memory.search({ query: 'ticket 1017', limit: 3 }))[0]?.summary).toContain('1017');
    expect((await memory.search({ query: 'expired TLS certificate', limit: 1 }))[0]?.summary).toContain('certificate');
  }, 600_000);

  it('refuses a record it cannot keep, and says why', async () => {
    await expect(memory.add({ id: '', text: 'x' })).rejects.toThrow(/id/);
    await expect(memory.add({ id: 'a', text: 'x', at: 'yesterday-ish' })).rejects.toThrow(/time/);
  });
});
