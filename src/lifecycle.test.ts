/**
 * openMemory's life: what an open that fails leaves behind, one process a home, the index written again when its
 * layout is not the logs', jobs one at a time, close() stopping what runs, forgetting a record whole. Over a stand-in
 * Qdrant and a stand-in embedding server on loopback (and a stand-in Qdrant binary) — no model, no network.
 */
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { spawnSync } from 'node:child_process';
import { INDEX_LAYOUT } from './core/qdrant-client.js';
import { alive, bootTime } from './core/qdrant-process.js';
import { RECALL_TRIM_EVERY, RecallBook } from './core/recall-log.js';
import { ltmLog, type MemoryItem } from './core/store.js';
import { openMemory, type Memory, type MemoryOptions } from './memory.js';
import { recordUid } from './records.js';

// Nothing is fetched: a test that reaches for the model or the Qdrant binary fails instead of downloading them.
vi.mock(import('./core/model.js'), async (orig) => ({ ...(await orig()), ensureModel: async () => { throw new Error('test: no model download'); } }));
vi.mock(import('./core/qdrant-binary.js'), async (orig) => ({ ...(await orig()), ensureQdrant: async () => { throw new Error('test: no Qdrant download'); } }));
vi.mock(import('./core/local-embedder.js'), async (orig) => ({ ...(await orig()), loadLocalEmbedder: async () => { throw new Error('test: no local model'); } }));

const DIMS = 768;
const wait = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
async function until(ok: () => boolean, ms = 5000): Promise<void> {
  for (const end = Date.now() + ms; !ok(); await wait(10)) if (Date.now() > end) throw new Error('timed out');
}

async function body(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const text = Buffer.concat(chunks).toString('utf8');
  return text ? JSON.parse(text) as Record<string, unknown> : {};
}

function listen(server: Server): Promise<string> {
  return new Promise((ok) => server.listen(0, '127.0.0.1', () => {
    const a = server.address();
    ok(`http://127.0.0.1:${typeof a === 'object' && a ? a.port : 0}`);
  }));
}

type Point = { id: string; payload: Record<string, unknown> };

/** Qdrant's REST API as far as memory uses it, in memory: points by id, deletes by item, every key point a search hit. */
function fakeQdrant() {
  const s = { exists: false, points: new Map<string, Point>(), delay: 0, upserts: 0, drops: 0 };
  const server = createServer(async (req, res) => {
    const b = await body(req);
    const path = new URL(req.url ?? '/', 'http://x').pathname;
    const send = (status: number, result: unknown): void => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify({ result })); };
    const m = /^\/collections\/[^/]+(\/.*)?$/.exec(path);
    if (!m) return send(200, { collections: [] });
    const rest = m[1] ?? '';
    if (!rest) {
      if (req.method === 'GET') return s.exists ? send(200, { config: { optimizer_config: { default_segment_number: 2 } }, payload_schema: {} }) : send(404, null);
      if (req.method === 'PUT') { s.exists = true; return send(200, true); }
      if (req.method === 'DELETE') { s.exists = false; s.points.clear(); s.drops++; return send(200, true); }
      return send(200, true);
    }
    if (!s.exists) return send(404, null);
    if (rest === '/index') return send(200, true);
    if (rest === '/points' && req.method === 'PUT') {
      s.upserts++;
      await wait(s.delay);
      if (!s.exists) return send(404, null);
      for (const p of b.points as Point[]) s.points.set(p.id, { id: p.id, payload: p.payload });
      return send(200, true);
    }
    if (rest === '/points') return send(200, (b.ids as string[]).flatMap((id) => (s.points.has(id) ? [s.points.get(id)!] : [])));
    if (rest === '/points/delete') {
      const items = ((b.filter as { must: Array<{ key: string; match: { any?: string[]; value?: string } }> }).must);
      const any = items.find((c) => c.key === 'item')?.match.any ?? [];
      const part = items.find((c) => c.key === 'part')?.match.value;
      for (const [id, p] of s.points) if (any.includes(String(p.payload.item)) && (!part || p.payload.part === part)) s.points.delete(id);
      return send(200, true);
    }
    if (rest === '/points/count') return send(200, { count: s.points.size });
    if (rest === '/points/payload') return send(200, true);
    if (rest === '/points/scroll') return send(200, { points: [] });
    if (rest === '/points/query') return send(200, { points: [] });
    if (rest === '/points/query/batch') return send(200, (b.searches as unknown[]).map(() => ({ points: [] })));
    if (rest === '/points/query/groups') {
      const keys = [...s.points.values()].filter((p) => p.payload.part === 'key' || p.payload.tier === 'stm');
      const groups = new Map<string, Point>();
      for (const p of keys) if (!groups.has(String(p.payload.item))) groups.set(String(p.payload.item), p);
      return send(200, { groups: [...groups].slice(0, Number(b.limit ?? 10)).map(([item, p], i) => ({ id: item, hits: [{ id: p.id, score: 0.9 - i * 0.01, payload: p.payload }] })) });
    }
    return send(200, true);
  });
  return { s, server };
}

/** An OpenAI-style embeddings server: a fixed unit-ish vector a text, from its characters. */
function fakeEmbedder() {
  const s = { calls: 0 };
  const server = createServer(async (req, res) => {
    const b = await body(req);
    s.calls++;
    const data = (b.input as string[]).map((t, index) => {
      let h = 7;
      for (const ch of t) h = (h * 31 + ch.charCodeAt(0)) % 100_003;
      return { index, embedding: Array.from({ length: DIMS }, (_, i) => ((h + i * 7919) % 1000) / 1000 + 0.001) };
    });
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ data }));
  });
  return { s, server };
}

const q = fakeQdrant();
const emb = fakeEmbedder();
const rr = { calls: 0 };
const reranker = createServer(async (req, res) => {
  const b = await body(req);
  rr.calls++;
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ results: (b.documents as string[]).map((_, index) => ({ index, relevance_score: 1 / (index + 1) })) }));
});
let qUrl = '';
let embedUrl = '';
let rerankUrl = '';
beforeAll(async () => { [qUrl, embedUrl, rerankUrl] = await Promise.all([listen(q.server), listen(emb.server), listen(reranker)]); });
afterAll(() => { q.server.close(); emb.server.close(); reranker.close(); });

const homes: string[] = [];
function home(): string {
  const d = mkdtempSync(join(tmpdir(), 'memtrails-life-'));
  homes.push(d);
  return d;
}
afterAll(() => { for (const d of homes) rmSync(d, { recursive: true, force: true }); });

/** A fresh index for each test: the fake is shared. */
function reset(): void { q.s.exists = false; q.s.points.clear(); q.s.delay = 0; q.s.upserts = 0; q.s.drops = 0; }

const opts = (dir: string, more: Partial<MemoryOptions> = {}): MemoryOptions => ({
  home: dir, embedding: { server: { url: embedUrl } }, qdrant: { url: qUrl }, rerank: false, ...more,
});
const records = (n: number, from = 0) => Array.from({ length: n }, (_, i) => ({ id: `r${from + i}`, text: `Record number ${from + i}: the gateway certificate expired on day ${from + i}.`, at: '2025-01-02' }));
const stmItems = (): string[] => [...new Set([...q.s.points.values()].filter((p) => p.payload.tier === 'stm').map((p) => String(p.payload.item)))];

/** A Qdrant stand-in binary that comes up, proves its key, and fails the collection call (`broken`). */
function brokenQdrantBinary(dir: string): string {
  const path = join(dir, 'fake-qdrant.cjs');
  writeFileSync(path, `#!/usr/bin/env node
const http = require('node:http'); const fs = require('node:fs');
const args = process.argv.slice(2);
const port = Number(/http_port: (\\d+)/.exec(fs.readFileSync(args[args.indexOf('--config-path') + 1], 'utf8'))[1]);
fs.appendFileSync('spawns.log', process.pid + '\\n');
const key = process.env.QDRANT__SERVICE__API_KEY;
http.createServer((req, res) => {
  if (req.url !== '/readyz' && req.headers['api-key'] !== key) { res.writeHead(401); return res.end('{}'); }
  if (req.url === '/readyz' || req.url === '/collections') { res.writeHead(200); return res.end('{"result":{"collections":[]}}'); }
  res.writeHead(500); res.end('{"status":{"error":"broken"}}');
}).listen(port, '127.0.0.1');
process.on('SIGTERM', () => process.exit(0));
`);
  chmodSync(path, 0o755);
  return path;
}

describe('a failed open', () => {
  it('stops the Qdrant it started and frees the home: nothing keeps the host alive', async () => {
    const dir = home();
    const bin = brokenQdrantBinary(mkdtempSync(join(tmpdir(), 'memtrails-bin-')));
    await expect(openMemory({ ...opts(dir), qdrant: { binary: bin } })).rejects.toThrow(/500/);
    const pid = Number(readFileSync(join(dir, 'qdrant', 'spawns.log'), 'utf8').trim());
    expect(pid).toBeGreaterThan(0);
    expect(alive(pid)).toBe(false);
    expect(existsSync(join(dir, 'memory.lock'))).toBe(false);
    // Free again: the same error, not "is open".
    await expect(openMemory({ ...opts(dir), qdrant: { binary: bin } })).rejects.toThrow(/500/);
  }, 20_000);

  it('rejects for a Qdrant binary that is not there, rather than throwing in the host', async () => {
    const dir = home();
    await expect(openMemory({ ...opts(dir), qdrant: { binary: 'memtrails-no-such-qdrant' } })).rejects.toThrow(/could not be started.*ENOENT/);
    expect(existsSync(join(dir, 'memory.lock'))).toBe(false);
  }, 10_000);
});

describe('one process a home', () => {
  it('refuses a second open while the first holds it, and opens once it is closed', async () => {
    reset();
    const dir = home();
    const a = await openMemory(opts(dir));
    await expect(openMemory(opts(dir))).rejects.toThrow(new RegExp(`is open in process ${process.pid}`));
    await a.close();
    expect(existsSync(join(dir, 'memory.lock'))).toBe(false);
    const b = await openMemory(opts(dir));
    await b.close();
  });

  it('takes over a lock whose process is gone, or that was written before a reboot', async () => {
    reset();
    const dir = home();
    const gone = spawnSync(process.execPath, ['-e', 'console.log(process.pid)'], { encoding: 'utf8' });
    const deadPid = Number(gone.stdout.trim());
    expect(alive(deadPid)).toBe(false);
    writeFileSync(join(dir, 'memory.lock'), JSON.stringify({ pid: deadPid, boot: bootTime() }));
    await (await openMemory(opts(dir))).close();
    // A live pid (this very process) but another boot: the pid was someone else's then.
    writeFileSync(join(dir, 'memory.lock'), JSON.stringify({ pid: process.pid, boot: bootTime() - 3_600_000 }));
    await (await openMemory(opts(dir))).close();
  });

  it('keeps the home private, every open', async () => {
    reset();
    const dir = home();
    chmodSync(dir, 0o755);
    await (await openMemory(opts(dir))).close();
    expect(statSync(dir).mode & 0o777).toBe(0o700);
    const fresh = join(home(), 'nested', 'mem');
    await (await openMemory(opts(fresh))).close();
    expect(statSync(fresh).mode & 0o777).toBe(0o700);
  });
});

describe('the index and the logs', () => {
  it('writes the index again when it was made in another layout, or in one not written down', async () => {
    reset();
    const dir = home();
    const m = await openMemory(opts(dir));
    await m.add(records(3));
    await m.close();
    expect(JSON.parse(readFileSync(join(dir, 'index.json'), 'utf8'))).toEqual({ layout: INDEX_LAYOUT });

    // An index from before the layout was written down, holding something the logs do not.
    rmSync(join(dir, 'index.json'));
    q.s.points.set('junk', { id: 'junk', payload: { item: 'stm:junk', tier: 'stm' } });
    await (await openMemory(opts(dir))).close();
    expect(q.s.points.has('junk')).toBe(false);
    expect(stmItems().length).toBe(3);
    expect(JSON.parse(readFileSync(join(dir, 'index.json'), 'utf8'))).toEqual({ layout: INDEX_LAYOUT });

    // The same layout: left as it is.
    const drops = q.s.drops;
    await (await openMemory(opts(dir))).close();
    expect(q.s.drops).toBe(drops);

    // Another layout: dropped and written again.
    writeFileSync(join(dir, 'index.json'), JSON.stringify({ layout: INDEX_LAYOUT - 1 }));
    q.s.points.set('junk', { id: 'junk', payload: { item: 'stm:junk', tier: 'stm' } });
    await (await openMemory(opts(dir))).close();
    expect(q.s.drops).toBe(drops + 1);
    expect(q.s.points.has('junk')).toBe(false);
    expect(stmItems().length).toBe(3);
  });

  it('forgets a record whole: its kept episode leaves the long-term log, so a rebuild does not bring it back', async () => {
    reset();
    const dir = home();
    let m = await openMemory(opts(dir));
    await m.add({ id: 'secret', text: 'The door code is 4711.', at: '2025-01-02' });
    await m.close();
    // What consolidation keeps of it: an episode, its own text.
    const uid = recordUid({ id: 'secret' });
    const ep: MemoryItem = {
      uid: `ep:${uid}`, kind: 'episode', scope: 'project', project: '', engine: 'record', session: '', cwd: null, where: '',
      ts: 1, firstTs: 1, title: 'door code', summary: 'The door code is 4711.', body: 'The door code is 4711.', files: [],
      appliesWhen: '', confidence: 'observed', importance: 2, status: 'active', provenance: [uid], seen: 1, fingerprint: 'x',
    };
    const log = ltmLog(join(dir, 'knowledge.jsonl'));
    await log.load();
    log.put([ep]);
    m = await openMemory(opts(dir));
    await m.rebuild();
    const items = (): string[] => [...q.s.points.values()].map((p) => String(p.payload.item));
    expect(items()).toContain(`ep:${uid}`);

    expect(await m.remove(['secret'])).toBe(1);
    expect(items()).not.toContain(`ep:${uid}`);
    await m.rebuild();
    expect(items()).not.toContain(`ep:${uid}`);
    await m.close();
    await log.load();
    expect(log.get(`ep:${uid}`)).toBeNull();
    expect(readFileSync(join(dir, 'knowledge.jsonl'), 'utf8')).toContain('"removed":true');
  });
});

describe('one job at a time', () => {
  it('a rebuild asked for while one runs waits for it, then writes everything: none of it lost to its drop', async () => {
    reset();
    const dir = home();
    const m = await openMemory(opts(dir));
    await m.add(records(70));
    q.s.delay = 20;
    const drops = q.s.drops;
    const first = m.rebuild();
    await until(() => q.s.drops > drops && stmItems().length > 0); // the first has written some of the new index
    const second = m.rebuild();
    await Promise.all([first, second]);
    expect(stmItems().length).toBe(70);
    expect(existsSync(join(dir, 'rebuild.json'))).toBe(false);
    await m.close();
  });

  it('two consolidations never ask the model at once', async () => {
    reset();
    const dir = home();
    let now = 0;
    let most = 0;
    const ask: MemoryOptions['llm'] = async () => {
      most = Math.max(most, ++now);
      await wait(50);
      now--;
      return { unavailable: 'test: the model is away' };
    };
    const m = await openMemory(opts(dir, { llm: ask, concurrency: 1 }));
    await m.add(records(1));
    await Promise.all([m.consolidate(), m.consolidate()]);
    expect(most).toBe(1);
    await m.close();
  });

  it('close() stops a running rebuild at its next step and waits for it; the next open finishes it', async () => {
    reset();
    const dir = home();
    let m: Memory = await openMemory(opts(dir));
    await m.add(records(200));
    q.s.delay = 30;
    const upserts = q.s.upserts;
    const running = m.rebuild();
    await until(() => q.s.upserts > upserts);
    await m.close();
    await running; // stopped, not failed
    const after = q.s.upserts;
    await wait(300);
    expect(q.s.upserts).toBe(after);
    expect(existsSync(join(dir, 'rebuild.json'))).toBe(true);
    expect(stmItems().length).toBeLessThan(200);
    await expect(m.rebuild()).rejects.toThrow(/closed/);

    q.s.delay = 0;
    m = await openMemory(opts(dir));
    expect(existsSync(join(dir, 'rebuild.json'))).toBe(false);
    expect(stmItems().length).toBe(200);
    await m.close();
  }, 20_000);
});

describe('the recall record', () => {
  it('is trimmed to its last days when memory opens, and close() waits for what was recorded', async () => {
    reset();
    const dir = home();
    mkdirSync(dir, { recursive: true });
    const old = Date.now() - 200 * 86_400_000;
    writeFileSync(join(dir, 'recall.jsonl'), `${JSON.stringify({ t: old, ev: 'open', item: 'stm:old' })}\n${JSON.stringify({ t: Date.now(), ev: 'open', item: 'stm:new' })}\n`);
    const m = await openMemory(opts(dir));
    await m.add(records(2));
    await m.search({ query: 'certificate' });
    await m.close();
    const lines = readFileSync(join(dir, 'recall.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l) as { item?: string; ev: string });
    expect(lines.map((l) => l.item ?? l.ev)).toEqual(['stm:new', 'search']);
  });
});

describe('the recall record, left open', () => {
  it('keeps to its cap without a restart: trimmed every so many events', async () => {
    const path = join(home(), 'recall.jsonl');
    const book = new RecallBook(path, Date.now, { ms: 86_400_000, bytes: 10_000 });
    for (let i = 0; i < RECALL_TRIM_EVERY; i++) book.record({ ev: 'open', item: `stm:${i}` });
    await book.settled();
    expect(statSync(path).size).toBeLessThanOrEqual(10_000);
  });
});

describe('servers the environment names', () => {
  it('embeds on MEMTRAILS_EMBED_URL and reranks on MEMTRAILS_RERANK_URL when the options name neither', async () => {
    reset();
    const dir = home();
    const before = { embed: process.env.MEMTRAILS_EMBED_URL, rerank: process.env.MEMTRAILS_RERANK_URL };
    process.env.MEMTRAILS_EMBED_URL = embedUrl;
    process.env.MEMTRAILS_RERANK_URL = rerankUrl;
    try {
      const calls = emb.s.calls;
      const m = await openMemory({ home: dir, qdrant: { url: qUrl } });
      await m.add(records(3));
      expect(emb.s.calls).toBeGreaterThan(calls);
      await m.search({ query: 'certificate' });
      expect(rr.calls).toBe(1);
      await m.close();
    } finally {
      for (const [k, v] of [['MEMTRAILS_EMBED_URL', before.embed], ['MEMTRAILS_RERANK_URL', before.rerank]] as const) {
        if (v === undefined) delete process.env[k]; else process.env[k] = v;
      }
    }
  });
});
