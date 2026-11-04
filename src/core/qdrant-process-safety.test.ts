/**
 * Memory's own Qdrant, run safely: a binary that is not there fails the start (and is not thrown in the host), a start
 * that fails leaves nothing running or scheduled, the child sees none of the host's secrets, what answers on the port
 * must prove it is ours before it gets the key, its log is private, and a Qdrant a dead host left on the storage is
 * stopped — an unrelated process with its pid never. Over a stand-in Qdrant binary (a Node script) on loopback.
 */
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterAll, afterEach, describe, expect, it } from 'vitest';
import { alive, bootTime, QdrantProcess } from './qdrant-process.js';

const wait = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
async function until(ok: () => boolean, ms = 5000): Promise<boolean> {
  for (const end = Date.now() + ms; !ok(); await wait(20)) if (Date.now() > end) return false;
  return true;
}

const root = mkdtempSync(join(tmpdir(), 'memtrails-qp-'));
afterAll(() => rmSync(root, { recursive: true, force: true }));

/** A stand-in Qdrant: `ok` checks its key like Qdrant; `anykey` takes any key (not ours); `exit` dies at once. Each
 *  start is a line in spawns.log, and its environment is env.json — both in the folder it runs in. */
function fakeQdrant(mode: 'ok' | 'anykey' | 'exit'): string {
  const path = join(root, `fake-qdrant-${mode}.cjs`);
  writeFileSync(path, `#!/usr/bin/env node
const http = require('node:http'); const fs = require('node:fs');
const args = process.argv.slice(2);
const port = Number(/http_port: (\\d+)/.exec(fs.readFileSync(args[args.indexOf('--config-path') + 1], 'utf8'))[1]);
fs.appendFileSync('spawns.log', process.pid + '\\n');
fs.writeFileSync('env.json', JSON.stringify(process.env));
if (${JSON.stringify(mode)} === 'exit') process.exit(1);
const key = process.env.QDRANT__SERVICE__API_KEY;
http.createServer((req, res) => {
  if (req.url !== '/readyz' && ${JSON.stringify(mode)} === 'ok' && req.headers['api-key'] !== key) { res.writeHead(401); return res.end('{}'); }
  res.writeHead(200); res.end('{"result":{"collections":[]}}');
}).listen(port, '127.0.0.1');
process.on('SIGTERM', () => process.exit(0));
`);
  chmodSync(path, 0o755);
  return path;
}

let n = 0;
const folder = (): string => join(root, `q${n++}`);
const spawns = (dir: string): number[] => (existsSync(join(dir, 'spawns.log')) ? readFileSync(join(dir, 'spawns.log'), 'utf8').trim().split('\n').map(Number) : []);

const started: QdrantProcess[] = [];
const others: ChildProcess[] = [];
afterEach(async () => {
  await Promise.all(started.splice(0).map((p) => p.stopAndWait()));
  for (const c of others.splice(0)) c.kill('SIGKILL');
});
function qdrant(binary: string, dir = folder()): QdrantProcess & { dir: string } {
  const p = Object.assign(new QdrantProcess({ binary, dir, log: () => {} }), { dir });
  started.push(p);
  return p;
}

describe('a start that fails', () => {
  it('rejects for a binary that is not there — never an uncaught spawn error — and starts nothing later', async () => {
    const p = qdrant('memtrails-no-such-qdrant');
    await expect(p.start()).rejects.toThrow(/could not be started.*ENOENT/);
    expect(p.current()).toBeNull();
  }, 10_000);

  it('schedules no restart when Qdrant dies while starting: whoever started it was told, and may be gone', async () => {
    const p = qdrant(fakeQdrant('exit'));
    await expect(p.start()).rejects.toThrow(/exited/);
    await wait(1500); // the first restart would come after 1 s
    expect(spawns(p.dir)).toHaveLength(1);
  }, 10_000);

  it('refuses what answers on the port but takes any key: it is not ours, and is stopped', async () => {
    const p = qdrant(fakeQdrant('anykey'));
    await expect(p.start()).rejects.toThrow(/not this memory's Qdrant: it took a wrong key/);
    const [pid] = spawns(p.dir);
    expect(await until(() => !alive(pid!))).toBe(true);
    expect(p.current()).toBeNull();
  }, 10_000);
});

describe('a running Qdrant', () => {
  it('is started again when it dies after it was up', async () => {
    const p = qdrant(fakeQdrant('ok'));
    await p.start();
    process.kill(spawns(p.dir)[0]!, 'SIGKILL');
    expect(await until(() => spawns(p.dir).length === 2 && p.current() !== null, 5000)).toBe(true);
  }, 10_000);

  it('gets none of the host\'s environment but what it needs: no secrets, no QDRANT__ overrides', async () => {
    const keep = { secret: process.env.MEMTRAILS_TEST_SECRET, host: process.env.QDRANT__SERVICE__HOST };
    process.env.MEMTRAILS_TEST_SECRET = 'hunter2';
    process.env.QDRANT__SERVICE__HOST = '0.0.0.0';
    try {
      const p = qdrant(fakeQdrant('ok'));
      const e = await p.start();
      const env = JSON.parse(readFileSync(join(p.dir, 'env.json'), 'utf8')) as Record<string, string>;
      expect(env.MEMTRAILS_TEST_SECRET).toBeUndefined();
      expect(env.QDRANT__SERVICE__HOST).toBeUndefined();
      expect(env.QDRANT__SERVICE__API_KEY).toBe(e.apiKey);
      expect(env.PATH).toBe(process.env.PATH);
    } finally {
      if (keep.secret === undefined) delete process.env.MEMTRAILS_TEST_SECRET; else process.env.MEMTRAILS_TEST_SECRET = keep.secret;
      if (keep.host === undefined) delete process.env.QDRANT__SERVICE__HOST; else process.env.QDRANT__SERVICE__HOST = keep.host;
    }
  });

  it('keeps its folder and its log private', async () => {
    const p = qdrant(fakeQdrant('ok'));
    mkdirSync(p.dir, { recursive: true });
    writeFileSync(join(p.dir, 'qdrant.log'), 'old\n', { mode: 0o644 });
    chmodSync(join(p.dir, 'qdrant.log'), 0o644);
    await p.start();
    expect(statSync(join(p.dir, 'qdrant.log')).mode & 0o777).toBe(0o600);
    const fresh = qdrant(fakeQdrant('ok'));
    await fresh.start();
    expect(statSync(fresh.dir).mode & 0o777).toBe(0o700);
    expect(statSync(join(fresh.dir, 'qdrant.log')).mode & 0o777).toBe(0o600);
  });
});

/** A host process that starts a Qdrant on `dir`, says so, then exits without stopping it (`exit`) or waits (`hang`). */
async function host(dir: string, binary: string, how: 'exit' | 'hang'): Promise<{ child: ChildProcess; qdrantPid: number }> {
  const script = join(root, 'host.mts');
  writeFileSync(script, `import { QdrantProcess } from ${JSON.stringify(resolve('src/core/qdrant-process.ts'))};
const p = new QdrantProcess({ binary: process.argv[2], dir: process.argv[3], log: () => {} });
await p.start();
process.stdout.write('up\\n');
if (process.argv[4] === 'exit') process.exit(0); else setInterval(() => {}, 1000);
`);
  const child = spawn(process.execPath, ['--import', 'tsx', script, binary, dir, how], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'inherit'] });
  others.push(child);
  await new Promise<void>((ok, fail) => {
    child.stdout!.on('data', (d: Buffer) => { if (d.toString().includes('up')) ok(); });
    child.once('exit', (code) => fail(new Error(`host exited ${code}`)));
  });
  return { child, qdrantPid: spawns(dir)[0]! };
}

describe('a host that goes without close()', () => {
  it('stops its Qdrant when it exits', async () => {
    const dir = folder();
    const { child, qdrantPid } = await host(dir, fakeQdrant('ok'), 'exit');
    await new Promise((ok) => (child.exitCode !== null ? ok(null) : child.once('exit', ok)));
    expect(await until(() => !alive(qdrantPid))).toBe(true);
  }, 30_000);

  it('killed outright, leaves a Qdrant the next start on that storage stops first', async () => {
    const dir = folder();
    const binary = fakeQdrant('ok');
    const { child, qdrantPid } = await host(dir, binary, 'hang');
    child.kill('SIGKILL');
    await new Promise((ok) => child.once('exit', ok));
    await wait(200);
    expect(alive(qdrantPid)).toBe(true); // orphaned, still on the storage
    const next = qdrant(binary, dir);
    await next.start();
    expect(alive(qdrantPid)).toBe(false);
    expect(spawns(dir)).toHaveLength(2);
  }, 30_000);

  it('never stops an unrelated process that has the recorded pid', async () => {
    const dir = folder();
    const stranger = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' });
    others.push(stranger);
    await until(() => !!stranger.pid);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'qdrant.pid'), JSON.stringify({ pid: stranger.pid, config: join(dir, 'config.yaml'), boot: bootTime() }));
    await qdrant(fakeQdrant('ok'), dir).start();
    expect(alive(stranger.pid!)).toBe(true);
    expect(stranger.exitCode).toBeNull();
  });
});

describe('alive', () => {
  it('tells a gone pid from a live one', () => {
    const gone = Number(spawnSync(process.execPath, ['-e', 'console.log(process.pid)'], { encoding: 'utf8' }).stdout.trim());
    expect(alive(gone)).toBe(false);
    expect(alive(process.pid)).toBe(true);
    expect(alive(0)).toBe(false);
  });
});
