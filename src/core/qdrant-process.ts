/**
 * Qdrant as memory runs it when you give it no server of your own: one process for this memory, on a loopback port
 * picked at start, behind an API key made for that run, gRPC off, telemetry off, stopped with `close()`.
 *
 * Tuned for a laptop: the index on disk, memory-mapped (qdrant-client.ts), one optimisation thread, two search threads
 * and ONE web worker: Qdrant's default is one a core, each waking ~4 times a second, so an idle index woke ~83 times a
 * second and spent 2.7–4 ms/s of CPU with 69 threads; with one it is 6.5 wake-ups a second and ~0.2 ms/s.
 *
 * A process that dies by itself is started again with a growing pause; five deaths in ten minutes stop the attempts
 * until the next `start()`. A start that fails leaves nothing running and nothing scheduled. It gets a minimal
 * environment, proves it is ours before it is handed anything, is stopped when the host exits, and one a crashed host
 * left on the same storage is stopped before the next starts.
 */
import type { ChildProcess } from 'node:child_process';
import { execFileSync, spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { chmodSync, closeSync, mkdirSync, openSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { uptime } from 'node:os';
import { join, resolve } from 'node:path';
import { rootCertificates } from 'node:tls';

export interface QdrantEndpoint {
  url: string;
  apiKey: string;
}

/** A port the OS says is free on loopback. */
async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.unref();
    s.once('error', reject);
    s.listen(0, '127.0.0.1', () => {
      const addr = s.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      s.close(() => (port ? resolve(port) : reject(new Error('no free port'))));
    });
  });
}

/** The config file: what environment variables cannot say (gRPC off is `grpc_port: null`). */
export function qdrantConfig(storage: string, port: number): string {
  return [
    'log_level: WARN',
    'telemetry_disabled: true',
    'service:',
    '  host: 127.0.0.1',
    `  http_port: ${port}`,
    '  grpc_port: null',
    '  enable_cors: false',
    '  max_workers: 1',
    'storage:',
    `  storage_path: ${JSON.stringify(join(storage, 'storage'))}`,
    `  snapshots_path: ${JSON.stringify(join(storage, 'snapshots'))}`,
    '  on_disk_payload: true',
    '  performance:',
    '    max_search_threads: 2',
    '  optimizers:',
    '    max_optimization_threads: 1',
    '',
  ].join('\n');
}

const LOG_MAX = 4 * 1024 * 1024;
/** Beside the storage: the pid of the Qdrant running on it, with its config and the machine's boot (reapLeftover). */
const PID_FILE = 'qdrant.pid';

/** The process's log beside its data, rotated once at start when it has grown past 4 MB. Private: it names paths. */
function openLog(dir: string): number {
  const path = join(dir, 'qdrant.log');
  try { if (statSync(path).size > LOG_MAX) renameSync(path, `${path}.1`); } catch { /* none yet */ }
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  const fd = openSync(path, 'a', 0o600);
  try { chmodSync(path, 0o600); } catch { /* not ours to change: written all the same */ }
  return fd;
}

/** When this machine started, in ms: a pid written down before a reboot names some other process now. */
export function bootTime(): number {
  return Date.now() - uptime() * 1000;
}

/** The same boot as `boot` (bootTime() of then): a minute apart at most, uptime being in whole seconds on some systems. */
export function sameBoot(boot: unknown): boolean {
  return typeof boot === 'number' && Math.abs(bootTime() - boot) < 60_000;
}

/** A process with this pid exists — one of another user's too (EPERM). */
export function alive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (err) { return (err as NodeJS.ErrnoException).code === 'EPERM'; }
}

const pause = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** A running process's command line, or null when it cannot be read (no /proc and no ps, Windows). */
function commandOf(pid: number): string | null {
  try { return readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' '); } catch { /* not Linux */ }
  if (process.platform === 'win32') return null;
  try {
    return execFileSync('ps', ['-ww', '-o', 'command=', '-p', String(pid)], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 5000 });
  } catch { return null; }
}

/** What Qdrant gets of the environment: what a process needs to run and the key. None of the host's secrets, and none
 *  of its QDRANT__* variables — they override the config (QDRANT__SERVICE__HOST=0.0.0.0 would open it to the network).
 *  Qdrant does not start without CA certificates, though it makes no outside call here, and a slim container has none:
 *  it gets the host's SSL_CERT_FILE, or `caFile`. */
function childEnv(apiKey: string, caFile: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const k of ['PATH', 'HOME', 'TMPDIR', 'TEMP', 'TMP', 'SystemRoot', 'USERPROFILE']) if (process.env[k] !== undefined) env[k] = process.env[k];
  return { ...env, SSL_CERT_FILE: process.env.SSL_CERT_FILE || caFile, QDRANT__SERVICE__API_KEY: apiKey, QDRANT__TELEMETRY_DISABLED: 'true' };
}

/** Every Qdrant this process started and has not seen exit: stopped when the host exits without `close()` (an 'exit'
 *  handler can only send the signal; a host killed outright leaves its Qdrant to the next start's reapLeftover). */
const running = new Set<ChildProcess>();
let exitHook = false;
function track(child: ChildProcess): void {
  running.add(child);
  child.once('exit', () => running.delete(child));
  if (exitHook) return;
  exitHook = true;
  process.on('exit', () => { for (const c of running) { try { c.kill('SIGTERM'); } catch { /* gone */ } } });
}

export class QdrantProcess {
  private child: ChildProcess | null = null;
  private endpoint: QdrantEndpoint | null = null;
  private stopping = false;
  private deaths: number[] = [];
  private restartTimer: ReturnType<typeof setTimeout> | null = null;
  /** A restart's start while it runs: stopAndWait waits for it, and for what it began to be gone. */
  private restarting: Promise<void> | null = null;
  /** Counts stop()s: a start that sees it move while it starts gives up — before it spawns too, where `child` is null. */
  private stops = 0;
  /** This instance's children that have not exited: stopAndWait waits for each, a restart's own included. */
  private readonly children = new Set<ChildProcess>();

  private readonly opts: { binary: string; dir: string; log: (line: string) => void; onDown?: (why: string) => void };

  /** `dir` may be relative: Qdrant runs inside it, so every path it is given is made absolute first. */
  constructor(opts: { binary: string; dir: string; log: (line: string) => void; onDown?: (why: string) => void }) {
    this.opts = { ...opts, dir: resolve(opts.dir), binary: opts.binary.includes('/') || opts.binary.includes('\\') ? resolve(opts.binary) : opts.binary };
  }

  /** Where it answers, once it does; null while it starts or after it stopped. */
  current(): QdrantEndpoint | null {
    return this.endpoint;
  }

  /** Up and proven ours (owned), or it throws — and then nothing of this start keeps running or is started again: the
   *  caller was told, and may not be there to stop it. */
  async start(): Promise<QdrantEndpoint> {
    this.stopping = false;
    const stops = this.stops;
    if (this.endpoint && this.child && this.child.exitCode === null) return this.endpoint;
    mkdirSync(this.opts.dir, { recursive: true, mode: 0o700 });
    await this.reapLeftover();
    const port = await freePort();
    if (this.stops !== stops) throw new Error('Qdrant was stopped while it started');
    const apiKey = randomBytes(24).toString('hex');
    const config = join(this.opts.dir, 'config.yaml');
    writeFileSync(config, qdrantConfig(this.opts.dir, port), { mode: 0o600 });
    const caFile = join(this.opts.dir, 'ca.pem');
    writeFileSync(caFile, `${rootCertificates.join('\n')}\n`, { mode: 0o600 }); // the ones Node carries
    const fd = openLog(this.opts.dir);
    let child: ChildProcess;
    try {
      child = spawn(this.opts.binary, ['--config-path', config, '--disable-telemetry'], {
        cwd: this.opts.dir,
        stdio: ['ignore', fd, fd],
        env: childEnv(apiKey, caFile),
      });
    } finally {
      closeSync(fd);
    }
    this.child = child;
    if (child.pid) {
      this.children.add(child);
      child.once('exit', () => this.children.delete(child));
    }
    // A binary that is missing or cannot run is an 'error', never an 'exit' — unheard, it is thrown in the host.
    let failed: Error | null = null;
    child.on('error', (err) => { failed ??= err; });
    child.once('exit', (code, signal) => this.exited(child, code, signal));
    if (child.pid) this.recordPid(child, config);
    const endpoint = { url: `http://127.0.0.1:${port}`, apiKey };
    try {
      await this.ready(endpoint, child, () => failed);
      await this.owned(endpoint, child);
      if (this.child !== child) throw new Error('Qdrant was stopped while it started');
    } catch (err) {
      if (this.child === child) this.child = null; // exited() then schedules nothing
      if (child.pid && child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      throw err;
    }
    this.endpoint = endpoint;
    this.opts.log(`[memory] Qdrant up on 127.0.0.1:${port}`);
    return endpoint;
  }

  /** The pid file, and the child tracked for the host's exit; the file goes when that child does. */
  private recordPid(child: ChildProcess, config: string): void {
    const file = join(this.opts.dir, PID_FILE);
    writeFileSync(file, JSON.stringify({ pid: child.pid, config, boot: bootTime() }), { mode: 0o600 });
    track(child);
    child.once('exit', () => {
      try { if ((JSON.parse(readFileSync(file, 'utf8')) as { pid?: number }).pid === child.pid) rmSync(file, { force: true }); } catch { /* none */ }
    });
  }

  /** A Qdrant left running on this storage by a host that died without `close()` (a crash, a kill): stopped before
   *  another starts on it. Only when the pid is still the process written down — this boot, its command naming this
   *  folder's config — never an unrelated process that has the pid since. */
  private async reapLeftover(): Promise<void> {
    const file = join(this.opts.dir, PID_FILE);
    let rec: { pid?: unknown; config?: unknown; boot?: unknown };
    try { rec = JSON.parse(readFileSync(file, 'utf8')) as typeof rec; } catch { return; }
    const pid = Number(rec.pid);
    const config = join(this.opts.dir, 'config.yaml');
    if (rec.config === config && sameBoot(rec.boot) && pid !== this.child?.pid && alive(pid) && commandOf(pid)?.includes(config)) {
      this.opts.log(`[memory] stopping the Qdrant left running on ${this.opts.dir} (pid ${pid})`);
      for (const [signal, ms] of [['SIGTERM', 10_000], ['SIGKILL', 2000]] as const) {
        try { process.kill(pid, signal); } catch { /* gone */ }
        for (const until = Date.now() + ms; alive(pid) && Date.now() < until;) await pause(50);
        if (!alive(pid)) break;
      }
    }
    rmSync(file, { force: true });
  }

  /** Answering `/readyz` within 30 s, or the start failed. /readyz asks no key, so none is sent: what answers may not be
   *  ours (owned). */
  private async ready(e: QdrantEndpoint, child: ChildProcess, failed: () => Error | null): Promise<void> {
    const until = Date.now() + 30_000;
    while (Date.now() < until) {
      const err = failed();
      if (err) throw new Error(`Qdrant could not be started (${this.opts.binary}): ${err.message}`);
      if (child.exitCode !== null || child.signalCode !== null) throw new Error(`Qdrant exited (${child.signalCode ?? child.exitCode}) while starting — see ${join(this.opts.dir, 'qdrant.log')}`);
      try {
        const r = await fetch(`${e.url}/readyz`, { signal: AbortSignal.timeout(1000) });
        await r.arrayBuffer().catch(() => undefined);
        if (r.ok) return;
      } catch { /* not listening yet */ }
      await pause(100);
    }
    throw new Error('Qdrant did not answer within 30 s');
  }

  /** The port was free when picked, not when Qdrant bound it: another local process may have taken it, answered
   *  /readyz, and would be handed the key and every document. Ours is still running, refuses a wrong key and takes
   *  this run's — the wrong key asked first, so a process that takes any key never sees the real one. */
  private async owned(e: QdrantEndpoint, child: ChildProcess): Promise<void> {
    const status = async (key: string): Promise<number> => {
      const r = await fetch(`${e.url}/collections`, { headers: { 'api-key': key }, signal: AbortSignal.timeout(5000) });
      await r.arrayBuffer().catch(() => undefined);
      return r.status;
    };
    const gone = (): boolean => child.exitCode !== null || child.signalCode !== null;
    const notOurs = `what answers on ${e.url} is not this memory's Qdrant`;
    if (gone()) throw new Error(`${notOurs}: ours exited — see ${join(this.opts.dir, 'qdrant.log')}`);
    const wrong = await status(randomBytes(24).toString('hex'));
    if (wrong !== 401 && wrong !== 403) throw new Error(`${notOurs}: it took a wrong key (${wrong})`);
    const right = await status(e.apiKey);
    if (right !== 200) throw new Error(`${notOurs}: it refused this run's key (${right})`);
    if (gone()) throw new Error(`${notOurs}: ours exited — see ${join(this.opts.dir, 'qdrant.log')}`);
  }

  private exited(child: ChildProcess, code: number | null, signal: NodeJS.Signals | null): void {
    if (this.child !== child) return;
    // One that never came up is its start()'s to report: that start rejects, and schedules nothing.
    const wasUp = this.endpoint !== null;
    this.child = null;
    this.endpoint = null;
    if (this.stopping || !wasUp) return;
    this.restartLater(`Qdrant stopped (${signal ?? code})`);
  }

  /** Started again after a growing pause; a restart that fails counts as another death. */
  private restartLater(why: string): void {
    this.opts.log(`[memory] ${why}`);
    const now = Date.now();
    this.deaths = [...this.deaths.filter((t) => now - t < 10 * 60_000), now];
    if (this.deaths.length >= 5) {
      this.opts.onDown?.(`${why}; five times in ten minutes, so it is left stopped`);
      return;
    }
    const wait = Math.min(60_000, 1000 * 2 ** (this.deaths.length - 1));
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      if (this.stopping) return;
      // Held until its start ends: a stop() meanwhile makes that start give up and stop what it began.
      const run: Promise<void> = this.start().then(() => undefined, (err: unknown) => {
        if (this.stopping) return; // given up for a stop: nothing is down
        this.opts.onDown?.((err as Error).message);
        this.restartLater((err as Error).message);
      }).finally(() => { if (this.restarting === run) this.restarting = null; });
      this.restarting = run;
    }, wait);
    this.restartTimer.unref?.();
  }

  stop(): void {
    this.stopping = true;
    this.stops++;
    if (this.restartTimer) { clearTimeout(this.restartTimer); this.restartTimer = null; }
    const c = this.child;
    this.child = null;
    this.endpoint = null;
    if (c) c.kill('SIGTERM');
  }

  /** Stop and wait for the process to be gone — before its storage is deleted. A restart that was starting one ends
   *  first, and the Qdrant it began is waited for too: none is left running after a close. */
  async stopAndWait(ms = 10_000): Promise<void> {
    const restart = this.restarting;
    this.stop();
    const within = (p: Promise<unknown>): Promise<void> => new Promise<void>((resolve) => {
      const t = setTimeout(resolve, ms);
      const done = (): void => { clearTimeout(t); resolve(); };
      p.then(done, done);
    });
    if (restart) await within(restart);
    await Promise.all([...this.children].filter((c) => c.exitCode === null && c.signalCode === null)
      .map((c) => within(new Promise((exited) => c.once('exit', exited)))));
  }
}
