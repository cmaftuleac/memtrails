/**
 * Qdrant as memory runs it when you give it no server of your own: one process for this memory, on a loopback port
 * picked at start, behind an API key made for that run, gRPC off, telemetry off, stopped with `close()`.
 *
 * Tuned for a laptop: the index on disk, memory-mapped (qdrant-client.ts), one optimisation thread, two search threads
 * and ONE web worker: Qdrant's default is one a core, each waking ~4 times a second, so an idle index woke ~83 times a
 * second and spent 2.7–4 ms/s of CPU with 69 threads; with one it is 6.5 wake-ups a second and ~0.2 ms/s.
 *
 * A process that dies by itself is started again with a growing pause; five deaths in ten minutes stop the attempts
 * until the next `start()`.
 */
import type { ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { closeSync, mkdirSync, openSync, statSync, renameSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';

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

/** The process's log beside its data, rotated once at start when it has grown past 4 MB. */
function openLog(dir: string): number {
  const path = join(dir, 'qdrant.log');
  try { if (statSync(path).size > LOG_MAX) renameSync(path, `${path}.1`); } catch { /* none yet */ }
  mkdirSync(dir, { recursive: true });
  return openSync(path, 'a');
}

export class QdrantProcess {
  private child: ChildProcess | null = null;
  private endpoint: QdrantEndpoint | null = null;
  private stopping = false;
  private deaths: number[] = [];
  private restartTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly opts: { binary: string; dir: string; log: (line: string) => void; onDown?: (why: string) => void };

  /** `dir` may be relative: Qdrant runs inside it, so every path it is given is made absolute first. */
  constructor(opts: { binary: string; dir: string; log: (line: string) => void; onDown?: (why: string) => void }) {
    this.opts = { ...opts, dir: resolve(opts.dir), binary: opts.binary.includes('/') || opts.binary.includes('\\') ? resolve(opts.binary) : opts.binary };
  }

  /** Where it answers, once it does; null while it starts or after it stopped. */
  current(): QdrantEndpoint | null {
    return this.endpoint;
  }

  async start(): Promise<QdrantEndpoint> {
    this.stopping = false;
    if (this.endpoint && this.child && this.child.exitCode === null) return this.endpoint;
    const port = await freePort();
    const apiKey = randomBytes(24).toString('hex');
    mkdirSync(this.opts.dir, { recursive: true });
    const config = join(this.opts.dir, 'config.yaml');
    writeFileSync(config, qdrantConfig(this.opts.dir, port), { mode: 0o600 });
    const fd = openLog(this.opts.dir);
    let child: ChildProcess;
    try {
      child = spawn(this.opts.binary, ['--config-path', config, '--disable-telemetry'], {
        cwd: this.opts.dir,
        stdio: ['ignore', fd, fd],
        // Only the key: the rest is in the config file.
        env: { ...process.env, QDRANT__SERVICE__API_KEY: apiKey, QDRANT__TELEMETRY_DISABLED: 'true' },
      });
    } finally {
      closeSync(fd);
    }
    this.child = child;
    child.once('exit', (code, signal) => this.exited(child, code, signal));
    const endpoint = { url: `http://127.0.0.1:${port}`, apiKey };
    await this.ready(endpoint, child);
    this.endpoint = endpoint;
    this.opts.log(`[memory] Qdrant up on 127.0.0.1:${port}`);
    return endpoint;
  }

  /** Answering `/readyz` within 30 s, or the start failed. */
  private async ready(e: QdrantEndpoint, child: ChildProcess): Promise<void> {
    const until = Date.now() + 30_000;
    while (Date.now() < until) {
      if (child.exitCode !== null) throw new Error(`Qdrant exited (${child.exitCode}) while starting — see ${join(this.opts.dir, 'qdrant.log')}`);
      try {
        const r = await fetch(`${e.url}/readyz`, { headers: { 'api-key': e.apiKey }, signal: AbortSignal.timeout(1000) });
        if (r.ok) return;
      } catch { /* not listening yet */ }
      await new Promise((r) => setTimeout(r, 100));
    }
    child.kill('SIGTERM');
    throw new Error('Qdrant did not answer within 30 s');
  }

  private exited(child: ChildProcess, code: number | null, signal: NodeJS.Signals | null): void {
    if (this.child !== child) return;
    this.child = null;
    this.endpoint = null;
    if (this.stopping) return;
    const why = `Qdrant stopped (${signal ?? code})`;
    this.opts.log(`[memory] ${why}`);
    const now = Date.now();
    this.deaths = [...this.deaths.filter((t) => now - t < 10 * 60_000), now];
    if (this.deaths.length >= 5) {
      this.opts.onDown?.(`${why}; five times in ten minutes, so it is left stopped`);
      return;
    }
    const pause = Math.min(60_000, 1000 * 2 ** (this.deaths.length - 1));
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      this.start().catch((err) => this.opts.onDown?.((err as Error).message));
    }, pause);
    this.restartTimer.unref?.();
  }

  stop(): void {
    this.stopping = true;
    if (this.restartTimer) { clearTimeout(this.restartTimer); this.restartTimer = null; }
    const c = this.child;
    this.child = null;
    this.endpoint = null;
    if (c) c.kill('SIGTERM');
  }

  /** Stop and wait for the process to be gone — before its storage is deleted. */
  async stopAndWait(ms = 10_000): Promise<void> {
    const c = this.child;
    this.stop();
    if (!c || c.exitCode !== null) return;
    await new Promise<void>((resolve) => {
      const t = setTimeout(resolve, ms);
      c.once('exit', () => { clearTimeout(t); resolve(); });
    });
  }
}
