/**
 * The pinned Qdrant (qdrant-binary.ts): every supported platform has a build with a pinned hash, and a download that
 * does not match is never unpacked or kept.
 */
import { createHash } from 'node:crypto';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, statSync, symlinkSync, truncateSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { ensureModel, MODEL_FILES, modelReady } from './model.js';
import { downloadVerified, ensureQdrant, QDRANT_ASSETS, qdrantAsset, qdrantBinaryPath, qdrantDir, qdrantReady, withLock } from './qdrant-binary.js';

describe('the pinned Qdrant', () => {
  it('has a build, a size and a SHA-256 for every supported platform', () => {
    for (const p of ['darwin-arm64', 'darwin-x64', 'linux-x64', 'linux-arm64', 'win32-x64']) {
      const a = QDRANT_ASSETS[p];
      expect(a, p).toBeDefined();
      expect(a!.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(a!.size).toBeGreaterThan(10_000_000);
    }
    expect(qdrantAsset('win32', 'x64')?.exe).toBe('qdrant.exe');
    expect(qdrantAsset('freebsd', 'x64')).toBeNull();
  });

  it('refuses a download whose size or hash is not the pinned one, and keeps nothing of it', async () => {
    const fetchImpl = (async () => new Response(new Uint8Array([1, 2, 3]), { status: 200 })) as unknown as typeof fetch;
    await expect(ensureQdrant({ fetchImpl })).rejects.toThrow(/expected/);
    const a = qdrantAsset()!;
    expect(existsSync(join(qdrantDir(), `${a.asset}.part`))).toBe(false);
    expect(existsSync(join(qdrantDir(), a.asset))).toBe(false);
    expect(qdrantReady()).toBe(false);
  });

  it('says why when the download fails', async () => {
    const fetchImpl = (async () => new Response('nope', { status: 404 })) as unknown as typeof fetch;
    await expect(ensureQdrant({ fetchImpl })).rejects.toThrow(/HTTP 404/);
  });

  it('downloads to a temp of its own, checked on disk: two at once never take each other\'s file', async () => {
    const bytes = new Uint8Array(200_000).map((_, i) => i % 251);
    const pin = { size: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
    const fetchImpl = (async () => new Response(bytes, { status: 200 })) as unknown as typeof fetch;
    const dir = mkdtempSync(join(tmpdir(), 'mt-dl-'));
    const [a, b] = await Promise.all([downloadVerified('https://x.test/f.bin', dir, pin, { fetchImpl }), downloadVerified('https://x.test/f.bin', dir, pin, { fetchImpl })]);
    expect(a.file).not.toBe(b.file);
    expect([existsSync(a.file), existsSync(b.file)]).toEqual([true, true]);
  });

  it('gives up on a download that stops sending, and leaves nothing behind', async () => {
    const fetchImpl = (async (_u: string, init: RequestInit) => new Response(new ReadableStream({
      start(c) { c.enqueue(new Uint8Array([1])); init.signal?.addEventListener('abort', () => c.error(init.signal!.reason)); },
    }), { status: 200 })) as unknown as typeof fetch;
    const dir = mkdtempSync(join(tmpdir(), 'mt-dl-'));
    await expect(downloadVerified('https://x.test/f.bin', dir, { size: 10, sha256: '0'.repeat(64) }, { fetchImpl, stallMs: 50 })).rejects.toThrow(/no data/);
    expect(readdirSync(dir)).toEqual([]);
  });

  it('runs one holder of a lock at a time, and takes over one whose process is gone', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'mt-lock-')), 'x');
    const order: string[] = [];
    const run = (n: string) => withLock(path, async () => { order.push(`${n}+`); await new Promise((ok) => setTimeout(ok, 30)); order.push(`${n}-`); }, { pollMs: 5 });
    await Promise.all([run('a'), run('b')]);
    expect(order.join(' ')).toMatch(/^(a\+ a- b\+ b-|b\+ b- a\+ a-)$/);
    const { writeFileSync } = await import('node:fs');
    writeFileSync(`${path}.lock`, `999999999 ${(await import('node:os')).hostname()} dead`);
    await expect(withLock(path, async () => 'taken', { pollMs: 5 })).resolves.toBe('taken');
  });
});

describe('a cache others could change', () => {
  const unix = process.platform !== 'win32';
  const notFound = (count: { n: number }) => (async () => { count.n++; return new Response('nope', { status: 404 }); }) as unknown as typeof fetch;
  /** `f` with MEMTRAILS_CACHE set to `cache`. */
  async function inCache<T>(cache: string, f: () => Promise<T> | T): Promise<T> {
    const was = process.env.MEMTRAILS_CACHE;
    process.env.MEMTRAILS_CACHE = cache;
    try {
      return await f();
    } finally {
      process.env.MEMTRAILS_CACHE = was;
    }
  }

  it.skipIf(!unix)('is made private before anything is fetched into it, and refused when it is a link', async () => {
    const root = mkdtempSync(join(tmpdir(), 'mt-cache-'));
    const open = join(root, 'open');
    mkdirSync(open);
    chmodSync(open, 0o777);
    const count = { n: 0 };
    await inCache(open, () => expect(ensureQdrant({ fetchImpl: notFound(count) })).rejects.toThrow(/HTTP 404/));
    expect(statSync(open).mode & 0o777).toBe(0o700);
    // A link to a folder: whoever can change the link can swap what it leads to — refused, nothing fetched.
    const link = join(root, 'link');
    symlinkSync(open, link);
    const before = count.n;
    await inCache(link, () => expect(ensureQdrant({ fetchImpl: notFound(count) })).rejects.toThrow(/not this user's alone/));
    expect(count.n).toBe(before);
  });

  /** A one-file tar whose entry belongs to `uid`, as an archive built elsewhere does. */
  function tarOwnedBy(uid: number, name: string, body: Uint8Array): Uint8Array<ArrayBuffer> {
    const header = new Uint8Array(512);
    const put = (at: number, text: string) => header.set(new TextEncoder().encode(text), at);
    const octal = (n: number, width: number) => `${n.toString(8).padStart(width - 1, '0')}\0`;
    put(0, name);
    put(100, octal(0o755, 8));
    put(108, octal(uid, 8));
    put(116, octal(uid, 8));
    put(124, octal(body.length, 12));
    put(136, octal(0, 12));
    put(148, ' '.repeat(8));
    put(156, '0');
    put(257, 'ustar\0' + '00');
    put(148, `${header.reduce((a, b) => a + b, 0).toString(8).padStart(6, '0')}\0 `);
    const out = new Uint8Array(512 + Math.ceil(body.length / 512) * 512 + 1024);
    out.set(header);
    out.set(body, 512);
    return out;
  }

  it.skipIf(!unix)('runs a binary unpacked as root, whoever owned it in the archive', async () => {
    const archive = tarOwnedBy(4242, 'qdrant', new TextEncoder().encode('#!/bin/sh\n'));
    const asset = { asset: 'qdrant-test.tar', size: archive.length, sha256: createHash('sha256').update(archive).digest('hex'), exe: 'qdrant' };
    const fetchImpl = (async () => new Response(archive, { status: 200 })) as unknown as typeof fetch;
    const bin = await inCache(join(mkdtempSync(join(tmpdir(), 'mt-root-')), 'cache'), () => ensureQdrant({ fetchImpl, asset }));
    expect(statSync(bin).uid).toBe(process.getuid!());
  });

  it.skipIf(!unix)('holds no binary that is ready while others could write to the cache above it', async () => {
    const cache = mkdtempSync(join(tmpdir(), 'mt-cache-'));
    await inCache(cache, () => {
      const a = qdrantAsset()!;
      mkdirSync(qdrantDir(), { recursive: true, mode: 0o755 });
      writeFileSync(qdrantBinaryPath(a)!, 'binary', { mode: 0o755 });
      writeFileSync(join(qdrantDir(), '.verified'), `${a.size}:${a.sha256}\n`, { mode: 0o644 });
      chmodSync(cache, 0o700);
      expect(qdrantReady(a)).toBe(true);
      chmodSync(cache, 0o777);
      expect(qdrantReady(a)).toBe(false);
    });
  });

  it.skipIf(!unix)('holds the model to the same: its folder made private or refused, and not ready while others could write to it', async () => {
    const root = mkdtempSync(join(tmpdir(), 'mt-model-'));
    const dir = join(root, 'model');
    mkdirSync(dir);
    chmodSync(dir, 0o777);
    const count = { n: 0 };
    await expect(ensureModel({ dir, fetchImpl: notFound(count) })).rejects.toThrow(/HTTP 404/);
    expect(statSync(dir).mode & 0o777).toBe(0o700);
    const link = join(root, 'link');
    symlinkSync(dir, link);
    const before = count.n;
    await expect(ensureModel({ dir: link, fetchImpl: notFound(count) })).rejects.toThrow(/not this user's alone/);
    expect(count.n).toBe(before);
    // A whole set in place (sparse files of the pinned sizes): ready while private, not once a folder is open to others.
    for (const f of MODEL_FILES) {
      mkdirSync(join(dir, f.rel, '..'), { recursive: true, mode: 0o700 });
      writeFileSync(join(dir, f.rel), '', { mode: 0o600 });
      truncateSync(join(dir, f.rel), f.size);
    }
    writeFileSync(join(dir, '.verified'), MODEL_FILES.map((f) => `${f.rel}:${f.size}:${f.sha256}`).join('\n'), { mode: 0o644 });
    expect(modelReady(dir)).toBe(true);
    chmodSync(join(dir, 'onnx'), 0o777);
    expect(modelReady(dir)).toBe(false);
  });
});
