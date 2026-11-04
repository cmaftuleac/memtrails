/**
 * The pinned Qdrant (qdrant-binary.ts): every supported platform has a build with a pinned hash, and a download that
 * does not match is never unpacked or kept.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { downloadVerified, ensureQdrant, QDRANT_ASSETS, qdrantAsset, qdrantDir, qdrantReady, withLock } from './qdrant-binary.js';

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
