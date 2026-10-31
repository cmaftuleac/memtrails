/**
 * The pinned Qdrant (qdrant-binary.ts): every supported platform has a build with a pinned hash, and a download that
 * does not match is never unpacked or kept.
 */
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { ensureQdrant, QDRANT_ASSETS, qdrantAsset, qdrantDir, qdrantReady } from './qdrant-binary.js';

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
});
