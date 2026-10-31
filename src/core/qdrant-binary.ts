/**
 * The Qdrant binary memory runs when you give it no Qdrant server of your own: one pinned release, fetched on first use
 * from Qdrant's GitHub release, verified against the size and SHA-256 pinned below, unpacked next to it (cache dir,
 * paths.ts), and trusted after that through a `.verified` marker.
 *
 * Qdrant publishes no checksums, so the hashes are pinned here from its release. Upgrading means a new version and new
 * hashes; Qdrant reads storage only from the minor version before it, so after a version change the index is written
 * again from the logs (rebuild.ts) rather than migrated: it is derived data.
 */
import { createHash } from 'node:crypto';
import { chmodSync, createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawn } from 'node:child_process';
import { cacheDir } from './paths.js';

export const QDRANT_VERSION = '1.20.0';

export interface QdrantAsset {
  asset: string;
  size: number;
  sha256: string;
  /** The executable's name inside the archive. */
  exe: string;
}

/** Per `${platform}-${arch}`. Linux takes the static musl build: it runs on any distribution. */
export const QDRANT_ASSETS: Readonly<Record<string, QdrantAsset>> = {
  'darwin-arm64': { asset: 'qdrant-aarch64-apple-darwin.tar.gz', size: 29_070_255, sha256: '531904e822050790bb8c1da6705f801ee48b91eb4c229d0d605ff06f6331b20a', exe: 'qdrant' },
  'darwin-x64': { asset: 'qdrant-x86_64-apple-darwin.tar.gz', size: 31_917_921, sha256: '94a2e6f6b8ef3a3b5756a11172fd66554ac753e0a71c1d8d8b5349356c084f4d', exe: 'qdrant' },
  'linux-arm64': { asset: 'qdrant-aarch64-unknown-linux-musl.tar.gz', size: 31_475_234, sha256: 'bd8a458b8ef8006ce2c603c5bd6d1d973e39d787f794008dd959112e747c3bb7', exe: 'qdrant' },
  'linux-x64': { asset: 'qdrant-x86_64-unknown-linux-musl.tar.gz', size: 33_773_335, sha256: 'b03a64616a1f41ddab78b480a318466541765f19835318113b1821cbea8df52c', exe: 'qdrant' },
  'win32-x64': { asset: 'qdrant-x86_64-pc-windows-msvc.zip', size: 30_801_664, sha256: '6d13553584fd0411f00637519e293331ff0aa1fb79bd9e30db45839bb75de2be', exe: 'qdrant.exe' },
};

export function qdrantAsset(platform: string = process.platform, arch: string = process.arch): QdrantAsset | null {
  return QDRANT_ASSETS[`${platform}-${arch}`] ?? null;
}

export function qdrantDir(): string {
  return join(cacheDir(), `qdrant-${QDRANT_VERSION}`);
}

export function qdrantBinaryPath(a: QdrantAsset | null = qdrantAsset()): string | null {
  return a ? join(qdrantDir(), a.exe) : null;
}

export const QDRANT_RELEASE_BASE = `https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}`;
const assetUrl = (a: QdrantAsset): string => `${QDRANT_RELEASE_BASE}/${a.asset}`;

/** The binary is in place and its archive was verified when it was unpacked. */
export function qdrantReady(a: QdrantAsset | null = qdrantAsset()): boolean {
  const bin = qdrantBinaryPath(a);
  if (!a || !bin || !existsSync(bin)) return false;
  try {
    return readFileSync(join(qdrantDir(), '.verified'), 'utf8').trim() === `${a.size}:${a.sha256}`;
  } catch {
    return false;
  }
}

/** Unpack one archive with the system's tar (bsdtar on macOS and Windows reads zip too). */
function untar(archive: string, into: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn('tar', ['-xf', archive, '-C', into], { stdio: 'ignore' });
    child.once('error', reject);
    child.once('exit', (code) => (code === 0 ? resolve() : reject(new Error(`tar exited ${code}`))));
  });
}

/**
 * Fetch, verify and unpack the binary unless it is already there. Throws on a network failure, a size or hash
 * mismatch (the download is deleted), or a platform Qdrant does not ship for; the caller retries with backoff.
 */
export async function ensureQdrant(opts: { signal?: AbortSignal; log?: (line: string) => void; fetchImpl?: typeof fetch } = {}): Promise<string> {
  const a = qdrantAsset();
  if (!a) throw new Error(`no Qdrant build for ${process.platform}-${process.arch}`);
  const bin = qdrantBinaryPath(a)!;
  if (qdrantReady(a)) return bin;
  const dir = qdrantDir();
  mkdirSync(dir, { recursive: true });
  const archive = join(dir, a.asset);
  const part = `${archive}.part`;
  rmSync(part, { force: true });
  opts.log?.(`[memtrails] fetching Qdrant ${QDRANT_VERSION} (${Math.round(a.size / 1e6)} MB)`);
  const res = await (opts.fetchImpl ?? fetch)(assetUrl(a), { signal: opts.signal, redirect: 'follow' });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${a.asset}`);
  const hash = createHash('sha256');
  let size = 0;
  try {
    await pipeline(
      Readable.fromWeb(res.body as never),
      async function* (src) {
        for await (const chunk of src) {
          const b = chunk as Buffer;
          hash.update(b);
          size += b.length;
          yield b;
        }
      },
      createWriteStream(part),
    );
  } catch (err) {
    rmSync(part, { force: true });
    throw err;
  }
  const digest = hash.digest('hex');
  if (size !== a.size || digest !== a.sha256) {
    rmSync(part, { force: true });
    throw new Error(`${a.asset}: got ${size} bytes sha256 ${digest}, expected ${a.size} ${a.sha256}`);
  }
  renameSync(part, archive);
  await untar(archive, dir);
  rmSync(archive, { force: true });
  if (!existsSync(bin) || statSync(bin).size === 0) throw new Error(`${a.asset} did not contain ${a.exe}`);
  if (process.platform !== 'win32') chmodSync(bin, 0o755);
  writeFileSync(join(dir, '.verified'), `${a.size}:${a.sha256}\n`);
  opts.log?.(`[memtrails] Qdrant ${QDRANT_VERSION} ready`);
  return bin;
}
