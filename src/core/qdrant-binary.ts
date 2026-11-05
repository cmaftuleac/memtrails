/**
 * The Qdrant binary memory runs when you give it no Qdrant server of your own: one pinned release, fetched on first use
 * from Qdrant's GitHub release, verified against the size and SHA-256 pinned below, unpacked next to it (cache dir,
 * paths.ts), and trusted after that through a `.verified` marker while the folder is this user's alone.
 *
 * Qdrant publishes no checksums, so the hashes are pinned here from its release. Upgrading means a new version and new
 * hashes; Qdrant reads storage only from the minor version before it, so after a version change the index is written
 * again from the logs (rebuild.ts) rather than migrated: it is derived data.
 */
import { createHash, randomBytes } from 'node:crypto';
import {
  chmodSync, closeSync, createReadStream, createWriteStream, existsSync, lstatSync, mkdirSync, mkdtempSync, openSync, readFileSync,
  renameSync, rmSync, statSync, utimesSync, writeFileSync, writeSync,
} from 'node:fs';
import { hostname } from 'node:os';
import { dirname, join } from 'node:path';
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

/**
 * Run only what this user alone could have put there: the cache, the folder, the binary and its marker owned by this
 * user, no symlink, writable by no one else. The archive's hash is pinned but not the unpacked binary's (that needs every
 * platform's archive to compute), so on a shared machine this is what stops a binary swapped in after unpacking.
 * Windows has no owner bits; there the user's profile ACLs keep others out.
 */
export function ownedSafely(path: string): boolean {
  if (typeof process.getuid !== 'function') return !lstatSync(path).isSymbolicLink();
  const st = lstatSync(path);
  return !st.isSymbolicLink() && st.uid === process.getuid() && (st.mode & 0o022) === 0;
}

/**
 * The folder made (0700) or made private, and this user's alone — or refused. What is verified in it is run or loaded
 * later: a folder others could write to lets them swap a binary between its check and its run, or a download between
 * its hash and its unpacking. (`mkdirSync`'s mode applies only to a folder it makes.)
 */
export function privateDir(dir: string): void {
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  chmodSync(dir, 0o700);
  if (!ownedSafely(dir)) {
    throw new Error(`${dir} is not this user's alone (its owner, its permissions, or it is a link): set MEMTRAILS_CACHE to a folder of your own`);
  }
}

/** The binary is in place, its archive was verified when it was unpacked, and no one else could have changed it since. */
export function qdrantReady(a: QdrantAsset | null = qdrantAsset()): boolean {
  const bin = qdrantBinaryPath(a);
  if (!a || !bin) return false;
  const marker = join(qdrantDir(), '.verified');
  try {
    return readFileSync(marker, 'utf8').trim() === `${a.size}:${a.sha256}` && statSync(bin).size > 0 &&
      [dirname(qdrantDir()), qdrantDir(), bin, marker].every(ownedSafely);
  } catch {
    return false;
  }
}

/** The system's tar; on Windows by full path, so a `tar.exe` in the working folder or on a planted PATH is not run. */
function tarProgram(): string {
  return process.platform === 'win32' ? join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe') : 'tar';
}

/**
 * Unpack one archive with the system's tar (bsdtar on macOS and Windows reads zip too). Run as root, tar would give
 * the files the archive's owner, and the binary would then not be this user's alone.
 */
function untar(archive: string, into: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(tarProgram(), ['--no-same-owner', '-xf', archive, '-C', into], { stdio: 'ignore' });
    child.once('error', reject);
    child.once('exit', (code) => (code === 0 ? resolve() : reject(new Error(`tar exited ${code}`))));
  });
}

/** A download given up after this long without a byte (the answer's headers included). */
export const DOWNLOAD_STALL_MS = 60_000;

export interface DownloadOptions { signal?: AbortSignal; fetchImpl?: typeof fetch; stallMs?: number }

/**
 * `url` fetched into a fresh temp folder inside `dir`, never to a fixed name: two first runs at once (two processes, or
 * two memories opened together in one) wrote the same `.part` and could rename each other's half-written file. The
 * file is then checked as it is on disk, not as it streamed past, against its pinned size and SHA-256. Returns the
 * file and its folder; the caller renames the file into place and removes the folder. Throws (leaving nothing) on a
 * failure, a mismatch, or `stallMs` without a byte.
 */
export async function downloadVerified(url: string, dir: string, pin: { size: number; sha256: string }, o: DownloadOptions = {}):
  Promise<{ file: string; tmp: string }> {
  mkdirSync(dir, { recursive: true });
  const tmp = mkdtempSync(join(dir, '.download-'));
  const file = join(tmp, 'file');
  const name = url.slice(url.lastIndexOf('/') + 1);
  const stallMs = o.stallMs ?? DOWNLOAD_STALL_MS;
  const stalled = new AbortController();
  let timer: NodeJS.Timeout | undefined;
  const arm = (): void => {
    clearTimeout(timer);
    timer = setTimeout(() => stalled.abort(new Error(`${name}: no data for ${Math.round(stallMs / 1000)} s`)), stallMs);
  };
  const signal = o.signal ? AbortSignal.any([o.signal, stalled.signal]) : stalled.signal;
  try {
    arm();
    const res = await (o.fetchImpl ?? fetch)(url, { signal, redirect: 'follow' });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${name}`);
    let size = 0;
    await pipeline(Readable.fromWeb(res.body as never), async function* (src) {
      for await (const chunk of src) {
        arm();
        size += (chunk as Buffer).length;
        if (size > pin.size) throw new Error(`${name}: more than the expected ${pin.size} bytes`);
        yield chunk as Buffer;
      }
    }, createWriteStream(file, { flags: 'wx', mode: 0o600 }), { signal });
    clearTimeout(timer);
    const hash = createHash('sha256');
    for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
    const got = { size: statSync(file).size, sha256: hash.digest('hex') };
    if (got.size !== pin.size || got.sha256 !== pin.sha256) {
      throw new Error(`${name}: got ${got.size} bytes sha256 ${got.sha256}, expected ${pin.size} ${pin.sha256}`);
    }
    return { file, tmp };
  } catch (err) {
    rmSync(tmp, { recursive: true, force: true });
    throw stalled.signal.aborted ? stalled.signal.reason : err;
  } finally {
    clearTimeout(timer);
  }
}

/** A lock not touched for this long is a dead holder's (the holder touches it every quarter of this). */
export const LOCK_STALE_MS = 120_000;

function lockStale(lock: string, staleMs: number): boolean {
  try {
    const [pid, host] = readFileSync(lock, 'utf8').split(' ');
    if (host === hostname() && Number(pid) !== process.pid) {
      try {
        process.kill(Number(pid), 0);
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code === 'ESRCH') return true; // its process is gone
      }
    }
    return Date.now() - statSync(lock).mtimeMs > staleMs;
  } catch {
    return false; // gone meanwhile: just try again
  }
}

/**
 * `f` run holding `<path>.lock`, created with O_EXCL: a second first run, in another process or this one, waits and
 * then finds the first one's result. A lock whose process is gone, or untouched for `staleMs`, is taken over. Nothing
 * rests on the lock for correctness (every file is verified in a private temp and renamed into place); it keeps two
 * runs from fetching the same hundreds of MB, and from swapping folders under each other.
 */
export async function withLock<T>(path: string, f: () => Promise<T>, o: { signal?: AbortSignal; staleMs?: number; pollMs?: number } = {}): Promise<T> {
  const lock = `${path}.lock`;
  const staleMs = o.staleMs ?? LOCK_STALE_MS;
  const mine = `${process.pid} ${hostname()} ${randomBytes(8).toString('hex')}`;
  mkdirSync(dirname(lock), { recursive: true });
  for (;;) {
    try {
      const fd = openSync(lock, 'wx', 0o600);
      writeSync(fd, mine);
      closeSync(fd);
      break;
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err;
    }
    if (lockStale(lock, staleMs)) {
      rmSync(lock, { force: true });
      continue;
    }
    o.signal?.throwIfAborted();
    await new Promise((ok) => setTimeout(ok, o.pollMs ?? 250));
  }
  const beat = setInterval(() => {
    try {
      utimesSync(lock, new Date(), new Date());
    } catch { /* taken over: nothing to keep alive */ }
  }, Math.max(10, staleMs / 4));
  beat.unref();
  try {
    return await f();
  } finally {
    clearInterval(beat);
    try {
      if (readFileSync(lock, 'utf8') === mine) rmSync(lock, { force: true }); // not one taken over meanwhile
    } catch { /* already gone */ }
  }
}

/**
 * Fetch, verify and unpack the binary unless it is already there. Throws on a network failure or stall, a size or
 * hash mismatch (the download is deleted), a platform Qdrant does not ship for, or a folder others could change; the
 * caller retries with backoff. `asset` is for tests: another archive with its own pins.
 */
export async function ensureQdrant(opts: DownloadOptions & { log?: (line: string) => void; asset?: QdrantAsset } = {}): Promise<string> {
  const a = opts.asset ?? qdrantAsset();
  if (!a) throw new Error(`no Qdrant build for ${process.platform}-${process.arch}`);
  const bin = qdrantBinaryPath(a)!;
  if (qdrantReady(a)) return bin;
  const dir = qdrantDir();
  const cache = dirname(dir);
  privateDir(cache);
  return withLock(dir, async () => {
    if (qdrantReady(a)) return bin; // another run finished it while this one waited
    opts.log?.(`[memtrails] fetching Qdrant ${QDRANT_VERSION} (${Math.round(a.size / 1e6)} MB)`);
    const { file, tmp } = await downloadVerified(assetUrl(a), cache, a, opts);
    try {
      // Unpacked into a fresh folder and moved into place whole: never a half-unpacked binary under the real name.
      const archive = join(tmp, a.asset);
      renameSync(file, archive);
      const unpacked = join(tmp, 'unpacked');
      mkdirSync(unpacked);
      await untar(archive, unpacked);
      const exe = join(unpacked, a.exe);
      if (!existsSync(exe) || !lstatSync(exe).isFile() || statSync(exe).size === 0) throw new Error(`${a.asset} did not contain ${a.exe}`);
      chmodSync(exe, 0o755);
      writeFileSync(join(unpacked, '.verified'), `${a.size}:${a.sha256}\n`, { mode: 0o644 });
      chmodSync(join(unpacked, '.verified'), 0o644);
      chmodSync(unpacked, 0o755);
      if (existsSync(dir)) renameSync(dir, join(tmp, 'old')); // unverified or not this user's alone; removed with tmp
      renameSync(unpacked, dir);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
    if (!qdrantReady(a)) throw new Error(`${dir} is not this user's alone (its owner or permissions): not running a binary others could change`);
    opts.log?.(`[memtrails] Qdrant ${QDRANT_VERSION} ready`);
    return bin;
  }, { signal: opts.signal });
}
