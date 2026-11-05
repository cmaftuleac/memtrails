/**
 * The embedding model memory runs on this machine: EmbeddingGemma-300m, 4-bit (onnx-community/embeddinggemma-300m-ONNX
 * on Hugging Face, one pinned revision), fetched on first use into the cache (paths.ts), each file checked against its
 * pinned size and SHA-256. 218 MB. The model is Google's, under the Gemma Terms of Use (ai.google.dev/gemma/terms);
 * memtrails does not ship it.
 */
import { chmodSync, existsSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { cacheDir } from './paths.js';
import { downloadVerified, ownedSafely, privateDir, withLock, type DownloadOptions } from './qdrant-binary.js';

export const MODEL_REPO = 'onnx-community/embeddinggemma-300m-ONNX';
export const MODEL_REVISION = '5090578d9565bb06545b4552f76e6bc2c93e4a66';
export const MODEL_FILES: ReadonlyArray<{ rel: string; size: number; sha256: string }> = [
  { rel: 'onnx/model_q4.onnx', size: 519_322, sha256: 'ad1dfee81a70f7944b9b9d1cc6e48075b832881cf33fab2f2b248be78f3f0043' },
  { rel: 'onnx/model_q4.onnx_data', size: 196_725_760, sha256: '599962c3143b040de2dd05e5975be3e9091dd067cacc6a8f7186e3203bab9e02' },
  { rel: 'tokenizer.json', size: 20_323_312, sha256: '4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47' },
  { rel: 'tokenizer_config.json', size: 1_156_830, sha256: '3ca953eea6c3c9fcda9cf3df22949ff18b216f7c74bd6459230f3f1013953f3a' },
];

export function defaultModelDir(): string {
  return join(cacheDir(), `embeddinggemma-300m-${MODEL_REVISION.slice(0, 8)}`);
}

const marker = (dir: string): string => join(dir, '.verified');
/** The model's folder and the folders its files are in. */
const folders = (dir: string): string[] => [...new Set([dir, ...MODEL_FILES.map((f) => dirname(join(dir, f.rel)))])];
const stamp = MODEL_FILES.map((f) => `${f.rel}:${f.size}:${f.sha256}`).join('\n');

/** Every file in place at its pinned size, the marker written after the last of them was verified, and no one else could
 *  have changed any of them since (the folders, the files and the marker this user's alone, as the Qdrant binary's). */
export function modelReady(dir = defaultModelDir()): boolean {
  try {
    return readFileSync(marker(dir), 'utf8') === stamp &&
      MODEL_FILES.every((f) => existsSync(join(dir, f.rel)) && statSync(join(dir, f.rel)).size === f.size) &&
      [...folders(dir), marker(dir), ...MODEL_FILES.map((f) => join(dir, f.rel))].every(ownedSafely);
  } catch {
    return false;
  }
}

/**
 * The model in `dir` (fetched and checked unless it is already there); the directory, ready for `modelDir`. Each file
 * is fetched to a temp of its own, checked on disk and renamed into place; one run at a time per `dir` (withLock), and
 * the marker is taken away first, so a set left half-replaced is never taken as ready.
 */
export async function ensureModel(opts: DownloadOptions & { dir?: string; log?: (line: string) => void } = {}): Promise<string> {
  const dir = opts.dir ?? defaultModelDir();
  if (modelReady(dir)) return dir;
  // Made private, or refused when others could change them: what is verified in them is loaded later. A set already
  // there in folders made before this is ready once they are private (checked again under the lock).
  if (!opts.dir) privateDir(cacheDir());
  for (const d of folders(dir)) privateDir(d);
  return withLock(dir, async () => {
    if (modelReady(dir)) return dir; // another run finished it while this one waited
    opts.log?.(`[memtrails] fetching ${MODEL_REPO} (218 MB, Gemma Terms of Use)`);
    rmSync(marker(dir), { force: true });
    for (const f of MODEL_FILES) {
      const path = join(dir, f.rel);
      const { file, tmp } = await downloadVerified(`https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${f.rel}`,
        dirname(path), f, opts);
      try {
        renameSync(file, path);
      } finally {
        rmSync(tmp, { recursive: true, force: true });
      }
    }
    writeFileSync(marker(dir), stamp, { mode: 0o644 });
    chmodSync(marker(dir), 0o644); // whatever the umask: a marker others could write is not trusted
    opts.log?.('[memtrails] embedding model ready');
    return dir;
  }, { signal: opts.signal });
}
