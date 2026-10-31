/**
 * The embedding model memory runs on this machine: EmbeddingGemma-300m, 4-bit (onnx-community/embeddinggemma-300m-ONNX
 * on Hugging Face, one pinned revision), fetched on first use into the cache (paths.ts), each file checked against its
 * pinned size and SHA-256. 218 MB. The model is Google's, under the Gemma Terms of Use (ai.google.dev/gemma/terms);
 * memtrails does not ship it.
 */
import { createHash } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { cacheDir } from './paths.js';

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
const stamp = MODEL_FILES.map((f) => `${f.rel}:${f.size}:${f.sha256}`).join('\n');

export function modelReady(dir = defaultModelDir()): boolean {
  try {
    return readFileSync(marker(dir), 'utf8') === stamp && MODEL_FILES.every((f) => existsSync(join(dir, f.rel)));
  } catch {
    return false;
  }
}

/** The model in `dir` (fetched and checked unless it is already there); the directory, ready for `modelDir`. */
export async function ensureModel(opts: { dir?: string; log?: (line: string) => void; fetchImpl?: typeof fetch; signal?: AbortSignal } = {}): Promise<string> {
  const dir = opts.dir ?? defaultModelDir();
  if (modelReady(dir)) return dir;
  opts.log?.(`[memtrails] fetching ${MODEL_REPO} (218 MB, Gemma Terms of Use)`);
  for (const f of MODEL_FILES) {
    const path = join(dir, f.rel);
    mkdirSync(dirname(path), { recursive: true });
    const part = `${path}.part`;
    rmSync(part, { force: true });
    const res = await (opts.fetchImpl ?? fetch)(`https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${f.rel}`, { signal: opts.signal, redirect: 'follow' });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${f.rel}`);
    const hash = createHash('sha256');
    let size = 0;
    try {
      await pipeline(Readable.fromWeb(res.body as never), async function* (src) {
        for await (const chunk of src) { const b = chunk as Buffer; hash.update(b); size += b.length; yield b; }
      }, createWriteStream(part));
    } catch (err) {
      rmSync(part, { force: true });
      throw err;
    }
    const digest = hash.digest('hex');
    if (size !== f.size || digest !== f.sha256) {
      rmSync(part, { force: true });
      throw new Error(`${f.rel}: got ${size} bytes sha256 ${digest}, expected ${f.size} ${f.sha256}`);
    }
    renameSync(part, path);
  }
  writeFileSync(marker(dir), stamp);
  opts.log?.('[memtrails] embedding model ready');
  return dir;
}
