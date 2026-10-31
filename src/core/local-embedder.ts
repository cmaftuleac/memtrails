/**
 * EmbeddingGemma on this machine, for a memory that needs no server: the same quantised model and the same prompts as
 * the remote workers, so vectors and an index move between the two. One run gives both what the search keeps (the
 * memory's vector) and what word by word keeps (a vector per word, the step before it is averaged: `last_hidden_state`).
 *
 * The model and its tokenizer are a directory you give it (onnx/model_q4.onnx, tokenizer.json, tokenizer_config.json).
 * `onnxruntime-node` and `@huggingface/tokenizers` are optional dependencies, imported only here. One inference at a
 * time; it works on the calling process, so a host with its own work should run it in a worker.
 */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { docText, queryText } from './embedder.js';
import { WORD_LIMIT, type WordEncoder, type WordVectors } from './words.js';

/** The slice of onnxruntime-node used here, typed locally so the build never needs the binding. */
interface OrtTensor { dims: readonly number[]; data: Float32Array }
interface OrtSession {
  outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}
export interface OrtModule {
  InferenceSession: { create(path: string, opts?: Record<string, unknown>): Promise<OrtSession> };
  Tensor: new (type: string, data: BigInt64Array, dims: number[]) => unknown;
}
export interface Tokens { encode(text: string): { ids?: number[]; input_ids?: number[] } | number[] }

export interface LocalEmbedderOptions {
  threads?: number;
  batch?: number;
  /** Tokens read of a text (the model's own limit is longer; memory's keys are short). */
  maxTokens?: number;
  /** Stand-ins for the two optional packages (tests). */
  ort?: OrtModule;
  tokenizer?: Tokens;
}

export interface LocalEmbedder {
  /** Already prompted texts → unit vectors, in order: RecallDeps.embed. */
  embed(texts: string[], timeoutMs?: number): Promise<Float32Array[]>;
  words: WordEncoder;
  close(): Promise<void>;
}

const BOS_EOS = 2;

function unit(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  const n = Math.sqrt(s) || 1;
  return v.map((x) => x / n);
}

async function importOptional<T>(name: string): Promise<T> {
  try {
    const mod = (await import(name)) as { default?: T } & T;
    return mod.default ?? mod;
  } catch (e) {
    throw new Error(`The local embedder needs the optional package ${name}: ${(e as Error).message}`);
  }
}

export async function loadLocalEmbedder(dir: string, opts: LocalEmbedderOptions = {}): Promise<LocalEmbedder> {
  const ort = opts.ort ?? await importOptional<OrtModule>('onnxruntime-node');
  const tokenizer = opts.tokenizer ?? await (async () => {
    const { Tokenizer } = await importOptional<{ Tokenizer: new (json: unknown, config: unknown) => Tokens }>('@huggingface/tokenizers');
    const json = async (f: string): Promise<unknown> => JSON.parse(await readFile(join(dir, f), 'utf8'));
    return new Tokenizer(await json('tokenizer.json'), await json('tokenizer_config.json'));
  })();
  const session = await ort.InferenceSession.create(join(dir, 'onnx', 'model_q4.onnx'), {
    executionProviders: ['cpu'], graphOptimizationLevel: 'all', executionMode: 'sequential', intraOpNumThreads: Math.max(1, opts.threads ?? 1),
  });
  const batch = Math.max(1, opts.batch ?? 8);
  const maxTokens = opts.maxTokens ?? 512;

  const tokens = (text: string, limit: number): { ids: number[]; cut: boolean } => {
    const e = tokenizer.encode(text);
    const ids = Array.isArray(e) ? e : (e.ids ?? e.input_ids ?? []);
    return { ids: ids.slice(0, limit), cut: ids.length > limit };
  };

  const run = async (encoded: number[][]): Promise<Record<string, OrtTensor>> => {
    const width = Math.max(1, ...encoded.map((e) => e.length));
    const ids = new BigInt64Array(encoded.length * width);
    const mask = new BigInt64Array(encoded.length * width);
    encoded.forEach((e, b) => e.forEach((id, i) => { ids[b * width + i] = BigInt(id); mask[b * width + i] = 1n; }));
    const dims = [encoded.length, width];
    return session.run({ input_ids: new ort.Tensor('int64', ids, dims), attention_mask: new ort.Tensor('int64', mask, dims) });
  };

  // One inference at a time, like lengths in one run: each text is padded to the run's longest.
  let queue: Promise<unknown> = Promise.resolve();
  const inOrder = <T>(job: () => Promise<T>): Promise<T> => {
    const next = queue.then(job, job);
    queue = next.catch(() => undefined);
    return next;
  };
  const batches = async <E extends { ids: number[] }, R>(items: E[], read: (out: Record<string, OrtTensor>, b: number, it: E) => R): Promise<R[]> => {
    const order = items.map((_, i) => i).sort((a, b) => items[a]!.ids.length - items[b]!.ids.length);
    const results = new Array<R>(items.length);
    for (let i = 0; i < order.length; i += batch) {
      const part = order.slice(i, i + batch);
      const out = await run(part.map((j) => items[j]!.ids));
      part.forEach((j, b) => { results[j] = read(out, b, items[j]!); });
    }
    return results;
  };

  return {
    embed: (texts) => inOrder(() => batches(texts.map((t) => tokens(t, maxTokens)), (out, b) => {
      const t = out['sentence_embedding'] ?? out[session.outputNames[session.outputNames.length - 1]!]!;
      const d = t.dims[t.dims.length - 1]!;
      return unit(Float32Array.from(t.data.subarray(b * d, (b + 1) * d)));
    })),
    words: (texts, kind) => inOrder(() => {
      const prompted = texts.map((t) => (kind === 'query' ? queryText(t) : docText('', t)));
      return batches(prompted.map((t) => tokens(t, WORD_LIMIT + BOS_EOS)), (out, b, it): WordVectors => {
        const h = out['last_hidden_state'];
        if (!h) throw new Error('This model gives no vector per word (last_hidden_state).');
        const [, width, dims] = h.dims as [number, number, number];
        // Without its first token, and its last unless the text was cut before it.
        const words = Math.max(0, it.ids.length - (it.cut ? 1 : BOS_EOS));
        const data = new Float32Array(words * dims);
        for (let w = 0; w < words; w++) {
          const from = (b * width + w + 1) * dims;
          data.set(unit(Float32Array.from(h.data.subarray(from, from + dims))), w * dims);
        }
        return { words, dims, data };
      });
    }),
    close: async () => { await queue; await session.release?.(); },
  };
}
