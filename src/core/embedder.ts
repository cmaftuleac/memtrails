/**
 * Embedding over HTTP, for a memory that embeds on a server: any service that speaks the OpenAI embeddings API
 * (`POST {url}/embeddings`, `{ model, input }`) and serves EmbeddingGemma-300m — text-embeddings-inference, Infinity,
 * Ollama (`embeddinggemma`), vLLM, your own. Vectors may come back as numbers or as little-endian float32 base64.
 *
 * Memory's prompts, indexes and measured thresholds are EmbeddingGemma's: a different model needs its own index
 * (`MemoryOptions.embed` with `dims`), and its results are not the measured ones.
 */
import { promisify } from 'node:util';
import { gzip } from 'node:zlib';

export const MEMORY_EMBEDDER = { id: 'embeddinggemma-300m-768', model: 'embeddinggemma-300m', dims: 768 } as const;

/** EmbeddingGemma's task prompts: a query and a document are embedded with different prefixes. */
export const docText = (title: string, text: string): string => `title: ${title || 'none'} | text: ${text}`;
export const queryText = (q: string): string => `task: search result | query: ${q}`;

export interface EmbedRoute {
  /** The API's base, e.g. `http://localhost:8080/v1` (`/embeddings` is added). */
  url: string;
  apiKey?: string;
  /** The model name the service knows it by. Default `embeddinggemma-300m`. */
  model?: string;
  /** Send bodies gzipped (a service that accepts `content-encoding: gzip`). */
  gzip?: boolean;
}

/** The route the environment names: MEMTRAILS_EMBED_URL, MEMTRAILS_EMBED_KEY, MEMTRAILS_EMBED_MODEL; null without one. */
export function resolveEmbedRoute(env: NodeJS.ProcessEnv = process.env): EmbedRoute | null {
  const url = env.MEMTRAILS_EMBED_URL?.trim();
  if (!url) return null;
  return { url, apiKey: env.MEMTRAILS_EMBED_KEY?.trim() || undefined, model: env.MEMTRAILS_EMBED_MODEL?.trim() || undefined };
}

/** No way to embed: neither a model directory nor a server was given. */
export class EmbeddingUnavailable extends Error {}

const MAX_INPUTS = 256;
const MAX_BYTES = 600_000; // JSON-escaped UTF-8 bytes in one request.
const MAX_TEXT = 24_000;
const IN_FLIGHT = 4;
const compress = promisify(gzip);

export interface EmbedderDeps {
  route?: () => EmbedRoute | null;
  fetchImpl?: typeof fetch;
  /** Total operation budget, including compression, retries and all batches (default 30 s). */
  timeoutMs?: number;
}

interface EmbeddingResponse {
  model?: string;
  data?: Array<{ index: number; embedding: number[] | string }>;
}

/** Validate the complete response before any vector reaches the index. Never mix dimensions or vector spaces. */
function vectors(body: EmbeddingResponse, count: number): Float32Array[] {
  const data = [...(body.data ?? [])].sort((a, b) => a.index - b.index);
  if (data.length !== count) throw new Error(`embeddings: ${data.length} vectors for ${count} texts`);
  return data.map(({ index, embedding }, n) => {
    if (index !== n) throw new Error('embeddings: missing or duplicate response index');
    let v: Float32Array;
    if (typeof embedding === 'string') {
      const bytes = Buffer.from(embedding, 'base64');
      if (bytes.length !== MEMORY_EMBEDDER.dims * 4 || bytes.toString('base64') !== embedding) throw new Error('embeddings: invalid base64 vector');
      v = new Float32Array(MEMORY_EMBEDDER.dims);
      for (let i = 0; i < v.length; i++) v[i] = bytes.readFloatLE(i * 4);
    } else {
      if (!Array.isArray(embedding) || embedding.length !== MEMORY_EMBEDDER.dims || embedding.some((x) => typeof x !== 'number')) throw new Error('embeddings: wrong vector dimensions');
      v = Float32Array.from(embedding);
    }
    const norm = Math.hypot(...v);
    if (!Number.isFinite(norm) || norm <= 0) throw new Error('embeddings: non-finite or zero vector');
    for (let i = 0; i < v.length; i++) v[i]! /= norm;
    return v;
  });
}

/** `embed(texts)` for memory: unit vectors in the order of `texts`, however many requests a batch needs. */
export function remoteEmbedder(d: EmbedderDeps = {}): (texts: string[], timeoutMs?: number) => Promise<Float32Array[]> {
  const route = d.route ?? (() => resolveEmbedRoute());
  return async (texts, timeoutMs = d.timeoutMs ?? 30_000) => {
    if (!texts.length) return [];
    const r = route();
    if (!r) throw new EmbeddingUnavailable('no embedder: give openMemory a modelDir, an embedServer, or set MEMTRAILS_EMBED_URL');
    const until = Date.now() + timeoutMs;
    const remaining = (): number => {
      const ms = Math.floor(until - Date.now());
      if (ms < 1) throw new Error('embeddings: operation deadline exceeded');
      return ms;
    };
    const input = texts.map((t) => t.slice(0, MAX_TEXT) || ' ');
    const parts: Array<{ at: number; texts: string[] }> = [];
    let cur: string[] = [];
    let bytes = 0;
    input.forEach((t, i) => {
      const size = Buffer.byteLength(JSON.stringify(t)) + 1;
      if (cur.length && (cur.length >= MAX_INPUTS || bytes + size > MAX_BYTES)) { parts.push({ at: i - cur.length, texts: cur }); cur = []; bytes = 0; }
      cur.push(t);
      bytes += size;
    });
    if (cur.length) parts.push({ at: input.length - cur.length, texts: cur });
    const call = async (part: { at: number; texts: string[] }): Promise<void> => {
      remaining();
      const json = Buffer.from(JSON.stringify({ model: r.model ?? MEMORY_EMBEDDER.model, input: part.texts }));
      const body = r.gzip ? await compress(json) : json;
      const url = `${r.url.replace(/\/+$/, '').replace(/\/embeddings$/, '')}/embeddings`;
      for (let attempt = 0; ; attempt++) {
        const res = await (d.fetchImpl ?? fetch)(url, {
          method: 'POST',
          signal: AbortSignal.timeout(remaining()),
          headers: { 'content-type': 'application/json', ...(r.gzip ? { 'content-encoding': 'gzip' } : {}), ...(r.apiKey ? { authorization: `Bearer ${r.apiKey}` } : {}) },
          body: new Uint8Array(body),
        });
        if (res.ok) {
          const payload = await res.json() as EmbeddingResponse;
          remaining();
          vectors(payload, part.texts.length).forEach((v, k) => { out[part.at + k] = v; });
          return;
        }
        const message = (await res.text()).slice(0, 200); // Drain the response before retrying (keep-alive socket).
        if ((res.status === 429 || res.status >= 500) && attempt < 3) {
          const delay = 500 * 2 ** attempt;
          if (remaining() <= delay) throw new Error('embeddings: operation deadline exceeded');
          await new Promise((ok) => setTimeout(ok, delay));
          continue;
        }
        throw new Error(`embeddings ${res.status}: ${message}`);
      }
    };
    const out: Float32Array[] = new Array(input.length);
    let next = 0;
    let failed = false;
    await Promise.all(Array.from({ length: Math.min(IN_FLIGHT, parts.length) }, async () => {
      while (!failed && next < parts.length) {
        try { await call(parts[next++]!); } catch (err) { failed = true; throw err; }
      }
    }));
    return out;
  };
}
