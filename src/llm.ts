/**
 * The language model memory asks, for consolidation, sleep and gathering — never for adding or searching. One structured
 * ask: a system text, a prompt and a JSON Schema the answer must follow (strict structured output).
 *
 * `openAiAsk` speaks the OpenAI Responses API, so any provider or local server that serves it works (`baseUrl`). Bring
 * any other model by passing your own `Ask`. Every call's tokens can be counted (`onUsage`).
 */
import { redactString } from './core/redact.js';

export interface LlmRequest {
  /** What this is for, in a word (`memory`). */
  purpose: string;
  system: string;
  prompt: string;
  /** The answer's JSON Schema. */
  schema: Record<string, unknown>;
  /** What the structured answer is called. */
  name?: string;
  model?: string;
  maxTokens?: number;
  timeoutMs?: number;
  /** What a failure cost, said after its reason ("so no summary was written"). */
  consequence?: string;
}

export type LlmAnswer =
  | { answer: Record<string, unknown>; model?: string }
  /** Why it could not be asked, or did not answer, in words a person can read. */
  | { unavailable: string }
  /** One for this purpose is already being asked. */
  | { busy: true };

export type Ask = (req: LlmRequest) => Promise<LlmAnswer>;

export interface LlmUsage { model: string; name?: string; inputTokens: number; cachedTokens: number; outputTokens: number }

export interface OpenAiAskOptions {
  /** Default: $MEMTRAILS_LLM_API_KEY, else $OPENAI_API_KEY — the latter only for OpenAI's own address. */
  apiKey?: string;
  /** Default: $MEMTRAILS_MODEL. */
  model?: string;
  /** Default: $MEMTRAILS_LLM_BASE_URL, else https://api.openai.com/v1. */
  baseUrl?: string;
  /** For reasoning models: how hard to think. Not sent when not given. */
  reasoningEffort?: 'minimal' | 'low' | 'medium' | 'high';
  /** Each answered call's tokens. */
  onUsage?: (u: LlmUsage) => void;
  /** Tries after a rate limit, a server error or a dropped connection (default 5), waiting longer each time. */
  retries?: number;
  fetchImpl?: typeof fetch;
}

const OPENAI = 'https://api.openai.com/v1';

const why = (reason: string, req: LlmRequest): string => `${reason}${req.consequence ? `, ${req.consequence}` : ''}`;

export function openAiAsk(o: OpenAiAskOptions = {}): Ask {
  return async (req) => {
    const model = req.model ?? o.model ?? process.env.MEMTRAILS_MODEL?.trim();
    const base = (o.baseUrl ?? (process.env.MEMTRAILS_LLM_BASE_URL?.trim() || OPENAI)).replace(/\/+$/, '');
    // An OpenAI key goes to OpenAI only: with another address set, it would hand the key to whoever serves that.
    const key = o.apiKey ?? (process.env.MEMTRAILS_LLM_API_KEY?.trim() || (base === OPENAI ? process.env.OPENAI_API_KEY?.trim() : undefined));
    if (!model) return { unavailable: why('no model is set (the model option, or MEMTRAILS_MODEL)', req) };
    if (!key) return { unavailable: why(`no API key is set (the apiKey option or MEMTRAILS_LLM_API_KEY${base === OPENAI ? ', or OPENAI_API_KEY' : ''})`, req) };
    const body = JSON.stringify({
      model, instructions: req.system, input: req.prompt, max_output_tokens: req.maxTokens ?? 12_000,
      ...(o.reasoningEffort ? { reasoning: { effort: o.reasoningEffort } } : {}),
      text: { format: { type: 'json_schema', name: req.name ?? 'memory', strict: true, schema: req.schema } },
    });
    const tries = Math.max(1, (o.retries ?? 5) + 1);
    let last = '';
    for (let attempt = 0; attempt < tries; attempt++) {
      if (attempt) await new Promise((ok) => setTimeout(ok, Math.min(60_000, 1000 * 2 ** (attempt - 1)) * (0.5 + Math.random())));
      let res: Response;
      try {
        res = await (o.fetchImpl ?? fetch)(`${base}/responses`, {
          method: 'POST', body, signal: AbortSignal.timeout(req.timeoutMs ?? 180_000),
          headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
        });
      } catch (err) {
        last = (err as Error).message || (err as Error).name;
        continue;
      }
      if (res.status === 429 || res.status >= 500) { last = `HTTP ${res.status}`; continue; }
      if (!res.ok) {
        // What the provider said (a wrong model name, a schema it refuses) — not only its status.
        const said = redactString((await res.text().catch(() => '')).replace(/\s+/g, ' ').trim()).slice(0, 300);
        return { unavailable: why(`the model answered HTTP ${res.status}${said ? `: ${said}` : ''}`, req) };
      }
      try {
        const r = (await res.json()) as {
          model?: string; output?: Array<{ content?: Array<{ text?: string }> }>;
          usage?: { input_tokens?: number; output_tokens?: number; input_tokens_details?: { cached_tokens?: number } };
        };
        o.onUsage?.({ model: r.model ?? model, name: req.name, inputTokens: r.usage?.input_tokens ?? 0,
          cachedTokens: r.usage?.input_tokens_details?.cached_tokens ?? 0, outputTokens: r.usage?.output_tokens ?? 0 });
        const text = (r.output ?? []).flatMap((x) => x.content ?? []).map((c) => c.text ?? '').join('');
        return { model: r.model ?? model, answer: JSON.parse(text) as Record<string, unknown> };
      } catch (err) {
        return { unavailable: why(`the model's answer could not be read (${(err as Error).message})`, req) };
      }
    }
    return { unavailable: why(`the model did not answer (${last})`, req) };
  };
}

/** The model the environment names (MEMTRAILS_MODEL and a key), asked when no `ask` is given. */
export const defaultAsk: Ask = (req) => openAiAsk()(req);
