/** The OpenAI-compatible ask (llm.ts): whose key goes where, and what a refused call says. */
import { afterEach, describe, expect, it } from 'vitest';
import { allOrAbort, openAiAsk, type LlmRequest } from './llm.js';

const req: LlmRequest = { purpose: 'memory', system: 's', prompt: 'p', schema: { type: 'object' } };
const env = { ...process.env };
afterEach(() => { process.env = { ...env }; });

function capture(status = 200, body = '{"output":[{"content":[{"text":"{}"}]}]}') {
  const seen: Array<{ url: string; auth: string }> = [];
  const fetchImpl = (async (url: string, init: RequestInit) => {
    seen.push({ url, auth: (init.headers as Record<string, string>).authorization });
    return new Response(body, { status });
  }) as unknown as typeof fetch;
  return { seen, fetchImpl };
}

describe('openAiAsk', () => {
  it('sends OPENAI_API_KEY to OpenAI, never to another address', async () => {
    process.env.OPENAI_API_KEY = 'openai-key';
    delete process.env.MEMTRAILS_LLM_API_KEY;
    const a = capture();
    expect(await openAiAsk({ model: 'm', fetchImpl: a.fetchImpl })(req)).toHaveProperty('answer');
    expect(a.seen[0]).toEqual({ url: 'https://api.openai.com/v1/responses', auth: 'Bearer openai-key' });
    const b = capture();
    const other = await openAiAsk({ model: 'm', baseUrl: 'https://llm.example/v1', fetchImpl: b.fetchImpl })(req);
    expect(other).toMatchObject({ unavailable: expect.stringContaining('no API key') });
    expect(b.seen).toEqual([]);
    process.env.MEMTRAILS_LLM_API_KEY = 'own-key';
    await openAiAsk({ model: 'm', baseUrl: 'https://llm.example/v1', fetchImpl: b.fetchImpl })(req);
    expect(b.seen[0]!.auth).toBe('Bearer own-key');
  });

  it('says what the provider said when it refuses, secrets taken out', async () => {
    const a = capture(400, '{"error":{"message":"model gpt-x does not exist; key sk-abcdefghijklmnop1234"}}');
    const r = await openAiAsk({ model: 'm', apiKey: 'k', fetchImpl: a.fetchImpl })(req);
    expect(r).toMatchObject({ unavailable: expect.stringContaining('model gpt-x does not exist') });
    expect(JSON.stringify(r)).not.toContain('sk-abcdefghijklmnop1234');
  });

  it('stops the call when the request\'s signal is aborted: no answer waited for, no retry', async () => {
    let calls = 0;
    const fetchImpl = (async (_url: string, init: RequestInit) => {
      calls++;
      await new Promise<void>((ok, fail) => {
        const t = setTimeout(ok, 2000);
        init.signal?.addEventListener('abort', () => { clearTimeout(t); fail(new DOMException('aborted', 'AbortError')); }, { once: true });
      });
      return new Response('{"output":[{"content":[{"text":"{}"}]}]}', { status: 200 });
    }) as unknown as typeof fetch;
    const c = new AbortController();
    setTimeout(() => c.abort(), 20);
    const started = Date.now();
    const r = await openAiAsk({ model: 'm', apiKey: 'k', fetchImpl })({ ...req, signal: c.signal, consequence: 'so nothing' });
    expect(r).toEqual({ unavailable: 'the ask was stopped, so nothing' });
    expect(calls).toBe(1);
    expect(Date.now() - started).toBeLessThan(1000);
  });
});

describe('allOrAbort', () => {
  it('aborts the others when one throws, and throws once they have all settled', async () => {
    const ended: string[] = [];
    const slow = (name: string) => (signal: AbortSignal) => new Promise<string>((ok) => {
      const t = setTimeout(() => { ended.push(name); ok(name); }, 2000);
      signal.addEventListener('abort', () => { clearTimeout(t); ended.push(`${name} aborted`); ok(name); }, { once: true });
    });
    const failing = async (): Promise<string> => { await new Promise((ok) => setTimeout(ok, 10)); throw new Error('broke'); };
    await expect(allOrAbort([slow('a'), failing, slow('b')])).rejects.toThrow('broke');
    expect(ended.sort()).toEqual(['a aborted', 'b aborted']);
    expect(await allOrAbort([async () => 1, async () => 2])).toEqual([1, 2]);
  });

  it('follows the caller\'s signal', async () => {
    const outer = new AbortController();
    outer.abort();
    expect(await allOrAbort([async (s) => s.aborted])).toEqual([false]);
    expect(await allOrAbort([async (s) => s.aborted], outer.signal)).toEqual([true]);
  });
});
