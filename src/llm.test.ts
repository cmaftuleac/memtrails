/** The OpenAI-compatible ask (llm.ts): whose key goes where, and what a refused call says. */
import { afterEach, describe, expect, it } from 'vitest';
import { openAiAsk, type LlmRequest } from './llm.js';

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
});
