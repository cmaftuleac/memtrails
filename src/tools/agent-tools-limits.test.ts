/**
 * The agent tools at their edges: arguments a model may send past the limits, an index that fails (not a memory that
 * is missing), and dates that do not exist.
 */
import { describe, expect, it } from 'vitest';
import { buildAgentTools, readableDate, type ToolProvider } from './agent-tools.js';

const provider = (over: Partial<ToolProvider> = {}): ToolProvider => ({
  search: async () => [],
  open: async () => null,
  explore: async () => ({ found: [], related: [], concepts: [] }),
  ...over,
});
const byName = (p: ToolProvider) => Object.fromEntries(buildAgentTools(p).map((t) => [t.name, t]));
const said = (r: { content: Array<{ text: string }> }): string => r.content[0]!.text;

describe('limits', () => {
  it('refuses a query, topic or id past 2000 characters and more than 50 collections — in the schema and in the handler', async () => {
    let searched = 0;
    const tools = byName(provider({ search: async () => { searched++; return []; } }));
    const long = 'x'.repeat(2001);
    for (const [name, args] of [
      ['memory_search', { query: long }],
      ['memory_search', { query: 'ok', collections: Array.from({ length: 51 }, (_, n) => `c${n}`) }],
      ['memory_open', { id: long }],
      ['memory_explore', { topic: long }],
    ] as const) {
      const r = await tools[name]!.handler(args);
      expect(r.isError, name).toBe(true);
      expect(said(r)).toMatch(/^Invalid arguments/);
    }
    expect(searched).toBe(0);
    expect(JSON.stringify(tools.memory_search!.jsonSchema)).toContain('"maxLength":2000');
    expect(JSON.stringify(tools.memory_search!.jsonSchema)).toContain('"maxItems":50');
    expect((await tools.memory_search!.handler({ query: 'x'.repeat(2000) })).isError).toBeUndefined();
  });
});

describe('memory_open', () => {
  it('says the index failed rather than that the memory is gone', async () => {
    const tools = byName(provider({ open: async () => { throw new Error('POST /collections/memory/points: 503 unavailable'); } }));
    const r = await tools.memory_open!.handler({ id: 'mem:1' });
    expect(r.isError).toBe(true);
    expect(said(r)).toContain('503 unavailable');
    expect(said(r)).not.toContain('There is no memory');
  });

  it('says a missing memory is missing', async () => {
    const r = await byName(provider()).memory_open!.handler({ id: 'mem:1' });
    expect(said(r)).toContain('There is no memory "mem:1"');
  });
});

describe('readable dates', () => {
  it('shows a date that does not exist as written', () => {
    expect(readableDate('2023-13')).toBe('2023-13');
    expect(readableDate('2023-02-30')).toBe('2023-02-30');
    expect(readableDate('2023-07-14')).toBe('Fri 14 Jul 2023');
    expect(readableDate('2023-07')).toBe('Jul 2023');
    expect(readableDate('2023')).toBe('2023');
    expect(readableDate('')).toBe('');
  });
});
