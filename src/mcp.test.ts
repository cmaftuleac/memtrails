/**
 * The MCP server (mcp.ts): the protocol as MCP clients speak it — the official SDK's client among them — the tools
 * listed before a memory is open, the memory opened at the first call and closed when idle or when the input ends, a
 * memory that cannot be opened as a tool error, and memory_save. Over a stand-in memory.
 */
import { PassThrough } from 'node:stream';
import { createInterface } from 'node:readline';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type { JSONRPCMessage } from '@modelcontextprotocol/sdk/types.js';
import { describe, expect, it } from 'vitest';
import type { Memory } from './memory.js';
import type { MemoryRecord } from './records.js';
import { MCP_PROTOCOL_VERSIONS, serveMcp, type McpServerOptions } from './mcp.js';

/** A stand-in memory: its tools answer with what they were asked; what is added and closed is kept. */
function fakeMemory(log: { added: MemoryRecord[]; closed: number }): Memory {
  const answer = (name: string) => async (args: unknown) => ({ content: [{ type: 'text' as const, text: `${name} ${JSON.stringify(args)}` }] });
  return {
    tools: () => ['memory_search', 'memory_open', 'memory_explore'].map((name) => ({ name, handler: answer(name) })),
    add: async (r: MemoryRecord | MemoryRecord[]) => { log.added.push(...(Array.isArray(r) ? r : [r])); return { added: 1, unchanged: 0 }; },
    close: async () => { log.closed++; },
  } as unknown as Memory;
}

/** A server over in-memory streams, and a way to talk to it one JSON line at a time. */
function serve(o: Partial<McpServerOptions> = {}) {
  const input = new PassThrough();
  const output = new PassThrough();
  const log = { added: [] as MemoryRecord[], closed: 0, opened: 0 };
  const done = serveMcp({
    version: '9.9.9',
    input,
    output,
    log: () => {},
    open: async () => { log.opened++; return fakeMemory(log); },
    ...o,
  });
  const answers = createInterface({ input: output })[Symbol.asyncIterator]();
  const next = async (): Promise<any> => JSON.parse((await answers.next()).value as string);
  let id = 0;
  const ask = async (method: string, params?: unknown): Promise<any> => {
    input.write(`${JSON.stringify({ jsonrpc: '2.0', id: ++id, method, params })}\n`);
    return next();
  };
  return { input, output, log, done, ask, next, end: () => { input.end(); return done; } };
}

describe('the MCP server', () => {
  it('answers initialize in the version the client asked for when it speaks it, else its newest', async () => {
    const s = serve();
    const a = await s.ask('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '1' } });
    expect(a.result.protocolVersion).toBe('2025-06-18');
    expect(a.result.serverInfo).toEqual({ name: 'memtrails', version: '9.9.9' });
    expect(a.result.capabilities).toEqual({ tools: {} });
    expect(a.result.instructions).toMatch(/Long-term memory/);
    const b = await s.ask('initialize', { protocolVersion: '1999-01-01' });
    expect(b.result.protocolVersion).toBe(MCP_PROTOCOL_VERSIONS[0]);
    await s.end();
  });

  it('lists the tools without opening the memory; read-only has no memory_save', async () => {
    const s = serve();
    const { result } = await s.ask('tools/list');
    expect(result.tools.map((t: { name: string }) => t.name)).toEqual(['memory_search', 'memory_open', 'memory_explore', 'memory_save']);
    expect(result.tools[0].inputSchema.type).toBe('object');
    expect(result.tools[0].inputSchema.required).toEqual(['query']);
    expect(result.tools[0].annotations.readOnlyHint).toBe(true);
    expect(result.tools[3].annotations.readOnlyHint).toBe(false);
    expect(s.log.opened).toBe(0);
    await s.end();
    const r = serve({ readOnly: true });
    expect((await r.ask('tools/list')).result.tools.map((t: { name: string }) => t.name)).not.toContain('memory_save');
    await r.end();
  });

  it('opens the memory once at the first call, and closes it when the input ends', async () => {
    const s = serve();
    const [a, b] = await Promise.all([
      s.ask('tools/call', { name: 'memory_search', arguments: { query: 'renewal' } }),
      s.ask('tools/call', { name: 'memory_open', arguments: { id: 'k:1' } }),
    ].map((p) => p));
    const texts = [a, b].map((x) => x.result.content[0].text).sort();
    expect(texts).toEqual(['memory_open {"id":"k:1"}', 'memory_search {"query":"renewal"}']);
    expect(s.log.opened).toBe(1);
    await s.end();
    expect(s.log.closed).toBe(1);
  });

  it('closes the memory when idle, and opens it again at the next call', async () => {
    const s = serve({ idleMs: 30 });
    await s.ask('tools/call', { name: 'memory_search', arguments: { query: 'a' } });
    await new Promise((r) => setTimeout(r, 120));
    expect(s.log.closed).toBe(1);
    await s.ask('tools/call', { name: 'memory_search', arguments: { query: 'b' } });
    expect(s.log.opened).toBe(2);
    await s.end();
    expect(s.log.closed).toBe(2);
  });

  it('says why when the memory cannot be opened, as a tool error, and tries again at the next call', async () => {
    let tries = 0;
    const s = serve({
      open: async () => {
        if (++tries === 1) throw new Error('memory /m is open in process 4242: close it there first');
        return fakeMemory({ added: [], closed: 0 });
      },
    });
    const a = await s.ask('tools/call', { name: 'memory_search', arguments: { query: 'a' } });
    expect(a.result).toEqual({ content: [{ type: 'text', text: 'memory /m is open in process 4242: close it there first' }], isError: true });
    const b = await s.ask('tools/call', { name: 'memory_search', arguments: { query: 'a' } });
    expect(b.result.isError).toBeUndefined();
    await s.end();
  });

  it('saves a note with memory_save, and refuses one without text', async () => {
    const s = serve();
    const a = await s.ask('tools/call', { name: 'memory_save', arguments: { text: 'Ana prefers the Friday deploy window.', title: 'Deploy window' } });
    expect(a.result.content[0].text).toMatch(/^Saved \(\d{4}-\d{2}-\d{2}, id /);
    expect(s.log.added).toHaveLength(1);
    expect(s.log.added[0]).toMatchObject({ source: 'mcp', kind: 'note', title: 'Deploy window', text: 'Ana prefers the Friday deploy window.' });
    const b = await s.ask('tools/call', { name: 'memory_save', arguments: {} });
    expect(b.result.isError).toBe(true);
    expect(b.result.content[0].text).toMatch(/^Invalid arguments: text/);
    await s.end();
  });

  it('answers JSON-RPC as the protocol says: unknown methods and tools, notifications, pings, parse errors, batches', async () => {
    const s = serve();
    expect((await s.ask('resources/list')).error.code).toBe(-32601);
    expect((await s.ask('tools/call', { name: 'nope', arguments: {} })).result).toEqual({ content: [{ type: 'text', text: 'Unknown tool: nope' }], isError: true });
    s.input.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`); // no answer
    expect(await s.ask('ping')).toEqual({ jsonrpc: '2.0', id: 3, result: {} });
    s.input.write('{not json\n');
    expect((await s.next()).error).toEqual({ code: -32700, message: 'Parse error' });
    s.input.write(`${JSON.stringify([{ jsonrpc: '2.0', id: 'x', method: 'ping' }, { jsonrpc: '2.0', method: 'notifications/cancelled' }])}\n`);
    expect(await s.next()).toEqual([{ jsonrpc: '2.0', id: 'x', result: {} }]);
    await s.end();
  });

  it('works with the official MCP SDK client: handshake, list, call', async () => {
    const s = serve();
    const answers = createInterface({ input: s.output });
    const transport: Transport = {
      start: async () => { answers.on('line', (line) => transport.onmessage?.(JSON.parse(line) as JSONRPCMessage)); },
      send: async (m) => { s.input.write(`${JSON.stringify(m)}\n`); },
      close: async () => { s.input.end(); transport.onclose?.(); },
    };
    const client = new Client({ name: 'test', version: '1.0.0' });
    await client.connect(transport);
    expect(client.getServerVersion()).toEqual({ name: 'memtrails', version: '9.9.9' });
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name)).toContain('memory_search');
    const r = await client.callTool({ name: 'memory_search', arguments: { query: 'why did renewal stop?' } });
    expect(r.content).toEqual([{ type: 'text', text: 'memory_search {"query":"why did renewal stop?"}' }]);
    await client.close();
    await s.done;
    expect(s.log.closed).toBe(1);
  });
});
