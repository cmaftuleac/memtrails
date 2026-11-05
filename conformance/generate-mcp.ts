/**
 * The MCP server's answer key (any port's server must give the same): what `serveMcp` answers to each message, in
 * order, over a stand-in memory — the handshake in a version it speaks and in one it does not, the tool list, the
 * errors the protocol and the tools' arguments call for.
 *
 *   npx tsx conformance/generate-mcp.ts      # mcp.json
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { PassThrough } from 'node:stream';
import { serveMcp } from '../src/mcp.js';
import type { Memory } from '../src/memory.js';

const OUT = join(import.meta.dirname, 'fixtures');
const VERSION = '0.0.0-fixture';

/** Each line sent, and whether it is answered (a notification is not). */
const SESSION: Array<{ send: string; answered: boolean }> = [
  { id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'key', version: '1' } } },
  { method: 'notifications/initialized' },
  { id: 2, method: 'initialize', params: { protocolVersion: '1999-01-01', capabilities: {}, clientInfo: { name: 'key', version: '1' } } },
  { id: 3, method: 'tools/list' },
  { id: 4, method: 'ping' },
  { id: 5, method: 'tools/call', params: { name: 'memory_save', arguments: {} } },
  { id: 6, method: 'tools/call', params: { name: 'memory_save', arguments: { text: '' } } },
  { id: 7, method: 'tools/call', params: { name: 'memory_save', arguments: { text: 1 } } },
  { id: 8, method: 'tools/call', params: { name: 'memory_save', arguments: { text: 'x', title: 'y'.repeat(201) } } },
  { id: 9, method: 'tools/call', params: { name: 'nope', arguments: {} } },
  { id: 10, method: 'tools/call', params: {} },
  { id: 'eleven', method: 'resources/list' },
  { id: 12, method: 7 },
  { method: 'notifications/cancelled', params: { requestId: 3 } },
].map((m) => ({ send: JSON.stringify({ jsonrpc: '2.0', ...m }), answered: 'id' in m }));
SESSION.push(
  { send: JSON.stringify({ id: 13, method: 'ping' }), answered: true },
  { send: '{not json', answered: true },
  { send: JSON.stringify([{ jsonrpc: '2.0', id: 14, method: 'ping' }, { jsonrpc: '2.0', method: 'notifications/initialized' }]), answered: true },
);

const input = new PassThrough();
const output = new PassThrough();
const stand = { tools: () => [], add: async () => ({ added: 1, unchanged: 0 }), close: async () => {} } as unknown as Memory;
const done = serveMcp({ version: VERSION, input, output, log: () => {}, open: async () => stand });
const answers = createInterface({ input: output })[Symbol.asyncIterator]();

const key: Array<{ send: string; answer: unknown }> = [];
for (const s of SESSION) {
  input.write(`${s.send}\n`);
  key.push({ send: s.send, answer: s.answered ? JSON.parse((await answers.next()).value as string) : null });
}
input.end();
await done;

writeFileSync(join(OUT, 'mcp.json'), `${JSON.stringify({ version: VERSION, session: key }, null, 1)}\n`);
console.log(`mcp.json: ${key.length} messages`);
