/**
 * A memory as an MCP server over stdio (newline-delimited JSON-RPC), for Claude Desktop, Claude Code, Cursor and any
 * other MCP client: the agent tools (memory_search, memory_open, memory_explore) and, unless read-only, memory_save.
 *
 * The memory is opened at the first tool call and closed again after `idleMs` without one, so the home is free in
 * between: for `memtrails add`, or another client on the same memory. A memory another process holds is a tool error
 * that says which process, not a server that will not start.
 */
import { randomBytes } from 'node:crypto';
import { createInterface } from 'node:readline';
import type { Readable, Writable } from 'node:stream';
import { z } from 'zod';
import type { Memory } from './memory.js';
import { buildAgentTools, type AgentTool } from './tools/agent-tools.js';

/** The protocol versions this server speaks, newest first; a client asking for another gets the newest. */
export const MCP_PROTOCOL_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];

export interface McpServerOptions {
  /** Opens the memory; called at the first tool call, and again after an idle close. */
  open: () => Promise<Memory>;
  /** Without memory_save. */
  readOnly?: boolean;
  /** Close the memory after this long without a tool call (default 5 minutes; 0 keeps it open). */
  idleMs?: number;
  /** The server's version, as clients show it. */
  version: string;
  input?: Readable;
  output?: Writable;
  log?: (line: string) => void;
}

type ToolResult = { content: Array<{ type: 'text'; text: string }>; isError?: boolean };
type Id = string | number | null;
interface Request { jsonrpc?: unknown; id?: Id; method?: unknown; params?: Record<string, unknown> }

const INSTRUCTIONS = 'Long-term memory: what this person or team worked on, wrote, decided and said, and what was learned '
  + 'from it. Search it before answering about past work, people, decisions or dates, and open a result to read it whole. '
  + 'Save what is worth remembering later: a decision and why, a preference, a fact about a person or a system.';

const SAVE_DESCRIPTION = 'Save something to memory, to be found later by memory_search: a decision and why, a preference, '
  + 'a fact about a person, a project or a system, what was learned. Write it to be understood on its own, later, by '
  + 'someone who was not here: name who and what, and the date when it matters.';

const READ_ONLY = { readOnlyHint: true, openWorldHint: false };

/** Serves until the input ends, then closes the memory. Resolves when it is closed. */
export async function serveMcp(o: McpServerOptions): Promise<void> {
  const input = o.input ?? process.stdin;
  const output = o.output ?? process.stdout;
  const log = o.log ?? ((line: string) => process.stderr.write(`${line}\n`));
  const idleMs = o.idleMs ?? 5 * 60_000;
  let memory: Promise<Memory> | null = null;
  let busy = 0;
  let idle: NodeJS.Timeout | null = null;

  const send = (message: unknown): void => { output.write(`${JSON.stringify(message)}\n`); };
  const closeMemory = async (): Promise<void> => {
    const m = memory;
    memory = null;
    if (m) await m.then((x) => x.close(), () => undefined);
  };
  const touch = (): void => {
    if (idle) clearTimeout(idle);
    idle = null;
    if (idleMs > 0 && busy === 0 && memory) {
      idle = setTimeout(() => { if (busy === 0) void closeMemory().catch((e: unknown) => log(`[memtrails] close: ${String(e)}`)); }, idleMs);
      idle.unref();
    }
  };
  const withMemory = async <T>(f: (m: Memory) => Promise<T>): Promise<T> => {
    busy++;
    if (idle) clearTimeout(idle);
    try {
      if (!memory) {
        memory = o.open();
        memory.catch(() => { memory = null; }); // the next call tries again
      }
      return await f(await memory);
    } finally {
      busy--;
      touch();
    }
  };

  // The tools' definitions do not need an open memory: they are listed before any call.
  const definitions = toolDefinitions(o.readOnly ?? false);
  const call = async (name: string, args: unknown): Promise<ToolResult> => {
    const d = definitions.find((t) => t.name === name);
    if (!d) return { content: [{ type: 'text', text: `Unknown tool: ${name}` }], isError: true };
    try {
      return await withMemory((m) => d.run(m, args));
    } catch (e) {
      return { content: [{ type: 'text', text: e instanceof Error ? e.message : String(e) }], isError: true };
    }
  };

  const handle = async (r: Request): Promise<unknown> => {
    const isCall = r.id !== undefined;
    if (r.jsonrpc !== '2.0' || typeof r.method !== 'string') {
      return isCall ? { jsonrpc: '2.0', id: r.id ?? null, error: { code: -32600, message: 'Invalid request' } } : undefined;
    }
    const ok = (result: unknown) => ({ jsonrpc: '2.0', id: r.id, result });
    switch (r.method) {
      case 'initialize': {
        const asked = r.params?.protocolVersion;
        const protocolVersion = typeof asked === 'string' && MCP_PROTOCOL_VERSIONS.includes(asked) ? asked : MCP_PROTOCOL_VERSIONS[0];
        return ok({ protocolVersion, capabilities: { tools: {} }, serverInfo: { name: 'memtrails', version: o.version }, instructions: INSTRUCTIONS });
      }
      case 'ping':
        return isCall ? ok({}) : undefined;
      case 'tools/list':
        return ok({ tools: definitions.map(({ name, description, inputSchema, annotations }) => ({ name, description, inputSchema, annotations })) });
      case 'tools/call': {
        const name = r.params?.name;
        if (typeof name !== 'string') return { jsonrpc: '2.0', id: r.id, error: { code: -32602, message: 'tools/call needs a tool name' } };
        return ok(await call(name, r.params?.arguments ?? {}));
      }
      default:
        // Notifications (initialized, cancelled, …) need no answer; an unknown request does.
        return isCall ? { jsonrpc: '2.0', id: r.id, error: { code: -32601, message: `Method not found: ${r.method}` } } : undefined;
    }
  };

  const pending = new Set<Promise<void>>();
  const lines = createInterface({ input, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch {
      send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } });
      continue;
    }
    // Answered as each finishes: a slow search does not hold up a ping.
    const p = (async () => {
      if (Array.isArray(message)) {
        const answers = (await Promise.all(message.map((m) => handle(asRequest(m))))).filter((a) => a !== undefined);
        if (answers.length) send(answers);
      } else {
        const answer = await handle(asRequest(message));
        if (answer !== undefined) send(answer);
      }
    })().catch((e: unknown) => log(`[memtrails] mcp: ${String(e)}`));
    pending.add(p);
    void p.finally(() => pending.delete(p));
  }
  await Promise.all(pending);
  if (idle) clearTimeout(idle);
  await closeMemory();
}

function asRequest(m: unknown): Request {
  return m && typeof m === 'object' && !Array.isArray(m) ? (m as Request) : {};
}

interface Definition {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: Record<string, boolean>;
  run: (m: Memory, args: unknown) => Promise<ToolResult>;
}

const saveInput = z.object({
  text: z.string().min(1).describe('What to remember, written to be understood on its own.'),
  title: z.string().max(200).optional().describe('A short title (default: the start of the text).'),
});

function toolDefinitions(readOnly: boolean): Definition[] {
  // The agent tools' names, texts and schemas, from a memory that is never searched: listing needs no open memory.
  const shapes = toolShapes();
  const defs: Definition[] = shapes.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: t.jsonSchema,
    annotations: READ_ONLY,
    run: (m, args) => {
      const live = m.tools().find((x) => x.name === t.name);
      if (!live) return Promise.resolve({ content: [{ type: 'text', text: `Unknown tool: ${t.name}` }], isError: true });
      return live.handler(args);
    },
  }));
  if (!readOnly) {
    defs.push({
      name: 'memory_save',
      description: SAVE_DESCRIPTION,
      inputSchema: z.toJSONSchema(saveInput) as Record<string, unknown>,
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
      run: async (m, args) => {
        const got = saveInput.safeParse(args ?? {});
        if (!got.success) {
          return { content: [{ type: 'text', text: `Invalid arguments: ${got.error.issues.map((i) => `${i.path.join('.') || 'input'}: ${i.message}`).join('; ')}` }], isError: true };
        }
        const now = new Date();
        const id = `${now.toISOString()}-${randomBytes(4).toString('hex')}`;
        await m.add({ id, source: 'mcp', kind: 'note', title: got.data.title, text: got.data.text, at: now.toISOString() });
        return { content: [{ type: 'text', text: `Saved (${now.toISOString().slice(0, 10)}, id ${id}).` }] };
      },
    });
  }
  return defs;
}

let shapesCache: AgentTool[] | null = null;
function toolShapes(): AgentTool[] {
  const never = (): never => { throw new Error('not a memory'); };
  return (shapesCache ??= buildAgentTools({ search: never, open: never, explore: never }));
}
