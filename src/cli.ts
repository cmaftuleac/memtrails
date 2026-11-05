#!/usr/bin/env node
/**
 * memtrails on the command line: `mcp` serves a memory to an MCP client, `add` fills it from files, mail, Slack or
 * git, `search` asks it. The memory is MEMTRAILS_HOME, or ~/.memtrails, unless --home names another.
 */
import { existsSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { extname, join, resolve } from 'node:path';
import { fromEml, fromFiles, fromGit, fromJsonl, fromMbox, fromSlackExport } from './importers/index.js';
import { serveMcp } from './mcp.js';
import { openMemory, type Memory } from './memory.js';
import type { MemoryRecord } from './records.js';

const VERSION = (createRequire(import.meta.url)('../package.json') as { version: string }).version;

const USAGE = `memtrails ${VERSION} — local memory for agents

  memtrails mcp [--read-only] [--idle MINUTES]   serve the memory to an MCP client over stdio
  memtrails add PATH... [--git REPO] [--collection NAME]
                                                 add folders of notes and documents, .mbox/.eml mail,
                                                 .jsonl records, a Slack export folder, a git history
  memtrails search QUERY... [--limit N]          search, best first

  --home DIR   the memory (default: MEMTRAILS_HOME, or ~/.memtrails)

MCP client configuration (Claude Desktop, Cursor, …):
  { "mcpServers": { "memory": { "command": "npx", "args": ["-y", "memtrails", "mcp"] } } }
Claude Code:
  claude mcp add memory -- npx -y memtrails mcp
`;

interface Args { command: string; rest: string[]; flags: Map<string, string[]>; switches: Set<string> }

const VALUED = new Set(['--home', '--git', '--collection', '--limit', '--idle']);
const SWITCHES = new Set(['--read-only', '--help', '-h', '--version', '-v']);

function parse(argv: string[]): Args {
  const flags = new Map<string, string[]>();
  const switches = new Set<string>();
  const rest: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    const eq = a.indexOf('=');
    const name = a.startsWith('--') && eq > 0 ? a.slice(0, eq) : a;
    if (VALUED.has(name)) {
      const value = eq > 0 && a.startsWith('--') ? a.slice(eq + 1) : argv[++i];
      if (value === undefined) throw new Error(`${name} needs a value`);
      flags.set(name, [...(flags.get(name) ?? []), value]);
    } else if (SWITCHES.has(a)) {
      switches.add(a);
    } else if (a.startsWith('-') && a !== '-') {
      throw new Error(`unknown option ${a}`);
    } else {
      rest.push(a);
    }
  }
  return { command: rest.shift() ?? '', rest, flags, switches };
}

const one = (a: Args, name: string): string | undefined => a.flags.get(name)?.at(-1);
const homeOf = (a: Args): string => resolve(one(a, '--home') ?? process.env.MEMTRAILS_HOME ?? join(homedir(), '.memtrails'));
const say = (line: string): void => { process.stderr.write(`${line}\n`); };

function number(a: Args, name: string, fallback: number, min: number): number {
  const raw = one(a, name);
  if (raw === undefined) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < min) throw new Error(`${name} must be a number of at least ${min}`);
  return n;
}

/** What a path holds, as records: the importer its kind calls for. */
function importerFor(path: string, collection: string | undefined): AsyncIterable<MemoryRecord> {
  if (!existsSync(path)) throw new Error(`${path}: not found`);
  if (statSync(path).isDirectory()) {
    if (existsSync(join(path, 'channels.json')) && existsSync(join(path, 'users.json'))) return fromSlackExport(path, { collection });
    return fromFiles(path, { collection, onSkip: (p, why) => say(`  skipped ${p}: ${why}`) });
  }
  switch (extname(path).toLowerCase()) {
    case '.mbox': return fromMbox(path, { collection });
    case '.eml': return fromEml(path, { collection });
    case '.jsonl': return fromJsonl(path, { collection, onSkip: (_line: string, why: string, n: number) => say(`  skipped line ${n}: ${why}`) });
    default: throw new Error(`${path}: give a folder of notes or documents, an .mbox or .eml file, a .jsonl file, or a Slack export folder`);
  }
}

async function withMemory<T>(a: Args, f: (m: Memory) => Promise<T>): Promise<T> {
  const memory = await openMemory({ home: homeOf(a), log: say });
  try {
    return await f(memory);
  } finally {
    await memory.close();
  }
}

async function add(a: Args): Promise<void> {
  const collection = one(a, '--collection');
  const sources: Array<[string, AsyncIterable<MemoryRecord>]> = [
    ...a.rest.map((p): [string, AsyncIterable<MemoryRecord>] => [p, importerFor(resolve(p), collection)]),
    ...(a.flags.get('--git') ?? []).map((p): [string, AsyncIterable<MemoryRecord>] => [`${p} (git)`, fromGit(resolve(p), { collection })]),
  ];
  if (!sources.length) throw new Error('add what? give one or more paths, or --git REPO');
  await withMemory(a, async (m) => {
    for (const [name, records] of sources) {
      say(`adding ${name}`);
      const r = await m.import(records, { onProgress: (n) => say(`  ${n} records`) });
      say(`  ${r.added} added, ${r.unchanged} unchanged`);
    }
    const s = m.stats();
    say(`${homeOf(a)}: ${s.records} records`);
  });
}

async function search(a: Args): Promise<void> {
  const query = a.rest.join(' ').trim();
  if (!query) throw new Error('search for what? give the question as words');
  const limit = number(a, '--limit', 10, 1);
  await withMemory(a, async (m) => { process.stdout.write(`${await m.searchText({ query, limit })}\n`); });
}

async function mcp(a: Args): Promise<void> {
  const home = homeOf(a);
  await serveMcp({
    version: VERSION,
    readOnly: a.switches.has('--read-only'),
    idleMs: number(a, '--idle', 5, 0) * 60_000,
    open: () => openMemory({ home, log: say }),
  });
}

async function main(): Promise<void> {
  const a = parse(process.argv.slice(2));
  if (a.switches.has('--version') || a.switches.has('-v')) return void process.stdout.write(`${VERSION}\n`);
  if (!a.command || a.command === 'help' || a.switches.has('--help') || a.switches.has('-h')) return void process.stdout.write(USAGE);
  switch (a.command) {
    case 'mcp': return mcp(a);
    case 'add': return add(a);
    case 'search': return search(a);
    default: throw new Error(`unknown command ${a.command}; try memtrails --help`);
  }
}

// Ends on its own, never by process.exit: exiting while ONNX Runtime's threads are still there aborts the process.
main().catch((e: unknown) => {
  say(`memtrails: ${e instanceof Error ? e.message : String(e)}`);
  process.exitCode = 1;
});
