/** The TypeScript side of the interop test (run.mjs runs it): write a memory, add one record to it, or read it.
 *
 *   tsx interop/ts.ts write <home> <corpus.json>
 *   tsx interop/ts.ts add   <home> <corpus.json>
 *   tsx interop/ts.ts read  <home> <queries.json>   → JSON on stdout
 */
import { readFileSync } from 'node:fs';
import { openMemory, type Conversation, type MemoryRecord, type RecalledItem, type SearchOptions } from '../src/index.js';

type Query = { name: string; op: 'search' | 'searchText' | 'room' | 'explore' | 'open'; q: Record<string, unknown> };

const hit = (h: RecalledItem) => ({ id: h.id, kind: h.kind, title: h.title, project: h.project, record: h.record?.id ?? null,
  summary: h.summary, match: h.match ?? null, score: h.score });

async function main(): Promise<void> {
  const [cmd, home, file] = process.argv.slice(2);
  if (!cmd || !home || !file) throw new Error('usage: ts.ts write|add|read <home> <file>');
  const input = JSON.parse(readFileSync(file, 'utf8'));
  const memory = await openMemory({ home, log: (line) => process.stderr.write(`[ts] ${line}\n`) });
  try {
    if (cmd === 'write') {
      const corpus = input as { records: MemoryRecord[]; conversations: Conversation[] };
      await memory.add(corpus.records);
      for (const c of corpus.conversations) await memory.addConversation(c);
    } else if (cmd === 'add') {
      await memory.add((input as { extra: MemoryRecord }).extra);
    } else if (cmd === 'read') {
      const results: Record<string, unknown> = {};
      for (const { name, op, q } of input as Query[]) {
        if (op === 'search') results[name] = { hits: (await memory.search(q as unknown as SearchOptions)).map(hit) };
        else if (op === 'searchText') results[name] = { text: await memory.searchText(q as unknown as SearchOptions) };
        else if (op === 'room') results[name] = { text: await memory.room(q as unknown as SearchOptions & { chars?: number }) };
        else if (op === 'explore') {
          const e = await memory.explore(q);
          results[name] = { found: e.found.map(hit), related: e.related.map((r) => ({ ...hit(r), via: r.via })), concepts: e.concepts };
        } else if (op === 'open') {
          const [first] = await memory.search(q as unknown as SearchOptions);
          results[name] = { id: first?.id ?? null, item: first ? await memory.open(first.id) : null };
        } else throw new Error(`unknown op ${op}`);
      }
      process.stdout.write(`${JSON.stringify({ lang: 'ts', stats: memory.stats(), results })}\n`);
    } else throw new Error(`unknown command ${cmd}`);
  } finally {
    await memory.close();
  }
}

main().catch((e) => {
  process.stderr.write(`[ts] ${e instanceof Error ? e.stack : String(e)}\n`);
  process.exit(1);
});
