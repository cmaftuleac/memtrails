/**
 * An agent's tools over a memory: `memory_search` (find), `memory_open` (read one whole, with the records it was learned
 * from) and `memory_explore` (from a topic or items to what is related but not found by search). Defined with zod
 * schemas, as MCP servers declare tools; `jsonSchema` gives the same input schema for function calling APIs.
 *
 *   const tools = memory.tools();   // hand them to an MCP server, or to a model's tool calling
 */
import { z } from 'zod';
import type { Explored, RecalledItem, SearchQuery } from '../core/recall.js';

type ToolResult = { content: Array<{ type: 'text'; text: string }>; isError?: boolean };
export interface AgentTool {
  name: string;
  description: string;
  inputSchema: Record<string, z.ZodType>;
  /** The input schema as JSON Schema, for function calling. */
  jsonSchema: Record<string, unknown>;
  handler: (args: any, extra?: unknown) => Promise<ToolResult>;
}
function tool<S extends Record<string, z.ZodType>>(name: string, description: string, inputSchema: S,
  handler: (args: z.infer<z.ZodObject<S>>, extra?: unknown) => Promise<ToolResult>): AgentTool {
  return { name, description, inputSchema, jsonSchema: z.toJSONSchema(z.object(inputSchema)) as Record<string, unknown>, handler: handler as AgentTool['handler'] };
}

const DESCRIPTIONS = {
  memory_search: 'Search memory: everything it was given (e-mails, messages, notes, documents, conversations, commits, …) and '
    + 'what was learned from them — facts, decisions, dated events, profiles of people and things. Best first; each result '
    + 'with its date and id. Ask in plain words, as you would ask a colleague who read it all. Name dates when the question '
    + 'names a time. Open a result with memory_open for all of it.',
  memory_open: 'Read one memory item whole, by the id memory_search or memory_explore gave: everything it holds, when it '
    + 'happened, and for something learned, the records it was learned from.',
  memory_explore: 'Explore memory around a topic, or around items you already found: it follows what they share — the same '
    + 'people, places, systems, names — and links between memories, to what a search for the words does not find. Returns '
    + 'what search found, what is related, and the concepts around it to explore next.',
};

/** What the tools read: a memory (openMemory) gives these. */
export interface ToolProvider {
  search(q: SearchQuery): Promise<RecalledItem[]>;
  open(id: string): Promise<Record<string, unknown> | null>;
  explore?(q: { topic?: string; from?: string[]; collections?: string[]; limit?: number }): Promise<Explored>;
}

type CrossMemoryHit = RecalledItem;

const day = (ts: number): string => (ts ? new Date(ts).toISOString().slice(0, 10) : '');
const hitLine = (h: CrossMemoryHit, n: number): string => `${n + 1}. ${h.title} (${h.kind === 'episode' ? 'earlier work' : h.kind} · ${day(h.ts)}${h.record?.author ? ` · ${h.record.author}` : ''} · id ${h.id})`;

const WEEKDAY = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** A date as a reader reckons with it — the weekday too ("the Friday before …"): 2023-07-14 → "Fri 14 Jul 2023". */
export function readableDate(d: string): string {
  const [y, m, dd] = d.split('-').map(Number);
  if (!y) return '';
  if (!m) return String(y);
  if (!dd) return `${MONTH[m - 1]} ${y}`;
  return `${WEEKDAY[new Date(Date.UTC(y, m - 1, dd)).getUTCDay()]} ${dd} ${MONTH[m - 1]} ${y}`;
}

/** How long something lasted, as a reader reckons with it: "since Feb 2022", "Nov 2021 – May 2022". */
export function lastedText(h: { startAt?: unknown; endAt?: unknown }): string {
  const [from, to] = [h.startAt, h.endAt].map((d) => (typeof d === 'string' && d ? readableDate(d) : ''));
  if (from && to) return `${from} – ${to}`;
  return from ? `since ${from}` : to ? `until ${to}` : '';
}

/** When a hit happened, for ordering by time: an observation's own date, else when it was learned. */
const when = (h: CrossMemoryHit): string => h.eventAt || day(h.ts);

/** All of a profile — what memory knows about one subject, one dated fact a line (core/profiles.ts) — up to this:
 *  measured, a profile cut like any other item lost most of its gain. */
export const PROFILE_SHOWN = 1500;
const cutAt = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n)}…` : s);

/** What memory_search shows: an observation as one dated line (when it happened, and when it was told when that
 *  differs), a profile whole (up to PROFILE_SHOWN), anything else as its title line and summary — shorter when many
 *  are asked for. Best first, or by time. */
export function searchText(hits: CrossMemoryHit[], order: 'relevance' | 'time' = 'relevance'): string {
  const list = order === 'time' ? [...hits].sort((a, b) => when(a).localeCompare(when(b))) : hits;
  // Many hits: one line each — the date, what it is, its summary — so a list question sees every one of them clearly.
  const compact = list.length > 20;
  const lines = list.map((h, n) => {
    if (h.kind === 'profile') return `${hitLine(h, n)}\n   ${cutAt(h.summary, PROFILE_SHOWN).replace(/\n/g, '\n   ')}`;
    if (h.kind !== 'observation') {
      // The passage that matched, when it was not the start: a fact said deep in a long request is what was found.
      const matched = h.match ? ` … ${h.match.replace(/\s+/g, ' ')}` : '';
      if (!compact) return `${hitLine(h, n)}\n   ${h.summary.slice(0, h.match ? 200 : 500)}${matched}`;
      const what = h.kind === 'episode' ? 'earlier work' : h.kind;
      return `${n + 1}. (${readableDate(day(h.ts))} · ${what}) ${h.summary.replace(/\s+/g, ' ').slice(0, h.match ? 150 : 300)}${matched} (id ${h.id})`;
    }
    const told = day(h.ts);
    const lasted = lastedText(h);
    const happened = [h.eventAt ? readableDate(h.eventAt) : '', lasted].filter(Boolean).join(', ');
    const at = happened ? `${happened}${h.eventAt !== told || lasted ? `; told ${readableDate(told)}` : ''}` : `told ${readableDate(told)}`;
    return `${n + 1}. (${at}) ${h.summary} (id ${h.id})`;
  });
  return `${order === 'time' ? 'Oldest first — the date is when it happened (and when it was told, when that differs).\n' : ''}${lines.join('\n')}`;
}

function memoryTools(p: ToolProvider): AgentTool[] {
  return [
    tool(
      'memory_search',
      DESCRIPTIONS.memory_search,
      {
        query: z.string().describe('What you need to know, in plain words.'),
        collections: z.array(z.string()).optional().describe('Only these collections (default: all).'),
        limit: z.number().int().min(1).max(200).optional().describe('How many of each to return — dated observations, and other memories (default 20; 100 to 200 when you gather everything on a topic, list or count things, or need their dates).'),
        order: z.enum(['relevance', 'time']).optional().describe('Best first (default), or oldest first — for what happened when, and in what order.'),
        from: z.string().regex(/^\d{4}(-\d{2}(-\d{2})?)?$/).optional().describe('Only what happened on or after this date (YYYY-MM-DD, YYYY-MM or YYYY) — when the request names a time ("in June 2023", "last week").'),
        to: z.string().regex(/^\d{4}(-\d{2}(-\d{2})?)?$/).optional().describe('Only what happened on or before this date (YYYY-MM-DD, YYYY-MM or YYYY).'),
      },
      async (args) => {
        try {
          const hits = await p.search({ query: args.query, collections: args.collections, limit: args.limit ?? 20, from: args.from, to: args.to });
          if (!hits.length) return text('Nothing in memory matches that.');
          return text(`${searchText(hits, args.order)}\n\nOpen one with memory_open and its id. It can be stale: check it against the current state before relying on it.`);
        } catch (err) {
          return text((err as Error).message, true);
        }
      },
    ),
    tool(
      'memory_open',
      DESCRIPTIONS.memory_open,
      { id: z.string().describe('The memory id, as memory_search or memory_explore gave it.') },
      async (args) => {
        const it = await p.open(args.id).catch(() => null);
        if (!it) return text(`There is no memory "${args.id}" (it may have been merged or cleared).`, true);
        const files = Array.isArray(it.files) && it.files.length ? `\nFiles: ${(it.files as string[]).join(', ')}` : '';
        const told = day(Number(it.ts ?? 0));
        const happened = [typeof it.eventAt === 'string' && it.eventAt ? readableDate(it.eventAt) : '', lastedText(it)].filter(Boolean).join(', ');
        const when = happened ? `happened ${happened}; told ${readableDate(told)}` : told;
        // The raw requests it was learned from, whole (the truth it is a key to), and the texts a merge folded into it.
        const sources = Array.isArray(it.sources) && it.sources.length
          ? `\n\n## Where it came from\n${(it.sources as Array<{ id: string; when: number; title: string; body: string }>).map((x) => `### ${day(x.when)} · ${x.title} (id ${x.id})\n${x.body}`).join('\n\n')}` : '';
        const absorbed = Array.isArray(it.absorbed) && it.absorbed.length
          ? `\n\n## Texts merged into it\n${(it.absorbed as Array<{ title: string; summary: string; body: string }>).map((x) => `### ${x.title}\n${x.summary}\n${x.body}`).join('\n\n')}` : '';
        return text(`# ${String(it.title ?? '')}\n(${String(it.kind ?? '')} · ${when} · ${String(it.project ?? '')})\n\n${String(it.summary ?? '')}\n\n${String(it.body ?? '')}${files}${sources}${absorbed}`);
      },
    ),
    ...(p.explore ? [tool(
      'memory_explore',
      DESCRIPTIONS.memory_explore,
      {
        topic: z.string().optional().describe('What to explore, in plain words — a system, a project, a problem, a name.'),
        from: z.array(z.string()).max(5).optional().describe('Memory ids to explore from, as memory_search or memory_explore gave them.'),
        collections: z.array(z.string()).optional().describe('Only these collections (default: all).'),
        limit: z.number().int().min(1).max(20).optional().describe('How many related items to return (default 10).'),
      },
      async (args) => {
        if (!args.topic?.trim() && !args.from?.length) return text('Give a topic, or memory ids to explore from.', true);
        try {
          const r = await p.explore!({ topic: args.topic, from: args.from, collections: args.collections, limit: args.limit ?? 10 });
          const parts: string[] = [];
          if (r.found.length) parts.push(`Found by search:\n${r.found.map(hitLine).join('\n')}`);
          if (r.related.length) parts.push(`Related — not found by search, reached through what they share:\n${r.related.map((h, n) => `${hitLine(h, n)} — ${h.via}\n   ${h.summary.slice(0, 300)}`).join('\n')}`);
          if (r.concepts.length) parts.push(`Concepts around it: ${r.concepts.join(', ')}`);
          if (!parts.length) return text('Nothing in memory reaches that.');
          return text(`${parts.join('\n\n')}\n\nOpen one with memory_open; explore further from an item (from: [its id]) or with one of the concepts as the topic. It can be stale: check it against the current state before relying on it.`);
        } catch (err) {
          return text((err as Error).message, true);
        }
      },
    )] : []),
  ];
}

const text = (t: string, isError = false) => ({ content: [{ type: 'text' as const, text: t }], ...(isError ? { isError: true } : {}) });
export function buildAgentTools(p: ToolProvider): AgentTool[] {
  return memoryTools(p);
}
