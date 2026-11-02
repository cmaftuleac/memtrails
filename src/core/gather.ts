/**
 * Gathering: a SET of memories for a question, not a top-k.
 *
 * A question that counts, lists, totals or compares ("how many doctors did I see", "what did I spend on luxury items")
 * needs every instance, and one ranked below the cut is a wrong answer however good the rest are — measured on
 * LongMemEval's multi-session questions: every fact our answers missed was in the store, as a dated observation, and not
 * in the room. Mem0 and Graphiti hand the reader a fixed number by similarity; here the fast model reads them:
 *   1. a wide search (memory_search at its limit: up to 200 observations and 200 other memories);
 *   2. the fast model reads every candidate, in parts at once, and keeps each one the answer needs — each instance of
 *      what is counted, what changed or cancelled one, the dates — and names what it did NOT see that could also count
 *      (other kinds, other words);
 *   3. those searches run, and what they find that is new is read the same way.
 * The kept set comes first, in rank order; the rest follows by rank (a reader still has them). Nothing is dropped: a part
 * the model did not answer keeps all of its candidates.
 */
import { redactString } from './redact.js';
import { defaultAsk, type Ask } from '../llm.js';
import { PROFILE_SHOWN } from './profiles.js';
import { memorySearch, MAX_SEARCH, type RecallDeps, type RecalledItem, type SearchQuery } from './recall.js';
import { cut } from './text.js';

/** Candidates the model reads in one ask. */
const PART = 40;
/** With notes, all of them in one ask, so a question's notes are written once: on LoCoMo the same accuracy as two parts
 *  of 40 (80.9 against 80.6), at 58% of the cost and 2.9 s against 7.5 s median. */
const NOTES_PART = 80;
/** Searches it may ask for. */
const MORE = 6;

const SYSTEM = `You pick, from someone's memories, every one needed to answer their question.

Keep a memory when it states something the answer needs. When the question counts, lists, totals, compares or orders
things, keep EVERY memory that is one instance of what is asked about — each purchase, visit, trip, item, person or
event — even one mentioned in passing inside another topic, and every memory that changes an instance (returned,
cancelled, sold, replaced, ended, a new amount) or dates it. When unsure, keep it: a missing instance makes the answer
wrong, an extra one does not.

In "searches", write up to ${MORE} short searches for instances the question could count that none of these memories
shows — other kinds, other words for the same thing (for "luxury items": "designer handbag", "jewelry", "watch") —
in the language of the question. Leave it empty when nothing is missing. Never translate.`;

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['keep', 'searches'],
  properties: {
    keep: { type: 'array', items: { type: 'string' } },
    searches: { type: 'array', items: { type: 'string' } },
  },
} as const;

/** With notes: the same reading also writes down what the kept memories say that answers the question, so a list spread
 *  over many memories reaches the reader as one line. */
const NOTES_RULE = `

In "notes", write what the memories you kept say that answers the question, in a few short lines: every instance with
its date as the memories give it ("Books Tim has read: Harry Potter (2023-08-02), The Name of the Wind (2023-08-11)"),
names, numbers and words as written. Only what they state, as plain statements: never write that something is not said,
not given, unclear or unknown — leave it out instead, since other memories may hold it — and never a guess. Leave notes
empty when the memories you kept say nothing about it. In the language of the question.`;

const NOTES_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['keep', 'searches', 'notes'],
  properties: { ...SCHEMA.properties, notes: { type: 'array', items: { type: 'string' } } },
} as const;

const day = (ts: number): string => (ts ? new Date(ts).toISOString().slice(0, 10) : '');

/** One candidate as the model reads it: when it happened (from when to when, for what lasted) and when it was told, its
 *  text, and the passage that matched. */
function line(h: RecalledItem, n: number): string {
  const happened = [h.eventAt, h.startAt && `from ${h.startAt}`, h.endAt && `to ${h.endAt}`].filter(Boolean).join(' ');
  const when = happened ? `${happened}; told ${day(h.ts)}` : day(h.ts);
  const text = h.kind === 'observation' ? h.summary : `${h.title} — ${cut(h.summary, h.kind === 'profile' ? PROFILE_SHOWN : 600)}`;
  return `[m${n}] (${when}) ${text}${h.match ? `\n    … ${cut(h.match, 600)}` : ''}`;
}

async function pick(ask: Ask, question: string, list: RecalledItem[], notes = false): Promise<{ keep: Set<string>; searches: string[]; notes: string[] }> {
  const parts: RecalledItem[][] = [];
  const size = notes ? NOTES_PART : PART;
  for (let i = 0; i < list.length; i += size) parts.push(list.slice(i, i + size));
  const answers = await Promise.all(parts.map(async (part) => {
    const prompt = `Question: ${question}\n\nMemories:\n${part.map((h, n) => line(h, n + 1)).join('\n')}`;
    const a = await ask({
      purpose: 'memory', system: notes ? SYSTEM + NOTES_RULE : SYSTEM, prompt: redactString(prompt),
      schema: (notes ? NOTES_SCHEMA : SCHEMA) as unknown as Record<string, unknown>,
      name: 'memory_gather', maxTokens: 40 * part.length + (notes ? 1400 : 600), timeoutMs: 120_000, consequence: 'so these memories are kept unread',
    });
    if (!('answer' in a)) return { keep: part.map((h) => h.id), searches: [] as string[], notes: [] as string[] };
    const got = a.answer as { keep?: unknown; searches?: unknown; notes?: unknown };
    const ids = Array.isArray(got.keep) ? got.keep.filter((x): x is string => typeof x === 'string') : [];
    const keep = ids.map((x) => part[Number(x.replace(/^m/, '')) - 1]?.id).filter((x): x is string => !!x);
    const searches = Array.isArray(got.searches) ? got.searches.filter((x): x is string => typeof x === 'string' && !!x.trim()) : [];
    const said = notes && Array.isArray(got.notes) ? got.notes.filter((x): x is string => typeof x === 'string' && !!x.trim()).map((x) => redactString(x.trim())) : [];
    return { keep, searches, notes: said };
  }));
  return { keep: new Set(answers.flatMap((a) => a.keep)), searches: [...new Set(answers.flatMap((a) => a.searches.map((s) => s.trim())))],
    notes: answers.flatMap((a) => a.notes) };
}

export interface Gathered {
  /** What the answer needs, best first, then the rest of what was found, by rank. */
  found: RecalledItem[];
  /** The ids kept. */
  kept: string[];
  /** The searches the model asked for. */
  searches: string[];
  /** With `notes`: what the kept memories say that answers the question, a few short lines (empty otherwise). Show them
   *  before the memories, as notes on them: on LoCoMo this took 79.5 to 80.9 (docs/BENCHMARKS.md). */
  notes: string[];
}

/** `from`/`to`: the time the question asks about ("on 10 April 2022", "in October 2023"). What happened then is searched
 *  for besides everything else and read first — not instead: a question also names dates it only counts from ("as of
 *  November 2023"), and a date filter alone would hide the event it counts. */
export async function memoryGather(d: RecallDeps, q: SearchQuery, o: { ask?: Ask; limit?: number; notes?: boolean }): Promise<Gathered> {
  const ask = o.ask ?? defaultAsk;
  const limit = o.limit ?? MAX_SEARCH;
  const { from, to, ...open } = q;
  const [inDates, wide] = await Promise.all([from || to ? memorySearch(d, { ...q, limit }) : [], memorySearch(d, { ...open, limit })]);
  const dated = new Set(inDates.map((h) => h.id));
  const first = [...inDates, ...wide.filter((h) => !dated.has(h.id))];
  const one = await pick(ask, q.query, first, o.notes);
  const searches = one.searches.slice(0, MORE);
  const have = new Set(first.map((h) => h.id));
  const more: RecalledItem[] = [];
  for (const found of await Promise.all(searches.map((s) => memorySearch(d, { ...open, query: s, limit: 30 })))) {
    for (const h of found) if (!have.has(h.id)) { have.add(h.id); more.push(h); }
  }
  const two = more.length ? await pick(ask, q.query, more, o.notes) : { keep: new Set<string>(), searches: [], notes: [] };
  const all = [...first, ...more];
  const kept = all.filter((h) => one.keep.has(h.id) || two.keep.has(h.id));
  const keptIds = new Set(kept.map((h) => h.id));
  return { found: [...kept, ...all.filter((h) => !keptIds.has(h.id))], kept: kept.map((h) => h.id), searches, notes: [...one.notes, ...two.notes] };
}
