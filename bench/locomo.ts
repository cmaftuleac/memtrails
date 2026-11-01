/**
 * LoCoMo with memtrails: each conversation built into its own memory from nothing (every session added as a
 * conversation; optionally consolidated and slept with a language model), then each question's context written as an
 * agent reads a search — memory_search's lines, best first until the room is full, shown oldest first.
 *
 *   npx tsx bench/locomo.ts --data locomo10.json --out runs/memtrails [--conversations conv-41,conv-42] \
 *       [--categories 1,2,3,4] [--room 6000] [--limit 20] [--consolidate] [--gather 40]
 *
 * `--gather n` builds each room from `gather` at limit n (a language model reads what was found and keeps what the
 * answer needs, shown first) instead of `search`. Writes <out>/locomo.jsonl ({ qid, question, answer, category, context, retrieval_ms }). Score it with bench/score.ts
 * (no model) and bench/judge.ts (a reader and a judge). LoCoMo: github.com/snap-research/locomo (CC BY-NC 4.0); bring
 * your own copy of locomo10.json.
 */
import { appendFileSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { openMemory, searchText, type Conversation } from '../src/index.js';

const { values: a } = parseArgs({ options: {
  data: { type: 'string' }, out: { type: 'string' }, conversations: { type: 'string' }, categories: { type: 'string', default: '1,2,3,4' },
  room: { type: 'string', default: '6000' }, limit: { type: 'string', default: '20' }, consolidate: { type: 'boolean', default: false },
  gather: { type: 'string', default: '0' }, threads: { type: 'string', default: '4' },
} });
if (!a.data || !a.out) throw new Error('usage: --data locomo10.json --out <dir> [...]');
const ROOM = Number(a.room);
const LIMIT = Number(a.limit);
const GATHER = Number(a.gather);
const CATS = a.categories!.split(',').map(Number);
const CATEGORY: Record<number, string> = { 1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop' };

/** "1:56 pm on 8 May, 2023" */
function locomoDate(s: string): number {
  const m = /(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})/i.exec(s);
  if (!m) throw new Error(`unreadable LoCoMo date: ${s}`);
  const month = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'].indexOf(m[5]!.toLowerCase());
  return Date.UTC(+m[6]!, month, +m[4]!, (+m[1]! % 12) + (m[3]!.toLowerCase() === 'pm' ? 12 : 0), +m[2]!);
}

const data = JSON.parse(readFileSync(a.data, 'utf8')) as Array<Record<string, any>>;
const want = a.conversations ? new Set(a.conversations.split(',')) : null;
mkdirSync(a.out, { recursive: true });
const file = join(a.out, 'locomo.jsonl');
rmSync(file, { force: true });

for (const conv of data.filter((c) => !want || want.has(c.sample_id))) {
  const unit = String(conv.sample_id);
  const c = conv.conversation as Record<string, any>;
  const memory = await openMemory({ home: join(a.out, 'homes', unit), embedding: { threads: Number(a.threads) } });
  const t0 = Date.now();
  // Whoever opens a session makes its requests; the other replies.
  for (let n = 1; c[`session_${n}`]; n++) {
    const turns = c[`session_${n}`] as Array<Record<string, string>>;
    const opener = turns[0]?.speaker;
    const session: Conversation = { id: `${unit}:session_${n}`, at: locomoDate(String(c[`session_${n}_date_time`])),
      turns: turns.map((t) => ({ role: t.speaker === opener ? 'user' : 'assistant',
        text: `${t.speaker}: ${t.text}${t.blip_caption ? ` [shared an image: ${t.blip_caption}]` : ''}` })) };
    await memory.addConversation(session);
  }
  if (a.consolidate) {
    await memory.consolidate();
    for (let night = 0; night < 50; night++) if (!(await memory.sleep()).looked) break;
  }
  console.log(JSON.stringify({ unit, ...memory.stats(), build_s: (Date.now() - t0) / 1000 }));
  for (const [i, q] of (conv.qa as Array<{ question: string; answer?: unknown; category: number }>).entries()) {
    if (!CATS.includes(q.category)) continue;
    const t = Date.now();
    const found = GATHER ? (await memory.gather({ query: q.question, limit: GATHER })).found : await memory.search({ query: q.question, limit: LIMIT });
    const shown: typeof found = [];
    let used = 0;
    for (const f of found) {
      const line = searchText([f]).length + 1;
      if (used + line > ROOM) continue;
      shown.push(f);
      used += line;
    }
    appendFileSync(file, `${JSON.stringify({ qid: `${unit}:${i}`, question: q.question, answer: String(q.answer ?? ''),
      category: CATEGORY[q.category], context: searchText(shown, 'time'), retrieval_ms: Date.now() - t })}\n`);
  }
  await memory.close();
}
