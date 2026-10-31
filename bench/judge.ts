/**
 * Judged answers: the same reader and the same judge for every system's contexts. The reader answers from the context
 * alone; the judge is LongMemEval's own yes/no prompt (github.com/xiaowu0162/LongMemEval, src/evaluation/evaluate_qa.py),
 * its temporal-reasoning variant for LoCoMo's temporal questions, the plain one for the rest — the strict score in
 * docs/BENCHMARKS.md. Mem0's lenient LoCoMo judge lives in Mem0's own benchmark repository; it is not copied here.
 *
 *   OPENAI_API_KEY=… npx tsx bench/judge.ts runs/memtrails/locomo.jsonl [--reader gpt-6-luna] [--judge gpt-5.5] [--chars 6000]
 *
 * Resumable: answers are appended to <run>.judged.jsonl and a question already judged is skipped. Prints accuracy by
 * category and the measured spend.
 */
import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { openAiAsk, type LlmUsage } from '../src/index.js';

const { values: a, positionals } = parseArgs({ allowPositionals: true, options: {
  reader: { type: 'string', default: process.env.BENCH_READER ?? 'gpt-6-luna' }, judge: { type: 'string', default: process.env.BENCH_JUDGE ?? 'gpt-5.5' },
  chars: { type: 'string', default: '28000' }, concurrency: { type: 'string', default: '16' },
  'price-in': { type: 'string', default: '0' }, 'price-out': { type: 'string', default: '0' },
} });
const run = positionals[0];
if (!run) throw new Error('usage: judge.ts <run.jsonl> [--reader m] [--judge m] [--chars n]');

const BASE = 'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. '
  + 'Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, '
  + 'you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. ';
const TEMPORAL = 'In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., '
  + "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. ";
const judgePrompt = (q: string, gold: string, response: string, temporal: boolean): string =>
  `${BASE}${temporal ? TEMPORAL : ''}\n\nQuestion: ${q}\n\nCorrect Answer: ${gold}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
const READ = 'You answer a question about past conversations and records, using ONLY the memory context retrieved for it. The context '
  + 'may be partial or contain unrelated items. Reason about dates from the dates given. If the context does not hold the answer, say you '
  + 'do not know. Answer concisely.';

let tokens = { in: 0, out: 0 };
const onUsage = (u: LlmUsage): void => { tokens = { in: tokens.in + u.inputTokens, out: tokens.out + u.outputTokens }; };
const reader = openAiAsk({ model: a.reader, reasoningEffort: 'low', onUsage });
const judge = openAiAsk({ model: a.judge, reasoningEffort: 'low', onUsage });

type Row = { qid: string; question: string; answer: string; category: string; context: string };
const rows = readFileSync(run, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as Row);
const out = run.replace(/\.jsonl$/, '') + '.judged.jsonl';
const done = new Map<string, { correct: boolean; category: string }>(existsSync(out)
  ? readFileSync(out, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)).map((r) => [r.qid, r]) : []);

async function one(r: Row): Promise<void> {
  const read = await reader({ purpose: 'bench', system: READ, prompt: `MEMORY CONTEXT:\n${r.context.slice(0, Number(a.chars))}\n\nQUESTION: ${r.question}`,
    schema: { type: 'object', additionalProperties: false, required: ['answer'], properties: { answer: { type: 'string' } } }, name: 'answer', maxTokens: 4000 });
  if (!('answer' in read)) throw new Error(`reader: ${'unavailable' in read ? read.unavailable : 'busy'}`);
  const response = String(read.answer.answer ?? '');
  const verdict = await judge({ purpose: 'bench', system: '', prompt: judgePrompt(r.question, r.answer, response, r.category === 'temporal'),
    schema: { type: 'object', additionalProperties: false, required: ['verdict'], properties: { verdict: { type: 'string', enum: ['yes', 'no'] } } }, name: 'verdict', maxTokens: 2000 });
  if (!('answer' in verdict)) throw new Error(`judge: ${'unavailable' in verdict ? verdict.unavailable : 'busy'}`);
  const row = { qid: r.qid, category: r.category, response, correct: verdict.answer.verdict === 'yes' };
  appendFileSync(out, `${JSON.stringify(row)}\n`);
  done.set(r.qid, row);
}

const todo = rows.filter((r) => !done.has(r.qid));
let next = 0;
await Promise.all(Array.from({ length: Number(a.concurrency) }, async () => {
  while (next < todo.length) await one(todo[next++]!);
}));
for (const cat of ['all', ...new Set(rows.map((r) => r.category))]) {
  const sub = rows.filter((r) => cat === 'all' || r.category === cat).map((r) => done.get(r.qid)!).filter(Boolean);
  console.log(`${cat.padEnd(12)} n=${String(sub.length).padStart(4)}  correct ${(100 * sub.filter((x) => x.correct).length / sub.length).toFixed(1)}`);
}
const usd = tokens.in * Number(a['price-in']) / 1e6 + tokens.out * Number(a['price-out']) / 1e6;
console.log(`tokens in ${tokens.in}, out ${tokens.out}${usd ? `, $${usd.toFixed(2)}` : ' (give --price-in/--price-out per million for the cost)'}`);
