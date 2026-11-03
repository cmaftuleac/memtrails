/**
 * Judged answers without LoCoMo's wrong golden answers. An independent audit (github.com/dial481/locomo-audit,
 * errors.json, CC BY-NC 4.0 — not redistributed here) checked every golden answer against its conversation; the
 * questions it found wrong (all but the wrong-citation-only ones) are dropped for every run alike, and nothing is
 * judged again. Prints accuracy over all questions and the audited ones, by category, and per question against
 * another run (a two-sided sign test). Mirrors docs/BENCHMARKS.md "the audited answers".
 *
 *   npx tsx bench/audited.ts --data locomo10.json --errors errors.json runs/a/locomo.judged.jsonl [runs/b/locomo.judged.jsonl]
 */
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { signTest } from './score.js';

const { values: a, positionals } = parseArgs({ allowPositionals: true, options: {
  data: { type: 'string' }, errors: { type: 'string' },
} });
const [first, other] = positionals;
if (!a.data || !a.errors || !first) throw new Error('usage: audited.ts --data locomo10.json --errors errors.json <run.judged.jsonl> [<other.judged.jsonl>]');

type Conv = { sample_id: string; qa: Array<{ question: string }> };
type AuditError = { question_id: string; question: string; error_type: string };
const convs = JSON.parse(readFileSync(a.data, 'utf8')) as Conv[];
const wrong = new Set<string>();
for (const e of JSON.parse(readFileSync(a.errors, 'utf8')) as AuditError[]) {
  if (e.error_type === 'WRONG_CITATION') continue; // the answer is right, only its evidence turn is not
  const conv = convs[Number(/^locomo_(\d+)_qa/.exec(e.question_id)?.[1])];
  conv?.qa.forEach((q, i) => { if (q.question.trim() === e.question.trim()) wrong.add(`${conv.sample_id}:${i}`); });
}

type Judged = { qid: string; category: string; correct: boolean };
const read = (path: string): Map<string, Judged> => new Map(readFileSync(path, 'utf8').split('\n').filter(Boolean)
  .map((l) => JSON.parse(l) as Judged).map((r) => [r.qid, r]));
const A = read(first);
const B = other ? read(other) : null;
const all = [...A.keys()].filter((q) => !B || B.has(q));
const kept = all.filter((q) => !wrong.has(q));
const pct = (rows: Map<string, Judged>, qs: string[]): string => (100 * qs.filter((q) => rows.get(q)!.correct).length / qs.length).toFixed(1);
console.log(`questions ${all.length}, golden answer wrong ${all.length - kept.length}, audited ${kept.length}`);
for (const cat of ['all', ...new Set(kept.map((q) => A.get(q)!.category))]) {
  const sub = kept.filter((q) => cat === 'all' || A.get(q)!.category === cat);
  let line = `${cat.padEnd(12)} n=${String(sub.length).padStart(4)}  audited ${pct(A, sub)}`;
  if (cat === 'all') line += `  (all questions ${pct(A, all)})`;
  if (B) {
    const t = signTest(sub.map((q) => A.get(q)!.correct), sub.map((q) => B.get(q)!.correct));
    line += `  vs ${pct(B, sub)}${cat === 'all' ? ` (all questions ${pct(B, all)})` : ''}  +${t.win}/-${t.loss} p=${t.p.toFixed(3)}`;
  }
  console.log(line);
}
