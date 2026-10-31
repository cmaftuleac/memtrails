/**
 * Do the gold answer's words reach the reader? No model: for each question, whether every word of the gold answer
 * (months shortened, days without a leading zero, a few filler words left out) is in the context, by category, and
 * per question against another run (a two-sided sign test). Mirrors docs/BENCHMARKS.md "the answer's words shown".
 *
 *   npx tsx bench/score.ts runs/memtrails/locomo.jsonl [runs/other/locomo.jsonl]
 */
import { readFileSync } from 'node:fs';

const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
const FILLER = new Set(['the', 'and', 'her', 'his', 'she', 'they', 'with', 'for', 'was', 'that', 'before', 'after', 'week', 'weekend',
  'first', 'last', 'around', 'about', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']);
const WORD = /[\p{L}\p{N}_]+/gu;

function dates(text: string): string {
  let t = text.toLowerCase();
  for (const m of MONTHS) t = t.replace(new RegExp(`\\b${m}\\b`, 'g'), m.slice(0, 3));
  return t.replace(/\b0(\d)\b/g, '$1');
}

/** The share of the answer's words found in the context. */
export function cover(context: string, answer: string): number {
  const words = (dates(answer).match(WORD) ?? []).filter((w) => (w.length > 2 || /^\d+$/.test(w)) && !FILLER.has(w));
  if (!words.length) return 1;
  const c = dates(context);
  const has = (w: string): boolean => new RegExp(`(?<![\\p{L}\\p{N}_])${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\p{L}\\p{N}_])`, 'u').test(c);
  return words.filter(has).length / words.length;
}

export function signTest(a: boolean[], b: boolean[]): { win: number; loss: number; p: number } {
  let win = 0;
  let loss = 0;
  a.forEach((x, i) => { if (x && !b[i]) win++; if (!x && b[i]) loss++; });
  const n = win + loss;
  if (!n) return { win, loss, p: 1 };
  let tail = 0;
  let choose = 1;
  for (let i = 0; i <= Math.min(win, loss); i++) { tail += choose; choose = (choose * (n - i)) / (i + 1); }
  return { win, loss, p: Math.min(1, (2 * tail) / 2 ** n) };
}

type Row = { qid: string; answer: string; category: string; context: string };
const read = (path: string): Map<string, Row> => new Map(readFileSync(path, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as Row).map((r) => [r.qid, r]));

if (process.argv[1]?.endsWith('score.ts')) {
  const [first, other] = process.argv.slice(2);
  if (!first) throw new Error('usage: score.ts <run.jsonl> [<other run.jsonl>]');
  const A = read(first);
  const B = other ? read(other) : null;
  const qs = [...A.keys()].filter((q) => !B || B.has(q));
  const full = (rows: Map<string, Row>, q: string): boolean => cover(rows.get(q)!.context, rows.get(q)!.answer) >= 0.99;
  for (const cat of ['all', ...new Set(qs.map((q) => A.get(q)!.category))]) {
    const sub = qs.filter((q) => cat === 'all' || A.get(q)!.category === cat);
    const fa = sub.map((q) => full(A, q));
    let line = `${cat.padEnd(12)} n=${String(sub.length).padStart(4)}  answer's words all shown ${(100 * fa.filter(Boolean).length / sub.length).toFixed(1)}`;
    if (B) {
      const fb = sub.map((q) => full(B, q));
      const t = signTest(fa, fb);
      line += `  vs ${(100 * fb.filter(Boolean).length / sub.length).toFixed(1)}  +${t.win}/-${t.loss} p=${t.p.toFixed(3)}`;
    }
    console.log(line);
  }
}
