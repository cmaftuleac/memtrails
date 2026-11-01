/**
 * Consolidation's reading of a conversation (extract.ts) — with a stand-in for the fast model. The gate is the
 * rule in code: nothing inferred, contested, ambiguous or trivial is kept, and no secret survives.
 */
import { describe, expect, it } from 'vitest';
import type { LlmAnswer, LlmRequest } from '../llm.js';
import type { ReadableRequest } from './episodes.js';
import { cleanObservations, EXTRACT_MAX_TOKENS, extractWindow, gate, partsOf, WINDOW_CHARS, windowsOf } from './extract.js';
import { renderForExtraction } from './episodes.js';

const req = (uid: string, request: string, extra: Partial<ReadableRequest> = {}): ReadableRequest => ({
  uid, typed: 'human', request, reply: 'done', said: 'done', steps: [], files: [], ts: 1_700_000_000_000, ...extra,
});

const item = (over: Record<string, unknown> = {}) => ({
  kind: 'procedure', title: 'Deploy the API', summary: 'Push to main; Jenkins deploys.', body: 'git push origin main',
  applies_when: 'Deploying the API', scope: 'project', requests: [1], confidence: 'verified', importance: 4,
  contested: false, ambiguous: false, ...over,
});

describe('the gate', () => {
  const uidOf = (n: number) => (n === 1 ? 'conv:s:0' : undefined);

  it('keeps verified and stated knowledge that matters, with its provenance', () => {
    const { kept, dropped } = gate([item(), item({ confidence: 'stated', title: 'Prefers Lingui', kind: 'preference' })], uidOf);
    expect(dropped).toBe(0);
    expect(kept.map((k) => k.title)).toEqual(['Deploy the API', 'Prefers Lingui']);
    expect(kept[0]!.provenance).toEqual(['conv:s:0']);
  });

  it('drops what was only inferred, contested, ambiguous, trivial or malformed', () => {
    const { kept, dropped } = gate([
      item({ confidence: 'inferred' }), item({ contested: true }), item({ ambiguous: true }), item({ importance: 1 }),
      item({ kind: 'gossip' }), item({ summary: '' }),
    ], uidOf);
    expect(kept).toEqual([]);
    expect(dropped).toBe(6);
  });

  it('keeps every retrieval cue the model wrote: non-empty, scrubbed, no repeats', () => {
    const cues = ['', 'how do we deploy the api', 'how do we deploy the api', ...Array.from({ length: 10 }, (_, i) => `cum fac deploy ${i}`),
      'token sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD'];
    const { kept } = gate([item({ cues })], uidOf);
    expect(kept[0]!.cues).toHaveLength(12);
    expect(kept[0]!.cues[0]).toBe('how do we deploy the api');
    expect(new Set(kept[0]!.cues).size).toBe(12);
    expect(kept[0]!.cues.join(' ')).not.toContain('sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD');
    expect(gate([item({ cues: 'not a list' })], uidOf).kept[0]!.cues).toEqual([]);
  });

  it('keeps up to four cues on an observation, scrubbed, and none when the model wrote none', () => {
    const cues = ['What instruments does Priya play?', 'When did Priya start the cello?', 'key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD', 'a', 'b'];
    const [o] = cleanObservations([{ text: 'Priya started the cello in May.', requests: [1], cues }], uidOf);
    expect(o!.cues).toHaveLength(4);
    expect(o!.cues.slice(0, 2)).toEqual(cues.slice(0, 2));
    expect(o!.cues.join(' ')).not.toContain('sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD');
    expect(cleanObservations([{ text: 'Priya has a brother.', requests: [1] }], uidOf)[0]!.cues).toEqual([]);
  });

  it('keeps a "this works" kind only from a request seen working; other kinds whatever the outcome', () => {
    const outcome = (o: 'success' | 'partial' | 'fail' | 'uncertain') => () => o;
    const kinds = ['procedure', 'snippet', 'template', 'pitfall', 'decision', 'fact', 'preference'];
    for (const o of ['fail', 'uncertain'] as const) {
      const { kept } = gate(kinds.map((kind) => item({ kind, title: kind })), uidOf, outcome(o));
      expect(kept.map((k) => k.kind)).toEqual(['decision', 'fact', 'preference']);
    }
    for (const o of ['success', 'partial'] as const) expect(gate(kinds.map((kind) => item({ kind, title: kind })), uidOf, outcome(o)).kept).toHaveLength(7);
    // No outcome known (a saved note, an older answer): nothing dropped for it.
    expect(gate([item()], uidOf).kept).toHaveLength(1);
  });

  it('scrubs a secret that slipped into the text', () => {
    const { kept } = gate([item({ body: 'export OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD' })], uidOf);
    expect(kept[0]!.body).not.toContain('sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD');
  });
});

describe('windows', () => {
  it('cuts a conversation into windows of whole requests', () => {
    const eps = Array.from({ length: 5 }, (_, i) => req(`u${i}`, 'x'.repeat(2500)));
    const wins = windowsOf(eps, 6000);
    expect(wins.flat().map((e) => e.uid)).toEqual(eps.map((e) => e.uid));
    expect(wins.every((w) => w.length >= 1)).toBe(true);
    expect(wins.length).toBeGreaterThan(1);
  });
});

describe('extractWindow', () => {
  it('asks the fast model (purpose memory) once and maps its answer to the requests', async () => {
    const asks: LlmRequest[] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      asks.push(a);
      return { answer: {
        segment_summary: 'Set up the deploy.',
        episodes: [{ request: 1, keep: true, title: 'Deploy set up', summary: 'The API deploys through Jenkins on push.' }, { request: 2, keep: false, title: '', summary: '' }],
        items: [item(), item({ confidence: 'inferred' })],
      } };
    };
    const x = await extractWindow([req('conv:s:0', 'set up the deploy'), req('conv:s:1', 'yes')], { cwd: '/p', engine: 'conversation', title: 'Deploy' }, { ask });
    expect(asks).toHaveLength(1);
    expect(asks[0]!.purpose).toBe('memory');
    expect(asks[0]!.maxTokens).toBe(EXTRACT_MAX_TOKENS); // room for a long window's answer: a cut one is no answer
    expect(asks[0]!.prompt).toContain('## Request 1');
    expect('candidates' in x && x.candidates.map((c) => c.title)).toEqual(['Deploy the API']);
    expect('episodes' in x && x.episodes.map((e) => e.uid)).toEqual(['conv:s:0']);
    expect('dropped' in x && x.dropped).toBe(1);
  });

  it('passes on that the model is busy or unavailable, so the worker can pause', async () => {
    const busy = await extractWindow([req('u', 'x')], { cwd: null, engine: 'e', title: 't' }, { ask: async () => ({ busy: true }) });
    expect(busy).toEqual({ busy: true });
    const off = await extractWindow([req('u', 'x')], { cwd: null, engine: 'e', title: 't' }, { ask: async () => ({ unavailable: 'no route' }) });
    expect(off).toEqual({ unavailable: 'no route' });
  });
});

describe('a request too long for one window', () => {
  it('is split at line ends into parts that hold every character, each once', () => {
    const text = Array.from({ length: 3000 }, (_, i) => `line ${i} ${'x'.repeat(i % 97)}`).join('\n') + '\n' + 'y'.repeat(25_000);
    const parts = partsOf(text, 10_000);
    expect(parts.length).toBeGreaterThan(5);
    expect(parts.every((p) => p.length <= 10_000)).toBe(true);
    expect(parts.join('\n').replace(/\n/g, '')).toBe(text.replace(/\n/g, ''));
  });

  it('is read part by part, in order — every character once — each part folded into the running episode', async () => {
    const steps = Array.from({ length: 900 }, (_, i) => `Bash: step ${i} ran the migration batch ${i}`);
    const said = Array.from({ length: 9000 }, (_, i) => `table${i} moved`).join(' '); // one 120k-character line
    const long = req('conv:s:0', 'migrate every table of the warehouse', { steps, said });
    const rendered = renderForExtraction(long, 1);
    expect(rendered.length).toBeGreaterThan(WINDOW_CHARS);
    const prompts: string[] = [];
    const ask = async (a: LlmRequest): Promise<LlmAnswer> => {
      prompts.push(a.prompt);
      const n = prompts.length;
      return { answer: {
        segment_summary: `part ${n}`,
        episodes: [{ request: 1, keep: true, title: `Warehouse migration, parts 1-${n}`, summary: `what parts 1 to ${n} show`, cues: [`cue ${n}`], concepts: [`c${n}`] }],
        items: [item({ title: `Item from part ${n}`, summary: `knowledge ${n}` })],
      } };
    };
    const x = await extractWindow([long], { cwd: '/p', engine: 'conversation', title: 'Migration' }, { ask });
    // Exactly the request's parts were read, each whole and in order: every character of the request, once.
    const parts = partsOf(rendered, WINDOW_CHARS - 2000);
    expect(parts.length).toBeGreaterThan(1);
    expect(prompts).toHaveLength(parts.length);
    prompts.forEach((prompt, k) => expect(prompt.endsWith(parts[k]!)).toBe(true));
    // Each later part is handed what the parts before it showed, to fold into.
    expect(prompts[0]).not.toContain('What the earlier parts showed');
    prompts.slice(1).forEach((prompt, k) => expect(prompt).toContain(`Summary: what parts 1 to ${k + 1} show`));
    if (!('episodes' in x)) throw new Error('no extraction');
    expect(x.candidates.map((c) => c.title)).toEqual(prompts.map((_, k) => `Item from part ${k + 1}`));
    expect(x.episodes).toEqual([expect.objectContaining({ uid: 'conv:s:0', summary: `what parts 1 to ${parts.length} show` })]);
    expect(x.episodes[0]!.cues).toEqual(prompts.map((_, k) => `cue ${k + 1}`));
  });

  it('waits whole for the next run when a part is not answered', async () => {
    const long = req('conv:s:0', 'big', { said: Array.from({ length: 12_000 }, (_, i) => `row${i} copied`).join(' ') });
    let n = 0;
    const ask = async (): Promise<LlmAnswer> => (++n === 2 ? { unavailable: 'no route' } : { answer: { segment_summary: '', episodes: [], items: [] } });
    expect(await extractWindow([long], { cwd: null, engine: 'e', title: 't' }, { ask })).toEqual({ unavailable: 'no route' });
  });
});

