/**
 * The fused search's channels (qdrant-client.ts) against a stand-in for Qdrant's HTTP API: what each channel asks for,
 * and how their scores add up.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { QdrantClient } from './qdrant-client.js';

type Body = { using: string; filter?: { must?: unknown[]; must_not?: unknown[] } };
const CUE = { key: 'part', match: { value: 'cue' } };

/** Answers each grouped query with `groups(body)`, recording the bodies. */
function stub(groups: (b: Body) => Array<{ id: string; score: number }>): Body[] {
  const bodies: Body[] = [];
  vi.stubGlobal('fetch', async (_url: string, init: { body: string }) => {
    const b = JSON.parse(init.body) as Body;
    bodies.push(b);
    return new Response(JSON.stringify({ result: { groups: groups(b).map((g) => ({ id: g.id, hits: [{ id: `${g.id}-p`, score: g.score }] })) } }));
  });
  return bodies;
}

const client = () => new QdrantClient(() => ({ url: 'http://qdrant', apiKey: 'k' }));
const cueOnly = (b: Body) => JSON.stringify(b.filter?.must ?? []).includes('"cue"');

afterEach(() => vi.unstubAllGlobals());

describe('fused search', () => {
  it('searches cues apart from the rest of each item, as a quieter channel', async () => {
    const bodies = stub(() => []);
    await client().fusedItems({ dense: [[1, 0]], sparse: { indices: [1], values: [1] }, filter: { must: [{ key: 'kind', match: { value: 'fact' } }] } });
    expect(bodies.map((b) => b.using)).toEqual(['dense', 'dense', 'bm25']);
    // The main dense channel leaves cues out, the cue channel reads only them; both keep the caller's filter.
    expect(bodies[0]!.filter!.must_not).toContainEqual(CUE);
    expect(bodies[0]!.filter!.must).toEqual([{ key: 'kind', match: { value: 'fact' } }]);
    expect(bodies[1]!.filter!.must).toEqual([{ key: 'kind', match: { value: 'fact' } }, CUE]);
    expect(bodies[2]!.filter).toEqual({ must: [{ key: 'kind', match: { value: 'fact' } }] });
  });

  it('weighs a match by a cue at a tenth of a match by the item itself', async () => {
    // Item a is the best by its own text, item b only by a cue; each channel's best counts its full weight.
    stub((b) => (b.using !== 'dense' ? [] : cueOnly(b) ? [{ id: 'b', score: 0.9 }, { id: 'a', score: 0.1 }] : [{ id: 'a', score: 0.9 }, { id: 'b', score: 0.1 }]));
    const got = await client().fusedItems({ dense: [[1, 0]] });
    expect(got.map((g) => g.item)).toEqual(['a', 'b']);
    expect(got[0]!.score).toBeCloseTo(0.9);
    expect(got[1]!.score).toBeCloseTo(0.1);
  });
});
