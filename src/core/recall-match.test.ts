/**
 * The passage search shows as `match` is one found beyond the key's start: a short record's only chunk is its whole
 * key, the two points tie, and whichever Qdrant returns first must not repeat the key as a "match".
 */
import { describe, expect, it } from 'vitest';
import { pointId, type PointStruct, type QdrantClient } from './qdrant-client.js';
import { memorySearch, type RecallDeps } from './recall.js';
import type { MemoryItem, StmEntry } from './store.js';
import { writeLtm, writeStm } from './writer.js';

/** Qdrant whose best point for every item is its first chunk — the tie broken the way that showed the quirk. */
function chunkFirst() {
  const points = new Map<string, PointStruct>();
  const client = {
    async upsert(ps: PointStruct[]) { for (const p of ps) points.set(p.id, p); },
    async deleteItems(items: string[]) { for (const [id, p] of points) if (items.includes(String(p.payload.item))) points.delete(id); },
    async retrieve(ids: string[]) { return ids.flatMap((id) => (points.has(id) ? [{ id, payload: points.get(id)!.payload }] : [])); },
    async fusedItems() {
      const keys = [...points.values()].filter((p) => p.payload.part === 'key');
      return keys.map((p) => {
        const item = String(p.payload.item);
        return { item, score: 1, point: points.has(pointId(item, 1)) ? pointId(item, 1) : pointId(item, 0) };
      });
    },
  };
  return client as unknown as QdrantClient;
}

const embed = async (texts: string[]) => texts.map(() => new Float32Array([1, 0, 0]));

const record = (uid: string, request: string, reply: string): StmEntry => ({
  uid, fingerprint: uid, engine: 'email', session: uid, turn: 0, cwd: null, project: '-p', title: 'Mail', ts: Date.parse('2026-04-02T00:00:00Z'),
  synthetic: false, typed: 'human', request, reply, said: reply, steps: [], files: [], done: false,
});
const fact = (uid: string, summary: string, body: string): MemoryItem => ({
  uid, kind: 'fact', scope: 'project', project: '-p', engine: 'email', session: 's', cwd: null, where: 'w', ts: 1, firstTs: 1,
  title: summary, summary, body, files: [], appliesWhen: '', confidence: 'stated', importance: 3, status: 'active',
  provenance: [], seen: 1, fingerprint: uid,
});

describe('search match', () => {
  it('is never the key again: a short record whose only chunk is its whole key has no match', async () => {
    const d: RecallDeps = { client: chunkFirst(), embed };
    await writeStm(d, [record('mail:1', 'Router password', 'It is on the sticker under the router.')]);
    const [hit] = await memorySearch(d, { query: 'router password', limit: 5 });
    expect(hit?.id).toBe('stm:mail:1');
    expect(hit?.summary).toBe('Router password It is on the sticker under the router.');
    expect(hit).not.toHaveProperty('match');
  });

  it('is never the start of the key: a body the summary begins with has no match', async () => {
    const d: RecallDeps = { client: chunkFirst(), embed };
    await writeLtm(d, [fact('mem:rack', 'The backup server moved to rack seven.\nIt was on Monday.', 'The backup server moved to rack seven.')]);
    const [hit] = await memorySearch(d, { query: 'backup server', limit: 5 });
    expect(hit?.id).toBe('mem:rack');
    expect(hit).not.toHaveProperty('match');
  });

  it('is still a passage found beyond the key\'s start', async () => {
    const d: RecallDeps = { client: chunkFirst(), embed };
    await writeLtm(d, [fact('mem:rack', 'The backup server moved.', 'It now sits in rack seven, beside the switch.')]);
    const [hit] = await memorySearch(d, { query: 'rack seven', limit: 5 });
    expect(hit?.match).toBe('It now sits in rack seven, beside the switch.');
  });
});
