/**
 * More of the answer key, for the Python port (any port may read it): what the TypeScript library gives where JavaScript
 * and Python differ underneath — a lone surrogate (half of a UTF-16 pair) in what is hashed, which Node writes as U+FFFD,
 * a JSON Lines id that is a number (String(1.0) is "1"), and a record whose metadata is an empty object.
 *
 *   npx tsx conformance/generate-python-2.ts      # python-2.json
 *
 * Every input here is invented.
 */
process.env.TZ = 'UTC';

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { profileSubjects } from '../src/core/profiles.js';
import { pointId } from '../src/core/qdrant-client.js';
import { fingerprint, type MemoryItem } from '../src/core/store.js';
import { fromJsonl } from '../src/importers/jsonl.js';
import { entryOf, recordUid } from '../src/records.js';

const OUT = join(import.meta.dirname, 'fixtures');
const NOW = Date.UTC(2026, 3, 2, 12, 0, 0);

/** Lone surrogates, a high and a low one, and a whole pair beside them (which stays a character). */
const LONE = 'note \uD83D end';
const LOW = '\uDE00 first';
const PAIR = 'ok 😀';

const item = (uid: string, concepts: string[]): MemoryItem => ({
  uid, kind: 'fact', scope: 'project', project: '-p', engine: 'conversation', session: 's', cwd: '/p', where: 'conv',
  ts: NOW - Number(uid.slice(-1)) * 1000, firstTs: NOW, title: `title ${uid} ${LONE}`, summary: `summary ${LOW}`, body: '',
  files: [], appliesWhen: '', confidence: 'verified', importance: 3, status: 'active', provenance: [], seen: 1, fingerprint: uid,
  concepts,
});

const dir = mkdtempSync(join(tmpdir(), 'memtrails-python-2-'));
try {
  const file = join(dir, 'ids.jsonl');
  const ids = ['1.0', '100000.0', '1e21', '1.5e-7', '-0.0', '12345678901234567890', '0.1', '1e300'];
  writeFileSync(file, `${ids.map((id) => `{"id":${id},"text":"a record"}`).join('\n')}\n`);
  const jsonlIds: string[] = [];
  for await (const r of fromJsonl(file)) jsonlIds.push(r.id);

  const subjects = profileSubjects(
    [item('mem:1', [`gate${LONE}`]), item('mem:2', [`gate${LONE}`]), item('mem:3', [`gate${LONE}`])],
    new Map([['-p', 15]]),
  );

  const data = {
    recordUid: [
      { source: LONE, id: LOW, uid: recordUid({ source: LONE, id: LOW }) },
      { source: null, id: PAIR, uid: recordUid({ id: PAIR }) },
    ],
    fingerprint: [
      { parts: [LONE, null, [LOW, PAIR]], fingerprint: fingerprint(LONE, null, [LOW, PAIR]) },
    ],
    pointId: [{ item: `stm:${LONE}`, n: 2, id: pointId(`stm:${LONE}`, 2) }],
    profileSubjects: subjects.map((s) => ({ uid: s.uid, concept: s.concept, fingerprint: s.fingerprint })),
    jsonlIds: { written: ids, read: jsonlIds },
    entries: [
      { record: { id: 'meta-empty', text: `A record ${LONE}`, metadata: {} } },
      { record: { id: LOW, source: LONE, text: `Another ${PAIR}`, at: '2026-04-01' } },
    ].map(({ record }) => ({ record, entry: entryOf(record, NOW) })),
  };
  writeFileSync(join(OUT, 'python-2.json'), `${JSON.stringify(data, null, 1)}\n`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
