/**
 * The framework retrievers' answer key (any port's adapters must give the same): what each found item becomes as a
 * document — its id, the text a model reads, its flat metadata.
 *
 *   npx tsx conformance/generate-retriever.ts      # retriever.json
 *
 * Every input here is invented.
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { RecalledItem } from '../src/core/recall.js';
import { retrievedDocument } from '../src/integrations/document.js';

const OUT = join(import.meta.dirname, 'fixtures');
const base = { where: '', project: 'default', engine: 'conversation' };

const ITEMS: RecalledItem[] = [
  { ...base, id: 'stm:1', title: 'Proxy certificate expired', summary: 'Proxy certificate expired Ana: Renewal stopped.', engine: 'email',
    ts: Date.UTC(2026, 3, 2, 9, 14), kind: 'email', score: 0.91,
    record: { id: '<CAF8w@mail.example>', source: 'work-mailbox', author: 'Ana Popescu', url: 'https://mail.example/1', metadata: { to: 'ops@example.com' } } },
  { ...base, id: 'k:2', title: 'Cron job removed', summary: 'The renewal cron job was removed.', ts: Date.UTC(2026, 3, 5), kind: 'observation',
    eventAt: '2026-03', score: 0.5, match: 'removed in March during the  server cleanup' },
  { ...base, id: 'k:3', title: 'Deploy freeze', summary: 'No deploys from the 20th to the 2nd.', ts: Date.UTC(2025, 11, 1), kind: 'observation',
    startAt: '2025-12-20', endAt: '2026-01-02', score: 0.25 },
  { ...base, id: 'k:4', title: 'Ana', summary: 'Ana Popescu runs the proxy and the certificates; prefers Tuesday deploys.\nOn call in April.',
    ts: Date.UTC(2026, 3, 6), kind: 'profile', score: 0.125 },
  { ...base, id: 'stm:5', title: 'Note', summary: 'Parola routerului este în caietul albastru.', ts: 0, kind: 'note', score: 0,
    record: { id: 'note-7', source: '', author: '' } },
  { ...base, id: 'stm:6', title: 'Lunch 🍝', summary: 'At the small Italian place 🍝, at one.', ts: Date.UTC(1999, 11, 31, 23, 59, 59, 999), kind: 'message',
    score: 1e-9, record: { id: 'chat-1#2' } },
];

writeFileSync(join(OUT, 'retriever.json'), `${JSON.stringify(ITEMS.map((item) => ({ item, document: retrievedDocument(item) })), null, 1)}\n`);
console.log(`retriever.json: ${ITEMS.length} items`);
