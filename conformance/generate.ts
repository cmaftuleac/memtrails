/**
 * The answer key every port of memtrails is held to: what the TypeScript library gives for fixed inputs, written to
 * conformance/fixtures/*.json. A port passes when it gives the same for the same inputs (vectors to 1e-6).
 *
 *   npx tsx conformance/generate.ts        # writes the fixtures; CI fails when they differ from what is committed
 *
 * Every input here is invented. No benchmark data (LoCoMo, LongMemEval) is used.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { calendarOf, episodeBody, renderForExtraction } from '../src/core/episodes.js';
import { pointId } from '../src/core/qdrant-client.js';
import { redactString } from '../src/core/redact.js';
import { sentenceSpans } from '../src/core/room.js';
import { fingerprint } from '../src/core/store.js';
import { children, conceptKey, cut, docSparse, firstWords, fold, querySparse, termId, tokens } from '../src/core/text.js';
import { maxSim, packWords } from '../src/core/words.js';
import { dateSpan, rawKey, stmTitle } from '../src/core/writer.js';
import { collectionKey, conversationEntries, entryOf, invalidRecord, recordUid, timeOf, type MemoryRecord } from '../src/records.js';
import { lastedText, readableDate, searchText } from '../src/tools/agent-tools.js';
import type { RecalledItem } from '../src/core/recall.js';

const OUT = join(import.meta.dirname, 'fixtures');
mkdirSync(OUT, { recursive: true });
const write = (name: string, data: unknown): void => writeFileSync(join(OUT, `${name}.json`), `${JSON.stringify(data, null, 1)}\n`);

/** Texts in many scripts and shapes: identifiers, paths, accents, emoji, scripts without spaces. */
const TEXTS = [
  '',
  'The proxy certificate expired on the 1st; renewal stopped because the cron job was removed.',
  'Run deploy.sh on billing_api; see apps/web/main.ts and services/edge/letsencrypt_renewal_hook.sh',
  'Baza de date a căzut azi-noapte — am repornit serviciul și am verificat jurnalele.',
  'Бэкап базы данных не прошёл: диск заполнен на 98%.',
  'Ελέγξαμε τον διακομιστή· η μνήμη είναι εντάξει.',
  '数据库备份失败了，因为磁盘已满。我们明天再试。',
  'データベースのバックアップは失敗しました。ディスクがいっぱいです。',
  'การสำรองข้อมูลล้มเหลวเพราะดิสก์เต็ม',
  'Café, naïve, résumé, Ångström, ﬁle (ligature), ＦＵＬＬＷＩＤＴＨ, Ⅻ',
  'Launch 🚀 went fine 👍🏽 — the 👨‍👩‍👧 photo is in /tmp/x.png',
  'a_b.c-d/e --flag=value ../../etc/passwd __init__.py v1.2.3 x',
  'Mixed: deploy-ul pe staging a mers, dar prod încă are vechiul .env',
  'A '.repeat(150) + 'very long text that keeps going so that it needs more than one child window to hold every word.',
  'hash 5f3c1a9e8b7d6c5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4',
  'Line one\nLine two\r\n\tTabbed line   with   spaces\n\n',
  'Ünïcödé ÄÖÜ ß ẞ İstanbul ı',
  'नमस्ते, सर्वर फिर से चालू है।',
  'مرحبا، الخادم يعمل الآن.',
  'שלום, השרת פועל.',
];

/** Strings for the secret filter: what must go, and what must stay. */
const SECRETS = [
  'my key is sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD ok',
  'GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123',
  'AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
  '{"api_key": "abc123def456", "password":"hunter2"}',
  'postgres://admin:p@ss/w0rd@db.internal:5432/app',
  'https://user:pa/ss@example.com/path?token=xyz&ok=1',
  'Authorization: Bearer abcdefghijklmnopqrstuvwxyz.0123',
  'Write to jane.doe@example.com about the invoice.',
  'see /Users/jane/projects/app and /home/bob/.ssh and C:\\Users\\Ana\\Desktop',
  '-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----',
  'the password field is required; token-based auth; max_tokens: 1200',
  'commit 3b18e512dba79e4c8300dd08aeb37f8e728b8dad fixed it',
  'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U',
  'AKIAIOSFODNN7EXAMPLE and ASIAIOSFODNN7EXAMPLE',
  'x'.repeat(60) + ' high entropy run',
  '',
];

const DATES = ['2024', '2024-02', '2024-02-29', '2023-02-29', '2024-02-30', '2024-13', '2024-00', '2024-12-31', 'April 2026', '', '2024-1-5'];
const TIMES: Array<string | number | undefined> = [
  '2026-04-02', '2026-04-02T10:30:00Z', '2026-04-02T10:30:00+03:00', '2026-04-02T10:30:00.123Z', 1712050200000,
  'Tue, 02 Apr 2024 10:30:00 +0000', 'Tue, 2 Apr 2024 10:30:00 GMT', 'not a date', '', undefined,
];

const NOW = Date.UTC(2026, 3, 2, 12, 0, 0);
const RECORDS: MemoryRecord[] = [
  { id: 'mail-1', kind: 'email', author: 'Ana', title: 'Proxy certificate expired', at: '2026-04-02',
    text: 'It expired on the 1st. Renewal stopped because the cron job was removed in March.', collection: 'Ops team', thread: 'cert' },
  { id: 'note-7', text: 'First line is the title\nand the rest is the body,\nover three lines.' },
  { id: 'n/ü:8', source: 'slack', url: 'https://example.com/m/8', text: '  spaced out  ', metadata: { channel: 'ops', pinned: true, score: 3 } },
  { id: '', text: 'no id' },
  { id: 'x', text: '', title: '' },
  { id: 'y', text: 'bad time', at: 'yesterday-ish' },
];

const CONVERSATION = {
  id: 'conv-1', source: 'chat', collection: 'personal', at: '2026-03-01T09:00:00Z',
  turns: [
    { role: 'assistant' as const, text: 'Hi! How can I help?' },
    { role: 'user' as const, text: 'I started the cello in May.', author: 'Priya' },
    { role: 'assistant' as const, text: 'Lovely. Lessons?' },
    { role: 'user' as const, text: 'Yes, every Tuesday.', at: '2026-03-01T09:05:00Z' },
    { role: 'user' as const, text: '   ' },
    { role: 'assistant' as const, text: 'Noted.' },
  ],
};

const HITS: RecalledItem[] = [
  { id: 'obs:1', kind: 'observation', title: 'cello', summary: 'Priya started the cello.', ts: Date.UTC(2026, 2, 1), eventAt: '2026-05', score: 0.9 } as unknown as RecalledItem,
  { id: 'mem:2', kind: 'fact', title: 'Lessons', summary: 'Lessons are on Tuesdays at the music school.', ts: Date.UTC(2026, 2, 2), score: 0.8 } as unknown as RecalledItem,
  { id: 'stm:3', kind: 'episode', title: 'Proxy', summary: 'The proxy certificate expired.', match: 'renewal   stopped', ts: Date.UTC(2026, 3, 2), score: 0.7 } as unknown as RecalledItem,
  { id: 'obs:4', kind: 'observation', title: 'job', summary: 'Rosa worked at the bakery.', ts: Date.UTC(2026, 1, 3), startAt: '2025-11', endAt: '2026-01', score: 0.6 } as unknown as RecalledItem,
];

/** Word vectors with a fixed pseudo-random fill, so ports need no random generator of their own. */
function vectors(seed: number, words: number, dims: number): { words: number; dims: number; data: number[] } {
  const data: number[] = [];
  let x = seed;
  for (let i = 0; i < words * dims; i++) { x = (x * 1103515245 + 12345) % 2147483648; data.push((x / 2147483648) * 2 - 1); }
  return { words, dims, data };
}

write('text', {
  note: 'String lengths and positions are in UTF-16 code units, as JavaScript counts them; termId hashes UTF-16 code units.',
  cases: TEXTS.map((t) => ({
    text: t, fold: fold(t), tokens: tokens(t), firstWords5: firstWords(t, 5), children: children(t, 12, 9),
    docSparse: docSparse(t), cut20: cut(t, 20), cut1: cut(t, 1), conceptKey: conceptKey(t),
  })),
  termIds: ['', 'a', 'proxy', 'bază', '数据库', '🚀', 'deploy.sh'].map((t) => ({ term: t, id: termId(t) })),
  querySparse: { texts: TEXTS.slice(1, 4), vector: querySparse(TEXTS.slice(1, 4)) },
  conceptKeys: [' Proxy.conf ', 'proxy.conf', 'acme-proxy:dev', '  Harbour   rowing club ', '"quoted"', 'Ångström!'].map((c) => ({ name: c, key: conceptKey(c) })),
  sentenceSpans: TEXTS.map((t) => ({ text: t, spans: sentenceSpans(t) })),
});

write('redact', SECRETS.map((s) => ({ input: s, output: redactString(s) })));

write('ids', {
  fingerprint: [[], ['a'], ['a', 'b'], ['a', ['b', 'c']], ['a', null, undefined, ''], ['ünï', '数据']].map((p) => ({ parts: p, fp: fingerprint(...(p as string[])) })),
  pointId: [['stm:abc', 0], ['stm:abc', 3], ['mem:xyz', 0], ['ep:ü数', 12]].map(([item, n]) => ({ item, n, id: pointId(item as string, n as number) })),
  recordUid: [{ id: 'mail-1' }, { id: 'mail-1', source: 'imap' }, { id: 'ü/数:1', source: '' }].map((r) => ({ record: r, uid: recordUid(r) })),
  collectionKey: [undefined, '', '  ', 'Ops team', 'a/b\\c:d', 'Équipe  Ops'].map((c) => ({ collection: c ?? null, key: collectionKey(c) })),
});

write('dates', {
  note: 'timeOf follows JavaScript Date.parse for ISO 8601 and RFC 2822 dates; anything else is undefined (null here).',
  dateSpan: DATES.map((d) => ({ date: d, span: dateSpan(d) })),
  readableDate: DATES.map((d) => ({ date: d, text: readableDate(d) })),
  lastedText: [{ startAt: '2022-02' }, { endAt: '2022-05-03' }, { startAt: '2021-11', endAt: '2022-05' }, {}].map((h) => ({ hit: h, text: lastedText(h) })),
  timeOf: TIMES.map((t) => ({ at: t ?? null, ms: timeOf(t) ?? null })),
  calendarOf: { times: [Date.UTC(2023, 6, 14, 10), Date.UTC(2023, 6, 16, 23, 59), Date.UTC(2024, 0, 1)], text: calendarOf([Date.UTC(2023, 6, 14, 10), Date.UTC(2023, 6, 16, 23, 59), Date.UTC(2024, 0, 1)]) },
});

const entries = RECORDS.filter((r) => !invalidRecord(r)).map((r, n) => entryOf(r, NOW, n));
const conv = conversationEntries(CONVERSATION, NOW);
write('records', {
  now: NOW,
  invalid: RECORDS.map((r) => ({ record: r, error: invalidRecord(r) })),
  entries: RECORDS.filter((r) => !invalidRecord(r)).map((r, n) => ({ record: r, turn: n, entry: entries[n] })),
  conversation: { input: CONVERSATION, entries: conv },
  rendered: [...entries, ...conv].map((e, n) => ({ uid: e.uid, rawKey: rawKey(e), stmTitle: stmTitle(e), episodeBody: episodeBody(e), extraction: renderForExtraction(e, n + 1) })),
});

const q = vectors(7, 4, 16);
const docs = [vectors(11, 9, 16), vectors(13, 200, 16), vectors(17, 1, 16)];
write('words', {
  note: 'packWords keeps the first `limit` words, one bit per number (sign); maxSim scores a query against packed words.',
  query: q,
  docs: docs.map((d) => {
    const packed = packWords({ ...d, data: Float32Array.from(d.data) });
    return { vectors: d, packed: { words: packed.words, dims: packed.dims, bits: Array.from(packed.bits) },
      maxSim: maxSim({ ...q, data: Float32Array.from(q.data) }, packed) };
  }),
});

write('search-text', {
  relevance: searchText(HITS, 'relevance'),
  time: searchText(HITS, 'time'),
  compact: searchText(Array.from({ length: 22 }, (_, i) => ({ ...HITS[i % HITS.length]!, id: `x:${i}` })), 'relevance'),
  hits: HITS,
});

console.log(`fixtures written to ${OUT}`);
