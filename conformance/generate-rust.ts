/**
 * More of the answer key, for the Rust port: what the TypeScript importers give for fixed inputs. The inputs are
 * written here, into conformance/inputs/importers/ (committed, so a port's tests read the very same bytes), and what
 * the importers make of them into conformance/fixtures/rust-importers.json.
 *
 *   npx tsx conformance/generate-rust.ts
 *
 * Every input here is invented. No benchmark data (LoCoMo, LongMemEval) is used. What depends on the machine — the
 * folder the inputs are in, a file's modification time — is written as a placeholder: `file://<root>/…`, `<mtime>`.
 */
import { mkdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fromEml, fromFiles, fromJsonl, fromMbox, fromSlackExport } from '../src/importers/index.js';
import { decodeEncodedWords } from '../src/importers/mbox.js';
import { slackText } from '../src/importers/slack.js';
import { frontMatter } from '../src/importers/files.js';
import { decodeBytes, decodeEntities, htmlTitle, htmlToText, splitText } from '../src/importers/text.js';
import type { MemoryRecord } from '../src/records.js';
import { createHash } from 'node:crypto';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import type { LlmAnswer, LlmRequest } from '../src/llm.js';
import { Consolidator } from '../src/core/consolidate.js';
import { extractWindow, observeWindow, windowsOf, partsOf } from '../src/core/extract.js';
import { memoryGather } from '../src/core/gather.js';
import { profileSubjects, writeProfiles } from '../src/core/profiles.js';
import { pointId } from '../src/core/qdrant-client.js';
import type { RecalledItem } from '../src/core/recall.js';
import { ltmLog, stmLog, type MemoryItem, type StmEntry } from '../src/core/store.js';
import { conversationEntries, entryOf } from '../src/records.js';
import { buildAgentTools } from '../src/tools/agent-tools.js';

const OUT = join(import.meta.dirname, 'fixtures');
const IN = join(import.meta.dirname, 'inputs', 'importers');
rmSync(IN, { recursive: true, force: true });
const put = (path: string, data: string | Uint8Array): void => {
  mkdirSync(join(IN, path, '..'), { recursive: true });
  writeFileSync(join(IN, path), data);
};
const json = (v: unknown): string => JSON.stringify(v, null, 2);
const b64 = (bytes: Uint8Array | string): string => Buffer.from(bytes).toString('base64');

// Kept byte for byte on every checkout: no line-ending conversion.
put('.gitattributes', '* -text\n');

// ── A folder of notes ───────────────────────────────────────────────────────────────────────────────────────────
put('files/plan.md', ['---', 'title: "Planul pe trimestrul trei: priorități"', 'date: 2024-07-01', 'author: Ioana Mărgărit',
  'tags: [buget, echipă, 3, true]', 'reviewers:', '  - Ana', '  - "Mihai"', 'summary: |', '  Prima linie', '  a doua', 'folded: >',
  '  unu', '  doi', 'draft: true', 'version: 3', 'ratio: 0.25', 'empty: ""', 'none: ~', 'note: value # a comment', 'nested:',
  '  inner: kept out', '---', '# Not the title', '', 'Bugetul pentru țară crește cu 5%.'].join('\n'));
put('files/toml.md', '+++\ntitle = "Din TOML"\ndate = 2023\nn = 2 # c\nlist = [a, "b", 7]\n+++\nCorp\r\nal doilea rând\r\n');
put('files/notes.org', '#+TITLE:  Notițe de birou\n* Head\nText.\n');
put('files/heading.org', 'text\n** Capitol 2\nmai mult\n');
put('files/title.rst', '=====\n Titlu \n=====\n\nCorpul documentului.\n');
put('files/doc.adoc', '== Sub\n= Documentul principal  \nText.\n');
put('files/setext.md', 'Planul   \n=====\nbody\n');
put('files/atx.md', 'intro\n# Plan   ##  \nbody\n');
put('files/page.html', `<!doctype html><html><head><title>Caf&eacute; &amp; Cr&egrave;me</title>
  <style>body { color: red }</style><script>var x = "<p>not text</p>";</script></head>
  <body><h1>Menu</h1><p>Prix&nbsp;: 3&nbsp;&euro; &mdash; <b>très</b>
  bon</p><ul><li>Thé</li><li>&#x1F375; matcha &#8211; 抹茶 &#150; &Alpha;&omega;&sigmaf; &scaron;&Uuml;&unknown;</li></ul>
  <!-- hidden --><table><tr><td>a</td><th>b</th></tr></table><pre>  x\r\n   y</pre><blockquote>quoted</blockquote>
  <p>Ünïcödé &lt;ok&gt;</p></body></html>`);
put('files/cp1252.txt', Uint8Array.from([0x43, 0x61, 0x66, 0xe9, 0x20, 0x96, 0x20, 0x80, 0x31, 0x30, 0x0a]));
put('files/utf16.txt', Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from('Text în UTF-16 🎉\n', 'utf16le')]));
put('files/binary.md', Uint8Array.from([0x23, 0x20, 0x41, 0, 0x42]));
put('files/big.md', `# Big\n\n${Array.from({ length: 12 }, (_, i) => `Paragraph ${i}: ${'lorem ipsum dolor sit amet '.repeat(12)}— ação, 漢字.`).join('\n\n')}`);
put('files/cjk.txt', '今日は晴れです。'.repeat(60) + '😀'.repeat(150));
put('files/sub dir/a #1%.md', '# Nume ciudat\n\ntext\n');
put('files/.hidden/x.md', '# hidden');
put('files/node_modules/pkg/README.md', '# dependency');
put('files/image.png', Uint8Array.from([0x89, 0x50, 0x4e, 0x47]));

// ── JSON Lines ──────────────────────────────────────────────────────────────────────────────────────────────────
put('records.jsonl', '﻿' + [
  JSON.stringify({ id: 'ja', text: '会議は火曜日に延期されました 🎉', title: '会議', at: '2024-05-01T10:00:00Z', metadata: { room: 'B', empty: '', n: 2, ok: null } }),
  '{ not json',
  JSON.stringify({ text: 'no id here' }),
  '',
  JSON.stringify({ id: 7, text: 'Întâlnirea de mâine e la București.', author: '', kind: 'note' }),
  JSON.stringify({ id: 'ar', text: 'تم تأجيل الاجتماع', at: 1714557600000, collection: 'echipa' }),
  JSON.stringify({ id: 'bad-time', text: 'x', at: 'whenever' }),
  JSON.stringify({ id: 'empty-time', text: 'x', at: '' }),
  JSON.stringify({ id: 'null-time', text: 'x', at: null }),
  JSON.stringify({ id: 'no-text', text: '', title: 'only a title' }),
  JSON.stringify({ id: 'blank', text: '   ' }),
  '[1, 2]',
  '"a string"',
].join('\r\n') + '\rlast\n');

// ── A Slack export ──────────────────────────────────────────────────────────────────────────────────────────────
put('slack/users.json', json([
  { id: 'U1', name: 'ana', real_name: 'Ana Popescu', profile: { real_name: 'Ana Popescu', display_name: 'ana' } },
  { id: 'U2', name: 'tanaka', profile: { real_name: '', display_name: '田中' } },
  { id: 'U3', name: 'omar' },
]));
put('slack/channels.json', json([{ id: 'C1', name: 'general' }, { id: 'C2', name: 'proiecte' }, { id: 'C3', name: '../outside' }]));
put('slack/dms.json', json([{ id: 'D1', members: ['U1', 'U9'] }]));
put('slack/general/2024-03-01.json', json([
  { type: 'message', subtype: 'channel_join', user: 'U3', text: '<@U3> has joined the channel', ts: '1709290000.000100' },
  { type: 'message', user: 'U1', text: 'Planul: <@U2> preia <#C2|proiecte>, vezi <https://example.org/doc|documentul> și <https://example.org/raw>. 1 &lt; 2 &amp;&amp; ok',
    ts: '1709290800.000200', thread_ts: '1709290800.000200', reply_count: 1 },
  { type: 'message', user: 'U2', text: '了解しました <@U1|ana> 👍 <@U7|ghost> <!subteam^S1|@ops> <!subteam^S2> <!channel> <mailto:a@example.org|Ana> <mailto:b@example.org> <#C9>',
    ts: '1709291000.000300', thread_ts: '1709290800.000200' },
  { type: 'message', user: 'U3', text: '', ts: '1709291100.000400', files: [{ name: 'تقرير.pdf', title: 'التقرير الشهري' }, { name: 'raw.txt' }] },
  { type: 'message', user: 'U3', text: '', ts: '1709291150.000450' },
  { type: 'message', subtype: 'bot_message', bot_id: 'B1', username: 'deploy-bot', text: 'Deployed <!here>',
    attachments: [{ title: 'Build 42', title_link: 'https://example.org/b/42' }, { fallback: 'see https://example.org/x', from_url: 'https://example.org/x' }],
    ts: '1709291200.000500' },
  { type: 'message', bot_id: 'B2', bot_profile: { name: 'Helper' }, text: 'beep', ts: '1709291300.000600' },
  { type: 'message', user: 'U5', user_profile: { real_name: 'Guest Person' }, text: 'hello', ts: '1709291400.000700' },
  { type: 'message', hidden: true, user: 'U1', text: 'hidden', ts: '1709291500.000800' },
  { type: 'event', user: 'U1', text: 'not a message', ts: '1709291600.000900' },
]));
put('slack/general/2024-03-02.json', json([
  { type: 'message', user: 'U1', text: 'Mulțumesc! <#C1>', ts: '1709377200.000100' },
  { type: 'message', subtype: 'channel_topic', user: 'U1', text: 'set the topic', ts: '1709377300.000200' },
]));
put('slack/general/notes.json', json([{ type: 'message', user: 'U1', text: 'not a day file', ts: '1' }]));
put('slack/proiecte/2024-03-01.json', json([{ type: 'message', user: 'U3', text: 'مرحبا بالجميع', ts: '1709290900.000100' }]));
put('slack/D1/2024-03-03.json', json([{ type: 'message', user: 'U9', text: 'direct', ts: '1709463600.000100' }]));
put('outside/2024-03-01.json', json([{ type: 'message', user: 'U1', text: 'outside', ts: '1709290000.000100' }]));

// ── Mail ────────────────────────────────────────────────────────────────────────────────────────────────────────
function qp(text: string): string {
  return text.split('\n').map((line) => {
    let out = '';
    let width = 0;
    for (const byte of Buffer.from(line, 'utf8')) {
      const piece = byte > 126 || byte === 61 ? `=${byte.toString(16).toUpperCase().padStart(2, '0')}` : String.fromCharCode(byte);
      if (width + piece.length > 72) { out += '=\n'; width = 0; }
      out += piece;
      width += piece.length;
    }
    return out.replace(/[ \t]$/, (c) => `=${c.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0')}`);
  }).join('\n');
}
const subjectBytes = Buffer.from('会議の議事録 — 2024年', 'utf8');
const plain = 'Bună ziua,\n\nAtașez procesul-verbal al ședinței — vă rog să-l citiți până mâine. 🙏\n\n> Ce s-a hotărât?\n\n-- \nAndré, București';
const first = [
  'From andre@example.org Mon Jan  1 10:00:00 2024',
  'From: =?iso-8859-1?Q?Andr=E9_Lef=E8vre?= <andre@example.org>',
  'To: "Ioana Mărgărit" <ioana@example.org>,', '\tbob@example.org',
  'Cc: =?utf-8?q?=C5=81ukasz?= <lukasz@example.org>',
  `Subject: =?UTF-8?B?${b64(subjectBytes.subarray(0, 4))}?=\n =?UTF-8?B?${b64(subjectBytes.subarray(4))}?=`,
  'Date: Mon, 1 Jan 2024 10:00:00 +0100 (CET)', 'Message-ID: <m1@example.org>',
  'Content-Type: multipart/mixed; boundary="outer"', '', 'preamble', '--outer',
  'Content-Type: multipart/alternative;', ' boundary=inner', '', '--inner',
  'Content-Type: text/plain; charset=utf-8', 'Content-Transfer-Encoding: quoted-printable', '', qp(plain), '--inner',
  'Content-Type: text/html; charset=utf-8', 'Content-Transfer-Encoding: base64', '',
  b64('<p>Bună ziua,</p><p>Atașez.</p>'), '--inner--', '--outer',
  'Content-Type: application/pdf', "Content-Disposition: attachment; filename*=UTF-8''%E8%AD%B0%E4%BA%8B%E9%8C%B2.pdf",
  'Content-Transfer-Encoding: base64', '', b64('%PDF-1.4 not really'), '--outer',
  'Content-Type: application/octet-stream; name="=?utf-8?B?' + b64('fișier.bin') + '?="', '', 'xx', '--outer',
  'Content-Type: text/plain; name*0*=utf-8\'\'p%C4%83rti; name*1=".txt"', 'Content-Disposition: attachment', '', 'data', '--outer--', '',
].join('\n');
const second = [
  'From ioana@example.org Mon Jan  1 12:00:00 2024', 'From: Ioana Mărgărit <ioana@example.org>', 'To: andre@example.org',
  'Subject: Re: =?UTF-8?B?5Lya6K2w44Gu6K2w5LqL6Yyy?=', 'Date: Mon, 1 Jan 2024 12:00:00 +0100', 'Message-ID: <m2@example.org>',
  'In-Reply-To: <m1@example.org>', 'References: <m0@example.org> <m1@example.org>',
  'Content-Type: text/plain; format=flowed; delsp=yes', 'Content-Transfer-Encoding: 8bit', '',
  'Mulțumesc, am citit tot ', 'documentul.', '>From the start it was clear.', '>>From a quoted line, kept escaped once.', '',
  'André Lefèvre wrote:', '> Bună ziua,', '',
].join('\n');
const cyrillic = Buffer.from([0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2, 0x2c, 0x20, 0xec, 0xe8, 0xf0]);
const third = Buffer.concat([Buffer.from([
  'From sergey@example.org Tue Jan  2 09:00:00 2024', 'From: sergey@example.org (Сергей)',
  'Subject: =?gb2312?B?' + b64(new Uint8Array([0xd6, 0xd0, 0xce, 0xc4])) + '?= =?koi8-r?Q?=F0=D2=C9=D7=C5=D4?=',
  'Date: Tue, 02 Jan 2024 09:00:00 GMT', 'Content-Type: text/html; charset=windows-1251', 'Content-Transfer-Encoding: base64', '',
  b64(Buffer.concat([Buffer.from('<p>'), cyrillic, Buffer.from('</p><blockquote>old text</blockquote>')])), '',
].join('\n')), Buffer.from('\n')]);
const fourth = [
  'From - Wed Jan  3 08:00:00 2024', 'From: "Dana" <dana@example.org>', 'Subject: Forwarded', 'Date: Wed, 3 Jan 2024 08:00:00 -0500 (EST)',
  'Content-Type: message/rfc822', '', 'From: Inner <inner@example.org>', 'Date: Tue, 2 Jan 2024 07:00:00 +0000',
  'Subject: The inner one', 'Content-Type: text/plain; charset=iso-8859-2', 'Content-Transfer-Encoding: quoted-printable', '',
  'Dzi=EAkuj=EA =\nbardzo', '',
].join('\n');
const fifth = ['From x@example.org Thu Jan  4 08:00:00 2024', 'From: x@example.org', '', '', ''].join('\n');
put('mail/Inbox.mbox', Buffer.concat([Buffer.from(`${first}\n${second}\n`), third, Buffer.from(`${fourth}\n${fifth}`)]));
put('mail/note.eml', first.split('\n').slice(1).join('\r\n'));
put('mail/bare.eml', 'Subject: =?utf-8?Q?Doar_un_subiect?=\r\nX-Note: plain\r\n\r\n');

// ── What the importers give ─────────────────────────────────────────────────────────────────────────────────────
async function all(it: AsyncIterable<MemoryRecord>): Promise<MemoryRecord[]> {
  const out: MemoryRecord[] = [];
  for await (const r of it) out.push(r);
  return out;
}

const root = resolve(IN);
const rootUrl = pathToFileURL(root).href;
/** The machine's parts as placeholders: the inputs' folder in a URL, a modification time. */
function stable(records: MemoryRecord[], dir?: string): unknown[] {
  return records.map((r) => {
    const out: Record<string, unknown> = { ...r };
    if (typeof r.url === 'string' && r.url.startsWith(rootUrl)) out.url = `file://<root>${r.url.slice(rootUrl.length)}`;
    if (dir && typeof r.at === 'number') {
      const path = join(dir, r.thread ?? r.id);
      try { if (Math.round(statSync(path).mtimeMs) === r.at) out.at = '<mtime>'; } catch { /* not a file */ }
    }
    return out;
  });
}

const skips: Array<{ line: number; reason: string }> = [];
const jsonl = await all(fromJsonl(join(IN, 'records.jsonl'), {
  source: 'notes', kind: 'record',
  onSkip: (_line, reason, n) => skips.push({ line: n, reason: reason.startsWith('not JSON') ? 'not JSON' : reason }),
}));
const fileSkips: Array<{ path: string; reason: string }> = [];
const files = await all(fromFiles(join(IN, 'files'), { collection: 'docs', maxChars: 1500, onSkip: (path, reason) => fileSkips.push({ path, reason }) }));
const mailSkips: number[] = [];

const result = {
  note: 'Inputs in conformance/inputs/importers. `file://<root>` is that folder as a file URL; `<mtime>` a file\'s modification time.',
  jsonl: { records: jsonl, skipped: skips },
  files: { options: { collection: 'docs', maxChars: 1500 }, records: stable(files, join(IN, 'files')), skipped: fileSkips },
  slack: {
    records: await all(fromSlackExport(join(IN, 'slack'), { workspaceUrl: 'https://example.slack.com/' })),
    filtered: await all(fromSlackExport(join(IN, 'slack'), { channels: ['#general', 'D1'], collection: 'team', skipBots: true, source: 'ws' })),
  },
  mbox: {
    records: await all(fromMbox(join(IN, 'mail', 'Inbox.mbox'), { onSkip: (_r, i) => mailSkips.push(i) })),
    kept: await all(fromMbox(join(IN, 'mail', 'Inbox.mbox'), { keepQuoted: true, keepSignature: true, collection: 'work', source: 'box' })),
    skipped: mailSkips,
  },
  eml: [...await all(fromEml(join(IN, 'mail', 'note.eml'))), ...await all(fromEml(join(IN, 'mail', 'bare.eml')))],
  html: [
    '<p>a</p><SCRIPT type="x">b</script ><p>c</p>', '<p>a</p><iframe src=x>keep<p>b</p>', 'a <script>var x = 1; <p>never</p>',
    '<pre>  open   <p>x</p>', '<![CDATA[x < y]]> z', 'a <![CDATA[ unclosed', '<ul><li>one<li>two</ul>x<br>y<hr>z',
    '<p>  lead  \t space </p>\n\n\n\n<p>end</p>', '<a href="x">link</a> <!x> <?php ?> < 3', '<p>a</p><!-- open comment',
  ].map((html) => ({ html, text: htmlToText(html), dropQuotes: htmlToText(html, { dropQuotes: true }), title: htmlTitle(html) ?? null })),
  titles: ['<title>T &amp; U</title>', '<h1>Only <i>h1</i></h1>', '<TITLE>Upper</TITLE >', '<title>  </title>'].map((html) => ({ html, title: htmlTitle(html) ?? null })),
  entities: ['&amp;&lt;&gt;&quot;&apos;', '&#233;&#xe9;&#X1F600;&#150;&#0;&#xD800;&#1114112;', '&eacute;&Ccedil;&zcaron;&qacute;&Theta;&thetasym;', '&nope; & ;&#;'].map((s) => ({ s, text: decodeEntities(s) })),
  bytes: [
    [[0x43, 0x61, 0x66, 0xc3, 0xa9], undefined], [[0x43, 0x61, 0x66, 0xe9], undefined], [[0xef, 0xbb, 0xbf, 0x41], 'utf-8'],
    [[0xd6, 0xd0, 0xce, 0xc4], 'gb2312'], [[0xf0, 0xd2], '"koi8-r"'], [[0xe9], 'iso-8859-1*fr'], [[0x41, 0xe9], 'x-nonsense'],
    [[0x41, 0x80], 'iso-2022-kr'], [[0xff, 0xfe, 0x41, 0x00], 'utf-16'], [[0x41], 'US-ASCII'],
  ].map(([bytes, charset]) => ({ bytes, charset: charset ?? null, text: decodeBytes(Uint8Array.from(bytes as number[]), charset as string | undefined) })),
  split: [
    ['Para one. '.repeat(30) + '\n\n' + 'Para two. '.repeat(30), 200], ['line '.repeat(80), 150], ['x'.repeat(250), 100],
    ['今日は晴れです。'.repeat(40) + '😀'.repeat(200), 150], ['short', 10],
  ].map(([text, max]) => ({ text, max, parts: splitText(text as string, max as number) })),
  encodedWords: ['=?utf-8?Q?Gr=C3=BC=C3=9Fe_aus?= =?utf-8?Q?_K=C3=B6ln?= und mehr', '=?unknown-x?Q?abc?=', 'a =?UTF-8?B?w6k=?= b =?iso-8859-1?q?=E9?=',
    '=?utf-8*ro?B?ZMOp?=', 'plain'].map((s) => ({ s, text: decodeEncodedWords(s) })),
  slackText: ['<@U1> <@U2|x> <#C1> <#C2|named> <!here> <!subteam^S1> <https://a.org|https://a.org> <mailto:x@y.org> <> &lt;&amp;'].map((s) => ({
    s, text: slackText(s, new Map([['U1', 'Ana']]), new Map([['C1', 'general']])),
  })),
  frontMatter: ['---\ntitle: T # a comment\nurl: http://x.org/a:b\nk:v\ncount: 3\n---\nbody', '+++\ntitle = "TOML"\nn = 2 # c\n+++\nbody',
    '---\nlist:\n  - a\n  - 1\nblock: >-\n  one\n\n  two\n---\n', 'no front matter', '---\nunclosed: yes\n'].map((s) => ({ s, ...frontMatter(s) })),
};

writeFileSync(join(OUT, 'rust-importers.json'), `${JSON.stringify(result, null, 1)}\n`);
console.log(`inputs written to ${IN}, fixtures to ${OUT}/rust-importers.json`);

// ── The core: every prompt built from fixed inputs, what the gate keeps of fixed answers, the agent tools ─────────
// Each language-model ask is recorded as it was asked; the answers are the ones written here (in the fixture too, so a
// port feeds its own code the same). Secrets in the answers are made of pieces, never a real key's shape.
const core = await (async () => {
  const NOW = Date.UTC(2026, 3, 2, 12);
  const key = ['sk-', 'abcdefghijklmnopqrstuvwx'].join('');
  const asked = (r: LlmRequest) => ({ purpose: r.purpose, system: r.system, prompt: r.prompt, schema: r.schema, name: r.name ?? null,
    maxTokens: r.maxTokens ?? null, timeoutMs: r.timeoutMs ?? null, consequence: r.consequence ?? null });
  const short = (r: LlmRequest) => ({ ...asked(r), prompt: { sha256: createHash('sha256').update(r.prompt).digest('hex'), length: r.prompt.length, start: r.prompt.slice(0, 400) } });
  const asker = (answers: unknown[]) => {
    const seen: LlmRequest[] = [];
    let n = 0;
    const ask = async (r: LlmRequest): Promise<LlmAnswer> => {
      seen.push(r);
      const a = answers[Math.min(n++, answers.length - 1)];
      return a === 'unavailable' ? { unavailable: 'the model is away' } : { answer: a as Record<string, unknown> };
    };
    return { ask, seen };
  };
  const records: MemoryRecord[] = [
    { id: 'm-1', kind: 'email', author: 'Ana', title: 'Backup server moved', at: '2026-03-30T09:00:00Z', thread: 't1', collection: 'Ops',
      text: `The backup server moved to rack seven on Monday. Key for the old one was ${key}; it is retired now.` },
    { id: 'm-2', kind: 'email', author: 'Ion', at: '2026-03-31T10:30:00Z', thread: 't1', collection: 'Ops',
      text: 'Restore test failed\nThe nightly restore test failed: the mount point /srv/backup was missing.' },
    { id: 'm-3', kind: 'note', at: '2026-04-01T08:15:00Z', thread: 't1', collection: 'Ops', url: 'https://notes.example/3',
      text: 'Checklist for the move: label the disks, update the inventory, tell the night shift.' },
  ];
  const win: StmEntry[] = records.map((r, n) => entryOf(r, NOW, n));
  const conv = conversationEntries({ id: 'c-7', collection: 'home', at: '2026-03-28T18:00:00Z', turns: [
    { role: 'user', text: 'We adopted a cat last Saturday, her name is Miso.' }, { role: 'assistant', text: 'Congratulations! How is she settling in?' },
    { role: 'user', text: 'Fine. Vet visit is booked for next Friday.' }, { role: 'assistant', text: 'Good idea.' },
  ] }, NOW);
  const meta = { cwd: null, engine: 'email', title: 'Backup server moved' };
  const extraction = {
    segment_summary: `Moved the backup server; old key ${key}.`,
    episodes: [
      { request: 1, outcome: 'success', keep: true, title: ' Backup server moved to rack 7 ', summary: 'The backup server moved to rack seven; the old key was retired.',
        cues: ['where is the backup server', 'where is the backup server', '', 'unde e serverul de backup'], concepts: ['Rack 7', 'rack 7 ', 'backup server', ''] },
      { request: 2, outcome: 'fail', keep: true, title: 'Restore failed', summary: '  ', cues: [], concepts: [] },
      { request: 3, outcome: 'partial', keep: false, title: '', summary: 'Checklist.', cues: [], concepts: [] },
      { request: 9, outcome: 'success', keep: true, title: 'Nowhere', summary: 'No such request.', cues: [], concepts: [] },
    ],
    items: [
      { kind: 'procedure', title: 'Moving a backup server', summary: 'Label the disks, update the inventory, tell the night shift.', body: 'Steps as in the checklist.',
        applies_when: 'When a server moves', scope: 'project', requests: [1, 3], confidence: 'verified', importance: 4, contested: false, ambiguous: false,
        cues: ['how do we move a server'], concepts: ['inventory', 'night shift'] },
      { kind: 'procedure', title: 'Restore test', summary: 'Run the restore test.', body: '', applies_when: '', scope: 'project', requests: [2],
        confidence: 'verified', importance: 4, contested: false, ambiguous: false, cues: [], concepts: [] },
      { kind: 'fact', title: 'Backup mount point', summary: `The backup mount point is /srv/backup (key ${key}).`, body: 'Mounted by fstab.', applies_when: 'Restores',
        scope: 'weird', requests: [2], confidence: 'stated', importance: 2.6, contested: false, ambiguous: false, cues: 'not a list', concepts: ['/srv/backup'] },
      { kind: 'fact', title: 'Guess', summary: 'Probably fine.', body: '', applies_when: '', scope: 'global', requests: [1], confidence: 'inferred', importance: 5,
        contested: false, ambiguous: false, cues: [], concepts: [] },
      { kind: 'gossip', title: 'x', summary: 'y', body: '', applies_when: '', scope: 'person', requests: [1], confidence: 'stated', importance: 3,
        contested: false, ambiguous: false, cues: [], concepts: [] },
    ],
    observations: [
      { text: 'Ana moved the backup server to rack seven.', about: 'Ana', event_date: ' 2026-03-30 ', start_date: 'soon', end_date: '2026-05', phase: 'nope',
        requests: [1, 9], concepts: ['Ana', 'rack seven'], cues: ['Where is the backup server?', 'Who moved it?', 'When?', 'Why?', 'Which rack?'] },
      { text: '', about: 'x', event_date: '', start_date: '', end_date: '', phase: 'done', requests: [1], concepts: [], cues: [] },
      { text: 'Orphan', about: '', event_date: '2026', start_date: '', end_date: '', phase: 'plan', requests: [7], concepts: [], cues: [] },
    ],
  };
  const a1 = asker([extraction]);
  const x1 = await extractWindow(win, meta, { ask: a1.ask, known: ['(2026-03-01) The old rack was rack three.'] });
  const a2 = asker([{ segment_summary: '', episodes: [], items: [] }]);
  await extractWindow([...win, ...conv], { cwd: '/srv', engine: 'email', title: 'Mixed' }, { ask: a2.ask, observations: false });
  const a3 = asker(['unavailable']);
  const x3 = await extractWindow(win, meta, { ask: a3.ask });
  // One request too long for a window: read in parts, each folded into the summary of the parts before it.
  const long = entryOf({ id: 'long-1', kind: 'document', title: 'Runbook', at: '2026-03-20T00:00:00Z',
    text: Array.from({ length: 1400 }, (_, i) => `Step ${i}: check the rack ${i % 9}, the cable ${i % 5} and the label on disk ${i % 13} before moving on.`).join('\n') }, NOW);
  const longAnswers = [0, 1, 2].map((k) => ({ segment_summary: `part ${k}`, episodes: [{ request: 1, outcome: k === 1 ? 'nope' : 'partial', keep: k === 0,
    title: k === 2 ? '' : `Runbook ${k}`, summary: `Steps so far ${k}.`, cues: [`cue ${k}`], concepts: [`Rack ${k}`] }], items: [], observations: [] }));
  const a4 = asker(longAnswers);
  const x4 = await extractWindow([long], meta, { ask: a4.ask });
  const observeAnswer = { observations: [{ text: 'Miso is the cat they adopted.', about: 'the person', event_date: '2026-03-21', start_date: '', end_date: '', phase: 'done',
    requests: [1], concepts: ['Miso'], cues: ['What is the cat called?'] }] };
  const a5 = asker([observeAnswer]);
  const x5 = await observeWindow(conv, win.slice(0, 2), { cwd: null, engine: 'conversation', title: 'Cat' }, { ask: a5.ask, chars: 300 });
  const a6 = asker([observeAnswer]);
  const x6 = await observeWindow(conv, win, { cwd: null, engine: 'conversation', title: 'Cat' }, { ask: a6.ask, chars: 7 });

  // Reconciling and cues: the asks a worker makes, over items made here.
  const mk = (uid: string, over: Partial<MemoryItem> = {}): MemoryItem => ({ uid, kind: 'fact', scope: 'project', project: 'Ops', engine: 'email', session: 't1',
    cwd: null, where: 'Backup server moved', ts: NOW, firstTs: NOW, title: `Title of ${uid}`, summary: `Summary of ${uid}: the backup server is in rack seven.`,
    body: `Body of ${uid}.`, files: [], appliesWhen: '', confidence: 'stated', importance: 3, status: 'active', provenance: [], seen: 1, fingerprint: uid, ...over });
  const pairs = [{ it: mk('mem:new'), known: mk('mem:old', { kind: 'decision', body: 'ü'.repeat(500) }) }, { it: mk('mem:a'), known: mk('mem:b') }];
  const relateAnswer = { pairs: [{ id: '[p1] NEW (fact)', relation: 'updates', title: ' Merged ', summary: ' Both. ', body: '' },
    { id: 'p2', relation: 'nonsense', title: '', summary: '', body: '' }, { id: 'x', relation: 'same', title: '', summary: '', body: '' }] };
  const a7 = asker([relateAnswer]);
  const worker = new Consolidator({ client: {} as never, embed: async () => [], stm: stmLog(join(tmpdir(), 'none.jsonl')), ltm: ltmLog(join(tmpdir(), 'none2.jsonl')),
    log: () => {}, aborted: () => false, ask: a7.ask });
  const decided = await worker.relate(pairs);
  const cueItems = [mk('mem:c1', { appliesWhen: 'When restoring' }), mk('mem:c2', { cues: ['kept cue'] }), mk('mem:c3', { kind: 'pitfall' })];
  const cueAnswer = { items: [{ id: 'i1', cues: ['  where is it  ', `key ${key}`], concepts: ['Rack 7', 'rack 7'] }, { id: '[i2] (fact)', cues: ['new'], concepts: ['x'] },
    { id: 'i3', cues: [], concepts: [] }, { id: 'i9', cues: ['lost'], concepts: [] }] };
  const a8 = asker([cueAnswer]);
  const w8 = new Consolidator({ client: {} as never, embed: async () => [], stm: stmLog(join(tmpdir(), 'none.jsonl')), ltm: ltmLog(join(tmpdir(), 'none2.jsonl')),
    log: () => {}, aborted: () => false, ask: a8.ask });
  const cued = await (w8 as unknown as { cueBatch(items: MemoryItem[]): Promise<{ items: MemoryItem[] } | { paused: string }> }).cueBatch(cueItems);

  // Profiles: the subjects of a set of items, and the ask for each, over a log of them.
  const profiled: MemoryItem[] = [
    ...Array.from({ length: 5 }, (_, i) => mk(`obs:m${i}`, { kind: 'observation', scope: 'person', project: 'home', ts: NOW - i * 86_400_000,
      eventAt: i === 2 ? '2026-03' : undefined, title: `Miso note ${i}`, summary: `Miso   did thing ${i}.\nSecond line.`, concepts: [i === 1 ? 'miso' : 'Miso', 'vet'] })),
    ...Array.from({ length: 30 }, (_, i) => mk(`mem:f${i}`, { project: 'home', concepts: [`topic ${i}`] })),
  ];
  const sizes = new Map([['home', 35]]);
  const subjects = profileSubjects(profiled, sizes);
  const dir = mkdtempSync(join(tmpdir(), 'mt-gen-'));
  const ltm = ltmLog(join(dir, 'knowledge.jsonl'));
  ltm.put(profiled);
  const a9 = asker([{ title: '  Miso the cat  ', facts: ['  2026-03: adopted ', '', 7, 'vet booked'] }]);
  const fakeClient = { deleteItems: async () => {}, upsert: async () => {}, deleteParts: async () => {}, setPayload: async () => {} };
  const p9 = await writeProfiles({ client: fakeClient as never, embed: async (t: string[]) => t.map(() => new Float32Array(768)), stm: stmLog(join(dir, 'r.jsonl')), ltm,
    log: () => {}, aborted: () => false, ask: a9.ask, now: () => NOW });
  const written = [...ltm.all()].filter((it) => it.kind === 'profile').map((it) => ({ ...it, learnedAt: null }));

  // Gathering over a stand-in index: the first search's observations and other memories, then what its own searches find.
  const hit = (uid: string, over: Record<string, unknown> = {}) => ({ item: uid, tier: 'ltm', status: 'active', kind: 'fact', project: 'home', ts: NOW - 86_400_000,
    title: `About ${uid}`, summary: `What ${uid} says about the plumber.`, where: 'w', engine: 'conversation', ...over });
  const payloads = new Map<string, Record<string, unknown>>();
  const obsHits = ['obs:g1', 'obs:g2'].map((u, i) => hit(u, { kind: 'observation', eventAt: i ? '2026-02' : '2026-01-05', startAt: i ? '2025-12' : undefined, summary: `Observation ${u}.` }));
  const restHits = [hit('mem:g3'), hit('pf:g4', { kind: 'profile', summary: 'P'.repeat(1700) }), hit('stm:g5', { tier: 'stm', engine: 'email', title: 'Plumber invoice', summary: 'Invoice 42 from the plumber.' })];
  const extraHits = [hit('mem:g6', { summary: 'Another plumber was called in March.' })];
  for (const h of [...obsHits, ...restHits, ...extraHits]) payloads.set(pointId(String(h.item), 0), h);
  let calls = 0;
  const gatherClient = {
    async fusedItems(q: { filter?: { must?: Array<{ key?: string }>; must_not?: Array<{ key?: string }> } }) {
      const n = calls++;
      const obs = (q.filter?.must ?? []).some((c) => c.key === 'kind');
      const list = obs ? obsHits : n >= 2 ? [...restHits, ...extraHits] : restHits;
      return list.map((h, i) => ({ item: String(h.item), score: 1 - i / 10 }));
    },
    async retrieve(ids: string[]) { return ids.flatMap((id) => (payloads.has(id) ? [{ id, payload: payloads.get(id)! }] : [])); },
  };
  const gatherDeps = { client: gatherClient as never, embed: async (t: string[]) => t.map(() => new Float32Array(768)) };
  const gatherAnswers = [{ keep: ['m1', '[m3] (2026-01-05)', 'm99', 'x'], searches: [' plumber invoice ', '', 'plumber invoice', 'boiler'], notes: [' Plumbers: one in January. ', ''] },
    { keep: ['m1'], searches: [], notes: [`key ${key}`] }];
  const a10 = asker(gatherAnswers);
  const g10 = await memoryGather(gatherDeps as never, { query: 'how many plumbers did we call?' }, { ask: a10.ask, notes: true });
  calls = 0;
  const a11 = asker(gatherAnswers);
  const g11 = await memoryGather(gatherDeps as never, { query: 'how many plumbers did we call?', limit: 40 }, { ask: a11.ask });
  calls = 0;
  const a12 = asker(['unavailable']);
  const g12 = await memoryGather(gatherDeps as never, { query: 'plumbers' }, { ask: a12.ask });

  // The agent tools over a stand-in memory.
  const hits: RecalledItem[] = [
    { id: 'obs:1', kind: 'observation', title: 'cello', summary: 'Priya started the cello.', ts: Date.UTC(2026, 2, 1), eventAt: '2026-05', score: 0.9 } as never,
    { id: 'stm:2', kind: 'email', title: 'Lessons', summary: 'Lessons on Tuesdays.', ts: Date.UTC(2026, 2, 2), record: { id: 'r2', author: 'Priya' }, score: 0.8 } as never,
  ];
  const opened = { title: 'Lessons', kind: 'fact', ts: Date.UTC(2026, 2, 2), eventAt: '2026-03-03', startAt: '2026-01', project: 'home', summary: 'Lessons on Tuesdays.',
    body: 'At the music school.', files: ['a.md', 'b.md'], sources: [{ id: 'stm:9', when: Date.UTC(2026, 2, 1), title: 'Chat', body: 'Asked: lessons?' }] };
  const explored = { found: hits, related: [{ ...hits[1]!, id: 'mem:7', summary: 'S'.repeat(400), via: 'shares "cello"' }], concepts: ['cello', 'music school'] };
  const searched: unknown[] = [];
  const tools = buildAgentTools({
    search: async (q) => { searched.push(q); if (q.query === 'fail') throw new Error('index down'); return q.query === 'none' ? [] : hits; },
    open: async (id) => { if (id === 'fail') throw new Error('index down'); return id === 'mem:1' ? opened : null; },
    explore: async (q) => (q.topic === 'nothing' ? { found: [], related: [], concepts: [] } : explored as never),
  });
  const calls2 = [
    ['memory_search', { query: 'cello' }], ['memory_search', { query: 'cello', order: 'time', limit: 5, collections: ['home'], from: '2026', to: '2026-12-31' }],
    ['memory_search', { query: 'none' }], ['memory_search', { query: 'fail' }], ['memory_search', {}], ['memory_search', null],
    ['memory_search', { query: 5, limit: 0, order: 'x', from: 'April', collections: ['a', 7] }], ['memory_search', { query: 'q', limit: 2.5, extra: true }],
    ['memory_search', { query: 'x'.repeat(2001) }], ['memory_search', 'text'], ['memory_search', { query: 'q', limit: 201, to: null }],
    ['memory_open', { id: 'mem:1' }], ['memory_open', { id: 'mem:2' }], ['memory_open', { id: 'fail' }], ['memory_open', {}],
    ['memory_explore', { topic: 'cello' }], ['memory_explore', {}], ['memory_explore', { topic: '  ', from: [] }], ['memory_explore', { topic: 'nothing' }],
    ['memory_explore', { from: ['a', 'b', 'c', 'd', 'e', 'f'], limit: 21 }],
  ] as const;
  const results = [];
  for (const [name, args] of calls2) results.push({ name, args, result: await tools.find((t) => t.name === name)!.handler(args) });
  return {
    note: 'Every ask recorded as asked (a long prompt by its SHA-256, length and start), with the answers given. `now` for the records.',
    now: NOW, records, conversation: conv.map((e) => e.uid), window: win, conversationEntries: conv, meta,
    extract: { answer: extraction, asked: a1.seen.map(asked), result: x1, withoutObservations: a2.seen.map(asked), unavailable: { asked: a3.seen.length, result: x3 } },
    long: { uid: long.uid, lines: 1400, answers: longAnswers, asked: a4.seen.map(short), result: x4 },
    windows: { of: windowsOf([...win, ...conv], 300).map((w) => w.map((e) => e.uid)), parts: partsOf('a\n' + 'b'.repeat(25) + '\nccc\n\ndd', 10) },
    observe: { answer: observeAnswer, asked: a5.seen.map(asked), result: x5, one: { asked: a6.seen.map(asked), result: x6 } },
    relate: { pairs, answer: relateAnswer, asked: a7.seen.map(asked), decided: Object.fromEntries(decided instanceof Map ? decided : []) },
    cues: { items: cueItems, answer: cueAnswer, asked: a8.seen.map(asked), result: 'items' in cued ? cued.items : cued },
    profiles: { items: profiled, subjects: subjects.map((s) => ({ ...s, read: s.read.map((it) => it.uid) })), asked: a9.seen.map(asked), report: p9, written },
    gather: { obsHits, restHits, extraHits, answers: gatherAnswers,
      notes: { asked: a10.seen.map(asked), result: g10 }, plain: { asked: a11.seen.map(asked), result: g11 }, unavailable: { asked: a12.seen.length, result: g12 } },
    tools: { schemas: tools.map((t) => ({ name: t.name, description: t.description, jsonSchema: t.jsonSchema })), hits, opened, explored, calls: results, searched },
  };
})();
writeFileSync(join(OUT, 'rust-core.json'), `${JSON.stringify(core, null, 1)}\n`);
console.log(`fixtures written to ${OUT}/rust-core.json`);
