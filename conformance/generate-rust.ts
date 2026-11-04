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
