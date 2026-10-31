import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { decodeEncodedWords, fromEml, fromMbox } from './mbox.js';

async function all<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

const b64 = (bytes: Uint8Array | string): string => Buffer.from(bytes).toString('base64');
const wrap = (s: string, n = 76): string => s.match(new RegExp(`.{1,${n}}`, 'g'))!.join('\n');

/** Quoted-printable, as a mail program writes it: bytes past ASCII, `=` and a line's last space as =XX, soft breaks under 76. */
function qp(text: string): string {
  return text.split('\n').map((line) => {
    let out = '';
    let width = 0;
    for (const byte of Buffer.from(line, 'utf8')) {
      const piece = byte > 126 || byte === 61 ? `=${byte.toString(16).toUpperCase().padStart(2, '0')}` : String.fromCharCode(byte);
      if (width + piece.length > 72) {
        out += '=\n';
        width = 0;
      }
      out += piece;
      width += piece.length;
    }
    return out.replace(/[ \t]$/, (c) => `=${c.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0')}`);
  }).join('\n');
}

// A Japanese subject cut in the middle of a character across two encoded-words on a folded line.
const subjectBytes = Buffer.from('会議の議事録 — 2024年', 'utf8');
const subject = `=?UTF-8?B?${b64(subjectBytes.subarray(0, 4))}?=\n =?UTF-8?B?${b64(subjectBytes.subarray(4))}?=`;
const plain = 'Bună ziua,\n\nAtașez procesul-verbal al ședinței — vă rog să-l citiți până mâine. Mulțumesc frumos pentru tot ajutorul! 🙏\n\n> Ce s-a hotărât?\n> Nimic încă.\n\n-- \nAndré, București';
const html = '<html><body><p>Bună ziua,</p><p>Atașez procesul-verbal.</p></body></html>';

const first = [
  'From andre@example.org Mon Jan  1 10:00:00 2024',
  'Return-Path: <andre@example.org>',
  'From: =?iso-8859-1?Q?Andr=E9_Lef=E8vre?= <andre@example.org>',
  'To: "Ioana Mărgărit" <ioana@example.org>,',
  '\tbob@example.org',
  'Cc: =?utf-8?q?=C5=81ukasz?= <lukasz@example.org>',
  `Subject: ${subject}`,
  'Date: Mon, 1 Jan 2024 10:00:00 +0100 (CET)',
  'Message-ID: <m1@example.org>',
  'MIME-Version: 1.0',
  'Content-Type: multipart/mixed; boundary="outer"',
  '',
  'This is a multi-part message in MIME format.',
  '--outer',
  'Content-Type: multipart/alternative;',
  ' boundary=inner',
  '',
  '--inner',
  'Content-Type: text/plain; charset=utf-8',
  'Content-Transfer-Encoding: quoted-printable',
  '',
  qp(plain),
  '--inner',
  'Content-Type: text/html; charset=utf-8',
  'Content-Transfer-Encoding: base64',
  '',
  wrap(b64(html)),
  '--inner--',
  '--outer',
  'Content-Type: application/pdf',
  "Content-Disposition: attachment; filename*=UTF-8''%E8%AD%B0%E4%BA%8B%E9%8C%B2.pdf",
  'Content-Transfer-Encoding: base64',
  '',
  b64('%PDF-1.4 not really'),
  '--outer--',
  '',
].join('\n');

const second = [
  'From ioana@example.org Mon Jan  1 12:00:00 2024',
  'From: Ioana Mărgărit <ioana@example.org>',
  'To: andre@example.org',
  'Subject: Re: =?UTF-8?B?5Lya6K2w44Gu6K2w5LqL6Yyy?=',
  'Date: Mon, 1 Jan 2024 12:00:00 +0100',
  'Message-ID: <m2@example.org>',
  'In-Reply-To: <m1@example.org>',
  'References: <m1@example.org>',
  'Content-Type: text/plain',
  'Content-Transfer-Encoding: 8bit',
  '',
  'Mulțumesc, am citit.',
  '>From the start it was clear.',
  '>>From a quoted line, kept escaped once.',
  '',
  'André Lefèvre wrote:',
  '> Bună ziua,',
  '> Atașez procesul-verbal.',
  '',
].join('\n');

// Windows-1251 Cyrillic in base64 HTML, no Message-ID, a quoted reply in a blockquote.
const cyrillic = Buffer.from([0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2, 0x2c, 0x20, 0xec, 0xe8, 0xf0]); // "Привет, мир"
const third = [
  'From sergey@example.org Tue Jan  2 09:00:00 2024',
  'From: sergey@example.org (Сергей)',
  'Subject: =?gb2312?B?' + b64(new Uint8Array([0xd6, 0xd0, 0xce, 0xc4])) + '?= =?koi8-r?Q?=F0=D2=C9=D7=C5=D4?=',
  'Date: Tue, 02 Jan 2024 09:00:00 GMT',
  'Content-Type: text/html; charset=windows-1251',
  'Content-Transfer-Encoding: base64',
  '',
  b64(Buffer.concat([Buffer.from('<p>'), cyrillic, Buffer.from('</p><blockquote>old text</blockquote>')])),
  '',
].join('\n');

describe('fromMbox', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memtrails-mbox-'));
  const path = join(dir, 'Inbox.mbox');
  writeFileSync(path, Buffer.from(first + '\n' + second + '\n' + third, 'utf8'));

  it('reads a thread of multipart, quoted-printable, base64 and encoded-word mail', async () => {
    const records = await all(fromMbox(path));
    expect(records).toHaveLength(3);
    const [a, b, c] = records;
    expect(a).toMatchObject({ id: 'm1@example.org', title: '会議の議事録 — 2024年', author: 'André Lefèvre', kind: 'email',
      thread: 'm1@example.org', source: 'Inbox', at: Date.parse('2024-01-01T09:00:00Z') });
    expect(a.text).toBe('Bună ziua,\n\nAtașez procesul-verbal al ședinței — vă rog să-l citiți până mâine. Mulțumesc frumos pentru tot ajutorul! 🙏\n\nAttachments: 議事録.pdf');
    expect(a.metadata).toEqual({ from: 'André Lefèvre <andre@example.org>', to: '"Ioana Mărgărit" <ioana@example.org>, bob@example.org',
      cc: 'Łukasz <lukasz@example.org>' });

    expect(b).toMatchObject({ id: 'm2@example.org', title: 'Re: 会議の議事録', author: 'Ioana Mărgărit', thread: 'm1@example.org' });
    expect(b.text).toBe('Mulțumesc, am citit.\nFrom the start it was clear.\n\nAndré Lefèvre wrote:');
    expect(b.metadata).toMatchObject({ inReplyTo: 'm1@example.org' });

    expect(c.title).toBe('中文Привет');
    expect(c.text).toBe('Привет, мир');
    expect(c.author).toBe('Сергей');
    expect(c.id).toMatch(/^[0-9a-f]{32}$/);
    expect(c.thread).toBe(c.id);
  });

  it('keeps quoted lines and the signature when asked', async () => {
    const [a, b, c] = await all(fromMbox(path, { keepQuoted: true, keepSignature: true, collection: 'work' }));
    expect(a.text).toContain('> Ce s-a hotărât?');
    expect(a.text).toContain('-- \nAndré, București');
    expect(b.text).toContain('>From a quoted line, kept escaped once.');
    expect(c.text).toBe('Привет, мир\n\nold text');
    expect(a.collection).toBe('work');
  });

  it('decodes encoded-words: Q underscores, words joined across spaces, plain text between kept', () => {
    expect(decodeEncodedWords('=?utf-8?Q?Gr=C3=BC=C3=9Fe_aus?= =?utf-8?Q?_K=C3=B6ln?= und mehr')).toBe('Grüße aus Köln und mehr');
    expect(decodeEncodedWords('=?unknown-x?Q?abc?=')).toBe('abc');
  });
});

describe('fromEml', () => {
  it('reads one message with CRLF line endings', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'memtrails-eml-'));
    const path = join(dir, 'note.eml');
    writeFileSync(path, first.split('\n').slice(1).join('\r\n'));
    const [r] = await all(fromEml(path));
    expect(r).toMatchObject({ id: 'm1@example.org', title: '会議の議事録 — 2024年', source: 'note' });
    expect(r.text.startsWith('Bună ziua,\n\nAtașez')).toBe(true);
  });
});
