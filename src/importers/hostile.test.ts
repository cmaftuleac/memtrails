/**
 * Importers read files someone else wrote: no input may make them take more than linear time (a run of unclosed tags,
 * a long line of spaces), and the linear scans keep what the regexes they replaced did.
 */
import { closeSync, mkdirSync, mkdtempSync, openSync, rmSync, symlinkSync, writeFileSync, writeSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { MemoryRecord } from '../records.js';
import { fromFiles } from './files.js';
import { fromJsonl } from './jsonl.js';
import { byteLines, LineTooLong, MAX_LINE_BYTES } from './lines.js';
import { fromEml, fromMbox } from './mbox.js';
import { fromSlackExport } from './slack.js';
import { htmlTitle, htmlToText } from './text.js';

async function all<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

const N = 200_000;
const timed = async (label: string, f: () => unknown): Promise<void> => {
  const t = performance.now();
  await f();
  expect(performance.now() - t, label).toBeLessThan(1000);
};

describe('HTML on hostile input', () => {
  it('stays linear on runs of unclosed tags', async () => {
    for (const open of ['<script', '<style', '<iframe', '<head', '<pre', '<![CDATA[', '<div', '<a', '<br', '<li', '</td', '<!x',
      '<title>', '<h1>', '<blockquote', '<<']) {
      const s = open.repeat(Math.ceil(N / open.length));
      await timed(open, () => {
        htmlToText(s, { dropQuotes: true });
        htmlTitle(s);
      });
    }
  });

  it('drops hidden elements, keeps <pre> as written, runs an unclosed script to the end', () => {
    expect(htmlToText('<p>a</p><SCRIPT type="x">b</script ><p>c</p>')).toBe('a\n\nc');
    expect(htmlToText('<p>a</p><iframe src=x>keep<p>b</p>')).toBe('a\nkeep\nb');
    expect(htmlToText('a <script>var x = 1; <p>never</p>')).toBe('a');
    expect(htmlToText('<head><title>t</title><script>s</script></head><p>x   y</p>')).toBe('x y');
    expect(htmlToText('<p>a   b</p><pre>  x\r\n   y</pre><p>c   d</p>')).toBe('a b\n\nx\ny\n\nc d'); // line breaks kept inside <pre> only
    expect(htmlToText('<pre>  open   <p>x</p>')).toBe('open\nx');
    expect(htmlToText('<![CDATA[x < y]]> z')).toBe('x < y z');
    expect(htmlToText('a <![CDATA[ unclosed')).toBe('a <![CDATA[ unclosed');
    expect(htmlToText('<blockquote>q<blockquote>qq</blockquote></blockquote>reply', { dropQuotes: true })).toBe('reply');
    expect(htmlToText('<p>a</p> 1 < 2')).toBe('a\n1 < 2');
  });

  it('finds the title, else the first h1, in the first 64 KB', () => {
    expect(htmlTitle('<html><title lang=ro>Caf&eacute; <b>bun</b></title ><h1>H</h1>')).toBe('Café bun');
    expect(htmlTitle('<title> </title><h1 class=x>Head<br>line</h1>')).toBe('Head line');
    expect(htmlTitle('<title>open')).toBeUndefined();
    expect(htmlTitle(`${' '.repeat(70_000)}<title>late</title>`)).toBeUndefined();
  });
});

describe('files on hostile input', () => {
  const sp = ' '.repeat(N);
  const files: Record<string, string> = {
    'atx.md': `# a${sp}b`, 'atx-close.md': `# a${sp}#`, 'setext.md': `a${sp}b\n===`, 'eq.md': `${'='.repeat(N)}x\n${'='.repeat(N)}`,
    'front.md': `---\ntitle: a${sp}b\nk: v${sp}x\n---\nbody`, 'front.toml.md': `+++\na${sp}\nb = c${sp}d\n+++\n`,
    'title.rst': `a${sp}b\n${'='.repeat(N)}x`, 'stars.org': `* ${'*'.repeat(N)}`, 'eq.adoc': '='.repeat(N), 'tags.html': '<title>'.repeat(N / 7),
    'hashes.md': `${'#'.repeat(N)}\n# ${'# '.repeat(N / 2)}`,
  };
  for (const [name, text] of Object.entries(files)) {
    it(`reads ${name} in linear time`, async () => {
      const dir = mkdtempSync(join(tmpdir(), 'memtrails-hostile-'));
      writeFileSync(join(dir, name), text);
      await timed(name, () => all(fromFiles(dir)));
    });
  }

  it('keeps the titles the regexes found', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'memtrails-titles-'));
    const cases: Record<string, [string, string]> = {
      'atx.md': ['intro\n# Plan   ##  \nbody', 'Plan'],
      'atx-hash.md': ['# C# tips', 'C# tips'],
      'atx-only.md': ['# ###', '###'],
      'setext.md': ['Planul   \n=====\nbody', 'Planul'],
      'over.rst': ['=====\n Titlu \n=====\n\ntext', 'Titlu'],
      'under.rst': ['Titlu\n~~~~~\n', 'Titlu'],
      'short.rst': ['Titlu\n==\nNot\n---\n', 'Not'],
      'doc.adoc': ['== Sub\n= Document  \n', 'Document'],
      'notes.org': ['#+TITLE:  Notițe\n* Head', 'Notițe'],
      'head.org': ['text\n** Capitol 2\n', 'Capitol 2'],
      'cr.md': ['intro\r# Old Mac\rbody', 'Old Mac'],
    };
    for (const [name, [text]] of Object.entries(cases)) writeFileSync(join(dir, name), text);
    const titles = Object.fromEntries((await all(fromFiles(dir))).map((r: MemoryRecord) => [r.id, r.title]));
    for (const [name, [, title]] of Object.entries(cases)) expect(titles[name], name).toBe(title);
  });

  it('reads front matter as before', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'memtrails-front-'));
    writeFileSync(join(dir, 'a.md'), '---\ntitle: T # a comment\nurl: http://x.org/a:b\nk:v\ncount: 3\n---\nbody');
    writeFileSync(join(dir, 'b.md'), '+++\ntitle = "TOML"\nn = 2 # c\n+++\nbody');
    const [a, b] = await all(fromFiles(dir));
    expect(a.title).toBe('T');
    expect(a.metadata).toEqual({ url: 'http://x.org/a:b', count: 3 });
    expect(b.title).toBe('TOML');
    expect(b.metadata).toEqual({ n: 2 });
  });
});

describe('e-mail on hostile input', () => {
  const sp = ' '.repeat(N);
  const emls: Record<string, string> = {
    'from-at.eml': `From: ${'a@'.repeat(N / 2)}(x\nSubject: s\n\nhi\n`,
    'from-spaces.eml': `From: a${sp}<b\nSubject: s\n\nhi\n`,
    'qp.eml': `From: a@b.c\nContent-Transfer-Encoding: quoted-printable\n\na${sp}b\n`,
    'date.eml': `From: a@b.c\nDate: ${'('.repeat(N)}\nSubject: s\n\nhi\n`,
  };
  for (const [name, text] of Object.entries(emls)) {
    it(`reads ${name} in linear time`, async () => {
      const dir = mkdtempSync(join(tmpdir(), 'memtrails-eml-'));
      writeFileSync(join(dir, name), text);
      await timed(name, () => all(fromEml(join(dir, name))));
    });
  }

  it('still names the sender and reads the date', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'memtrails-eml-'));
    const authors: string[] = [];
    for (const [i, from] of ['"Pop, Ana" <ana@x.org>', '<ana@x.org>', 'ana@x.org (Ana Pop)', 'ana@x.org ()', 'Ana'].entries()) {
      writeFileSync(join(dir, `${i}.eml`), `From: ${from}\nDate: Fri, 1 Mar 2024 10:00:00 +0000 (UTC)\nSubject: s\n\nhi\n`);
      const [r] = await all(fromEml(join(dir, `${i}.eml`)));
      authors.push(r.author!);
      expect(r.at).toBe(Date.parse('2024-03-01T10:00:00Z'));
    }
    expect(authors).toEqual(['Pop, Ana', 'ana@x.org', 'Ana Pop', 'ana@x.org ()', 'Ana']);
  });
});

describe('a Slack export naming folders outside itself', () => {
  it('reads no conversation folder outside the export', async () => {
    const base = mkdtempSync(join(tmpdir(), 'memtrails-slack-escape-'));
    const root = join(base, 'export');
    const outside = join(base, 'secret');
    mkdirSync(join(root, 'general'), { recursive: true });
    mkdirSync(outside);
    writeFileSync(join(outside, '2024-01-01.json'), JSON.stringify([{ type: 'message', user: 'U1', text: 'outside', ts: '1704100000.0001' }]));
    writeFileSync(join(root, 'general', '2024-01-01.json'), JSON.stringify([{ type: 'message', user: 'U1', text: 'inside', ts: '1704100000.0002' }]));
    writeFileSync(join(root, 'channels.json'), JSON.stringify([
      { id: 'C1', name: 'general' }, { id: 'C2', name: '../secret' }, { id: 'C3', name: '..' }, { id: 'C4', name: outside },
      { id: 'C5', name: 'a/../../secret' }, { id: 'C6', name: '..\\secret' },
    ]));
    writeFileSync(join(root, 'dms.json'), JSON.stringify([{ id: '../secret', members: ['U1'] }]));
    const texts = (await all(fromSlackExport(root))).map((r) => r.text);
    expect(texts).toEqual(['inside']);
  });
});

describe('a note swapped for a link after the folder was listed', () => {
  it.skipIf(process.platform === 'win32')('is not followed', async () => {
    const base = mkdtempSync(join(tmpdir(), 'memtrails-files-link-'));
    const notes = join(base, 'notes');
    mkdirSync(notes);
    writeFileSync(join(notes, 'a.md'), '# A\nfine');
    const secret = join(base, 'secret.md');
    writeFileSync(secret, '# Secret\nthe private key');
    const skipped: string[] = [];
    // `exclude` is asked just before a file is read: the note becomes a link to what is outside the folder then.
    const records = await all(fromFiles(notes, {
      exclude: () => {
        rmSync(join(notes, 'a.md'));
        symlinkSync(secret, join(notes, 'a.md'));
        return false;
      },
      onSkip: (path, why) => skipped.push(`${path}: ${why}`),
    }));
    expect(records.map((r) => r.text)).toEqual([]);
    expect(skipped).toEqual(['a.md: not a file: it became a link after the folder was read']);
  });
});

/** A file of `head`, then a line of `long` bytes, then `tail`, written a megabyte at a time. */
function withLongLine(path: string, head: string, long: number, tail: string): void {
  const fd = openSync(path, 'w');
  writeSync(fd, head);
  const mb = Buffer.alloc(1 << 20, 'x');
  for (let left = long; left > 0; left -= mb.length) writeSync(fd, left >= mb.length ? mb : mb.subarray(0, left));
  writeSync(fd, tail);
  closeSync(fd);
}

describe('a line longer than 64 MiB', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memtrails-long-line-'));
  const lines = async (text: string, max: number) => {
    const path = join(dir, 'lines.txt');
    writeFileSync(path, text);
    const out: string[] = [];
    for await (const l of byteLines(path, max)) out.push(l instanceof LineTooLong ? `! ${l.reason}` : l.toString());
    return out;
  };

  it('lines end as readline ends them, and one past the bound is passed over', async () => {
    expect(await lines('a\r\nb\rc\n\nd', 100)).toEqual(['a', 'b', 'c', '', 'd']);
    expect(await lines('a\r\r\n', 100)).toEqual(['a', '']);
    expect(await lines('', 100)).toEqual([]);
    expect(await lines('short\nfar too long a line\nafter\n', 8)).toEqual(['short', '! a line longer than 8 bytes (19)', 'after']);
  });

  it('JSON Lines: skipped with why, and the next line read', { timeout: 60_000 }, async () => {
    const path = join(dir, 'big.jsonl');
    withLongLine(path, '{"id":"a","text":"before"}\n', MAX_LINE_BYTES + 1, '\n{"id":"b","text":"after"}\n');
    const skipped: string[] = [];
    const stats = { lines: 0, records: 0, skipped: 0 };
    const records = await all(fromJsonl(path, { onSkip: (line, why, n) => skipped.push(`${n} ${JSON.stringify(line)}: ${why}`), stats }));
    expect(records.map((r) => r.id)).toEqual(['a', 'b']);
    expect(skipped).toEqual([`2 "": a line longer than ${MAX_LINE_BYTES} bytes (${MAX_LINE_BYTES + 1})`]);
    expect(stats).toEqual({ lines: 3, records: 2, skipped: 1 });
  });

  it('mbox: the message is past its size from that line, and the next message read', { timeout: 60_000 }, async () => {
    const path = join(dir, 'big.mbox');
    const msg = (n: number, body: string) => `From a@b.c Mon Jan  1 00:00:0${n} 2024\nFrom: a@b.c\nSubject: m${n}\nMessage-ID: <m${n}@b.c>\n\n${body}`;
    withLongLine(path, `${msg(1, 'kept\n')}`, MAX_LINE_BYTES + 1, `\nnot kept\n\n${msg(2, 'second\n')}`);
    const records = await all(fromMbox(path, { maxMessageBytes: 4 * MAX_LINE_BYTES }));
    expect(records.map((r) => [r.title, r.text])).toEqual([['m1', 'kept'], ['m2', 'second']]);
  });
});
