import { mkdirSync, mkdtempSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { fromFiles } from './files.js';
import { htmlToText, splitText } from './text.js';

async function all<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

describe('fromFiles', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memtrails-files-'));
  mkdirSync(join(dir, 'notițe', '日本'), { recursive: true });
  mkdirSync(join(dir, '.git'));
  mkdirSync(join(dir, 'node_modules', 'pkg'), { recursive: true });
  writeFileSync(join(dir, 'notițe', 'plan.md'), [
    '---',
    'title: "Planul pe trimestrul trei: priorități"',
    'date: 2024-07-01',
    'author: Ioana Mărgărit',
    'tags: [buget, echipă]',
    'reviewers:',
    '  - Ana',
    '  - Mihai',
    'draft: true',
    'version: 3',
    '---',
    '# Not the title',
    '',
    'Bugetul pentru țară crește cu 5%.',
  ].join('\n'));
  writeFileSync(join(dir, 'notițe', '日本', 'メモ.md'), 'Intro line\n\n# 東京オフィスの移転\n\n来月、新しいオフィスに移ります。🚚\n');
  writeFileSync(join(dir, 'readme.txt'), 'Ελληνικά: καλημέρα κόσμε.\n');
  utimesSync(join(dir, 'readme.txt'), new Date('2023-02-03T04:05:06Z'), new Date('2023-02-03T04:05:06Z'));
  writeFileSync(join(dir, 'page.html'), `<!doctype html><html><head><title>Caf&eacute; &amp; Cr&egrave;me</title>
    <style>body { color: red }</style><script>var secret = "<p>not text</p>";</script></head>
    <body><h1>Menu</h1><p>Prix&nbsp;: 3&nbsp;&euro; &mdash; <b>très</b>
    bon</p><ul><li>Thé</li><li>&#x1F375; matcha &#8211; 抹茶</li></ul><!-- hidden --><p>Ünïcödé &lt;ok&gt;</p></body></html>`);
  writeFileSync(join(dir, '.git', 'config.md'), '# hidden');
  writeFileSync(join(dir, '.secret.md'), '# hidden');
  writeFileSync(join(dir, 'node_modules', 'pkg', 'README.md'), '# dependency');
  writeFileSync(join(dir, 'image.png'), Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0]));
  const paragraph = (n: number): string => `Paragraph ${n}: ${'lorem ipsum dolor sit amet '.repeat(12)}— ação, naïve, 漢字.`;
  writeFileSync(join(dir, 'big.md'), `# Big\n\n${Array.from({ length: 30 }, (_, i) => paragraph(i)).join('\n\n')}`);

  it('reads text files, skipping dot folders, node_modules and other extensions', async () => {
    const records = await all(fromFiles(dir, { collection: 'docs', maxChars: 2000 }));
    const ids = records.map((r) => r.id);
    expect(ids.filter((id) => !id.startsWith('big.md'))).toEqual(['notițe/plan.md', 'notițe/日本/メモ.md', 'page.html', 'readme.txt']);
    expect(records.every((r) => r.collection === 'docs' && r.kind === 'note')).toBe(true);

    const plan = records.find((r) => r.id === 'notițe/plan.md')!;
    expect(plan.title).toBe('Planul pe trimestrul trei: priorități');
    expect(plan.author).toBe('Ioana Mărgărit');
    expect(plan.at).toBe(Date.parse('2024-07-01'));
    expect(plan.text.startsWith('# Not the title')).toBe(true);
    expect(plan.metadata).toEqual({ date: '2024-07-01', author: 'Ioana Mărgărit', tags: 'buget, echipă', reviewers: 'Ana, Mihai', draft: true, version: 3 });

    expect(records.find((r) => r.id === 'notițe/日本/メモ.md')!.title).toBe('東京オフィスの移転');
    const readme = records.find((r) => r.id === 'readme.txt')!;
    expect(readme.title).toBe('readme');
    expect(readme.at).toBe(Date.parse('2023-02-03T04:05:06Z'));
    expect(readme.url).toMatch(/^file:\/\//);

    const page = records.find((r) => r.id === 'page.html')!;
    expect(page.title).toBe('Café & Crème');
    expect(page.text).not.toMatch(/secret|color|hidden|<p>|Café/);
    expect(page.text).toContain('Prix : 3 € — très bon');
    expect(page.text).toContain('- Thé\n- 🍵 matcha – 抹茶');
    expect(page.text).toContain('Ünïcödé <ok>');
  });

  it('cuts a large file into parts that share a thread and keep every word', async () => {
    const parts = (await all(fromFiles(dir, { maxChars: 2000 }))).filter((r) => r.id.startsWith('big.md'));
    expect(parts.length).toBeGreaterThan(3);
    expect(parts.map((p) => p.id)).toEqual(parts.map((_, i) => `big.md#${i + 1}`));
    expect(parts.every((p) => p.thread === 'big.md' && p.text.length <= 2000)).toBe(true);
    expect(parts[0].title).toBe(`Big (1/${parts.length})`);
    expect(parts[1].metadata).toMatchObject({ part: 2, parts: parts.length });
    const words = (s: string): string[] => s.split(/\s+/).filter(Boolean);
    expect(words(parts.map((p) => p.text).join('\n\n')).length).toBe(words(`# Big\n\n${Array.from({ length: 30 }, (_, i) => paragraph(i)).join('\n\n')}`).length);
  });

  it('cuts text without spaces at a sentence end, and never inside a character', () => {
    const cjk = '今日は晴れです。'.repeat(40) + '😀'.repeat(200);
    const parts = splitText(cjk, 150);
    expect(parts.join('')).toBe(cjk);
    expect(parts[0].endsWith('。')).toBe(true);
    for (const p of parts) expect(p).not.toMatch(/^[\udc00-\udfff]|[\ud800-\udbff]$/);
  });

  it('drops HTML blockquotes when asked', () => {
    expect(htmlToText('<p>Yes.</p><blockquote>Old <blockquote>older</blockquote> text</blockquote><p>Bye</p>', { dropQuotes: true })).toBe('Yes.\n\nBye');
  });
});
