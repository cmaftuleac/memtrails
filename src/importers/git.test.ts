import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { fromGit } from './git.js';

async function all<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

describe('fromGit', () => {
  const repo = mkdtempSync(join(tmpdir(), 'memtrails-git-'));
  const env = { ...process.env, GIT_CONFIG_GLOBAL: '/dev/null', GIT_CONFIG_NOSYSTEM: '1' };
  const git = (args: string[], extra: Record<string, string> = {}): string =>
    execFileSync('git', args, { cwd: repo, env: { ...env, ...extra }, encoding: 'utf8' });
  // `git commit -F -` reads the message from standard input; `--cleanup=verbatim` keeps it exactly as written.
  const commitWith = (name: string, email: string, date: string, message: string): void => {
    execFileSync('git', ['-c', `user.name=${name}`, '-c', `user.email=${email}`, '-c', 'commit.gpgsign=false', 'commit', '-q',
      '--cleanup=verbatim', '-F', '-'], { cwd: repo, env: { ...env, GIT_AUTHOR_DATE: date, GIT_COMMITTER_DATE: date }, input: message });
  };
  git(['init', '-q', '-b', 'main']);
  mkdirSync(join(repo, 'notițe'));
  writeFileSync(join(repo, 'notițe', '日本語.md'), 'こんにちは\n');
  writeFileSync(join(repo, 'README.md'), 'hello\n');
  git(['add', '-A']);
  commitWith('Ștefan Müller-Łukasiewicz 王小明', 'stefan@example.org', '2024-03-01T10:00:00+02:00',
    'Adaugă notițele în japoneză 🗾\n\nCorpul mesajului: café, naïve, 漢字.\n');
  writeFileSync(join(repo, 'README.md'), 'hello again\n');
  git(['add', '-A']);
  commitWith('Ana', 'ana@example.org', '2024-03-02T11:30:00Z', [
    'Tricky message with %H and --- and \u001e\u001f in it',
    '',
    '\u001e0123456789abcdef0123456789abcdef',
    'commit 0000000000000000000000000000000000000000',
    'Author: Nobody <nobody@example.org>',
    '\u0000-less but \u001fstill odd',
    '',
    'Files: not really',
  ].join('\n').replace('\u0000', ''));

  it('reads every commit, newest first, whatever its message holds', async () => {
    const records = await all(fromGit(repo));
    expect(records).toHaveLength(2);
    const [second, first] = records;
    expect(second.title).toBe('Tricky message with %H and --- and \u001e\u001f in it');
    expect(second.text).toContain('commit 0000000000000000000000000000000000000000');
    expect(second.text).toContain('\u001e0123456789abcdef');
    expect(second.text.endsWith('Files: README.md')).toBe(true);
    expect(second.author).toBe('Ana');
    expect(second.at).toBe('2024-03-02T11:30:00Z');

    expect(first.id).toMatch(/^[0-9a-f]{40}$/);
    expect(first.metadata).toMatchObject({ sha: first.id, email: 'stefan@example.org' });
    expect(first.author).toBe('Ștefan Müller-Łukasiewicz 王小明');
    expect(first.title).toBe('Adaugă notițele în japoneză 🗾');
    expect(first.text).toBe('Adaugă notițele în japoneză 🗾\n\nCorpul mesajului: café, naïve, 漢字.\n\nFiles: README.md, notițe/日本語.md');
    expect(Date.parse(String(first.at))).toBe(Date.parse('2024-03-01T08:00:00Z'));
    expect(first).toMatchObject({ kind: 'commit', thread: basename(repo), source: basename(repo) });
  });

  it('takes maxCount, paths, since, thread and a link', async () => {
    expect(await all(fromGit(repo, { maxCount: 1 }))).toHaveLength(1);
    const onlyNotes = await all(fromGit(repo, { paths: ['notițe'], thread: 'history', url: (sha) => `https://example.org/c/${sha}` }));
    expect(onlyNotes.map((r) => r.title)).toEqual(['Adaugă notițele în japoneză 🗾']);
    expect(onlyNotes[0]).toMatchObject({ thread: 'history', url: `https://example.org/c/${onlyNotes[0].id}` });
    expect(await all(fromGit(repo, { since: new Date('2024-03-02T00:00:00Z') }))).toHaveLength(1);
    expect(await all(fromGit(repo, { branch: 'main', maxFiles: 1 }))).toHaveLength(2);
    const capped = await all(fromGit(repo, { maxFiles: 1 }));
    expect(capped[1].text.endsWith('Files: README.md, … and 1 more')).toBe(true);
  });

  it('stops git when the reader stops early, and fails loudly on a bad revision', async () => {
    for await (const r of fromGit(repo)) {
      expect(r.id).toMatch(/^[0-9a-f]{40}$/);
      break;
    }
    await expect(all(fromGit(repo, { rev: 'no-such-branch' }))).rejects.toThrow(/git log failed/);
  });

  it('reads an empty repository as nothing', async () => {
    const empty = mkdtempSync(join(tmpdir(), 'memtrails-git-empty-'));
    execFileSync('git', ['init', '-q'], { cwd: empty, env });
    expect(await all(fromGit(empty))).toEqual([]);
  });
});
