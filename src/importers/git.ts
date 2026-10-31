/**
 * A Git repository's history: one record a commit, read from `git log` as it runs (never the whole log at once).
 * Each commit is framed by markers carrying a random token made for the run, so no message text can be taken for one.
 *
 * id: the full sha. title: the subject. text: the whole message, then `Files: …` (the paths it changed, at most
 * `maxFiles`). author: the author's name. at: the author date. kind `commit`; thread: `opts.thread` or the repository's
 * name; source: `opts.source` or the repository's name; metadata { sha, email, parents }.
 */
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { basename, resolve } from 'node:path';
import type { MemoryRecord } from '../records.js';
import { record } from './text.js';

export interface GitOptions {
  /** Only commits after this (anything `git log --since` reads, a Date, or milliseconds). */
  since?: string | Date | number;
  /** Only commits before this. */
  until?: string | Date | number;
  /** The branch, tag or range to read (`main`, `v1.0..v2.0`). Default: HEAD. `rev` is the same. */
  branch?: string;
  rev?: string;
  /** Every branch and tag. */
  all?: boolean;
  /** At most this many commits. */
  maxCount?: number;
  /** Only commits that changed these paths. */
  paths?: string[];
  /** Leave out merge commits. */
  noMerges?: boolean;
  /** At most this many changed paths listed in a record's text. Default 50. */
  maxFiles?: number;
  /** Default: the repository's name. */
  thread?: string;
  source?: string;
  collection?: string;
  /** Where to open a commit, e.g. `(sha) => \`https://example.org/repo/commit/${sha}\``. */
  url?: (sha: string) => string;
  /** The git program. Default `git`. */
  git?: string;
}

function when(t: string | Date | number): string {
  return t instanceof Date ? t.toISOString() : typeof t === 'number' ? new Date(t).toISOString() : t;
}

/** A path as git writes it when it has unusual characters: `"a\tb\303\251"` back to its text. */
function unquote(path: string): string {
  if (!(path.length > 1 && path.startsWith('"') && path.endsWith('"'))) return path;
  const bytes: number[] = [];
  const s = path.slice(1, -1);
  const esc: Record<string, number> = { a: 7, b: 8, t: 9, n: 10, v: 11, f: 12, r: 13, '"': 34, '\\': 92 };
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '\\' && /[0-7]{3}/.test(s.slice(i + 1, i + 4))) {
      bytes.push(parseInt(s.slice(i + 1, i + 4), 8));
      i += 3;
    } else if (s[i] === '\\' && s[i + 1] in esc) {
      bytes.push(esc[s[++i]]);
    } else {
      bytes.push(...Buffer.from(s[i], 'utf8'));
    }
  }
  return Buffer.from(bytes).toString('utf8');
}

/** Records from a repository's commits, newest first (as `git log` gives them). */
export async function* fromGit(repoDir: string, opts: GitOptions = {}): AsyncIterable<MemoryRecord> {
  const name = basename(resolve(repoDir));
  const token = randomBytes(16).toString('hex');
  const start = `\u001e${token}\n`;
  const end = `\u001f${token}`;
  const args = ['-c', 'core.quotePath=false', '-c', 'log.showSignature=false', 'log', '--encoding=UTF-8', '--no-color',
    '--no-ext-diff', '--name-only', `--format=%x1e${token}%n%H%x00%an%x00%ae%x00%aI%x00%P%x00%B%x1f${token}`];
  if (opts.maxCount !== undefined) args.push(`--max-count=${Math.max(0, Math.floor(opts.maxCount))}`);
  if (opts.since !== undefined) args.push(`--since=${when(opts.since)}`);
  if (opts.until !== undefined) args.push(`--until=${when(opts.until)}`);
  if (opts.noMerges) args.push('--no-merges');
  if (opts.all) args.push('--all');
  const rev = opts.rev ?? opts.branch;
  if (rev) {
    if (rev.startsWith('-')) throw new Error(`not a revision: ${rev}`);
    args.push(rev);
  }
  args.push('--', ...(opts.paths ?? []));
  const maxFiles = opts.maxFiles ?? 50;

  const child = spawn(opts.git ?? 'git', args, { cwd: repoDir, stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (d: string) => {
    if (stderr.length < 4000) stderr += d;
  });
  const exited = new Promise<number | null>((done, fail) => {
    child.on('error', fail);
    child.on('close', (code) => done(code));
  });
  exited.catch(() => undefined); // read below; this only keeps an early stop from leaving it unhandled

  const toRecord = (chunk: string): MemoryRecord | null => {
    const close = chunk.lastIndexOf(end);
    const head = close >= 0 ? chunk.slice(0, close) : chunk;
    const [sha, author, email, date, parents, ...message] = head.split('\u0000');
    if (!sha || !/^[0-9a-f]{7,64}$/.test(sha)) return null;
    const body = message.join('\u0000').replace(/\r\n?/g, '\n').trim();
    const subject = (body.split(/\n\s*\n/)[0] ?? '').replace(/\s*\n\s*/g, ' ').trim();
    const files = close >= 0 ? chunk.slice(close + end.length).split('\n').filter((l) => l.trim()).map(unquote) : [];
    const listed = files.slice(0, maxFiles).join(', ') + (files.length > maxFiles ? `, … and ${files.length - maxFiles} more` : '');
    const text = [body, files.length ? `Files: ${listed}` : ''].filter(Boolean).join('\n\n');
    return record({ id: sha, title: subject || sha.slice(0, 12), text, kind: 'commit', author, at: date || undefined,
      thread: opts.thread ?? name, source: opts.source ?? name, collection: opts.collection, url: opts.url?.(sha),
      metadata: { sha, email, parents } });
  };

  let buffer = '';
  try {
    child.stdout.setEncoding('utf8');
    for await (const piece of child.stdout as AsyncIterable<string>) {
      buffer += piece;
      let next = buffer.indexOf(start, 1);
      while (next > 0) {
        const r = toRecord(buffer.slice(buffer.startsWith(start) ? start.length : 0, next));
        if (r) yield r;
        buffer = buffer.slice(next);
        next = buffer.indexOf(start, 1);
      }
    }
    if (buffer.startsWith(start)) {
      const r = toRecord(buffer.slice(start.length));
      if (r) yield r;
    }
  } finally {
    if (child.exitCode === null && child.signalCode === null) child.kill();
  }
  const code = await exited;
  if (code !== 0 && !/does not have any commits|bad default revision 'HEAD'/.test(stderr)) {
    throw new Error(`git log failed (${code}) in ${repoDir}: ${stderr.trim()}`);
  }
}
