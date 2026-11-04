/**
 * A folder of text files: notes, documents, saved pages. Every file whose extension is in `include` becomes a record,
 * read one at a time; dot folders, dot files and `node_modules` are passed by.
 *
 * id: the path inside the folder (with `/`). title: the Markdown front matter's `title`, else the first `# heading`
 * (an Org `#+TITLE:`, an RST underlined title, an HTML `<title>` or `<h1>`), else the file's name. text: the file, front
 * matter taken off, HTML reduced to text. at: the front matter's `date`, else when the file was last changed. The other
 * front-matter fields go into metadata (`author` also to the record's author). A file longer than `maxChars` is cut
 * into parts with ids `path#1`, `path#2`, … sharing the thread `path`.
 */
import { readdir, readFile, stat } from 'node:fs/promises';
import { basename, extname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { timeOf, type MemoryRecord } from '../records.js';
import { decodeBytes, htmlTitle, htmlToText, record, splitText } from './text.js';

/** The extensions read by default: text formats only. */
export const TEXT_EXTENSIONS = ['.md', '.markdown', '.mdx', '.txt', '.text', '.html', '.htm', '.xhtml', '.rst', '.org', '.adoc',
  '.asciidoc', '.textile', '.wiki', '.tex'];

export interface FilesOptions {
  /** Extensions to read, with or without the dot. Default: TEXT_EXTENSIONS. */
  include?: string[];
  /** Folder names never entered (dot folders never are). Default: `node_modules`. */
  skipFolders?: string[];
  /** Leave out a file by its path inside the folder. */
  exclude?: (path: string) => boolean;
  /** Longer files are cut into parts of at most this many characters. Default 20,000. */
  maxChars?: number;
  /** Larger files are not read (reported through `onSkip`). Default 64 MiB. */
  maxBytes?: number;
  collection?: string;
  /** Default: the folder's name. */
  source?: string;
  /** Default `note`. */
  kind?: string;
  /** Called for each file left out, with why. */
  onSkip?: (path: string, reason: string) => void;
}

type Scalar = string | number | boolean | null;

function scalar(raw: string): Scalar {
  const v = raw.trim();
  if (/^(["']).*\1$/s.test(v)) return v.slice(1, -1).replace(/\\"/g, '"').replace(/''/g, "'");
  if (/^(true|false)$/i.test(v)) return v.toLowerCase() === 'true';
  if (/^(null|~)$/i.test(v) || v === '') return null;
  if (/^-?(0|[1-9]\d{0,14})(\.\d+)?$/.test(v)) return Number(v);
  return v;
}

function listValue(items: string[]): string {
  return items.map((i) => scalar(i)).filter((i) => i !== null && i !== '').join(', ');
}

/**
 * One front-matter line as key and value: TOML `key = value`, YAML `key: value` (a `:` then a space or the end). Split
 * at the first `=` or `:` by hand: `^([^=]*?)\s*=` tried a run of spaces at every position, quadratic on a long line.
 */
function keyValue(line: string, toml: boolean): [string, string] | null {
  const at = line.indexOf(toml ? '=' : ':');
  if (at <= 0 || (toml ? /[\s=#[]/ : /[\s:#-]/).test(line[0])) return null;
  const value = line.slice(at + 1);
  if (!toml && value && !/^\s/.test(value)) return null;
  return [line.slice(0, at).trimEnd(), value];
}

/** YAML (`---`) or TOML (`+++`) front matter, flat fields only: scalars, lists (joined with `, `) and block text. */
export function frontMatter(text: string): { fields: Record<string, Scalar>; body: string } {
  const m = /^(---|\+\+\+)[ \t]*\r?\n([\s\S]*?)\r?\n(?:---|\+\+\+|\.\.\.)[ \t]*(?:\r?\n|$)/.exec(text);
  if (!m) return { fields: {}, body: text };
  const toml = m[1] === '+++';
  const fields: Record<string, Scalar> = {};
  const lines = m[2].split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const kv = keyValue(lines[i], toml);
    if (!kv) continue;
    const key = kv[0].trim().replace(/^["']|["']$/g, '');
    const value = kv[1].trim();
    if (value.startsWith('[') && value.endsWith(']')) {
      fields[key] = listValue(value.slice(1, -1).split(','));
    } else if (!toml && /^[|>][-+]?$/.test(value)) {
      const block: string[] = [];
      while (i + 1 < lines.length && (/^\s/.test(lines[i + 1]) || !lines[i + 1].trim())) block.push(lines[++i].trim());
      fields[key] = block.join(value.startsWith('|') ? '\n' : ' ').trim();
    } else if (!toml && !value) {
      const items: string[] = [];
      while (i + 1 < lines.length && /^\s*-\s+/.test(lines[i + 1])) items.push(lines[++i].replace(/^\s*-\s+/, ''));
      while (i + 1 < lines.length && /^\s+\S/.test(lines[i + 1])) i++; // a nested map: not kept
      if (items.length) fields[key] = listValue(items);
    } else {
      const comment = value.search(/\s#/); // not /\s+#.*$/: quadratic on a long run of spaces
      fields[key] = scalar(comment < 0 ? value : value.slice(0, comment));
    }
  }
  return { fields, body: text.slice(m[0].length) };
}

/** Title rules see at most this much of a file, a line at a time: per-line regexes over a whole hostile file were slow. */
const TITLE_SCAN = 64 * 1024;

/** An RST underline (or overline): one punctuation character, three or more times, then spaces at most. */
function rstRule(line: string | undefined): boolean {
  const t = line?.trimEnd() ?? ''; // not /[ \t]+$/: quadratic on a long run of spaces
  return t.length >= 3 && '=-~^"\'`#*+'.includes(t[0]) && t === t[0].repeat(t.length);
}

function titleOf(ext: string, body: string, raw: string): string | undefined {
  if (ext === '.html' || ext === '.htm' || ext === '.xhtml') return htmlTitle(raw);
  const lines = body.slice(0, TITLE_SCAN).split(/\r\n?|\n/);
  const first = (f: (line: string, i: number) => string | undefined): string | undefined => {
    for (let i = 0; i < lines.length; i++) {
      const t = f(lines[i], i);
      if (t !== undefined) return t;
    }
    return undefined;
  };
  if (ext === '.org') {
    return first((l) => (/^#\+title:/i.test(l) && l.length > 8 ? l.slice(8).trim() : undefined)) ||
      first((l) => (/^\*+[ \t]/.test(l) && l.replace(/^\*+/, '').trim() ? l.replace(/^\*+/, '').trim() : undefined));
  }
  if (ext === '.rst') return first((l, i) => (l.trim() && rstRule(lines[i + 1]) ? l.trim() : undefined));
  if (ext === '.adoc' || ext === '.asciidoc') return first((l) => (/^=[ \t]/.test(l) && l.slice(1).trim() ? l.slice(1).trim() : undefined));
  // `# Title`, a closing run of `#` after a space dropped (by hand: `[ \t]+#+$` is quadratic on a run of spaces).
  const atx = first((l) => {
    if (!/^#[ \t]/.test(l)) return undefined;
    const t = l.slice(1).trim();
    let j = t.length;
    while (j > 0 && t[j - 1] === '#') j--;
    const title = j > 0 && j < t.length && /[ \t]/.test(t[j - 1]) ? t.slice(0, j).trimEnd() : t;
    return title || undefined;
  });
  if (atx) return atx;
  // A Setext `Title\n=====`.
  return first((l, i) => (/^\S/.test(l) && /^=+[ \t]*$/.test(lines[i + 1] ?? '') ? l.trimEnd() : undefined));
}

async function* walk(root: string, rel: string, o: { include: Set<string>; skipFolders: Set<string> }): AsyncGenerator<string> {
  const entries = await readdir(join(root, rel), { withFileTypes: true });
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const e of entries) {
    if (e.name.startsWith('.')) continue;
    const path = rel ? `${rel}/${e.name}` : e.name;
    if (e.isDirectory()) {
      if (!o.skipFolders.has(e.name)) yield* walk(root, path, o);
    } else if (e.isFile() && o.include.has(extname(e.name).toLowerCase())) {
      yield path;
    }
  }
}

function decodeFile(bytes: Buffer): string | null {
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return new TextDecoder('utf-16le').decode(bytes);
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return new TextDecoder('utf-16be').decode(bytes);
  if (bytes.subarray(0, 8192).includes(0)) return null; // binary
  return decodeBytes(bytes);
}

/** Records from the text files in a folder and its subfolders. */
export async function* fromFiles(dir: string, opts: FilesOptions = {}): AsyncIterable<MemoryRecord> {
  const root = resolve(dir);
  const include = new Set((opts.include ?? TEXT_EXTENSIONS).map((e) => (e.startsWith('.') ? e : `.${e}`).toLowerCase()));
  const skipFolders = new Set(opts.skipFolders ?? ['node_modules']);
  const maxChars = opts.maxChars ?? 20_000;
  const maxBytes = opts.maxBytes ?? 64 * 1024 * 1024;
  const source = opts.source ?? basename(root);
  for await (const path of walk(root, '', { include, skipFolders })) {
    if (opts.exclude?.(path)) continue;
    const abs = join(root, path);
    const info = await stat(abs);
    if (info.size > maxBytes) {
      opts.onSkip?.(path, `larger than ${maxBytes} bytes`);
      continue;
    }
    const raw = decodeFile(await readFile(abs));
    if (raw === null) {
      opts.onSkip?.(path, 'not text');
      continue;
    }
    const ext = extname(path).toLowerCase();
    const isHtml = ext === '.html' || ext === '.htm' || ext === '.xhtml';
    const { fields, body } = isHtml ? { fields: {} as Record<string, Scalar>, body: raw } : frontMatter(raw);
    const fmTitle = typeof fields.title === 'string' || typeof fields.title === 'number' ? String(fields.title).trim() : '';
    const title = fmTitle || titleOf(ext, body, raw) || basename(path, extname(path));
    const text = isHtml ? htmlToText(raw) : body.replace(/\r\n?/g, '\n').trim();
    const dated = typeof fields.date === 'string' || typeof fields.date === 'number' ? timeOf(String(fields.date)) : undefined;
    const at = dated ?? info.mtimeMs;
    const author = typeof fields.author === 'string' ? fields.author : undefined;
    const metadata = Object.fromEntries(Object.entries(fields).filter(([k]) => k !== 'title'));
    const base: MemoryRecord = { id: path, text, title, kind: opts.kind ?? 'note', source, collection: opts.collection, author,
      at: Math.round(at), url: pathToFileURL(abs).href };
    if (!text && !title) continue;
    const parts = splitText(text, maxChars);
    if (parts.length === 1) {
      yield record({ ...base, metadata });
      continue;
    }
    for (let i = 0; i < parts.length; i++) {
      yield record({ ...base, id: `${path}#${i + 1}`, thread: path, text: parts[i], title: `${title} (${i + 1}/${parts.length})`,
        metadata: { ...metadata, part: i + 1, parts: parts.length } });
    }
  }
}
