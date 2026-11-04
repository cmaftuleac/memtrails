/**
 * E-mail: an mbox file (RFC 4155: each message opens with a `From ` line; `>From ` in a body is unescaped), read a
 * message at a time, or one `.eml` file. Headers are unfolded and their RFC 2047 encoded-words (Q and B, any charset)
 * decoded; MIME parts are taken apart (text/plain preferred, else text/html reduced to text), base64 and
 * quoted-printable decoded, and every charset the platform knows read as such.
 *
 * id: the Message-ID (else a hash of the message). title: the Subject. text: the body, with the names of attachments
 * after it; lines quoted from earlier messages (`>`) and the signature (after `-- `) are left out unless `keepQuoted` /
 * `keepSignature`, since the earlier message is a record of its own. author: the sender's name, else address. at: the
 * Date. kind `email`. thread: the first id in References, else In-Reply-To, else the message's own id.
 * metadata { from, to, cc, inReplyTo }.
 */
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { basename, extname } from 'node:path';
import { createInterface } from 'node:readline';
import { timeOf, type MemoryRecord } from '../records.js';
import { decodeBytes, htmlToText, record } from './text.js';

export interface MboxOptions {
  /** Keep lines quoted from earlier messages (`> …`). */
  keepQuoted?: boolean;
  /** Keep the signature (what follows a `-- ` line). */
  keepSignature?: boolean;
  /** Default: the file's name without its extension. */
  source?: string;
  collection?: string;
  /** A message's body past this many bytes is not read (its headers are). Default 32 MiB. */
  maxMessageBytes?: number;
  /** Called for each message left out (no subject and no text), with its place in the file. */
  onSkip?: (reason: string, index: number) => void;
}

/** One MIME entity: headers (first of each name, raw bytes as Latin-1 characters, unfolded) and body lines. */
interface Entity { headers: Map<string, string>; body: string[] }
interface Content { text: string; html: boolean; attachments: string[] }

function entityOf(lines: string[]): Entity {
  const headers = new Map<string, string>();
  let i = 0;
  let name = '';
  for (; i < lines.length; i++) {
    const line = lines[i];
    if (line === '' || line === '\r') break;
    if (/^[ \t]/.test(line)) {
      if (name) headers.set(name, `${headers.get(name)}${line.replace(/^[ \t]+/, ' ')}`);
      continue;
    }
    const colon = line.indexOf(':');
    if (colon <= 0) continue;
    const key = line.slice(0, colon).trim().toLowerCase();
    if (headers.has(key)) {
      name = ''; // a later copy of a header: the first is kept
      continue;
    }
    name = key;
    headers.set(key, line.slice(colon + 1).trim());
  }
  return { headers, body: lines.slice(i + 1) };
}

function qBytes(s: string): number[] {
  const out: number[] = [];
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '_') out.push(0x20);
    else if (s[i] === '=' && /^[0-9A-Fa-f]{2}$/.test(s.slice(i + 1, i + 3))) {
      out.push(parseInt(s.slice(i + 1, i + 3), 16));
      i += 2;
    } else out.push(s.charCodeAt(i) & 0xff);
  }
  return out;
}

/** RFC 2047 encoded-words decoded; words side by side join with no space, and their bytes are read together. */
export function decodeEncodedWords(s: string): string {
  let out = '';
  let last = 0;
  let pending: { charset: string; bytes: number[] } | null = null;
  const flush = (): void => {
    if (pending) out += decodeBytes(Uint8Array.from(pending.bytes), pending.charset);
    pending = null;
  };
  for (const m of s.matchAll(/=\?([^?\s]+)\?([bBqQ])\?([^?\s]*)\?=/g)) {
    const gap = s.slice(last, m.index);
    if (!pending || !/^\s*$/.test(gap)) {
      flush();
      out += gap;
    }
    const charset = m[1].replace(/\*.*$/, '').toLowerCase();
    const bytes = m[2].toUpperCase() === 'B' ? [...Buffer.from(m[3], 'base64')] : qBytes(m[3]);
    if (pending && pending.charset !== charset) flush();
    if (pending) pending.bytes.push(...bytes);
    else pending = { charset, bytes };
    last = m.index + m[0].length;
  }
  flush();
  return out + s.slice(last);
}

/** A raw header as text: bytes sent unencoded read as UTF-8 (else Windows-1252), then encoded-words decoded. */
function headerText(raw: string | undefined): string {
  if (!raw) return '';
  return decodeEncodedWords(decodeBytes(Buffer.from(raw, 'latin1'))).replace(/\s+/g, ' ').trim();
}

/** `type/sub; a=1; b="x y"` as its value and parameters (RFC 2231 continued and encoded ones joined and decoded). */
function parseParams(raw: string | undefined): { value: string; params: Record<string, string> } {
  const pieces: string[] = [];
  let cur = '';
  let quoted = false;
  for (const ch of raw ?? '') {
    if (ch === '"') quoted = !quoted;
    if (ch === ';' && !quoted) {
      pieces.push(cur);
      cur = '';
    } else cur += ch;
  }
  pieces.push(cur);
  const value = (pieces.shift() ?? '').trim().toLowerCase();
  const params: Record<string, string> = {};
  const sections: Record<string, Array<{ n: number; encoded: boolean; v: string }>> = {};
  for (const p of pieces) {
    const eq = p.indexOf('=');
    if (eq <= 0) continue;
    const key = p.slice(0, eq).trim().toLowerCase();
    let v = p.slice(eq + 1).trim();
    if (v.startsWith('"') && v.endsWith('"') && v.length > 1) v = v.slice(1, -1).replace(/\\(.)/g, '$1');
    const ext = /^(.+?)\*(\d+)?(\*)?$/.exec(key);
    if (ext) (sections[ext[1]] ??= []).push({ n: Number(ext[2] ?? 0), encoded: ext[2] === undefined || !!ext[3], v });
    else params[key] = key === 'name' || key === 'filename' ? headerText(v) : v;
  }
  for (const [key, list] of Object.entries(sections)) {
    list.sort((a, b) => a.n - b.n);
    let charset = '';
    const bytes: number[] = [];
    for (const [i, s] of list.entries()) {
      let v = s.v;
      if (s.encoded && i === 0) {
        const m = /^([^']*)'[^']*'(.*)$/.exec(v);
        if (m) [charset, v] = [m[1], m[2]];
      }
      if (s.encoded) bytes.push(...Buffer.from(v.replace(/%([0-9A-Fa-f]{2})/g, (_, h: string) => String.fromCharCode(parseInt(h, 16))), 'latin1'));
      else bytes.push(...Buffer.from(v, 'latin1'));
    }
    params[key] = decodeBytes(Uint8Array.from(bytes), charset);
  }
  return { value, params };
}

function quotedPrintable(lines: string[]): Buffer {
  let s = '';
  for (const [i, raw] of lines.entries()) {
    let end = raw.length;
    while (end > 0 && (raw[end - 1] === ' ' || raw[end - 1] === '\t' || raw[end - 1] === '\r')) end--; // /[ \t\r]+$/ is quadratic on a run of spaces
    const line = raw.slice(0, end);
    if (line.endsWith('=')) s += line.slice(0, -1);
    else s += i < lines.length - 1 ? `${line}\n` : line;
  }
  return Buffer.from(s.replace(/=([0-9A-Fa-f]{2})/g, (_, h: string) => String.fromCharCode(parseInt(h, 16))), 'latin1');
}

function bodyBytes(e: Entity): Buffer {
  const cte = (e.headers.get('content-transfer-encoding') ?? '').trim().toLowerCase();
  if (cte === 'base64') return Buffer.from(e.body.join(''), 'base64');
  if (cte === 'quoted-printable') return quotedPrintable(e.body);
  return Buffer.from(e.body.join('\n'), 'latin1');
}

/** `format=flowed` (RFC 3676) text with its soft line breaks joined. */
function unflow(text: string, delsp: boolean): string {
  const out: string[] = [];
  let cur: string | null = null;
  for (const raw of text.split('\n')) {
    const quote = /^>+/.exec(raw)?.[0] ?? '';
    let line = raw.slice(quote.length);
    if (line.startsWith(' ')) line = line.slice(1);
    const soft = line.endsWith(' ') && line !== '-- ';
    if (soft && delsp) line = line.slice(0, -1);
    cur = cur === null ? (quote ? `${quote} ${line}` : line) : cur + line;
    if (!soft) {
      out.push(cur);
      cur = null;
    }
  }
  if (cur !== null) out.push(cur);
  return out.join('\n');
}

function clean(text: string, html: boolean, opts: MboxOptions): string {
  let lines = text.split('\n');
  if (!opts.keepSignature) {
    const sig = lines.findIndex((l) => l === '-- ' || (html && l === '--'));
    if (sig >= 0) lines = lines.slice(0, sig);
  }
  if (!opts.keepQuoted) lines = lines.filter((l) => !/^ ?>/.test(l));
  return lines.map((l) => (l === '-- ' ? l : l.trimEnd())).join('\n').replace(/\n{3,}/g, '\n\n').trim();
}

function contentOf(e: Entity, opts: MboxOptions, depth: number): Content {
  const type = parseParams(e.headers.get('content-type') ?? 'text/plain');
  const disposition = parseParams(e.headers.get('content-disposition'));
  const name = disposition.params.filename ?? type.params.name;
  const attached = disposition.value === 'attachment';
  const mime = type.value || 'text/plain';
  if (mime.startsWith('multipart/') && type.params.boundary && depth < 30) {
    const kids = splitParts(e.body, type.params.boundary).map((k) => contentOf(k, opts, depth + 1));
    const attachments = kids.flatMap((k) => k.attachments);
    if (mime === 'multipart/alternative') {
      const pick = kids.find((k) => k.text && !k.html) ?? kids.find((k) => k.text);
      return { text: pick?.text ?? '', html: pick?.html ?? false, attachments: [...new Set(attachments)] };
    }
    const said = kids.filter((k) => k.text);
    return { text: said.map((k) => k.text).join('\n\n'), html: said.length > 0 && said.every((k) => k.html), attachments };
  }
  if (mime === 'message/rfc822' && depth < 30) {
    const inner = entityOf(bodyBytes(e).toString('latin1').split(/\r?\n/));
    const c = contentOf(inner, opts, depth + 1);
    const head = [['From', 'from'], ['Date', 'date'], ['Subject', 'subject']]
      .map(([label, key]) => (inner.headers.has(key) ? `${label}: ${headerText(inner.headers.get(key))}` : ''))
      .filter(Boolean);
    return { text: [...head, '', c.text].join('\n').trim(), html: false, attachments: c.attachments };
  }
  if (!attached && (mime === 'text/plain' || mime === 'text/html' || mime === 'text/markdown')) {
    const raw = decodeBytes(bodyBytes(e), type.params.charset).replace(/\r\n?/g, '\n');
    if (mime === 'text/html') return { text: clean(htmlToText(raw, { dropQuotes: !opts.keepQuoted }), true, opts), html: true, attachments: [] };
    const flowed = type.params.format?.toLowerCase() === 'flowed';
    const text = flowed ? unflow(raw, type.params.delsp?.toLowerCase() === 'yes') : raw;
    return { text: clean(text, false, opts), html: false, attachments: [] };
  }
  return { text: '', html: false, attachments: name ? [name] : [] };
}

function splitParts(body: string[], boundary: string): Entity[] {
  const open = `--${boundary}`;
  const close = `--${boundary}--`;
  const parts: string[][] = [];
  let cur: string[] | null = null;
  for (const line of body) {
    const t = line.trimEnd();
    if (t === close) break;
    if (t === open) {
      if (cur) parts.push(cur);
      cur = [];
    } else if (cur) cur.push(line);
  }
  if (cur) parts.push(cur);
  return parts.map((p) => (p[0] === '' ? { headers: new Map<string, string>(), body: p.slice(1) } : entityOf(p)));
}

function ids(raw: string | undefined): string[] {
  const v = raw ?? '';
  const bracketed = [...v.matchAll(/<([^<>\s]+)>/g)].map((m) => m[1]);
  return bracketed.length ? bracketed : v.split(/\s+/).filter((x) => x.includes('@'));
}

/** `Name <address>` → Name, `address (Name)` → Name. By hand: `^\s*(\S+@\S+)\s*\(` is quadratic on `a@a@a@…`. */
function personOf(from: string): string {
  const t = from.trim();
  const lt = t.lastIndexOf('<');
  if (t.endsWith('>') && lt >= 0 && !t.slice(lt + 1, -1).includes('>')) {
    return t.slice(0, lt).trim().replace(/^"(.*)"$/, '$1').trim() || t.slice(lt + 1, -1).trim();
  }
  const open = t.indexOf('(');
  const address = open > 0 ? t.slice(0, open).trim() : '';
  if (t.endsWith(')') && t.length - open > 2 && /^\S+$/.test(address) && address.indexOf('@') > 0 && !address.endsWith('@')) {
    return t.slice(open + 1, -1).trim();
  }
  return t;
}

/** `(comments)` taken out of a header, up to its last `)`: past it, each `(` would scan to the end and fail. */
function dropComments(v: string): string {
  const end = v.lastIndexOf(')') + 1;
  return v.slice(0, end).replace(/\([^)]*\)/g, ' ') + v.slice(end);
}

/** One message's lines (raw bytes as Latin-1 characters, no line endings) as a record, or null when it says nothing. */
function messageRecord(lines: string[], fromLine: string, source: string, opts: MboxOptions): MemoryRecord | null {
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  const e = entityOf(lines);
  const h = (key: string): string => headerText(e.headers.get(key));
  const content = contentOf(e, opts, 0);
  const attachments = [...new Set(content.attachments)];
  const text = [content.text, attachments.length ? `Attachments: ${attachments.join(', ')}` : ''].filter(Boolean).join('\n\n');
  const subject = h('subject');
  if (!text && !subject) return null;
  const from = h('from');
  const date = dropComments(h('date')).trim();
  const at = timeOf(date) ?? timeOf(fromLine.replace(/^From \S+\s+/, '').trim());
  const own = ids(e.headers.get('message-id'))[0];
  const id = own ?? createHash('sha256').update([from, date, subject, text.slice(0, 4000)].join('\u0000')).digest('hex').slice(0, 32);
  const inReplyTo = ids(e.headers.get('in-reply-to'))[0];
  return record({
    id, title: subject, text, kind: 'email', source, collection: opts.collection, author: from ? personOf(from) : undefined, at,
    thread: ids(e.headers.get('references'))[0] ?? inReplyTo ?? id,
    metadata: { from, to: h('to'), cc: h('cc'), inReplyTo },
  });
}

/** Records from an mbox file, a message at a time. */
export async function* fromMbox(path: string, opts: MboxOptions = {}): AsyncIterable<MemoryRecord> {
  const source = opts.source ?? basename(path, extname(path));
  const max = opts.maxMessageBytes ?? 32 * 1024 * 1024;
  // Latin-1 keeps every byte as one character: each part is decoded later in its own charset.
  const lines = createInterface({ input: createReadStream(path, { encoding: 'latin1' }), crlfDelay: Infinity });
  let msg: string[] | null = null;
  let fromLine = '';
  let size = 0;
  let previousBlank = true;
  let index = 0;
  const finish = (): MemoryRecord | null => {
    if (!msg) return null;
    const r = messageRecord(msg, fromLine, source, opts);
    if (!r) opts.onSkip?.('no subject and no text', index);
    index++;
    return r;
  };
  for await (const line of lines) {
    // A new message: a `From ` line at the start or after an empty line, carrying a time (`From - Mon Jan 1 00:00:00 2024`).
    if (line.startsWith('From ') && previousBlank && (msg === null || /\d:\d\d/.test(line))) {
      const r = finish();
      if (r) yield r;
      msg = [];
      fromLine = line;
      size = 0;
      previousBlank = false;
      continue;
    }
    previousBlank = line === '' || line === '\r';
    if (msg === null) {
      if (!line.trim()) continue;
      msg = []; // a file that does not open with a `From ` line: one message from its first line
    }
    size += line.length + 1;
    if (size > max) continue;
    msg.push(/^>+From /.test(line) ? line.slice(1) : line);
  }
  const r = finish();
  if (r) yield r;
}

/** The record of one `.eml` file (an RFC 5322 message). */
export async function* fromEml(path: string, opts: MboxOptions = {}): AsyncIterable<MemoryRecord> {
  const lines = (await readFile(path)).toString('latin1').split(/\r?\n/);
  const fromLine = lines[0]?.startsWith('From ') ? (lines.shift() ?? '') : '';
  const r = messageRecord(lines, fromLine, opts.source ?? basename(path, extname(path)), opts);
  if (r) yield r;
  else opts.onSkip?.('no subject and no text', 0);
}
