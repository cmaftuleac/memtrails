/**
 * Small text tools the importers share: HTML reduced to text, character entities, bytes in any charset, long texts cut
 * into parts. No dependency; any language and script is kept as written.
 */
import type { MemoryRecord } from '../records.js';

const NAMED: Record<string, string> = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ensp: ' ', emsp: ' ', thinsp: ' ', shy: '',
  zwnj: '‌', zwj: '‍', lrm: '‎', rlm: '‏',
  copy: '©', reg: '®', trade: '™', hellip: '…', mdash: '—', ndash: '–', minus: '−',
  lsquo: '‘', rsquo: '’', sbquo: '‚', ldquo: '“', rdquo: '”', bdquo: '„', laquo: '«', raquo: '»', lsaquo: '‹', rsaquo: '›',
  euro: '€', pound: '£', yen: '¥', cent: '¢', curren: '¤', deg: '°', middot: '·', bull: '•', times: '×', divide: '÷',
  sect: '§', para: '¶', iexcl: '¡', iquest: '¿', szlig: 'ß', aelig: 'æ', AElig: 'Æ', oelig: 'œ', OElig: 'Œ',
  oslash: 'ø', Oslash: 'Ø', eth: 'ð', ETH: 'Ð', thorn: 'þ', THORN: 'Þ', ordf: 'ª', ordm: 'º', micro: 'µ', plusmn: '±',
  frac12: '½', frac14: '¼', frac34: '¾', sup1: '¹', sup2: '²', sup3: '³', dagger: '†', Dagger: '‡', permil: '‰',
  prime: '′', Prime: '″', le: '≤', ge: '≥', ne: '≠', infin: '∞', asymp: '≈', larr: '←', rarr: '→', uarr: '↑', darr: '↓',
  harr: '↔', hearts: '♥', not: '¬', brvbar: '¦', uml: '¨', acute: '´', cedil: '¸', macr: '¯', circ: 'ˆ', tilde: '˜',
};
// Greek letters by name (Alpha … Omega, alpha … omega, sigmaf).
const GREEK = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi',
  'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'];
GREEK.forEach((name, i) => {
  const offset = i + (i >= 17 ? 1 : 0);
  NAMED[name] = String.fromCodePoint(0x391 + offset);
  NAMED[name.toLowerCase()] = String.fromCodePoint(0x3b1 + offset);
});
NAMED.sigmaf = 'ς';
// A Latin letter with a mark: &eacute; &Uuml; &ccedil; &scaron; … composed from the letter and a combining mark.
const MARKS: Record<string, string> = {
  acute: '́', grave: '̀', circ: '̂', tilde: '̃', uml: '̈', ring: '̊', cedil: '̧',
  caron: '̌', macr: '̄', breve: '̆', ogon: '̨', dot: '̇', dblac: '̋',
};
const win1252 = new TextDecoder('windows-1252');

function namedEntity(name: string): string | undefined {
  if (name in NAMED) return NAMED[name];
  const m = /^([a-zA-Z])(acute|grave|circ|tilde|uml|ring|cedil|caron|macr|breve|ogon|dot|dblac)$/.exec(name);
  if (!m) return undefined;
  const composed = (m[1] + MARKS[m[2]]).normalize('NFC');
  return [...composed].length === 1 ? composed : undefined;
}

function codePoint(n: number): string {
  if (n >= 0x80 && n <= 0x9f) return win1252.decode(new Uint8Array([n])); // what browsers do with &#150; and the like
  if (!Number.isFinite(n) || n <= 0 || n > 0x10ffff || (n >= 0xd800 && n <= 0xdfff)) return '�';
  return String.fromCodePoint(n);
}

/** `&amp;`, `&#233;`, `&#x1F600;`, `&eacute;` and the other common names, decoded; unknown names are left as written. */
export function decodeEntities(s: string): string {
  if (!s.includes('&')) return s;
  return s.replace(/&(#[xX][0-9a-fA-F]{1,6}|#\d{1,7}|[a-zA-Z][a-zA-Z0-9]{1,31});/g, (whole, ref: string) => {
    if (ref[0] === '#') return codePoint(ref[1] === 'x' || ref[1] === 'X' ? parseInt(ref.slice(2), 16) : parseInt(ref.slice(1), 10));
    return namedEntity(ref) ?? whole;
  });
}

const BLOCK = 'address|article|aside|blockquote|caption|center|dd|details|dialog|div|dl|dt|fieldset|figcaption|figure|footer|form|' +
  'h[1-6]|header|hr|main|nav|ol|p|pre|section|summary|table|tbody|thead|tfoot|tr|ul';

/**
 * `f` applied to `s` up to its last `>`, the rest kept. Every tag pattern here ends in `[^>]*>`: past the last `>` each
 * `<` is a start that scans to the end and fails, so a run of unclosed `<a<a<a…` was quadratic; before it, each scan
 * stops at the next `>` and matches.
 */
function beforeLastGt(s: string, f: (head: string) => string): string {
  const end = s.lastIndexOf('>') + 1;
  return end === s.length ? f(s) : f(s.slice(0, end)) + s.slice(end);
}

/** Removes every `<tag …>…</tag>` (nested ones too), for `blockquote`: the quoted earlier message in an HTML reply. */
function dropElements(html: string, tag: string): string {
  const re = new RegExp(`<(/?)${tag}\\b[^>]*>`, 'gi');
  const head = html.slice(0, html.lastIndexOf('>') + 1); // see beforeLastGt
  let out = '';
  let depth = 0;
  let from = 0;
  for (let m = re.exec(head); m; m = re.exec(head)) {
    if (!m[1]) {
      if (depth === 0) out += html.slice(from, m.index);
      depth++;
    } else if (depth > 0) {
      depth--;
      if (depth === 0) from = m.index + m[0].length;
    }
  }
  return depth === 0 ? out + html.slice(from) : out;
}

/**
 * Each `<tag …>…</tag>` found by `open` (the tag's name in group 1) replaced by `f(name, element)`, scanning once:
 * `<(script|…)\b[\s\S]*?<\/\1>` rescanned to the end for every unclosed opening tag, quadratic on `<script<script…`.
 * An unclosed one has `element` null; `f` then gives what replaces it and everything after it, or null to leave it as
 * written. A tag with no closing tag after it has none after a later opening tag either, so it is looked for once.
 * The text between elements goes through `outside`.
 */
function replaceElements(s: string, open: RegExp, f: (name: string, element: string | null) => string | null,
  outside: (text: string) => string = (t) => t): string {
  const closes = new Map<string, RegExp>();
  const unclosed = new Set<string>();
  let out = '';
  let from = 0;
  open.lastIndex = 0;
  for (let m = open.exec(s); m; m = open.exec(s)) {
    const name = m[1].toLowerCase();
    let end = -1;
    if (!unclosed.has(name)) {
      let close = closes.get(name);
      if (!close) closes.set(name, (close = new RegExp(`</${name}\\s*>`, 'gi')));
      close.lastIndex = m.index + m[0].length;
      const c = close.exec(s);
      if (c) end = c.index + c[0].length;
      else unclosed.add(name);
    }
    const replaced = f(name, end < 0 ? null : s.slice(m.index, end));
    if (replaced === null) continue; // left as written: look on after the opening tag
    out += outside(s.slice(from, m.index)) + replaced;
    if (end < 0) return out;
    from = open.lastIndex = end;
  }
  return out + outside(s.slice(from));
}

/** `<![CDATA[…]]>` as its text; an unclosed one, and everything after it, left as written. */
function unwrapCdata(s: string): string {
  let out = '';
  let from = 0;
  for (let i = s.indexOf('<![CDATA[', from); i >= 0; i = s.indexOf('<![CDATA[', from)) {
    const j = s.indexOf(']]>', i + 9);
    if (j < 0) break;
    out += s.slice(from, i) + s.slice(i + 9, j);
    from = j + 3;
  }
  return out + s.slice(from);
}

const HIDDEN = /<(script|style|head|noscript|template|svg|math|iframe|object)\b/gi;
const PRE = /<(pre)\b/gi;

/** Title regexes see at most this much of a page: a title is near the top, and a hostile page is not read to its end. */
const TITLE_SCAN = 64 * 1024;

/** The `<title>` of an HTML page, else its first `<h1>`, as text; undefined when it has neither. */
export function htmlTitle(html: string): string | undefined {
  const top = html.slice(0, TITLE_SCAN);
  const head = top.slice(0, top.lastIndexOf('>') + 1); // see beforeLastGt
  for (const tag of ['title', 'h1']) {
    const open = new RegExp(`<${tag}\\b[^>]*>`, 'i').exec(head);
    if (!open) continue;
    const bodyStart = open.index + open[0].length;
    const close = new RegExp(`</${tag}\\s*>`, 'gi');
    close.lastIndex = bodyStart;
    const end = close.exec(head);
    if (!end) continue;
    const inner = beforeLastGt(head.slice(bodyStart, end.index), (h) => h.replace(/<[^>]*>/g, ' '));
    const t = decodeEntities(inner).replace(/\s+/g, ' ').trim();
    if (t) return t;
  }
  return undefined;
}

/**
 * HTML as plain text: scripts, styles, comments and the head dropped, blocks and line breaks kept as lines, list items
 * as `- `, tags removed, entities decoded, `<pre>` kept as written. `dropQuotes` also drops `<blockquote>`s.
 */
export function htmlToText(html: string, opts: { dropQuotes?: boolean } = {}): string {
  let s = unwrapCdata(html.replace(/<!--[\s\S]*?(-->|$)/g, ' '));
  // Hidden elements dropped; a script or style left open runs to the end, any other unclosed one is left to the tag rules.
  s = replaceElements(s, HIDDEN, (name, element) => (element !== null || name === 'script' || name === 'style' ? ' ' : null));
  if (opts.dropQuotes) s = dropElements(s, 'blockquote');
  // Whitespace in the source is one space, except inside <pre>.
  s = replaceElements(s, PRE, (_name, element) => element?.replace(/\r\n?/g, '\n') ?? null, (t) => t.replace(/\s+/g, ' '));
  s = beforeLastGt(s, (h) => h
    .replace(/<br\b[^>]*>/gi, '\n')
    .replace(/<li\b[^>]*>/gi, '\n- ')
    .replace(new RegExp(`</?(?:${BLOCK})\\b[^>]*>`, 'gi'), '\n')
    .replace(/<\/t[dh]\s*>/gi, '\t')
    .replace(/<\/?[a-zA-Z][^>]*>/g, '')
    .replace(/<[!?][^>]*>/g, ''));
  s = decodeEntities(s);
  return tidy(s);
}

/** Lines trimmed at the end, spaces and tabs inside a line kept to one, at most one empty line in a row. */
export function tidy(s: string): string {
  return s
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((l) => l.replace(/[ \t ]+/g, ' ').replace(/^ (?=\S)/, '').trimEnd())
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** Bytes as text in the charset named (any the platform knows); unnamed, ASCII or unknown: UTF-8, else Windows-1252. */
export function decodeBytes(bytes: Uint8Array, charset?: string): string {
  const label = (charset ?? '').trim().replace(/^["']|["']$/g, '').replace(/\*.*$/, '').toLowerCase();
  if (label && !/^(us-?ascii|ascii|ansi_x3\.4-1968|utf-?8|unicode-1-1-utf-8|x-unknown|unknown(-8bit)?|default)$/.test(label)) {
    try {
      return new TextDecoder(label).decode(bytes);
    } catch {
      // A charset the platform does not know: read it as below.
    }
  }
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return win1252.decode(bytes);
  }
}

/**
 * A long text in parts of at most `maxChars`, each cut at a paragraph, else a line, else a sentence's end (in any
 * script), else a space; never inside a character.
 */
export function splitText(text: string, maxChars: number): string[] {
  const max = Math.max(100, Math.floor(maxChars));
  if (text.length <= max) return [text];
  const parts: string[] = [];
  let rest = text;
  while (rest.length > max) {
    const window = rest.slice(0, max);
    const half = Math.floor(max / 2);
    let cut = window.lastIndexOf('\n\n');
    if (cut < half) cut = window.lastIndexOf('\n');
    if (cut < half) {
      const ends = [...window.slice(half).matchAll(/\p{Sentence_Terminal}+/gu)].at(-1);
      cut = ends ? half + ends.index + ends[0].length : -1;
    }
    if (cut < half) cut = window.search(/\s\S*$/);
    if (cut < half) cut = max;
    const code = rest.charCodeAt(cut - 1);
    if (code >= 0xd800 && code <= 0xdbff) cut--; // keep a surrogate pair together
    const part = rest.slice(0, cut).trim();
    if (part) parts.push(part);
    rest = rest.slice(cut);
  }
  if (rest.trim()) parts.push(rest.trim());
  return parts;
}

/** A record whose fields, and metadata values, may be left undefined. */
export type RecordDraft = Omit<MemoryRecord, 'metadata'> & { metadata?: Record<string, string | number | boolean | null | undefined> };

/** A record without its empty fields. */
export function record(r: RecordDraft): MemoryRecord {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(r)) {
    if (v === undefined || (v === '' && k !== 'text')) continue;
    if (k === 'metadata' && v && typeof v === 'object') {
      const m = Object.fromEntries(Object.entries(v).filter(([, x]) => x !== undefined && x !== ''));
      if (Object.keys(m).length) out.metadata = m;
      continue;
    }
    out[k] = v;
  }
  return out as unknown as MemoryRecord;
}
