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

/** Removes every `<tag …>…</tag>` (nested ones too), for `blockquote`: the quoted earlier message in an HTML reply. */
function dropElements(html: string, tag: string): string {
  const re = new RegExp(`<(/?)${tag}\\b[^>]*>`, 'gi');
  let out = '';
  let depth = 0;
  let from = 0;
  for (let m = re.exec(html); m; m = re.exec(html)) {
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

/** The `<title>` of an HTML page, else its first `<h1>`, as text; undefined when it has neither. */
export function htmlTitle(html: string): string | undefined {
  for (const re of [/<title\b[^>]*>([\s\S]*?)<\/title\s*>/i, /<h1\b[^>]*>([\s\S]*?)<\/h1\s*>/i]) {
    const m = re.exec(html);
    const t = m ? decodeEntities(m[1].replace(/<[^>]*>/g, ' ')).replace(/\s+/g, ' ').trim() : '';
    if (t) return t;
  }
  return undefined;
}

/**
 * HTML as plain text: scripts, styles, comments and the head dropped, blocks and line breaks kept as lines, list items
 * as `- `, tags removed, entities decoded, `<pre>` kept as written. `dropQuotes` also drops `<blockquote>`s.
 */
export function htmlToText(html: string, opts: { dropQuotes?: boolean } = {}): string {
  let s = html
    .replace(/<!--[\s\S]*?(-->|$)/g, ' ')
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/<(script|style|head|noscript|template|svg|math|iframe|object)\b[\s\S]*?<\/\1\s*>/gi, ' ')
    .replace(/<(script|style)\b[\s\S]*$/i, ' '); // one left open runs to the end
  if (opts.dropQuotes) s = dropElements(s, 'blockquote');
  // Whitespace in the source is one space, except inside <pre>.
  s = s.split(/(<pre\b[\s\S]*?<\/pre\s*>)/i)
    .map((piece, i) => (i % 2 ? piece.replace(/\r\n?/g, '\n') : piece.replace(/\s+/g, ' ')))
    .join('');
  s = s
    .replace(/<br\b[^>]*>/gi, '\n')
    .replace(/<li\b[^>]*>/gi, '\n- ')
    .replace(new RegExp(`</?(?:${BLOCK})\\b[^>]*>`, 'gi'), '\n')
    .replace(/<\/t[dh]\s*>/gi, '\t')
    .replace(/<\/?[a-zA-Z][^>]*>/g, '')
    .replace(/<[!?][^>]*>/g, '');
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
