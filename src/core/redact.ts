/**
 * Secrets are scrubbed from what memory sends to a language model and from what consolidation keeps: provider keys,
 * tokens, credentials in URLs, private keys, e-mail local parts, JWTs, `key=value` secrets; home directories become `~`;
 * a long high-entropy run last. The records you add are kept as you wrote them.
 *
 * The input can be anything a record held, so every pattern is bounded: no rule may take more than linear time on a
 * hostile string (a long run of what almost matches).
 */

/**
 * A name that says its value is a secret: `password`, `GITHUB_TOKEN`, `AWS_SECRET_ACCESS_KEY`, `PGPASSWORD`,
 * `client_secret`, `AccountKey`. The secret word must end the name (`token_count`, `max_tokens` are not secrets), and
 * any prefix is allowed (`_` is a word character, so `\btoken` alone misses `GITHUB_TOKEN`).
 */
const SECRET_NAME = String.raw`[A-Za-z0-9_.-]{0,64}?(?:api[_-]?key|apikey|(?:access|secret|private|account|signing|encryption|master|client)[_-]?key|token|secret|password|passwd|passphrase|authorization)`;
/** `name=value`, `name: value`, `"name": "value"`, `Authorization: token X` (the scheme word is kept, its credential not). */
const KEY_VALUE = new RegExp(
  String.raw`\b(${SECRET_NAME})(["']?[ \t]{0,16}[=:][ \t]{0,16})(?:(["'])((?:[^"'\\\n]|\\.){1,512})|((?:(?:Bearer|Basic|Digest|token|Bot)[ \t]{1,16})?)([^\s&"';]+))`,
  'gi',
);

const SECRET_PATTERNS: Array<[RegExp, string]> = [
  [/\bsk-[A-Za-z0-9_-]{16,}/g, '<redacted:sk>'],
  [/\bsk_(?:live|test)_[A-Za-z0-9]{16,}/g, '<redacted:stripe>'],
  [/\bpk_(?:live|test)_[A-Za-z0-9]{16,}/g, '<redacted:stripe-pub>'],
  [/\bgithub_pat_[A-Za-z0-9_]{20,}/g, '<redacted:github>'],
  [/\bgh[porsu]_[A-Za-z0-9]{20,}/g, '<redacted:github>'],
  [/\bglpat-[A-Za-z0-9_-]{20,}/g, '<redacted:gitlab>'],
  [/\bhf_[A-Za-z0-9]{30,}/g, '<redacted:huggingface>'],
  [/\bnpm_[A-Za-z0-9]{36,}/g, '<redacted:npm>'],
  [/\bxox[abprs]-[A-Za-z0-9-]{10,}/g, '<redacted:slack>'],
  [/\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g, '<redacted:aws>'],
  [/\bAIza[A-Za-z0-9_-]{20,}/g, '<redacted:google>'],
  [/\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}/g, '<redacted:sendgrid>'],
  // A Telegram bot token: the bot's number, then 35 characters.
  [/\b\d{8,10}:[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])/g, '<redacted:telegram>'],
  // `scheme://user:password@host`: the password runs to the LAST `@` before the host, so `p@ss` and `pa/ss` go whole.
  [/\b([a-z][a-z0-9+.-]{0,31}:\/\/)([^\s/:@]{0,256}):([^\s]{1,256})@(?=[^\s@/?#]{0,253}(?:[/?#\s]|$))/gi, '$1$2:<redacted>@'],
  [/https:\/\/hooks\.slack\.com\/services\/[A-Za-z0-9/]+/g, 'https://hooks.slack.com/services/<redacted>'],
  [/https:\/\/(?:canary\.|ptb\.)?discord(?:app)?\.com\/api\/webhooks\/[0-9]+\/[A-Za-z0-9_-]+/g, '<redacted:discord-webhook>'],
  // Local part and domain bounded as RFC 5321 bounds them: unbounded, a long run of `a.a.a…` without an `@` is quadratic.
  [/\b[A-Za-z0-9._%+-]{1,64}@([A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,63})\b/g, '<redacted:email>@$1'],
  // Bounded: every `eyJ` in one long run is a start that scans it again. A longer JWT is caught as entropy below.
  [/\beyJ[A-Za-z0-9_-]{8,1024}\.[A-Za-z0-9_-]{8,8192}\.[A-Za-z0-9_-]{8,}/g, '<redacted:jwt>'],
  [/\b(Bearer|Basic)\s{1,16}[A-Za-z0-9._~+/=-]{12,}/gi, '$1 <redacted>'],
  [/([?&](?:token|key|api_key|apikey|secret|password|access_token|sig)=)[^&\s"']+/gi, '$1<redacted>'],
];

const HOME_DIR_PATTERNS: Array<[RegExp, string]> = [
  [/\/Users\/[^/\s"')\]]+/g, '~'],
  [/\/home\/[^/\s"')\]]+/g, '~'],
  [/[A-Za-z]:\\Users\\[^\\\s"')\]]+/gi, '~'],
];

/** Conservative on purpose (48+): digests and minified names are shorter, and a git sha (40 hex) stays. */
const HIGH_ENTROPY = /\b[A-Za-z0-9_-]{48,}\b/g;

const PEM_BEGIN = /-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----/g;
const PEM_END = /-----END [A-Z ]{0,40}PRIVATE KEY-----/g;

/** Every PEM private key; one left unterminated is cut to the end. A scan, not `BEGIN[\s\S]*?END`: many BEGINs without an END made that quadratic. */
function dropPrivateKeys(s: string): string {
  if (!s.includes('PRIVATE KEY-----')) return s;
  let out = '';
  let from = 0;
  PEM_BEGIN.lastIndex = 0;
  for (let m = PEM_BEGIN.exec(s); m; m = PEM_BEGIN.exec(s)) {
    PEM_END.lastIndex = m.index + m[0].length;
    const end = PEM_END.exec(s);
    out += `${s.slice(from, m.index)}<redacted:private-key>`;
    if (!end) return out;
    from = PEM_BEGIN.lastIndex = end.index + end[0].length;
  }
  return out + s.slice(from);
}

/** Scrub one string. Never throws; returns `''` for a non-string. */
export function redactString(input: unknown): string {
  if (typeof input !== 'string' || input.length === 0) return '';
  let out = input;
  try {
    out = dropPrivateKeys(out);
    for (const [re, replacement] of SECRET_PATTERNS) out = out.replace(re, replacement);
    out = out.replace(KEY_VALUE, (_w, name: string, sep: string, quote: string | undefined, _q, scheme: string | undefined) =>
      quote ? `${name}${sep}${quote}<redacted>` : `${name}${sep}${scheme ?? ''}<redacted>`);
    for (const [re, replacement] of HOME_DIR_PATTERNS) out = out.replace(re, replacement);
    out = out.replace(HIGH_ENTROPY, '<redacted:entropy>');
  } catch {
    /* a pathological input must not break the caller */
  }
  return out;
}
