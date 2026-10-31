/**
 * Secrets are scrubbed from what memory sends to a language model and from what consolidation keeps: provider keys,
 * tokens, credentials in URLs, private keys, e-mail local parts, JWTs, `key=value` secrets; home directories become `~`;
 * a long high-entropy run last. The records you add are kept as you wrote them.
 */
const SECRET_PATTERNS: Array<[RegExp, string]> = [
  [/\bsk-[A-Za-z0-9_-]{16,}/g, '<redacted:sk>'],
  [/\bsk_(?:live|test)_[A-Za-z0-9]{16,}/g, '<redacted:stripe>'],
  [/\bpk_(?:live|test)_[A-Za-z0-9]{16,}/g, '<redacted:stripe-pub>'],
  [/\bgithub_pat_[A-Za-z0-9_]{20,}/g, '<redacted:github>'],
  [/\bgh[porsu]_[A-Za-z0-9]{20,}/g, '<redacted:github>'],
  [/\bxox[abprs]-[A-Za-z0-9-]{10,}/g, '<redacted:slack>'],
  [/\bAKIA[0-9A-Z]{16}\b/g, '<redacted:aws>'],
  [/\bAIza[A-Za-z0-9_-]{20,}/g, '<redacted:google>'],
  [/\b([a-z][a-z0-9+.-]*:\/\/)([^\s/:@]*):([^\s/@]+)@/gi, '$1$2:<redacted>@'],
  [/https:\/\/hooks\.slack\.com\/services\/[A-Za-z0-9/]+/g, 'https://hooks.slack.com/services/<redacted>'],
  [/https:\/\/(?:canary\.|ptb\.)?discord(?:app)?\.com\/api\/webhooks\/[0-9]+\/[A-Za-z0-9_-]+/g, '<redacted:discord-webhook>'],
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, '<redacted:private-key>'],
  [/\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b/g, '<redacted:email>@$1'],
  [/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g, '<redacted:jwt>'],
  [/\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}/gi, '$1 <redacted>'],
  [/([?&](?:token|key|api_key|apikey|secret|password|access_token)=)[^&\s"']+/gi, '$1<redacted>'],
  [/\b(api[_-]?key|token|secret|password|passwd|authorization)\b(\s*[=:]\s*)[^\s&"']+/gi, '$1$2<redacted>'],
];

const HOME_DIR_PATTERNS: Array<[RegExp, string]> = [
  [/\/Users\/[^/\s"')\]]+/g, '~'],
  [/\/home\/[^/\s"')\]]+/g, '~'],
  [/[A-Za-z]:\\Users\\[^\\\s"')\]]+/gi, '~'],
];

/** Conservative on purpose (48+): digests and minified names are shorter. */
const HIGH_ENTROPY = /\b[A-Za-z0-9_-]{48,}\b/g;

/** Scrub one string. Never throws; returns `''` for a non-string. */
export function redactString(input: unknown): string {
  if (typeof input !== 'string' || input.length === 0) return '';
  let out = input;
  try {
    for (const [re, replacement] of SECRET_PATTERNS) out = out.replace(re, replacement);
    for (const [re, replacement] of HOME_DIR_PATTERNS) out = out.replace(re, replacement);
    out = out.replace(HIGH_ENTROPY, '<redacted:entropy>');
  } catch {
    /* a pathological input must not break the caller */
  }
  return out;
}
