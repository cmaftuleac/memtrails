"""
Secrets are scrubbed from what memory sends to a language model and from what consolidation keeps: provider keys,
tokens, credentials in URLs, private keys, e-mail local parts, JWTs, ``key=value`` secrets; home directories become
``~``; a long high-entropy run last. The records you add are kept as you wrote them.

The input can be anything a record held, so every pattern is bounded: no rule may take more than linear time on a
hostile string (a long run of what almost matches).

The patterns are the reference's JavaScript ones, read as JavaScript reads them: ``\\b``, ``\\d`` and case-insensitive
matching are ASCII-only (the ``ASCII`` flag), ``\\s`` is JavaScript's whitespace, ``$`` is the end of the text.
"""

from __future__ import annotations

from typing import Any

import regex

from .._js import JS_SPACE

_A = regex.ASCII
_AI = regex.ASCII | regex.IGNORECASE
_WS = f"[{JS_SPACE}]"
_NOT_WS = f"[^{JS_SPACE}]"
#: JavaScript's ``.``: anything but a line terminator.
_DOT = "[^\n\r  ]"

#: A name that says its value is a secret: ``password``, ``GITHUB_TOKEN``, ``AWS_SECRET_ACCESS_KEY``, ``PGPASSWORD``,
#: ``client_secret``, ``AccountKey``. The secret word must end the name (``token_count``, ``max_tokens`` are not
#: secrets), and any prefix is allowed (``_`` is a word character, so ``\btoken`` alone misses ``GITHUB_TOKEN``).
SECRET_NAME = (
    r"[A-Za-z0-9_.-]{0,64}?(?:api[_-]?key|apikey|(?:access|secret|private|account|signing|encryption|master|client)"
    r"[_-]?key|token|secret|password|passwd|passphrase|authorization)"
)
#: ``name=value``, ``name: value``, ``"name": "value"``, ``Authorization: token X`` (the scheme word is kept, its
#: credential not).
_KEY_VALUE = regex.compile(
    rf"\b({SECRET_NAME})([\"']?[ \t]{{0,16}}[=:][ \t]{{0,16}})(?:([\"'])((?:[^\"'\\\n]|\\{_DOT}){{1,512}})"
    rf"|((?:(?:Bearer|Basic|Digest|token|Bot)[ \t]{{1,16}})?)([^{JS_SPACE}&\"';]+))",
    _AI,
)

_SECRET_PATTERNS: list[tuple[regex.Pattern[str], str]] = [
    (regex.compile(r"\bsk-[A-Za-z0-9_-]{16,}", _A), "<redacted:sk>"),
    (regex.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}", _A), "<redacted:stripe>"),
    (regex.compile(r"\bpk_(?:live|test)_[A-Za-z0-9]{16,}", _A), "<redacted:stripe-pub>"),
    (regex.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}", _A), "<redacted:github>"),
    (regex.compile(r"\bgh[porsu]_[A-Za-z0-9]{20,}", _A), "<redacted:github>"),
    (regex.compile(r"\bglpat-[A-Za-z0-9_-]{20,}", _A), "<redacted:gitlab>"),
    (regex.compile(r"\bhf_[A-Za-z0-9]{30,}", _A), "<redacted:huggingface>"),
    (regex.compile(r"\bnpm_[A-Za-z0-9]{36,}", _A), "<redacted:npm>"),
    (regex.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}", _A), "<redacted:slack>"),
    (regex.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", _A), "<redacted:aws>"),
    (regex.compile(r"\bAIza[A-Za-z0-9_-]{20,}", _A), "<redacted:google>"),
    (regex.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}", _A), "<redacted:sendgrid>"),
    # A Telegram bot token: the bot's number, then 35 characters.
    (regex.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])", _A), "<redacted:telegram>"),
    # `scheme://user:password@host`: the password runs to the LAST `@` before the host, so `p@ss` and `pa/ss` go whole.
    (
        regex.compile(
            rf"\b([a-z][a-z0-9+.-]{{0,31}}://)([^{JS_SPACE}/:@]{{0,256}}):({_NOT_WS}{{1,256}})@"
            rf"(?=[^{JS_SPACE}@/?#]{{0,253}}(?:[/?#{JS_SPACE}]|\Z))",
            _AI,
        ),
        r"\1\2:<redacted>@",
    ),
    (regex.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+", _A), "https://hooks.slack.com/services/<redacted>"),
    (
        regex.compile(r"https://(?:canary\.|ptb\.)?discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_-]+", _A),
        "<redacted:discord-webhook>",
    ),
    # Local part and domain bounded as RFC 5321 bounds them: unbounded, a long run of `a.a.a…` without an `@` is quadratic.
    (regex.compile(r"\b[A-Za-z0-9._%+-]{1,64}@([A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,63})\b", _A), r"<redacted:email>@\1"),
    # Bounded: every `eyJ` in one long run is a start that scans it again. A longer JWT is caught as entropy below.
    (regex.compile(r"\beyJ[A-Za-z0-9_-]{8,1024}\.[A-Za-z0-9_-]{8,8192}\.[A-Za-z0-9_-]{8,}", _A), "<redacted:jwt>"),
    (regex.compile(rf"\b(Bearer|Basic){_WS}{{1,16}}[A-Za-z0-9._~+/=-]{{12,}}", _AI), r"\1 <redacted>"),
    (
        regex.compile(r"([?&](?:token|key|api_key|apikey|secret|password|access_token|sig)=)[^&" + JS_SPACE + "\"']+", _AI),
        r"\1<redacted>",
    ),
]

_HOME_DIR_PATTERNS: list[tuple[regex.Pattern[str], str]] = [
    (regex.compile(f"/Users/[^/{JS_SPACE}\"')\\]]+", _A), "~"),
    (regex.compile(f"/home/[^/{JS_SPACE}\"')\\]]+", _A), "~"),
    (regex.compile(f"[A-Za-z]:\\\\Users\\\\[^\\\\{JS_SPACE}\"')\\]]+", _AI), "~"),
]

#: Conservative on purpose (48+): digests and minified names are shorter, and a git sha (40 hex) stays.
_HIGH_ENTROPY = regex.compile(r"\b[A-Za-z0-9_-]{48,}\b", _A)

_PEM_BEGIN = regex.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----", _A)
_PEM_END = regex.compile(r"-----END [A-Z ]{0,40}PRIVATE KEY-----", _A)


def _drop_private_keys(s: str) -> str:
    """Every PEM private key; one left unterminated is cut to the end. A scan, not ``BEGIN[\\s\\S]*?END``: many BEGINs
    without an END made that quadratic."""
    if "PRIVATE KEY-----" not in s:
        return s
    out = ""
    start = 0
    pos = 0
    while True:
        m = _PEM_BEGIN.search(s, pos)
        if not m:
            break
        end = _PEM_END.search(s, m.end())
        out += f"{s[start : m.start()]}<redacted:private-key>"
        if not end:
            return out
        start = pos = end.end()
    return out + s[start:]


#: Where a secret word starts. A ``key=value`` match has one within 64 characters of its own start (its name's
#: prefix is at most 64), so the rule is tried only there: the same matches as a scan of every position, without
#: trying 64 prefixes at each place of a long text that holds no secret word at all.
_SECRET_WORD = regex.compile(
    r"(?=api[_-]?key|apikey|(?:access|secret|private|account|signing|encryption|master|client)[_-]?key|token|secret"
    r"|password|passwd|passphrase|authorization)",
    _AI,
)


def _key_values(s: str) -> str:
    out: list[str] = []
    done = 0
    tried = 0
    for w in _SECRET_WORD.finditer(s):
        start = max(done, tried, w.start() - 64)
        tried = w.start() + 1
        for p in range(start, w.start() + 1):
            m = _KEY_VALUE.match(s, p)
            if not m:
                continue
            name, sep, quote, scheme = m.group(1), m.group(2), m.group(3), m.group(5)
            out.append(s[done:p])
            out.append(f"{name}{sep}{quote}<redacted>" if quote else f"{name}{sep}{scheme or ''}<redacted>")
            done = m.end()
            break
    out.append(s[done:])
    return "".join(out)


def redact_string(value: Any) -> str:
    """Scrub one string. Never raises; returns ``''`` for anything that is not a string."""
    if not isinstance(value, str) or not value:
        return ""
    out = value
    try:
        out = _drop_private_keys(out)
        for pattern, replacement in _SECRET_PATTERNS:
            out = pattern.sub(replacement, out)
        out = _key_values(out)
        for pattern, replacement in _HOME_DIR_PATTERNS:
            out = pattern.sub(replacement, out)
        out = _HIGH_ENTROPY.sub("<redacted:entropy>", out)
    except Exception:  # noqa: BLE001 - a pathological input must not break the caller
        return out
    return out
