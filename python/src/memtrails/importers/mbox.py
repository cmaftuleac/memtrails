"""
E-mail: an mbox file (RFC 4155: each message opens with a ``From `` line; ``>From `` in a body is unescaped), read a
message at a time, or one ``.eml`` file. Headers are unfolded and their RFC 2047 encoded-words (Q and B, any charset)
decoded; MIME parts are taken apart (text/plain preferred, else text/html reduced to text), base64 and quoted-printable
decoded, and every charset the platform knows read as such.

id: the Message-ID (else a hash of the message). title: the Subject. text: the body, with the names of attachments after
it; lines quoted from earlier messages (``>``) and the signature (after ``-- ``) are left out unless ``keep_quoted`` /
``keep_signature``, since the earlier message is a record of its own. author: the sender's name, else address. at: the
Date. kind ``email``. thread: the first id in References, else In-Reply-To, else the message's own id. metadata
``{from, to, cc, inReplyTo}``.
"""

from __future__ import annotations

import asyncio
import binascii
import hashlib
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import regex

from .._js import JS_SPACE, UNDEFINED, js_trim, js_trim_end, u16slice, utf8
from ..records import time_of
from .text import LineTooLong, decode_bytes, html_to_text, read_lines, record, undef

_WS = regex.compile(f"[{JS_SPACE}]+")


def latin1_bytes(s: str) -> bytes:
    """``Buffer.from(s, 'latin1')``: each character's low byte."""
    return bytes(ord(c) & 0xFF for c in s)


_B64 = regex.compile(r"[A-Za-z0-9+/\-_]")


def base64_bytes(s: str) -> bytes:
    """``Buffer.from(s, 'base64')``: lenient, as Node is — anything not base64 skipped, the URL-safe alphabet read too,
    padding optional, a stray last character dropped."""
    chars = "".join(_B64.findall(s.split("=", 1)[0])).replace("-", "+").replace("_", "/")
    if len(chars) % 4 == 1:
        chars = chars[:-1]
    try:
        return binascii.a2b_base64(chars + "=" * (-len(chars) % 4))
    except binascii.Error:
        return b""


@dataclass
class _Entity:
    """One MIME entity: headers (first of each name, raw bytes as Latin-1 characters, unfolded) and body lines."""

    headers: dict[str, str]
    body: list[str]


@dataclass
class _Content:
    text: str
    html: bool
    attachments: list[str] = field(default_factory=list)


@dataclass
class _Options:
    keep_quoted: bool
    keep_signature: bool


def _entity_of(lines: list[str]) -> _Entity:
    headers: dict[str, str] = {}
    i = 0
    name = ""
    while i < len(lines):
        line = lines[i]
        if line in ("", "\r"):
            break
        if line[:1] in (" ", "\t"):
            if name:
                headers[name] = headers[name] + " " + line.lstrip(" \t")
            i += 1
            continue
        colon = line.find(":")
        if colon <= 0:
            i += 1
            continue
        key = js_trim(line[:colon]).lower()
        if key in headers:
            name = ""  # a later copy of a header: the first is kept
            i += 1
            continue
        name = key
        headers[key] = js_trim(line[colon + 1 :])
        i += 1
    return _Entity(headers, lines[i + 1 :])


_HEX2 = regex.compile(r"^[0-9A-Fa-f]{2}$")


def _q_bytes(s: str) -> bytes:
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "_":
            out.append(0x20)
        elif s[i] == "=" and _HEX2.match(s[i + 1 : i + 3]):
            out.append(int(s[i + 1 : i + 3], 16))
            i += 2
        else:
            out.append(ord(s[i]) & 0xFF)
        i += 1
    return bytes(out)


_ENCODED_WORD = regex.compile(r"=\?([^?\s]+)\?([bBqQ])\?([^?\s]*)\?=")


def decode_encoded_words(s: str) -> str:
    """RFC 2047 encoded-words decoded; words side by side join with no space, and their bytes are read together."""
    out = ""
    last = 0
    pending: list[Any] | None = None  # [charset, bytearray]

    def flush() -> None:
        nonlocal out, pending
        if pending is not None:
            out += decode_bytes(bytes(pending[1]), pending[0])
        pending = None

    for m in _ENCODED_WORD.finditer(s):
        gap = s[last : m.start()]
        if pending is None or js_trim(gap) or (gap != "" and not regex.fullmatch(f"[{JS_SPACE}]*", gap)):
            flush()
            out += gap
        charset = regex.sub(r"\*.*$", "", m.group(1)).lower()
        data = base64_bytes(m.group(3)) if m.group(2).upper() == "B" else _q_bytes(m.group(3))
        if pending is not None and pending[0] != charset:
            flush()
        if pending is not None:
            pending[1] += data
        else:
            pending = [charset, bytearray(data)]
        last = m.end()
    flush()
    return out + s[last:]


def header_text(raw: str | None) -> str:
    """A raw header as text: bytes sent unencoded read as UTF-8 (else Windows-1252), then encoded-words decoded."""
    if not raw:
        return ""
    return js_trim(_WS.sub(" ", decode_encoded_words(decode_bytes(latin1_bytes(raw)))))


_RFC2231 = regex.compile(r"^(.+?)\*([0-9]+)?(\*)?$")
_PERCENT = regex.compile(r"%([0-9A-Fa-f]{2})")


def _parse_params(raw: str | None) -> tuple[str, dict[str, str]]:
    """``type/sub; a=1; b="x y"`` as its value and parameters (RFC 2231 continued and encoded ones joined and decoded)."""
    pieces: list[str] = []
    cur = ""
    quoted = False
    for ch in raw or "":
        if ch == '"':
            quoted = not quoted
        if ch == ";" and not quoted:
            pieces.append(cur)
            cur = ""
        else:
            cur += ch
    pieces.append(cur)
    value = js_trim(pieces.pop(0) if pieces else "").lower()
    params: dict[str, str] = {}
    sections: dict[str, list[tuple[int, bool, str]]] = {}
    for p in pieces:
        eq = p.find("=")
        if eq <= 0:
            continue
        key = js_trim(p[:eq]).lower()
        v = js_trim(p[eq + 1 :])
        if v.startswith('"') and v.endswith('"') and len(v) > 1:
            v = regex.sub(r"\\([^\n\r\u2028\u2029])", r"\1", v[1:-1])
        ext = _RFC2231.match(key)
        if ext:
            sections.setdefault(ext.group(1), []).append((int(ext.group(2) or 0), ext.group(2) is None or bool(ext.group(3)), v))
        else:
            params[key] = header_text(v) if key in ("name", "filename") else v
    for key, listed in sections.items():
        listed.sort(key=lambda s: s[0])
        charset = ""
        data = bytearray()
        for i, (_, encoded, val) in enumerate(listed):
            v = val
            if encoded and i == 0:
                m = regex.match(r"^([^']*)'[^']*'(.*)$", v, regex.DOTALL)
                if m:
                    charset, v = m.group(1), m.group(2)
            if encoded:
                data += latin1_bytes(_PERCENT.sub(lambda h: chr(int(h.group(1), 16)), v))
            else:
                data += latin1_bytes(v)
        params[key] = decode_bytes(bytes(data), charset)
    return value, params


def _quoted_printable(lines: list[str]) -> bytes:
    s = ""
    for i, raw in enumerate(lines):
        end = len(raw)
        while end > 0 and raw[end - 1] in " \t\r":
            end -= 1  # /[ \t\r]+$/ is quadratic on a run of spaces
        line = raw[:end]
        if line.endswith("="):
            s += line[:-1]
        else:
            s += f"{line}\n" if i < len(lines) - 1 else line
    return latin1_bytes(regex.sub(r"=([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), s))


def _body_bytes(e: _Entity) -> bytes:
    cte = js_trim(e.headers.get("content-transfer-encoding", "")).lower()
    if cte == "base64":
        return base64_bytes("".join(e.body))
    if cte == "quoted-printable":
        return _quoted_printable(e.body)
    return latin1_bytes("\n".join(e.body))


def _unflow(text: str, delsp: bool) -> str:
    """``format=flowed`` (RFC 3676) text with its soft line breaks joined."""
    out: list[str] = []
    cur: str | None = None
    for raw in text.split("\n"):
        quote = regex.match(r"^>+", raw)
        q = quote.group(0) if quote else ""
        line = raw[len(q) :]
        if line.startswith(" "):
            line = line[1:]
        soft = line.endswith(" ") and line != "-- "
        if soft and delsp:
            line = line[:-1]
        cur = (f"{q} {line}" if q else line) if cur is None else cur + line
        if not soft:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return "\n".join(out)


_QUOTED = regex.compile(r"^ ?>")


def _clean(text: str, html: bool, opts: _Options) -> str:
    lines = text.split("\n")
    if not opts.keep_signature:
        sig = next((i for i, x in enumerate(lines) if x == "-- " or (html and x == "--")), -1)
        if sig >= 0:
            lines = lines[:sig]
    if not opts.keep_quoted:
        lines = [x for x in lines if not _QUOTED.match(x)]
    joined = "\n".join(x if x == "-- " else js_trim_end(x) for x in lines)
    return js_trim(regex.sub(r"\n{3,}", "\n\n", joined))


def _content_of(e: _Entity, opts: _Options, depth: int) -> _Content:
    mime, tparams = _parse_params(e.headers.get("content-type", "text/plain"))
    disposition, dparams = _parse_params(e.headers.get("content-disposition"))
    name = dparams.get("filename") or tparams.get("name")
    attached = disposition == "attachment"
    mime = mime or "text/plain"
    if mime.startswith("multipart/") and tparams.get("boundary") and depth < 30:
        kids = [_content_of(k, opts, depth + 1) for k in _split_parts(e.body, tparams["boundary"])]
        attachments = [a for k in kids for a in k.attachments]
        if mime == "multipart/alternative":
            pick = next((k for k in kids if k.text and not k.html), None) or next((k for k in kids if k.text), None)
            return _Content(pick.text if pick else "", pick.html if pick else False, list(dict.fromkeys(attachments)))
        said = [k for k in kids if k.text]
        return _Content("\n\n".join(k.text for k in said), bool(said) and all(k.html for k in said), attachments)
    if mime == "message/rfc822" and depth < 30:
        inner = _entity_of(regex.split(r"\r?\n", _body_bytes(e).decode("latin-1")))
        c = _content_of(inner, opts, depth + 1)
        head = [
            f"{label}: {header_text(inner.headers.get(key))}"
            for label, key in (("From", "from"), ("Date", "date"), ("Subject", "subject"))
            if key in inner.headers
        ]
        return _Content(js_trim("\n".join([*head, "", c.text])), False, c.attachments)
    if not attached and mime in ("text/plain", "text/html", "text/markdown"):
        raw = regex.sub(r"\r\n?", "\n", decode_bytes(_body_bytes(e), tparams.get("charset")))
        if mime == "text/html":
            return _Content(_clean(html_to_text(raw, drop_quotes=not opts.keep_quoted), True, opts), True)
        flowed = (tparams.get("format") or "").lower() == "flowed"
        text = _unflow(raw, (tparams.get("delsp") or "").lower() == "yes") if flowed else raw
        return _Content(_clean(text, False, opts), False)
    return _Content("", False, [name] if name else [])


def _split_parts(body: list[str], boundary: str) -> list[_Entity]:
    opening = f"--{boundary}"
    closing = f"--{boundary}--"
    parts: list[list[str]] = []
    cur: list[str] | None = None
    for line in body:
        t = js_trim_end(line)
        if t == closing:
            break
        if t == opening:
            if cur is not None:
                parts.append(cur)
            cur = []
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        parts.append(cur)
    return [_Entity({}, p[1:]) if p and p[0] == "" else _entity_of(p) for p in parts]


_BRACKETED = regex.compile(f"<([^<>{JS_SPACE}]+)>")


def _ids(raw: str | None) -> list[str]:
    v = raw or ""
    bracketed = [m.group(1) for m in _BRACKETED.finditer(v)]
    return bracketed or [x for x in regex.split(f"[{JS_SPACE}]+", v) if "@" in x]


def _person_of(sender: str) -> str:
    """``Name <address>`` → Name, ``address (Name)`` → Name. By hand: ``^\\s*(\\S+@\\S+)\\s*\\(`` is quadratic on
    ``a@a@a@…``."""
    t = js_trim(sender)
    lt = t.rfind("<")
    if t.endswith(">") and lt >= 0 and ">" not in t[lt + 1 : -1]:
        name = js_trim(regex.sub(r'^"(.*)"$', r"\1", js_trim(t[:lt]), flags=regex.DOTALL))
        return name or js_trim(t[lt + 1 : -1])
    opening = t.find("(")
    address = js_trim(t[:opening]) if opening > 0 else ""
    if (
        t.endswith(")")
        and len(t) - opening > 2
        and address
        and not regex.search(f"[{JS_SPACE}]", address)
        and address.find("@") > 0
        and not address.endswith("@")
    ):
        return js_trim(t[opening + 1 : -1])
    return t


def _drop_comments(v: str) -> str:
    """``(comments)`` taken out of a header, up to its last ``)``: past it, each ``(`` would scan to the end and fail."""
    end = v.rfind(")") + 1
    return regex.sub(r"\([^)]*\)", " ", v[:end]) + v[end:]


def _message_record(lines: list[str], from_line: str, source: str, opts: _Options, collection: str | None) -> dict[str, Any] | None:
    """One message's lines (raw bytes as Latin-1 characters, no line endings) as a record, or None when it says nothing."""
    while lines and not js_trim(lines[-1]):
        lines.pop()
    e = _entity_of(lines)

    def h(key: str) -> str:
        return header_text(e.headers.get(key))

    content = _content_of(e, opts, 0)
    attachments = list(dict.fromkeys(content.attachments))
    text = "\n\n".join(x for x in (content.text, f"Attachments: {', '.join(attachments)}" if attachments else "") if x)
    subject = h("subject")
    if not text and not subject:
        return None
    sender = h("from")
    date = js_trim(_drop_comments(h("date")))
    at = time_of(date)
    if at is None:
        at = time_of(js_trim(regex.sub(r"^From [^\s]+\s+", "", from_line)))
    own = next(iter(_ids(e.headers.get("message-id"))), None)
    rid = own or hashlib.sha256(utf8("\0".join([sender, date, subject, u16slice(text, 0, 4000)]))).hexdigest()[:32]
    in_reply_to = next(iter(_ids(e.headers.get("in-reply-to"))), None)
    return record(
        {
            "id": rid,
            "title": subject,
            "text": text,
            "kind": "email",
            "source": source,
            "collection": undef(collection),
            "author": _person_of(sender) if sender else UNDEFINED,
            "at": undef(at),
            "thread": next(iter(_ids(e.headers.get("references"))), None) or in_reply_to or rid,
            "metadata": {"from": sender, "to": h("to"), "cc": h("cc"), "inReplyTo": undef(in_reply_to)},
        }
    )


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


_TIME_IN_FROM = regex.compile(r"[0-9]:[0-9][0-9]")
_ESCAPED_FROM = regex.compile(r"^>+From ")


async def from_mbox(
    path: str | os.PathLike[str],
    keep_quoted: bool = False,
    keep_signature: bool = False,
    source: str | None = None,
    collection: str | None = None,
    max_message_bytes: int = 32 * 1024 * 1024,
    on_skip: Callable[[str, int], None] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Records from an mbox file, a message at a time (``async for record in from_mbox(path)``). ``keep_quoted``: keep lines quoted from earlier messages;
    ``keep_signature``: keep what follows a ``-- `` line; ``source``: default the file's name without its extension; a
    message's body past ``max_message_bytes`` is not read (its headers are); ``on_skip(reason, index)`` for each message
    left out."""
    p = os.fspath(path)
    src = source if source is not None else _stem(p)
    opts = _Options(keep_quoted, keep_signature)
    msg: list[str] | None = None
    from_line = ""
    size = 0
    previous_blank = True
    index = 0

    def finish() -> dict[str, Any] | None:
        nonlocal index
        if msg is None:
            return None
        r = _message_record(msg, from_line, src, opts, collection)
        if r is None and on_skip:
            on_skip("no subject and no text", index)
        index += 1
        return r

    # Latin-1 keeps every byte as one character: each part is decoded later in its own charset.
    async for line in read_lines(p, "latin-1"):
        # A line past the bound (64 MiB): the message is past its size from here, as a long one is.
        if isinstance(line, LineTooLong):
            size = max_message_bytes + 1
            previous_blank = False
            continue
        # A new message: a `From ` line at the start or after an empty line, carrying a time.
        if line.startswith("From ") and previous_blank and (msg is None or _TIME_IN_FROM.search(line)):
            r = finish()
            if r:
                yield r
            msg = []
            from_line = line
            size = 0
            previous_blank = False
            continue
        previous_blank = line in ("", "\r")
        if msg is None:
            if not js_trim(line):
                continue
            msg = []  # a file that does not open with a `From ` line: one message from its first line
        size += len(line) + 1
        if size > max_message_bytes:
            continue
        msg.append(line[1:] if _ESCAPED_FROM.match(line) else line)
    r = finish()
    if r:
        yield r


async def from_eml(
    path: str | os.PathLike[str],
    keep_quoted: bool = False,
    keep_signature: bool = False,
    source: str | None = None,
    collection: str | None = None,
    on_skip: Callable[[str, int], None] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """The record of one ``.eml`` file (an RFC 5322 message)."""
    p = os.fspath(path)
    data = await asyncio.to_thread(_read, p)
    lines = regex.split(r"\r?\n", data.decode("latin-1"))
    from_line = lines.pop(0) if lines and lines[0].startswith("From ") else ""
    r = _message_record(lines, from_line, source if source is not None else _stem(p), _Options(keep_quoted, keep_signature), collection)
    if r:
        yield r
    elif on_skip:
        on_skip("no subject and no text", 0)
