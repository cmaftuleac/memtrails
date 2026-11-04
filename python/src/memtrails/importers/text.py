"""
Small text tools the importers share: HTML reduced to text, character entities, bytes in any charset, long texts cut
into parts. No dependency; any language and script is kept as written. Every pattern is the reference's, and none may
take more than linear time on a hostile page.
"""

from __future__ import annotations

import asyncio
import codecs
import unicodedata
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import regex

from .._js import JS_SPACE, from16, js_trim, js_trim_end, to16

_NAMED: dict[str, str] = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "nbsp": " ",
    "ensp": " ",
    "emsp": " ",
    "thinsp": " ",
    "shy": "",
    "zwnj": "\u200c",
    "zwj": "\u200d",
    "lrm": "\u200e",
    "rlm": "\u200f",
    "copy": "\u00a9",
    "reg": "\u00ae",
    "trade": "\u2122",
    "hellip": "\u2026",
    "mdash": "\u2014",
    "ndash": "\u2013",
    "minus": "\u2212",
    "lsquo": "\u2018",
    "rsquo": "\u2019",
    "sbquo": "\u201a",
    "ldquo": "\u201c",
    "rdquo": "\u201d",
    "bdquo": "\u201e",
    "laquo": "\u00ab",
    "raquo": "\u00bb",
    "lsaquo": "\u2039",
    "rsaquo": "\u203a",
    "euro": "\u20ac",
    "pound": "\u00a3",
    "yen": "\u00a5",
    "cent": "\u00a2",
    "curren": "\u00a4",
    "deg": "\u00b0",
    "middot": "\u00b7",
    "bull": "\u2022",
    "times": "\u00d7",
    "divide": "\u00f7",
    "sect": "\u00a7",
    "para": "\u00b6",
    "iexcl": "\u00a1",
    "iquest": "\u00bf",
    "szlig": "\u00df",
    "aelig": "\u00e6",
    "AElig": "\u00c6",
    "oelig": "\u0153",
    "OElig": "\u0152",
    "oslash": "\u00f8",
    "Oslash": "\u00d8",
    "eth": "\u00f0",
    "ETH": "\u00d0",
    "thorn": "\u00fe",
    "THORN": "\u00de",
    "ordf": "\u00aa",
    "ordm": "\u00ba",
    "micro": "\u00b5",
    "plusmn": "\u00b1",
    "frac12": "\u00bd",
    "frac14": "\u00bc",
    "frac34": "\u00be",
    "sup1": "\u00b9",
    "sup2": "\u00b2",
    "sup3": "\u00b3",
    "dagger": "\u2020",
    "Dagger": "\u2021",
    "permil": "\u2030",
    "prime": "\u2032",
    "Prime": "\u2033",
    "le": "\u2264",
    "ge": "\u2265",
    "ne": "\u2260",
    "infin": "\u221e",
    "asymp": "\u2248",
    "larr": "\u2190",
    "rarr": "\u2192",
    "uarr": "\u2191",
    "darr": "\u2193",
    "harr": "\u2194",
    "hearts": "\u2665",
    "not": "\u00ac",
    "brvbar": "\u00a6",
    "uml": "\u00a8",
    "acute": "\u00b4",
    "cedil": "\u00b8",
    "macr": "\u00af",
    "circ": "\u02c6",
    "tilde": "\u02dc",
}
# Greek letters by name (Alpha … Omega, alpha … omega, sigmaf).
for _i, _name in enumerate(
    [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "Eta",
        "Theta",
        "Iota",
        "Kappa",
        "Lambda",
        "Mu",
        "Nu",
        "Xi",
        "Omicron",
        "Pi",
        "Rho",
        "Sigma",
        "Tau",
        "Upsilon",
        "Phi",
        "Chi",
        "Psi",
        "Omega",
    ]
):
    _offset = _i + (1 if _i >= 17 else 0)
    _NAMED[_name] = chr(0x391 + _offset)
    _NAMED[_name.lower()] = chr(0x3B1 + _offset)
_NAMED["sigmaf"] = "ς"
# A Latin letter with a mark: &eacute; &Uuml; &ccedil; &scaron; … composed from the letter and a combining mark.
_MARKS = {
    "acute": "́",
    "grave": "̀",
    "circ": "̂",
    "tilde": "̃",
    "uml": "̈",
    "ring": "̊",
    "cedil": "̧",
    "caron": "̌",
    "macr": "̄",
    "breve": "̆",
    "ogon": "̨",
    "dot": "̇",
    "dblac": "̋",
}
_MARKED = regex.compile(r"^([a-zA-Z])(acute|grave|circ|tilde|uml|ring|cedil|caron|macr|breve|ogon|dot|dblac)$")


def _c1(err: UnicodeError) -> tuple[str, int]:
    assert isinstance(err, UnicodeDecodeError)
    return chr(err.object[err.start]), err.start + 1


codecs.register_error("memtrails-c1", _c1)


def win1252(b: bytes) -> str:
    """Bytes as the WHATWG windows-1252 decoder reads them: the five bytes Python's cp1252 leaves undefined are the C1
    controls of the same number."""
    return b.decode("cp1252", "memtrails-c1")


def _named_entity(name: str) -> str | None:
    if name in _NAMED:
        return _NAMED[name]
    m = _MARKED.match(name)
    if not m:
        return None
    composed = unicodedata.normalize("NFC", m.group(1) + _MARKS[m.group(2)])
    return composed if len(composed) == 1 else None


def _code_point(n: int) -> str:
    if 0x80 <= n <= 0x9F:
        return win1252(bytes([n]))  # what browsers do with &#150; and the like
    if n <= 0 or n > 0x10FFFF or 0xD800 <= n <= 0xDFFF:
        return "�"
    return chr(n)


_ENTITY = regex.compile(r"&(#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7}|[a-zA-Z][a-zA-Z0-9]{1,31});")


def decode_entities(s: str) -> str:
    """``&amp;``, ``&#233;``, ``&#x1F600;``, ``&eacute;`` and the other common names, decoded; unknown names are left as
    written."""
    if "&" not in s:
        return s

    def one(m: regex.Match[str]) -> str:
        ref = m.group(1)
        if ref[0] == "#":
            return _code_point(int(ref[2:], 16) if ref[1] in "xX" else int(ref[1:], 10))
        found = _named_entity(ref)
        return m.group(0) if found is None else found

    return _ENTITY.sub(one, s)


_BLOCK = (
    "address|article|aside|blockquote|caption|center|dd|details|dialog|div|dl|dt|fieldset|figcaption|figure|footer|form|"
    "h[1-6]|header|hr|main|nav|ol|p|pre|section|summary|table|tbody|thead|tfoot|tr|ul"
)
_I = regex.IGNORECASE | regex.ASCII


def _before_last_gt(s: str, f: Callable[[str], str]) -> str:
    """``f`` applied to ``s`` up to its last ``>``, the rest kept. Every tag pattern here ends in ``[^>]*>``: past the last
    ``>`` each ``<`` is a start that scans to the end and fails, so a run of unclosed ``<a<a<a…`` was quadratic; before
    it, each scan stops at the next ``>`` and matches."""
    end = s.rfind(">") + 1
    return f(s) if end == len(s) else f(s[:end]) + s[end:]


def _drop_elements(html: str, tag: str) -> str:
    """Removes every ``<tag …>…</tag>`` (nested ones too), for ``blockquote``: the quoted earlier message in an HTML
    reply."""
    tags = regex.compile(f"<(/?){tag}\\b[^>]*>", _I)
    head = html[: html.rfind(">") + 1]  # see _before_last_gt
    out = ""
    depth = 0
    start = 0
    for m in tags.finditer(head):
        if not m.group(1):
            if depth == 0:
                out += html[start : m.start()]
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                start = m.end()
    return out + html[start:] if depth == 0 else out


def _replace_elements(
    s: str, open_tag: regex.Pattern[str], f: Callable[[str, str | None], str | None], outside: Callable[[str], str] = lambda t: t
) -> str:
    """Each ``<tag …>…</tag>`` found by ``open_tag`` (the tag's name in group 1) replaced by ``f(name, element)``,
    scanning once: an unclosed one has ``element`` None; ``f`` then gives what replaces it and everything after it, or
    None to leave it as written. A tag with no closing tag after it has none after a later opening tag either, so it is
    looked for once. The text between elements goes through ``outside``."""
    closes: dict[str, regex.Pattern[str]] = {}
    unclosed: set[str] = set()
    out = ""
    start = 0
    pos = 0
    while True:
        m = open_tag.search(s, pos)
        if not m:
            break
        pos = m.end()
        name = m.group(1).lower()
        end = -1
        if name not in unclosed:
            close = closes.setdefault(name, regex.compile(f"</{name}[{JS_SPACE}]*>", _I))
            c = close.search(s, m.end())
            if c:
                end = c.end()
            else:
                unclosed.add(name)
        replaced = f(name, None if end < 0 else s[m.start() : end])
        if replaced is None:
            continue  # left as written: look on after the opening tag
        out += outside(s[start : m.start()]) + replaced
        if end < 0:
            return out
        start = pos = end
    return out + outside(s[start:])


def _unwrap_cdata(s: str) -> str:
    """``<![CDATA[…]]>`` as its text; an unclosed one, and everything after it, left as written."""
    out = ""
    start = 0
    i = s.find("<![CDATA[", start)
    while i >= 0:
        j = s.find("]]>", i + 9)
        if j < 0:
            break
        out += s[start:i] + s[i + 9 : j]
        start = j + 3
        i = s.find("<![CDATA[", start)
    return out + s[start:]


_HIDDEN = regex.compile(r"<(script|style|head|noscript|template|svg|math|iframe|object)\b", _I)
_PRE = regex.compile(r"<(pre)\b", _I)
_COMMENT = regex.compile(r"<!--[\s\S]*?(?:-->|\Z)")
_WS = regex.compile(f"[{JS_SPACE}]+")
_CRLF = regex.compile(r"\r\n?")
_BR = regex.compile(r"<br\b[^>]*>", _I)
_LI = regex.compile(r"<li\b[^>]*>", _I)
_BLOCKS = regex.compile(f"</?(?:{_BLOCK})\\b[^>]*>", _I)
_CELL = regex.compile(f"</t[dh][{JS_SPACE}]*>", _I)
_TAG = regex.compile(r"</?[a-zA-Z][^>]*>")
_DECL = regex.compile(r"<[!?][^>]*>")

#: Title regexes see at most this much of a page: a title is near the top, and a hostile page is not read to its end.
_TITLE_SCAN = 64 * 1024


def html_title(html: str) -> str | None:
    """The ``<title>`` of an HTML page, else its first ``<h1>``, as text; None when it has neither."""
    top = from16(to16(html)[:_TITLE_SCAN])
    head = top[: top.rfind(">") + 1]  # see _before_last_gt
    for tag in ("title", "h1"):
        opened = regex.compile(f"<{tag}\\b[^>]*>", _I).search(head)
        if not opened:
            continue
        end = regex.compile(f"</{tag}[{JS_SPACE}]*>", _I).search(head, opened.end())
        if not end:
            continue
        inner = _before_last_gt(head[opened.end() : end.start()], lambda h: regex.sub(r"<[^>]*>", " ", h))
        t = js_trim(_WS.sub(" ", decode_entities(inner)))
        if t:
            return t
    return None


def html_to_text(html: str, drop_quotes: bool = False) -> str:
    """HTML as plain text: scripts, styles, comments and the head dropped, blocks and line breaks kept as lines, list
    items as ``- ``, tags removed, entities decoded, ``<pre>`` kept as written. ``drop_quotes`` also drops
    ``<blockquote>``s."""
    s = _unwrap_cdata(_COMMENT.sub(" ", html))
    # Hidden elements dropped; a script or style left open runs to the end, any other unclosed one is left to the tag rules.
    s = _replace_elements(s, _HIDDEN, lambda name, element: " " if element is not None or name in ("script", "style") else None)
    if drop_quotes:
        s = _drop_elements(s, "blockquote")
    # Whitespace in the source is one space, except inside <pre>.
    s = _replace_elements(
        s, _PRE, lambda _name, element: _CRLF.sub("\n", element) if element is not None else None, lambda t: _WS.sub(" ", t)
    )

    def tags(h: str) -> str:
        h = _BR.sub("\n", h)
        h = _LI.sub("\n- ", h)
        h = _BLOCKS.sub("\n", h)
        h = _CELL.sub("\t", h)
        h = _TAG.sub("", h)
        return _DECL.sub("", h)

    s = _before_last_gt(s, tags)
    return tidy(decode_entities(s))


_INNER = regex.compile("[ \t ]+")
_LEAD = regex.compile(f"^ (?=[^{JS_SPACE}])")
_GAPS = regex.compile(r"\n{3,}")


def tidy(s: str) -> str:
    """Lines trimmed at the end, spaces and tabs inside a line kept to one, at most one empty line in a row."""
    lines = [js_trim_end(_LEAD.sub("", _INNER.sub(" ", line))) for line in _CRLF.sub("\n", s).split("\n")]
    return js_trim(_GAPS.sub("\n\n", "\n".join(lines)))


_LABEL_PLAIN = regex.compile(r"^(us-?ascii|ascii|ansi_x3\.4-1968|utf-?8|unicode-1-1-utf-8|x-unknown|unknown(-8bit)?|default)$")
#: WHATWG labels whose decoder is not Python's codec of the same name (TextDecoder reads latin1 as windows-1252, and so
#: on).
_WHATWG = {
    "latin1": "cp1252",
    "iso-8859-1": "cp1252",
    "iso8859-1": "cp1252",
    "iso_8859-1": "cp1252",
    "l1": "cp1252",
    "cp819": "cp1252",
    "ibm819": "cp1252",
    "iso-ir-100": "cp1252",
    "csisolatin1": "cp1252",
    "iso-8859-9": "cp1254",
    "iso8859-9": "cp1254",
    "latin5": "cp1254",
    "l5": "cp1254",
    "iso-8859-11": "cp874",
    "tis-620": "cp874",
    "windows-874": "cp874",
    "gb2312": "gbk",
    "x-gbk": "gbk",
    "chinese": "gbk",
    "csgb2312": "gbk",
    "gb_2312-80": "gbk",
    "iso-ir-58": "gbk",
    "big5": "big5hkscs",
    "big5-hkscs": "big5hkscs",
    "cn-big5": "big5hkscs",
    "x-x-big5": "big5hkscs",
    "shift_jis": "cp932",
    "sjis": "cp932",
    "ms_kanji": "cp932",
    "windows-31j": "cp932",
    "x-sjis": "cp932",
    "csshiftjis": "cp932",
    "euc-kr": "cp949",
    "ks_c_5601-1987": "cp949",
    "ks_c_5601-1989": "cp949",
    "korean": "cp949",
    "windows-949": "cp949",
    "cseuckr": "cp949",
    "macintosh": "mac_roman",
    "mac": "mac_roman",
    "x-mac-roman": "mac_roman",
    "csmacintosh": "mac_roman",
    "x-mac-cyrillic": "mac_cyrillic",
    "koi8": "koi8_r",
    "koi8_r": "koi8_r",
    "cskoi8r": "koi8_r",
    "koi8-ru": "koi8_u",
    "utf-16": "utf-16-le",
    "utf-16le": "utf-16-le",
    "utf-16be": "utf-16-be",
    "ucs-2": "utf-16-le",
    "unicode": "utf-16-le",
}
#: Labels TextDecoder does not know: Python would decode them, the reference reads them as below.
_NOT_WHATWG = {"utf-32", "utf-32le", "utf-32be", "utf-7", "iso-8859-12"}


def _codec(label: str) -> str | None:
    if label in _NOT_WHATWG:
        return None
    name = _WHATWG.get(label, label)
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def decode_bytes(data: bytes, charset: str | None = None) -> str:
    """Bytes as text in the charset named (any WHATWG label); unnamed, ASCII or unknown: UTF-8, else Windows-1252."""
    label = regex.sub(r"\*.*$", "", regex.sub(r"^[\"']|[\"']$", "", js_trim(charset or ""))).lower()
    if label and not _LABEL_PLAIN.match(label):
        codec = _codec(label)
        if codec == "cp1252":
            return win1252(data)
        if codec is not None:
            # TextDecoder drops a byte order mark that matches its encoding.
            bom = {"utf-16-le": b"\xff\xfe", "utf-16-be": b"\xfe\xff", "utf-8": b"\xef\xbb\xbf"}.get(codec)
            if bom and data.startswith(bom):
                data = data[len(bom) :]
            return data.decode(codec, "replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return win1252(data)


_TERMINAL = regex.compile(r"\p{Sentence_Terminal}+")
_LAST_SPACE = regex.compile(f"[{JS_SPACE}][^{JS_SPACE}]*\\Z")


def split_text(text: str, max_chars: int) -> list[str]:
    """A long text in parts of at most ``max_chars`` UTF-16 units, each cut at a paragraph, else a line, else a sentence's
    end (in any script), else a space; never inside a character."""
    top = max(100, int(max_chars))
    rest = to16(text)
    if len(rest) <= top:
        return [text]
    parts: list[str] = []
    while len(rest) > top:
        window = rest[:top]
        half = top // 2
        cut = window.rfind("\n\n")
        if cut < half:
            cut = window.rfind("\n")
        if cut < half:
            ends = list(_TERMINAL.finditer(from16(window[half:])))
            cut = half + len(to16(from16(window[half:])[: ends[-1].end()])) if ends else -1
        if cut < half:
            m = _LAST_SPACE.search(window)
            cut = m.start() if m else -1
        if cut < half:
            cut = top
        if 0xD800 <= ord(rest[cut - 1]) <= 0xDBFF:
            cut -= 1  # keep a surrogate pair together
        part = js_trim(from16(rest[:cut]))
        if part:
            parts.append(part)
        rest = rest[cut:]
    if js_trim(from16(rest)):
        parts.append(js_trim(from16(rest)))
    return parts


def record(r: Mapping[str, Any]) -> dict[str, Any]:
    """A record without its empty fields (None or ``''``, the text excepted), and without empty metadata values."""
    out: dict[str, Any] = {}
    for k, v in r.items():
        if v is None or (v == "" and k != "text"):
            continue
        if k == "metadata" and isinstance(v, Mapping):
            m = {mk: mv for mk, mv in v.items() if mv is not None and mv != ""}
            if m:
                out["metadata"] = m
            continue
        out[k] = v
    return out


_NEWLINE = regex.compile(r"\r\n|\r|\n")


async def read_lines(path: str, encoding: str) -> AsyncIterator[str]:
    """A file's lines as Node's readline gives them (a break at ``\\n``, ``\\r\\n`` or a lone ``\\r``; no empty line after
    the last break), decoded as it is read, a megabyte at a time, off the event loop."""
    decoder = codecs.getincrementaldecoder(encoding)("replace")
    rest = ""
    f = await asyncio.to_thread(open, path, "rb")
    try:
        while True:
            chunk: bytes = await asyncio.to_thread(f.read, 1 << 20)
            text = rest + decoder.decode(chunk, final=not chunk)
            if not chunk:
                rest = text
                break
            # A '\r' at the end may be the first half of a '\r\n': kept for the next chunk.
            held = "\r" if text.endswith("\r") else ""
            parts = _NEWLINE.split(text[:-1] if held else text)
            rest = parts.pop() + held
            for line in parts:
                yield line
    finally:
        f.close()
    if rest:
        for line in _NEWLINE.split(rest[:-1] if rest.endswith(("\r", "\n")) else rest):
            yield line
