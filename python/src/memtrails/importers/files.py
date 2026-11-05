"""
A folder of text files: notes, documents, saved pages. Every file whose extension is in ``include`` becomes a record,
read one at a time (off the event loop); dot folders, dot files and ``node_modules`` are passed by.

id: the path inside the folder (with ``/``). title: the Markdown front matter's ``title``, else the first ``# heading``
(an Org ``#+TITLE:``, an RST underlined title, an HTML ``<title>`` or ``<h1>``), else the file's name. text: the file,
front matter taken off, HTML reduced to text. at: the front matter's ``date``, else when the file was last changed. The
other front-matter fields go into metadata (``author`` also to the record's author). A file longer than ``max_chars``
is cut into parts with ids ``path#1``, ``path#2``, … sharing the thread ``path``.
"""

from __future__ import annotations

import asyncio
import errno
import math
import os
import stat
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import regex

from .._js import JS_SPACE, from16, js_trim, js_trim_end, num_str, to16
from ..records import time_of
from .text import decode_bytes, html_title, html_to_text, record, split_text, undef

#: The extensions read by default: text formats only.
TEXT_EXTENSIONS = [
    ".md",
    ".markdown",
    ".mdx",
    ".txt",
    ".text",
    ".html",
    ".htm",
    ".xhtml",
    ".rst",
    ".org",
    ".adoc",
    ".asciidoc",
    ".textile",
    ".wiki",
    ".tex",
]

Scalar = str | int | float | bool | None
_QUOTED = regex.compile(r"^([\"']).*\1$", regex.DOTALL)
_BOOL = regex.compile(r"^(true|false)$", regex.IGNORECASE)
_NULL = regex.compile(r"^(null|~)$", regex.IGNORECASE)
_NUMBER = regex.compile(r"^-?(0|[1-9][0-9]{0,14})(\.[0-9]+)?$")


def _scalar(raw: str) -> Scalar:
    v = js_trim(raw)
    if _QUOTED.match(v):
        return v[1:-1].replace('\\"', '"').replace("''", "'")
    if _BOOL.match(v):
        return v.lower() == "true"
    if _NULL.match(v) or v == "":
        return None
    if _NUMBER.match(v):
        n = float(v)
        return int(n) if n.is_integer() else n
    return v


def _list_value(items: Sequence[str]) -> str:
    return ", ".join(
        num_str(x)
        if isinstance(x, (int, float)) and not isinstance(x, bool)
        else ("true" if x is True else "false" if x is False else str(x))
        for x in (_scalar(i) for i in items)
        if x is not None and x != ""
    )


def _key_value(line: str, toml: bool) -> tuple[str, str] | None:
    """One front-matter line as key and value: TOML ``key = value``, YAML ``key: value`` (a ``:`` then a space or the
    end). Split at the first ``=`` or ``:`` by hand: a regex tried a run of spaces at every position, quadratic on a long
    line."""
    at = line.find("=" if toml else ":")
    if at <= 0 or regex.match(r"[\s=#[]" if toml else r"[\s:#-]", line[0]):
        return None
    value = line[at + 1 :]
    if not toml and value and not regex.match(f"[{JS_SPACE}]", value[0]):
        return None
    return js_trim_end(line[:at]), value


_FRONT = regex.compile(r"^(---|\+\+\+)[ \t]*\r?\n([\s\S]*?)\r?\n(?:---|\+\+\+|\.\.\.)[ \t]*(?:\r?\n|\Z)")
_LINE_SPLIT = regex.compile(r"\r?\n")
_INDENTED = regex.compile(f"^[{JS_SPACE}]")
_ITEM = regex.compile(f"^[{JS_SPACE}]*-[{JS_SPACE}]+")
_NESTED = regex.compile(f"^[{JS_SPACE}]+[^{JS_SPACE}]")
_COMMENT = regex.compile(f"[{JS_SPACE}]#")


def front_matter(text: str) -> tuple[dict[str, Scalar], str]:
    """YAML (``---``) or TOML (``+++``) front matter, flat fields only: scalars, lists (joined with ``, ``) and block
    text. Returns the fields and the body after it."""
    m = _FRONT.match(text)
    if not m:
        return {}, text
    toml = m.group(1) == "+++"
    fields: dict[str, Scalar] = {}
    lines = _LINE_SPLIT.split(m.group(2))
    i = 0
    while i < len(lines):
        kv = _key_value(lines[i], toml)
        if not kv:
            i += 1
            continue
        key = regex.sub(r"^[\"']|[\"']$", "", js_trim(kv[0]))
        value = js_trim(kv[1])
        if value.startswith("[") and value.endswith("]"):
            fields[key] = _list_value(value[1:-1].split(","))
        elif not toml and regex.match(r"^[|>][-+]?$", value):
            block: list[str] = []
            while i + 1 < len(lines) and (_INDENTED.match(lines[i + 1]) or not js_trim(lines[i + 1])):
                i += 1
                block.append(js_trim(lines[i]))
            fields[key] = js_trim(("\n" if value.startswith("|") else " ").join(block))
        elif not toml and not value:
            items: list[str] = []
            while i + 1 < len(lines) and _ITEM.match(lines[i + 1]):
                i += 1
                items.append(_ITEM.sub("", lines[i], count=1))
            while i + 1 < len(lines) and _NESTED.match(lines[i + 1]):
                i += 1  # a nested map: not kept
            if items:
                fields[key] = _list_value(items)
        else:
            comment = _COMMENT.search(value)  # not /\s+#.*$/: quadratic on a long run of spaces
            fields[key] = _scalar(value if comment is None else value[: comment.start()])
        i += 1
    return fields, text[m.end() :]


#: Title rules see at most this much of a file, a line at a time: per-line regexes over a whole hostile file were slow.
_TITLE_SCAN = 64 * 1024


def _rst_rule(line: str | None) -> bool:
    """An RST underline (or overline): one punctuation character, three or more times, then spaces at most."""
    t = js_trim_end(line or "")
    return len(t) >= 3 and t[0] in "=-~^\"'`#*+" and t == t[0] * len(t)


def _title_of(ext: str, body: str, raw: str) -> str | None:
    if ext in (".html", ".htm", ".xhtml"):
        return html_title(raw)
    lines = regex.split(r"\r\n?|\n", from16(to16(body)[:_TITLE_SCAN]))

    def first(f: Callable[[str, int], str | None]) -> str | None:
        for i, line in enumerate(lines):
            t = f(line, i)
            if t is not None:
                return t
        return None

    def at(i: int) -> str | None:
        return lines[i] if i < len(lines) else None

    if ext == ".org":
        return first(lambda x, _i: js_trim(x[8:]) if regex.match(r"^#\+title:", x, regex.IGNORECASE) and len(x) > 8 else None) or first(
            lambda x, _i: (
                js_trim(regex.sub(r"^\*+", "", x)) if regex.match(r"^\*+[ \t]", x) and js_trim(regex.sub(r"^\*+", "", x)) else None
            )
        )
    if ext == ".rst":
        return first(lambda x, i: js_trim(x) if js_trim(x) and _rst_rule(at(i + 1)) else None)
    if ext in (".adoc", ".asciidoc"):
        return first(lambda x, _i: js_trim(x[1:]) if regex.match(r"^=[ \t]", x) and js_trim(x[1:]) else None)

    def atx(x: str, _i: int) -> str | None:
        """``# Title``, a closing run of ``#`` after a space dropped (by hand: a regex is quadratic on a run of spaces)."""
        if not regex.match(r"^#[ \t]", x):
            return None
        t = js_trim(x[1:])
        j = len(t)
        while j > 0 and t[j - 1] == "#":
            j -= 1
        title = js_trim_end(t[:j]) if 0 < j < len(t) and t[j - 1] in " \t" else t
        return title or None

    found = first(atx)
    if found:
        return found
    # A Setext `Title\n=====`.
    return first(lambda x, i: js_trim_end(x) if regex.match(f"^[^{JS_SPACE}]", x) and regex.match(r"^=+[ \t]*$", at(i + 1) or "") else None)


def _walk(root: str, rel: str, include: set[str], skip_folders: set[str]) -> Iterator[str]:
    names = sorted(os.listdir(os.path.join(root, rel)), key=lambda n: n.encode("utf-16-be", "surrogatepass"))
    for name in names:
        if name.startswith("."):
            continue
        path = f"{rel}/{name}" if rel else name
        full = os.path.join(root, path)
        if os.path.isdir(full) and not os.path.islink(full):
            if name not in skip_folders:
                yield from _walk(root, path, include, skip_folders)
        elif os.path.isfile(full) and not os.path.islink(full) and os.path.splitext(name)[1].lower() in include:
            yield path


def _decode_file(data: bytes) -> str | None:
    if data[:2] == b"\xff\xfe":
        return data[2:].decode("utf-16-le", "replace")
    if data[:2] == b"\xfe\xff":
        return data[2:].decode("utf-16-be", "replace")
    if b"\0" in data[:8192]:
        return None  # binary
    return decode_bytes(data)


def _open_no_follow(path: str) -> int | None:
    """A file opened for reading, never through a link in its last part (``O_NOFOLLOW``), and without waiting on a pipe
    swapped in for it (``O_NONBLOCK``, which a regular file ignores); None when it is a link. Where there is no
    ``O_NOFOLLOW`` (Windows), a link is refused by looking first."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        if os.path.islink(path):
            return None
        return os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        return os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0))
    except OSError as err:
        if err.errno == errno.ELOOP:
            return None
        raise


def _read_file(path: str, max_bytes: int) -> tuple[float, str | None, str | None]:
    """Its time, its text (None when it is not text), and why it was left out (None when it was read). Opened without
    following a link, and measured and read from what was opened: a note swapped for a link after the folder was
    listed (to a key outside it, say) is not read through it."""
    fd = _open_no_follow(path)
    if fd is None:
        return 0, None, "not a file: it became a link after the folder was read"
    with os.fdopen(fd, "rb") as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode):
            return 0, None, "not a file: it changed after the folder was read"
        if info.st_size > max_bytes:
            return 0, None, f"larger than {max_bytes} bytes"
        return info.st_mtime * 1000, _decode_file(f.read()), None


async def from_files(
    directory: str | os.PathLike[str],
    include: Sequence[str] | None = None,
    skip_folders: Sequence[str] | None = None,
    exclude: Callable[[str], bool] | None = None,
    max_chars: int = 20_000,
    max_bytes: int = 64 * 1024 * 1024,
    collection: str | None = None,
    source: str | None = None,
    kind: str = "note",
    on_skip: Callable[[str, str], None] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Records from the text files in a folder and its subfolders. ``include``: extensions to read (default
    TEXT_EXTENSIONS); ``skip_folders``: never entered (default ``node_modules``; dot folders never are);
    ``exclude(path)``; longer files are cut into parts of ``max_chars``; larger than ``max_bytes`` are not read;
    ``on_skip(path, why)`` for each file left out."""
    root = os.path.abspath(os.fspath(directory))
    wanted = {(e if e.startswith(".") else f".{e}").lower() for e in (include if include is not None else TEXT_EXTENSIONS)}
    skipped = set(skip_folders if skip_folders is not None else ["node_modules"])
    src = source if source is not None else os.path.basename(root)
    paths = await asyncio.to_thread(lambda: list(_walk(root, "", wanted, skipped)))
    for path in paths:
        if exclude and exclude(path):
            continue
        full = os.path.join(root, path)
        mtime, raw, left_out = await asyncio.to_thread(_read_file, full, max_bytes)
        if left_out is not None:
            if on_skip:
                on_skip(path, left_out)
            continue
        if raw is None:
            if on_skip:
                on_skip(path, "not text")
            continue
        ext = os.path.splitext(path)[1].lower()
        is_html = ext in (".html", ".htm", ".xhtml")
        fields, body = ({}, raw) if is_html else front_matter(raw)
        own = fields.get("title")
        fm_title = (
            js_trim(own if isinstance(own, str) else num_str(own))
            if isinstance(own, (str, int, float)) and not isinstance(own, bool)
            else ""
        )
        title = fm_title or _title_of(ext, body, raw) or os.path.splitext(os.path.basename(path))[0]
        text = html_to_text(raw) if is_html else js_trim(regex.sub(r"\r\n?", "\n", body))
        date = fields.get("date")
        dated = (
            time_of(num_str(date) if isinstance(date, (int, float)) and not isinstance(date, bool) else date)
            if isinstance(date, (str, int, float)) and not isinstance(date, bool)
            else None
        )
        at = dated if dated is not None else mtime
        author = fields.get("author") if isinstance(fields.get("author"), str) else None
        metadata = {k: v for k, v in fields.items() if k != "title"}
        base: dict[str, Any] = {
            "id": path,
            "text": text,
            "title": title,
            "kind": kind,
            "source": src,
            "collection": undef(collection),
            "author": undef(author),
            "at": math.floor(at + 0.5),
            "url": Path(full).as_uri(),
        }
        if not text and not title:
            continue
        parts = split_text(text, max_chars)
        if len(parts) == 1:
            yield record({**base, "metadata": metadata})
            continue
        for i, part in enumerate(parts):
            yield record(
                {
                    **base,
                    "id": f"{path}#{i + 1}",
                    "thread": path,
                    "text": part,
                    "title": f"{title} ({i + 1}/{len(parts)})",
                    "metadata": {**metadata, "part": i + 1, "parts": len(parts)},
                }
            )
