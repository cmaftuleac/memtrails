"""
A Git repository's history: one record a commit, read from ``git log`` as it runs (never the whole log at once). Each
commit is framed by markers carrying a random token made for the run, so no message text can be taken for one.

id: the full sha. title: the subject. text: the whole message, then ``Files: …`` (the paths it changed, at most
``max_files``). author: the author's name. at: the author date. kind ``commit``; thread: ``thread`` or the repository's
name; source: ``source`` or the repository's name; metadata ``{sha, email, parents}``.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import secrets
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime
from typing import Any

import regex

from .._js import iso, js_trim, local_ms
from .text import record


def _when(t: str | datetime | int | float) -> str:
    if isinstance(t, datetime):
        return iso(local_ms(t))
    if isinstance(t, (int, float)):
        return iso(t)
    return t


_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}
_OCTAL = regex.compile(r"[0-7]{3}")


def _unquote(path: str) -> str:
    """A path as git writes it when it has unusual characters: ``"a\\tb\\303\\251"`` back to its text."""
    if not (len(path) > 1 and path.startswith('"') and path.endswith('"')):
        return path
    out = bytearray()
    s = path[1:-1]
    i = 0
    while i < len(s):
        if s[i] == "\\" and _OCTAL.match(s[i + 1 : i + 4]):
            out.append(int(s[i + 1 : i + 4], 8))
            i += 4
        elif s[i] == "\\" and s[i + 1 : i + 2] in _ESCAPES:
            out.append(_ESCAPES[s[i + 1]])
            i += 2
        else:
            out += s[i].encode("utf-8", "surrogatepass")
            i += 1
    return out.decode("utf-8", "replace")


_SHA = regex.compile(r"^[0-9a-f]{7,64}$")
_PARAGRAPH = regex.compile(r"\n[\t\n\v\f\r    -     　﻿]*\n")
_FOLD = regex.compile(r"[\t\n\v\f\r    -     　﻿]*\n[\t\n\v\f\r    -     　﻿]*")
_EMPTY_HISTORY = regex.compile(r"does not have any commits|bad default revision 'HEAD'")


async def from_git(
    repo_dir: str | os.PathLike[str],
    since: str | datetime | int | float | None = None,
    until: str | datetime | int | float | None = None,
    branch: str | None = None,
    rev: str | None = None,
    all: bool = False,
    max_count: int | None = None,
    paths: Sequence[str] | None = None,
    no_merges: bool = False,
    max_files: int = 50,
    thread: str | None = None,
    source: str | None = None,
    collection: str | None = None,
    url: Callable[[str], str] | None = None,
    git: str = "git",
) -> AsyncIterator[dict[str, Any]]:
    """Records from a repository's commits, newest first (as ``git log`` gives them). ``since``/``until``: anything
    ``git log --since`` reads, a datetime, or milliseconds; ``branch``/``rev``: the branch, tag or range to read;
    ``all``: every branch and tag; ``paths``: only commits that changed these; ``url(sha)``: where to open a commit."""
    root = os.path.abspath(os.fspath(repo_dir))
    name = os.path.basename(root)
    token = secrets.token_hex(16)
    start = f"\u001e{token}\n"
    end = f"\u001f{token}"
    args = [
        "-c",
        "core.quotePath=false",
        "-c",
        "log.showSignature=false",
        "log",
        "--encoding=UTF-8",
        "--no-color",
        "--no-ext-diff",
        "--name-only",
        f"--format=%x1e{token}%n%H%x00%an%x00%ae%x00%aI%x00%P%x00%B%x1f{token}",
    ]
    if max_count is not None:
        args.append(f"--max-count={max(0, int(max_count))}")
    if since is not None:
        args.append(f"--since={_when(since)}")
    if until is not None:
        args.append(f"--until={_when(until)}")
    if no_merges:
        args.append("--no-merges")
    if all:
        args.append("--all")
    revision = rev if rev is not None else branch
    if revision:
        if revision.startswith("-"):
            raise ValueError(f"not a revision: {revision}")
        args.append(revision)
    args += ["--", *(paths or [])]

    def to_record(chunk: str) -> dict[str, Any] | None:
        close = chunk.rfind(end)
        head = chunk[:close] if close >= 0 else chunk
        fields = head.split("\0")
        sha = fields[0] if fields else ""
        if not sha or not _SHA.match(sha):
            return None
        author, email, date, parents = [*fields, "", "", "", ""][1:5]
        body = js_trim(regex.sub(r"\r\n?", "\n", "\0".join(fields[5:])))
        subject = js_trim(_FOLD.sub(" ", _PARAGRAPH.split(body)[0] if body else ""))
        files = [_unquote(x) for x in chunk[close + len(end) :].split("\n") if js_trim(x)] if close >= 0 else []
        listed = ", ".join(files[:max_files]) + (f", … and {len(files) - max_files} more" if len(files) > max_files else "")
        text = "\n\n".join(x for x in (body, f"Files: {listed}" if files else "") if x)
        return record(
            {
                "id": sha,
                "title": subject or sha[:12],
                "text": text,
                "kind": "commit",
                "author": author,
                "at": date or None,
                "thread": thread if thread is not None else name,
                "source": source if source is not None else name,
                "collection": collection,
                "url": url(sha) if url else None,
                "metadata": {"sha": sha, "email": email, "parents": parents},
            }
        )

    # `-C`, not a working folder: on Windows a command is looked up in its working folder first, so running in the
    # repository would run a `git.exe` planted in the repository being imported.
    child = await asyncio.create_subprocess_exec(
        git, "-C", root, *args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    assert child.stdout is not None and child.stderr is not None
    err_pipe = child.stderr

    async def drain() -> bytes:
        """All of what git says on stderr is read (a full pipe would stop it); the start is kept."""
        kept = b""
        while chunk := await err_pipe.read(1 << 14):
            if len(kept) < 4000:
                kept += chunk
        return kept

    errors = asyncio.ensure_future(drain())
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        while True:
            piece = await child.stdout.read(1 << 16)
            if not piece:
                break
            buffer += decoder.decode(piece)
            nxt = buffer.find(start, 1)
            while nxt > 0:
                r = to_record(buffer[len(start) if buffer.startswith(start) else 0 : nxt])
                if r:
                    yield r
                buffer = buffer[nxt:]
                nxt = buffer.find(start, 1)
        buffer += decoder.decode(b"", final=True)
        if buffer.startswith(start):
            r = to_record(buffer[len(start) :])
            if r:
                yield r
    finally:
        if child.returncode is None:
            try:
                child.kill()
            except ProcessLookupError:
                pass
        code = await child.wait()
        stderr = (await errors).decode("utf-8", "replace")
    if code != 0 and not _EMPTY_HISTORY.search(stderr):
        raise RuntimeError(f"git log failed ({code}) in {root}: {js_trim(stderr)}")
