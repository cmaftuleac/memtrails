"""
What memtrails stores: records. Anything with text is a record — an e-mail, a chat message, a note, a memo, a news
article, a commit message, a ticket, a page of a document. Records that belong together (an e-mail thread, a chat
thread, a meeting's notes) share a ``thread``; a conversation between a person and an assistant is a thread of turns.

Every record is searchable the moment it is added, with no model asked. Consolidation (optional, with a language
model) reads threads in windows and keeps typed knowledge from them; the records themselves are never lost.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, TypedDict

import regex

from ._js import UNDEFINED, date_parse, intish, js_trim, local_ms, num_str, stringify

Scalar = str | int | float | bool | None
#: When something was written: milliseconds since the epoch, an ISO or RFC 2822 date, or a ``datetime`` (a naive one
#: is local time, as a JavaScript Date reads it).
TimeLike = int | float | str | datetime


class _RecordBase(TypedDict):
    #: Stable and unique within its ``source``: the same id added again replaces the record (unchanged: nothing is done).
    id: str
    #: The content.
    text: str


class MemoryRecord(_RecordBase, total=False):
    """One piece of information to remember."""

    #: A subject, headline or name; when absent, the first line of the text.
    title: str
    #: What it is, in one word of your choosing: ``email``, ``message``, ``note``, ``memo``, ``article``, ``commit``, …
    kind: str
    #: Where it came from: a mailbox, a workspace, a feed, a repository. Part of the record's identity.
    source: str
    #: What it belongs to, for searches scoped to one collection: a project, a team, a customer, a person.
    collection: str
    #: Records that belong together share a thread (an e-mail thread, a chat thread, a repository's history).
    thread: str
    #: Who wrote it.
    author: str
    #: When it was written. Default: when it was added.
    at: TimeLike
    #: Where to open it.
    url: str
    #: Anything else, kept with the record and returned with it. Not searched.
    metadata: dict[str, Scalar]


class _TurnBase(TypedDict):
    role: Literal["user", "assistant"]
    text: str


class Turn(_TurnBase, total=False):
    author: str
    at: TimeLike


class _ConversationBase(TypedDict):
    #: Stable: the same id added again is read again, not doubled.
    id: str
    turns: list[Turn]


class Conversation(_ConversationBase, total=False):
    """A conversation: turns between a person and an assistant (or any two parties), in order."""

    collection: str
    source: str
    #: When it began (each turn may carry its own ``at``).
    at: TimeLike


#: The collection records without one belong to.
DEFAULT_COLLECTION = "default"

#: A record as memory keeps it in ``records.jsonl`` (core/store.py StmEntry): a JSON object.
StmEntry = dict[str, Any]


def time_of(at: Any) -> int | float | None:
    """A time as milliseconds; None for nothing or something unreadable."""
    if at is None or at == "" or at is UNDEFINED or isinstance(at, bool):
        return None
    if isinstance(at, datetime):
        t = local_ms(at)
    elif isinstance(at, (int, float)):
        t = float(at)
    elif isinstance(at, str):
        t = date_parse(at)
    else:
        return None
    return intish(t) if math.isfinite(t) else None


_COLLECTION_SEP = regex.compile(r"[\t\n\v\f\r    -     　﻿/\\:]+")


def collection_key(collection: str | None) -> str:
    """A collection as the index keys it: one path segment, any script kept."""
    c = js_trim(collection or "")
    return _COLLECTION_SEP.sub("-", c) if c else DEFAULT_COLLECTION


def _hash(*parts: str | None) -> str:
    return hashlib.sha256("\0".join(p or "" for p in parts).encode("utf-8", "surrogatepass")).hexdigest()


def record_uid(r: Mapping[str, Any]) -> str:
    """A record's id in memory: its source and id, hashed (any characters may be in either)."""
    return _hash(r.get("source") or "", r["id"])[:32]


def _text(v: Any) -> str:
    return v if isinstance(v, str) else ""


def invalid_record(r: Any) -> str | None:
    """Checks a record and says what is wrong with it, or None."""
    if not isinstance(r, Mapping):
        return "a record must be an object"
    rid = r.get("id")
    if not isinstance(rid, str) or not js_trim(rid):
        return "a record needs an id"
    if not isinstance(r.get("text"), str):
        return f"record {rid}: text must be a string"
    if not js_trim(r["text"]) and not js_trim(_text(r.get("title"))):
        return f"record {rid}: it has no text and no title"
    if "at" in r and r["at"] is not UNDEFINED and time_of(r["at"]) is None:
        at = r["at"]
        shown = (
            "null"
            if at is None
            else num_str(at)
            if isinstance(at, (int, float)) and not isinstance(at, bool)
            else str(at).lower()
            if isinstance(at, bool)
            else str(at)
        )
        return f"record {rid}: its time cannot be read ({shown})"
    return None


def _first_line(text: str) -> str:
    return next((x for x in (js_trim(line) for line in text.split("\n")) if x), "")


def entry_of(r: Mapping[str, Any], now: int | float, turn: int = 0) -> StmEntry:
    """A record as memory keeps it: the title is what it is about (the first line when there is none), the text after
    the author is the content. Searchable as written: its start is its key, and every word of it is in a chunk."""
    text = js_trim(r["text"])
    first = _first_line(text)
    own_title = js_trim(_text(r.get("title")))
    title = own_title or first
    # Without a title its first line is the title, and not said twice.
    body = text if own_title else js_trim(text[text.find(first) + len(first) :])
    author = js_trim(_text(r.get("author")))
    by = f"{author}: " if author else ""
    uid = record_uid(r)
    reply = f"{by}{body}"
    t = time_of(r.get("at"))
    metadata = r.get("metadata")
    record: dict[str, Any] = {"id": r["id"]}
    for key in ("source", "author", "url"):
        if r.get(key):
            record[key] = r[key]
    if metadata:
        record["metadata"] = metadata
    return {
        "uid": uid,
        "fingerprint": _hash(
            title,
            reply,
            r.get("url"),
            r.get("kind"),
            r.get("collection"),
            r.get("thread"),
            num_str(t) if t is not None else "",
            stringify(metadata if metadata is not None else {}),
        ),
        "engine": js_trim(_text(r.get("kind"))) or "record",
        "session": js_trim(_text(r.get("thread"))) or uid,
        "turn": turn,
        "cwd": None,
        "project": collection_key(r.get("collection")),
        "title": title,
        "ts": t if t is not None else now,
        "synthetic": False,
        "typed": "human",
        "request": title,
        "reply": reply,
        "said": reply,
        "steps": [],
        "files": [r["url"]] if r.get("url") else [],
        "done": False,
        "record": record,
    }


def conversation_entries(c: Mapping[str, Any], now: int | float) -> list[StmEntry]:
    """A conversation's turns as records: each time the person speaks, what they said is the request and what the
    other side answered until they speak again is its reply — one record a request, so a question finds the exchange
    it was in. Turns before the first request (an assistant that opened) are the first record's own."""
    began = time_of(c.get("at"))
    out: list[StmEntry] = []
    request: list[str] = []
    reply: list[str] = []
    at: int | float | None = None

    def flush() -> None:
        nonlocal request, reply, at
        if not request and not reply:
            return
        n = len(out)
        # A time only the caller gave is part of the fingerprint: one defaulted from when it is added would make the
        # same conversation added again a changed one — read, embedded and consolidated again.
        given = at if at is not None else (None if began is None else began + n * 60_000)
        asked = "\n".join(request)
        answered = "\n".join(reply)
        title = _first_line_kept(asked) or _first_line_kept(answered) or c["id"]
        draft: dict[str, Any] = {"id": f"{c['id']}#{n}", "thread": c["id"], "kind": "conversation", "title": title, "text": asked}
        for key in ("source", "collection"):
            if c.get(key) is not None:
                draft[key] = c[key]
        if given is not None:
            draft["at"] = given
        e = entry_of(draft, now, n)
        out.append(
            {
                **e,
                "ts": given if given is not None else now + n * 60_000,
                "request": asked or e["title"],
                "reply": answered,
                "said": answered,
            }
        )
        request, reply, at = [], [], None

    for t in c["turns"]:
        author = js_trim(_text(t.get("author")))
        said = js_trim(t["text"])
        text = f"{author + ': ' if author else ''}{said}"
        if not said:
            continue
        if t["role"] == "user":
            if reply:
                flush()
            request.append(text)
        else:
            reply.append(text)
        if at is None:
            at = time_of(t.get("at"))
    flush()
    return [{**e, "fingerprint": _hash(e["fingerprint"], e["request"], e["reply"])} for e in out]


def _first_line_kept(text: str) -> str:
    """``text.split('\\n').find((l) => l.trim())?.trim()``."""
    return next((js_trim(line) for line in text.split("\n") if js_trim(line)), "")
