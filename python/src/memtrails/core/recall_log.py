"""
What recall showed, for what, and what was opened after it — the record a model that learns from USE reads.

Learned only from what memory says and how it is laid out, two light models did not beat the first stage on the answers
shown (docs/RESEARCH.md). What works in the literature is learned from use — links that strengthen when memories are
recalled together and fade when not, rankers trained on what was opened. The usage counts (store.py UsageBook) say how
often, never with what nor for what; this says both.

One line per event in ``recall.jsonl`` beside the logs — on this machine only, cleared with the index, never sent:
  {"t", "ev": "search", "project", "query", "shown": [ids]}       memory_search
  {"t", "ev": "open",   "item"}                                    memory_open
Appended in order, a line at a time; a write that fails is dropped (recall never waits on, or fails for, its record).
Kept: the last RECALL_KEEP_MS, at most RECALL_KEEP_BYTES of the newest (``trim()``: when memory opens, and every
RECALL_TRIM_EVERY events).
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from typing import Any

from .._fs import append_all, write_file
from .._js import now_ms, parse, stringify, u16slice

#: A query kept up to this: the start of a long message is what the search embedded first.
RECALL_QUERY_CHARS = 2000
RECALL_KEEP_MS = 180 * 24 * 60 * 60_000
RECALL_KEEP_BYTES = 32 * 1024 * 1024
#: Events between trims of a book left open for long (a server): the cap holds without a restart.
RECALL_TRIM_EVERY = 1000


class RecallBook:
    """The record, appended a line at a time. Writes are done in order under a lock: a line is a small append, and what
    is recorded is there when ``record`` returns; a trim, which reads the whole file, runs on a thread."""

    def __init__(
        self, path: str, now: Callable[[], float] = now_ms, keep_ms: float = RECALL_KEEP_MS, keep_bytes: int = RECALL_KEEP_BYTES
    ) -> None:
        self._path = path
        self._now = now
        self._keep_ms = keep_ms
        self._keep_bytes = keep_bytes
        self._lock = threading.Lock()
        self._appended = 0
        self._trimming: asyncio.Task[int] | None = None

    def record(self, e: dict[str, Any]) -> None:
        """``e``: ``{"ev": "search", "project"?: …, "query": …, "shown": [...]}`` or ``{"ev": "open", "item": …}``."""
        line: dict[str, Any] = {"t": self._now(), **e}
        if "query" in e:
            line["query"] = u16slice(e["query"], 0, RECALL_QUERY_CHARS)
        data = (stringify(line) + "\n").encode("utf-8")
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
                append_all(self._path, data)
            except OSError:
                pass  # recall never fails for its record
            self._appended += 1
            due = self._appended % RECALL_TRIM_EVERY == 0
        if due:
            # A trim reads the whole record: on a thread when there is an event loop to keep free.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self.trim()
                return
            self._trimming = loop.create_task(asyncio.to_thread(self.trim))

    def trim(self) -> int:
        """The events older than ``keep_ms`` dropped, then the oldest until the rest fit ``keep_bytes`` — rewritten
        between appends (a new file renamed over the old). How many were dropped; 0 on any failure, the record as it
        was."""
        with self._lock:
            try:
                with open(self._path, "rb") as f:
                    text = f.read().decode("utf-8", "replace")
            except OSError:
                return 0
            try:
                lines = [x for x in text.split("\n") if x]
                since = self._now() - self._keep_ms

                def recent(line: str) -> bool:
                    try:
                        t = parse(line).get("t")
                        return isinstance(t, (int, float)) and t >= since
                    except (ValueError, AttributeError):
                        return False

                kept = [x for x in lines if recent(x)]
                size = sum(len(x.encode("utf-8")) + 1 for x in kept)
                start = 0
                while size > self._keep_bytes and start < len(kept):
                    size -= len(kept[start].encode("utf-8")) + 1
                    start += 1
                kept = kept[start:]
                if len(kept) == len(lines):
                    return 0
                nxt = f"{self._path}.trim"
                write_file(nxt, "".join(f"{x}\n" for x in kept).encode("utf-8"))
                os.replace(nxt, self._path)
                return len(lines) - len(kept)
            except OSError:
                return 0
