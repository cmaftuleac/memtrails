"""
What a memory holds, on disk, as two append-only logs in its home:

  - ``records.jsonl`` — SHORT-TERM memory: every record as it was added (records.py), kept for good; also the queue the
    consolidation worker reads (``done``);
  - ``knowledge.jsonl`` — LONG-TERM memory: what consolidation kept (consolidate.py) — typed items, dated observations,
    kept episodes, profiles — with their keys, status (active / superseded / archived) and provenance.

The logs are the truth; Qdrant is derived from them (rebuilt without asking a model, after a Qdrant upgrade or a lost
index). In memory: each record's place in its file and a small projection — never the text. A later line for the
same uid replaces the earlier one; a log is rewritten without replaced lines once it is twice what is live.

``usage.json``: how often each long-term item was shown and opened (the sleep pass's "use", MemRL-style).
``recall.jsonl``: what each search showed, for what, and what was opened after (recall_log.py).

The files are the reference library's, byte for byte: a memory written by one opens in the other.

The records are JSON objects (dicts). A short-term entry (``StmEntry``): uid, fingerprint, engine (the record's kind,
``conversation`` for an exchange), session (its thread), turn, cwd, project (its collection key), title, ts,
synthetic, typed, request, reply, said, steps, files, done, and ``record`` (its own id, source, author, url,
metadata). A long-term item (``MemoryItem``): uid, kind, scope, project, engine, session, cwd, where, ts, firstTs,
learnedAt, title, summary, body, files, appliesWhen, confidence, importance, status (active, superseded, archived,
merged), supersededBy, supersededAt, outcome, mergedInto, absorbed, provenance, seen, fingerprint, cues, origin,
related, concepts, and for an observation eventAt, startAt, endAt, phase, about.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Generic, TypeVar

from .._fs import append_all, write_all, write_file
from .._js import now_ms, parse, stringify, utf8

#: A record of the short-term log (see above).
StmEntry = dict[str, Any]
#: An item of the long-term log (see above).
MemoryItem = dict[str, Any]

M = TypeVar("M")


def fingerprint(*parts: str | Sequence[str] | None) -> str:
    """SHA-1 over the parts, each followed by NUL (a list joined by U+0001), first 16 hex."""
    h = hashlib.sha1()
    for p in parts:
        text = p if isinstance(p, str) else "" if p is None else "\u0001".join(p)
        h.update(utf8(text))  # a lone surrogate as U+FFFD, as Node hashes it
        h.update(b"\0")
    return h.hexdigest()[:16]


#: What a compaction holds before it writes.
_WRITE_BYTES = 1 << 20


def _dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


class JsonlLog(Generic[M]):
    """An append-only JSONL log, read back by byte offset: a later record for the same uid replaces the earlier one.
    ``project`` is what stays in memory for each record.

    A small append or read is done where it is asked for; a load or a compaction, which reads the whole file, is run on
    a thread (``await asyncio.to_thread(log.load)``). A lock keeps a put from landing in the middle of a load; a
    compaction holds it only to take its snapshot and to swap the new file in, so a get or put on the event loop never
    waits for the whole file to be read and written."""

    def __init__(self, path: str, project: Callable[[dict[str, Any]], M]) -> None:
        self._path = path
        self._project = project
        self._live: dict[str, tuple[int, int, M]] = {}
        self._lines = 0
        self._end = 0
        self._lock = threading.RLock()
        #: One compaction at a time; a load since a compaction's snapshot voids it (``_loads``).
        self._compacting = threading.Lock()
        self._loads = 0

    @property
    def path(self) -> str:
        return self._path

    def size(self) -> int:
        return len(self._live)

    def meta(self, uid: str) -> M | None:
        s = self._live.get(uid)
        return s[2] if s else None

    def entries(self) -> Iterator[tuple[str, M]]:
        for uid, s in list(self._live.items()):
            yield uid, s[2]

    def _load(self) -> None:
        self._loads += 1
        self._live.clear()
        self._lines = 0
        self._end = 0
        if not os.path.exists(self._path):
            return

        def take(line: bytes, at: int, length: int) -> bool:
            if not line:
                return True
            try:
                rec = parse(line.decode("utf-8", "replace"))
            except ValueError:
                return False
            if isinstance(rec, dict):
                if rec.get("removed"):
                    self._live.pop(rec.get("uid"), None)  # type: ignore[arg-type]
                else:
                    self._live[rec.get("uid")] = (at, length, self._project(rec))  # type: ignore[index]
            self._lines += 1
            return True

        # By bytes, not decoded lines: a torn multi-byte character must not shift every offset after it.
        at = 0
        tail = b""
        with open(self._path, "rb") as f:
            for raw in f:
                if not raw.endswith(b"\n"):
                    tail = raw
                    break
                line = raw[:-1]
                take(line, at, len(raw))
                at += len(raw)
        # A crash cut the last line short of its '\n': whole but for it, it is ended; torn, it goes. Either way what
        # is added next starts on a line of its own, at the offset the index gives it.
        if tail:
            try:
                # Its length without the '\n' it is given: readable whether or not that write lands.
                if take(tail, at, len(tail)):
                    append_all(self._path, b"\n")
                else:
                    os.truncate(self._path, at)
            except OSError:
                pass  # read-only: nothing is added after it either
        self._end = os.stat(self._path).st_size

    def _get(self, uid: str) -> dict[str, Any] | None:
        s = self._live.get(uid)
        if not s:
            return None
        try:
            with open(self._path, "rb") as f:
                return self._read(f, s)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read(f: Any, s: tuple[int, int, Any]) -> dict[str, Any]:
        f.seek(s[0])
        rec = parse(f.read(s[1]).decode("utf-8", "replace").rstrip())
        if not isinstance(rec, dict):
            raise ValueError("not a record")
        return rec

    def _put(self, recs: Sequence[dict[str, Any]]) -> None:
        if not recs:
            return
        _dir(self._path)
        if not self._end and os.path.exists(self._path):
            self._end = os.stat(self._path).st_size
        chunks: list[bytes] = []
        slots: list[tuple[str, tuple[int, int, M]]] = []
        at = self._end
        for rec in recs:
            b = (stringify(rec) + "\n").encode("utf-8", "surrogatepass")
            slots.append((rec["uid"], (at, len(b), self._project(rec))))
            chunks.append(b)
            at += len(b)
        # Written first, then indexed: a write that fails leaves the index pointing at nothing past the file's end.
        append_all(self._path, b"".join(chunks))
        self._live.update(slots)
        self._end = at
        self._lines += len(recs)

    def _remove(self, uids: Sequence[str]) -> None:
        gone = [u for u in uids if u in self._live]
        if not gone:
            return
        _dir(self._path)
        b = "".join(stringify({"uid": u, "removed": True}) + "\n" for u in gone).encode("utf-8", "surrogatepass")
        append_all(self._path, b)
        if not self._end:
            self._end = os.stat(self._path).st_size - len(b)
        self._end += len(b)
        self._lines += len(gone)
        for u in gone:
            self._live.pop(u, None)

    def load(self) -> None:
        """Reads the file's index in (each record's place and projection, never its text)."""
        with self._lock:
            self._load()

    def get(self, uid: str) -> dict[str, Any] | None:
        with self._lock:
            return self._get(uid)

    def put(self, recs: Sequence[dict[str, Any]]) -> None:
        with self._lock:
            self._put(recs)

    def remove(self, uids: Sequence[str]) -> None:
        """Forget these records: gone from memory, and from the file when it is next compacted."""
        with self._lock:
            self._remove(uids)

    def compact(self) -> None:
        """Rewrites the log without replaced lines once it is twice what is live (see _compact). One running already
        does it: this returns."""
        if not self._compacting.acquire(blocking=False):
            return
        try:
            self._compact()
        finally:
            self._compacting.release()

    def all(self) -> Iterator[dict[str, Any]]:
        """Every live record, in file order."""
        for uid, _ in sorted(self._live.items(), key=lambda kv: kv[1][0]):
            r = self.get(uid)
            if r is not None:
                yield r

    def _compact(self) -> None:
        """Rewrite the log without replaced lines once it is twice what is live. Record by record into a file beside it,
        made durable and then renamed over it: a log of any size, and a crash leaves the old one whole. A record that
        cannot be read is never dropped by a rewrite: it raises, and the log stays as it is.

        The records are copied from a snapshot of the index, without the lock: puts and removes go on meanwhile, at the
        end of the file. At the swap, under the lock, what they appended is carried over after the copied records as it
        is, and their places moved with it."""
        with self._lock:
            if self._lines < 2 * len(self._live) + 1000:
                return
            if not self._end and os.path.exists(self._path):
                self._end = os.stat(self._path).st_size
            order = sorted(self._live.items(), key=lambda kv: kv[1][0])
            snap_end, snap_lines, loads = self._end, self._lines, self._loads
        tmp = f"{self._path}.compact"
        nxt: dict[str, tuple[int, int]] = {}
        out: int | None = None
        swapped = False
        try:
            with open(self._path, "rb") as src:
                out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                at = 0
                parts: list[bytes] = []
                size = 0
                for uid, slot in order:
                    try:
                        rec: dict[str, Any] | None = self._read(src, slot)
                    except (OSError, ValueError):
                        rec = None
                    if rec is None or rec.get("uid") != uid:
                        raise RuntimeError(f"{self._path}: record {uid} cannot be read at {slot[0]}; not compacted")
                    b = (stringify(rec) + "\n").encode("utf-8", "surrogatepass")
                    nxt[uid] = (at, len(b))
                    parts.append(b)
                    size += len(b)
                    at += len(b)
                    if size >= _WRITE_BYTES:
                        write_all(out, b"".join(parts))
                        parts, size = [], 0
                if size:
                    write_all(out, b"".join(parts))
            os.fsync(out)
            with self._lock:
                if self._loads != loads:
                    return  # a load replaced the snapshot: the next compaction starts from what it read
                tail = b""
                if self._end > snap_end:
                    with open(self._path, "rb") as src:
                        src.seek(snap_end)
                        tail = src.read(self._end - snap_end)
                    if len(tail) != self._end - snap_end:
                        raise RuntimeError(f"{self._path}: shorter than its index; not compacted")
                    write_all(out, tail)
                    os.fsync(out)
                os.close(out)
                out = None
                os.replace(tmp, self._path)
                swapped = True
                # A place before the snapshot's end is a copied record's (puts and removes only append); one after it
                # was appended meanwhile, and moved with the tail.
                shift = at - snap_end
                self._live = {
                    uid: (*nxt[uid], meta) if slot_at < snap_end else (slot_at + shift, length, meta)
                    for uid, (slot_at, length, meta) in self._live.items()
                }
                self._end = at + len(tail)
                self._lines = len(nxt) + self._lines - snap_lines
        finally:
            if out is not None:
                os.close(out)
            if not swapped:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass


def stm_meta_of(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "fp": r.get("fingerprint"),
        "done": r.get("done"),
        "ts": r.get("ts"),
        "session": r.get("session"),
        "project": r.get("project"),
        "synthetic": r.get("synthetic"),
    }


def ltm_meta_of(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "fp": r.get("fingerprint"),
        "status": r.get("status"),
        "kind": r.get("kind"),
        "scope": r.get("scope"),
        "project": r.get("project"),
        "ts": r.get("ts"),
        "importance": r.get("importance"),
        "cued": bool(r.get("cues")),
        "mapped": bool(r.get("concepts")),
        "learnedAt": r.get("learnedAt"),
    }


StmStore = JsonlLog[dict[str, Any]]
LtmStore = JsonlLog[dict[str, Any]]


def stm_log(path: str) -> StmStore:
    return JsonlLog(path, stm_meta_of)


def ltm_log(path: str) -> LtmStore:
    return JsonlLog(path, ltm_meta_of)


class UsageBook:
    """How often each long-term item was shown with a turn and opened by an agent. Saved at most once a minute (a timer
    on the event loop that noted it), and when the memory closes."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._dirty = False
        self._timer: asyncio.TimerHandle | None = None
        try:
            with open(path, encoding="utf-8") as f:
                data = parse(f.read())
            self._data: dict[str, dict[str, Any]] = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            self._data = {}

    def of(self, uid: str) -> dict[str, Any]:
        u = self._data.get(uid)
        return dict(u) if u else {"shown": 0, "opened": 0, "lastUsed": 0}

    def note(self, uids: Sequence[str], what: str, now: int | float | None = None) -> None:
        t = now_ms() if now is None else now
        for uid in uids:
            u = self._data.setdefault(uid, {"shown": 0, "opened": 0, "lastUsed": 0})
            u[what] = u.get(what, 0) + 1
            u["lastUsed"] = t
        self._dirty = True
        if self._timer is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no event loop: saved when it is closed
            self._timer = loop.call_later(60.0, self._tick)

    def _tick(self) -> None:
        self._timer = None
        self.flush()

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            _dir(self._path)
            write_file(self._path, stringify(self._data).encode("utf-8"))
            self._dirty = False
        except OSError:
            pass  # read-only: kept in memory

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.flush()


def learned_at(m: dict[str, Any]) -> Any:
    """When an item was written to long-term memory. ``ts`` is its source's own time — for history imported today,
    long ago — so "learned since the last sleep" and "unused for half a year" read this. An item written before the
    field existed has only ``ts``."""
    v = m.get("learnedAt")
    return m.get("ts") if v is None else v
