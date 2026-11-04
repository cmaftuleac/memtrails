"""
Consolidation — LONG-TERM memory. Like the brain: short-term memory keeps everything as it came; long-term memory
filters, stores, reorganises and reprioritises.

It reads the queue (the records log) a thread at a time — one that has settled (quiet for an hour, or a day old),
newest first — in windows of whole records (extract.py: one language-model ask a window). What survives the gate is
RECONCILED with what long-term memory already holds before it is written, so nothing is stored twice:

  - cosine >= 0.93 to a known item (0.97 for episodes) → the same memory: its provenance and ``seen`` grow, it is
    refreshed — no new item;
  - 0.80–0.93 → one fast-model decision a batch: same / updates / contradicts / related / different → merge, rewrite,
    supersede (the old one stays in the log, out of recall), add and link the two, or add;
  - below → added.

Then the window's entries are marked done and leave the short-term index: what mattered of them is long-term now.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import regex

from .._js import cut, iso_day, js_trim, now_ms, u16len
from ..llm import Ask, LlmAnswer, LlmRequest, ask_with, default_ask
from .embedder import Embed, Vector, doc_text
from .episodes import episode_body
from .extract import (
    WINDOW_CHARS,
    Candidate,
    Extraction,
    Observation,
    clean_concepts,
    clean_cues,
    extract_window,
    observe_window,
    windows_of,
)
from .qdrant_client import Filter
from .recall import RecallDeps, memory_search
from .redact import redact_string
from .store import LtmStore, MemoryItem, StmEntry, StmStore, fingerprint, learned_at
from .text import concept_key
from .writer import WriterDeps, embed_budget, write_concepts, write_cues, write_ltm

T = TypeVar("T")

SAME = 0.93
SAME_EPISODE = 0.97
#: An observation is the same only when it says the same thing (this close) about the same date: two dinners on two days
#: are two observations, and a list question needs both.
SAME_OBSERVATION = 0.97
RELATED = 0.80
#: A conversation is consolidated once it has been quiet this long — or its oldest waiting entry is a day old.
SETTLED_MS = 60 * 60_000
#: Extraction calls in flight: they are independent, and one at a time made a backfill take days (~86 s a window).
EXTRACT_CONCURRENCY = 8
_DAY_MS = 24 * 60 * 60_000

#: Requests of a window searched for, and the passages of other conversations kept to name what it refers to.
_RESOLVE_QUERIES = 4
_RESOLVE_KEPT = 8

#: Items given their cues and concepts after a run's windows (the backlog: saved notes, what was learned before
#: them), and how many go in one ask.
CUE_BACKLOG = 40
_CUE_BATCH = 10


@dataclass
class ConsolidateDeps:
    client: Any
    embed: Embed
    stm: StmStore
    ltm: LtmStore
    log: Callable[[str], None] = lambda _line: None
    aborted: Callable[[], bool] = lambda: False
    ask: Ask | None = None
    now: Callable[[], float] | None = None
    #: Windows the fast model extracts at once (EXTRACT_CONCURRENCY when not given).
    concurrency: int | None = None
    #: Windows reconciled at once (1, in order, when not given). A backfill sets more: a window may then not see what
    #: one reconciled beside it wrote — the duplicates that leaves are the sleep pass's to merge — and an item two
    #: windows touch at once is changed under its lock, from what it is by then.
    reconcile_concurrency: int | None = None
    #: A window's size in characters (WINDOW_CHARS when not given).
    window_chars: int | None = None
    #: Observations read apart, in windows of this many characters (observe_window); absent: in the window's one ask.
    observe_chars: int | None = None
    #: Each window read with what other conversations say that it may refer to (known_for): "the book you
    #: recommended" written as the book's name.
    resolve: bool = False
    #: What a window's reading may keep, checked before anything is reconciled or written. An answer it cannot give
    #: leaves the window for the next run, as an extraction that was not answered does.
    guard: Callable[[list[StmEntry], Extraction], Awaitable[Extraction | LlmAnswer]] | None = None

    def time(self) -> float:
        return self.now() if self.now else now_ms()


def new_report() -> dict[str, Any]:
    """A consolidation's report: windows, added, merged, updated, superseded, linked, dropped, episodes, observed,
    cued, and ``paused`` (why the model did not read) when it did not."""
    return {
        "windows": 0,
        "added": 0,
        "merged": 0,
        "updated": 0,
        "superseded": 0,
        "linked": 0,
        "dropped": 0,
        "episodes": 0,
        "observed": 0,
        "cued": 0,
    }


CUE_SYSTEM = """You index a person's long-term memory of their work with AI agents. For EACH item:
- cues: 4 to 8 requests the person might type to an agent LATER, in other work, when this item would be needed. Vary
  them: short and long, direct and indirect (describe the task, not the memory), never copying the title. Write them in
  the language the item is written in — when that is not English, make one or two of them English.
- concepts: the 3 to 12 most specific things it is about — files, folders, repositories, services, servers, tools,
  commands, APIs, libraries, models, products, settings, error names, people or organisations, project-specific terms
  — named the way the item names them (other terms in the item's language, never translated). No generic words.
Keep code, identifiers, paths and commands exactly as written. Return every id with its cues and concepts."""
_CUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "cues", "concepts"],
                "properties": {
                    "id": {"type": "string"},
                    "cues": {"type": "array", "items": {"type": "string"}},
                    "concepts": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

RELATIONS = ["same", "updates", "contradicts", "related", "different"]
#: Links an item keeps to others about the same subject (exploration walks them), newest last.
MAX_LINKS = 24

RELATE_SYSTEM = """You keep a person's long-term memory free of duplicates and stale knowledge without ever losing what it
knows. For EACH pair, compare the NEW memory with the KNOWN one and answer:
- same: they say the same thing (one may be more detailed) — give the merged version
- updates: NEW refines or extends KNOWN without contradicting it — give the merged version
- contradicts: NEW replaces KNOWN — the text shows KNOWN stopped being true (a decision reversed, a value changed, a
  thing removed or renamed)
- related: the same subject or system, different facts that belong together — reading one, you would want the other
- different: they are about different things
Rules:
- Different numbers, dates, versions, names, paths, commands or qualifiers mean they are NOT the same: updates when one
  extends the other, related otherwise.
- contradicts needs explicit evidence in the texts that KNOWN was replaced or ended; being newer is not evidence. When
  unsure between contradicts and related, answer related.
- For same and updates, write title (at most 12 words), summary (about 150 words, standing alone) and body. The merged
  body keeps EVERY fact of both — every value, path, command, error, code, reason and caveat, exactly as written —
  never shortened or dropped to be brief. For contradicts, related and different leave them empty.
- Never include secret values. Never translate."""
_RELATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "relation", "title", "summary", "body"],
                "properties": {
                    "id": {"type": "string"},
                    "relation": {"type": "string", "enum": RELATIONS},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
        },
    },
}

#: A decision on a pair: relation, title, summary, body.
Decision = dict[str, str]
#: Pairs asked about together: at most 8, and at most RELATE_CHARS of their texts — a long pair on its own.
RELATE_CHARS = 60_000

P = TypeVar("P", bound=Mapping[str, Any])


def _u16(s: str) -> int:
    return u16len(s)


def relate_batches(pairs: Sequence[P]) -> list[list[P]]:
    def size(p: Mapping[str, Any]) -> int:
        return _u16(p["it"]["summary"]) + _u16(p["it"]["body"]) + _u16(p["known"]["summary"]) + _u16(p["known"]["body"])

    out: list[list[P]] = []
    cur: list[P] = []
    chars = 0
    for p in pairs:
        if cur and (len(cur) >= 8 or chars + size(p) > RELATE_CHARS):
            out.append(cur)
            cur, chars = [], 0
        cur.append(p)
        chars += size(p)
    if cur:
        out.append(cur)
    return out


def stamped(it: MemoryItem, at: float) -> MemoryItem:
    return {**it, "learnedAt": at}


def label_no(id: str, letter: str) -> int:
    """The number of a label the model wrote back: ``p3``, or the line it labels quoted whole (``[p3] NEW …``) — 0 for
    none."""
    m = regex.search(rf"\b{letter}([0-9]+)", id, regex.ASCII)
    return int(m.group(1)) if m else 0


def _with(base: dict[str, Any], **optional: Any) -> dict[str, Any]:
    for k, v in optional.items():
        if v:
            base[k] = v
    return base


def item_of(c: Candidate, src: StmEntry, now: float, sources: Sequence[StmEntry] | None = None) -> MemoryItem:
    """One remembered candidate as a long-term item (not yet reconciled)."""
    project = src["project"] if c["scope"] == "project" else c["scope"]
    # The files its requests changed, from their recorded steps — by code, not the model: paths are what summaries lose.
    files = list(dict.fromkeys(f for e in (sources if sources is not None else [src]) for f in e["files"]))
    it = {
        "uid": f"mem:{fingerprint(c['kind'], c['scope'], project, c['title'], c['summary'])}",
        "kind": c["kind"],
        "scope": c["scope"],
        "project": src["project"],
        "engine": src["engine"],
        "session": src["session"],
        "cwd": src.get("cwd"),
        "where": src["title"],
        "ts": src["ts"] or now,
        "firstTs": src["ts"] or now,
        "title": c["title"],
        "summary": c["summary"],
        "body": c["body"],
        "files": files,
        "appliesWhen": c["appliesWhen"],
        "confidence": c["confidence"],
        "importance": c["importance"],
        "status": "active",
        "provenance": c["provenance"],
        "seen": 1,
        "fingerprint": fingerprint(c["kind"], c["title"], c["summary"], c["body"]),
        "cues": c["cues"],
        "concepts": c["concepts"],
    }
    return stamped(_with(it, origin=c.get("origin")), now)


def episode_item(e: StmEntry, key: Mapping[str, Any], now: float) -> MemoryItem:
    it = {
        "uid": f"ep:{e['uid']}",
        "kind": "episode",
        "scope": "project",
        "project": e["project"],
        "engine": e["engine"],
        "session": e["session"],
        "cwd": e.get("cwd"),
        "where": e["title"],
        "ts": e["ts"],
        "firstTs": e["ts"],
        "title": key.get("title") or e["title"],
        "summary": key["summary"],
        "body": episode_body(e),
        "files": e["files"],
        "appliesWhen": "",
        "confidence": "observed",
        "importance": 3 if e["files"] else 2,
        "status": "active",
        "provenance": [e["uid"]],
        "seen": 1,
        "fingerprint": fingerprint(e["request"], e["reply"], e["files"]),
        "cues": key.get("cues") or [],
        "concepts": key.get("concepts") or [],
    }
    return stamped(_with(it, outcome=key.get("outcome"), origin=key.get("origin")), now)


def observation_item(o: Observation, src: StmEntry, now: float) -> MemoryItem:
    """An observation as a long-term item: its text is its key, its date beside when it was said."""
    first = o["provenance"][0] if o["provenance"] else src["uid"]
    it: dict[str, Any] = {
        "uid": f"obs:{fingerprint(o['text'], o.get('eventAt'), first)}",
        "kind": "observation",
        "scope": "person",
        "project": src["project"],
        "engine": src["engine"],
        "session": src["session"],
        "cwd": src.get("cwd"),
        "where": src["title"],
        "ts": src["ts"],
        "firstTs": src["ts"],
        "title": cut(o["text"], 100),
        "summary": o["text"],
        "body": "",
        "files": [],
        "appliesWhen": "",
        "confidence": "stated",
        "importance": 2,
        "status": "active",
        "provenance": o["provenance"],
        "seen": 1,
        "fingerprint": fingerprint("observation", o["text"], o.get("eventAt")),
        "cues": o.get("cues") or [],
        "concepts": o["concepts"],
    }
    _with(it, eventAt=o.get("eventAt"), startAt=o.get("startAt"), endAt=o.get("endAt"))
    it["phase"] = o["phase"]
    return stamped(_with(it, about=o.get("about"), origin=o.get("origin")), now)


def _cosine(a: Vector, b: Vector) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    aa, bb = float(x @ x), float(y @ y)
    return float(x @ y) / math.sqrt(aa * bb) if aa and bb else 0.0


def _scope_filter(it: Mapping[str, Any]) -> Filter:
    """Where an item's duplicates can be: its project for project knowledge, anywhere for the global and the personal."""
    must: list[Any] = [{"key": "tier", "match": {"value": "ltm"}}, {"key": "status", "match": {"value": "active"}}]
    if it["scope"] == "project":
        must.append({"key": "project", "match": {"value": it["project"]}})
    else:
        must.append({"key": "scope", "match": {"any": ["global", "person"]}})
    # Observations are reconciled among themselves (observe): a fact is never merged into a dated event — nor into a
    # profile, which is written again from its items (profiles.py).
    return {"must": must, "must_not": [{"key": "kind", "match": {"any": ["observation", "profile"]}}]}


def _uniq(xs: Sequence[T]) -> list[T]:
    return list(dict.fromkeys(xs))


async def _after(before: asyncio.Future[None] | None, gate: asyncio.Future[None]) -> None:
    """A holder's turn: done once the one before it is, and it has let go."""
    if before is not None:
        await asyncio.shield(before)
    await asyncio.shield(gate)


def _retrieve_exception(t: asyncio.Future[Any]) -> None:
    if not t.cancelled():
        t.exception()


class Consolidator:
    def __init__(self, d: ConsolidateDeps) -> None:
        self.d = d
        self._vecs: dict[str, Vector] = {}
        #: Windows whose extraction failed, by their first request: how many times in a row.
        self._failed: dict[str, int] = {}
        #: Windows read in smaller parts from now on, by their first request: at most this many requests a part.
        self._narrowed: dict[str, int] = {}
        #: An item changed by one window at a time (windows reconciled at once): the last holder's turn, by uid.
        self._locks: dict[str, asyncio.Future[None]] = {}

    async def locked(self, uids: Sequence[str], run: Callable[[], Awaitable[T]]) -> T:
        """Runs with the items locked — taken in uid order, so two windows never wait on each other (sleep's answers
        too)."""
        loop = asyncio.get_running_loop()
        release: list[Callable[[], None]] = []
        for uid in sorted(set(uids)):
            before = self._locks.get(uid)
            gate: asyncio.Future[None] = loop.create_future()
            mine = asyncio.ensure_future(_after(before, gate))
            self._locks[uid] = mine
            if before is not None:
                await asyncio.shield(before)

            def free(u: str = uid, g: asyncio.Future[None] = gate, m: asyncio.Future[None] = mine) -> None:
                if not g.done():
                    g.set_result(None)
                if self._locks.get(u) is m:
                    del self._locks[u]

            release.append(free)
        try:
            return await run()
        finally:
            for r in release:
                r()

    def _latest(self, it: Mapping[str, Any]) -> MemoryItem | None:
        """The item as it is now, or None when it stopped being active (merged or superseded by a window beside this
        one)."""
        now = self.d.ltm.get(it["uid"])
        return now if now and now.get("status") == "active" else None

    async def _embed(self, texts: Sequence[str], timeout_ms: float | None = None) -> list[Vector]:
        """Embed with a per-run cache: the key is embedded once for reconcile and reused when it is written."""
        missing = _uniq([t for t in texts if t not in self._vecs])
        if missing:
            got = await self.d.embed(missing, timeout_ms)
            for i, v in enumerate(got):
                self._vecs[missing[i]] = v
        return [self._vecs[t] for t in texts]

    @property
    def _writer(self) -> WriterDeps:
        return WriterDeps(client=self.d.client, embed=self._embed)

    def pending(self, now: float | None = None) -> list[dict[str, Any]]:
        """Conversations with requests waiting, settled, newest first."""
        t = self.d.time() if now is None else now
        by: dict[str, dict[str, Any]] = {}
        for uid, m in self.d.stm.entries():
            if m.get("done") or m.get("synthetic"):
                continue
            s = by.setdefault(m["session"], {"uids": [], "first": math.inf, "last": 0})
            s["uids"].append(uid)
            s["first"] = min(s["first"], m["ts"])
            s["last"] = max(s["last"], m["ts"])
        # Quiet for an hour, or waiting a day since its oldest unread entry: a thread that never goes quiet (a message
        # every half hour) is still read.
        out = [
            {"session": session, "uids": s["uids"], "last": s["last"]}
            for session, s in by.items()
            if t - s["last"] >= SETTLED_MS or t - s["first"] >= _DAY_MS
        ]
        return sorted(out, key=lambda x: -x["last"])

    async def run(self, max_windows: int = 20) -> dict[str, Any]:
        """Consolidate up to ``max_windows`` windows. Stops early when the model pauses or the worker is stopped. The
        fast model extracts several windows at once (they are independent); each is reconciled in order, newest
        conversation first, so a later item still sees what an earlier one wrote."""
        r = new_report()
        wins: list[list[StmEntry]] = []
        for conv in self.pending():
            entries = sorted((e for e in (self.d.stm.get(u) for u in conv["uids"]) if e), key=lambda e: e["turn"])
            for w in windows_of(entries, self.d.window_chars or WINDOW_CHARS):
                for win in self._narrow(w):
                    if len(wins) < max_windows:
                        wins.append(win)
            if len(wins) >= max_windows:
                break
        limit = max(1, self.d.concurrency or EXTRACT_CONCURRENCY)
        extracting: list[asyncio.Task[Any]] = []
        # A pool: the next window starts when ANY one is read, not when the oldest is — one slow answer held every
        # other place idle while the windows after it waited their turn. They are still reconciled in order.
        state = {"next": 0, "reading": 0, "stopped": False}

        def launch() -> None:
            while not state["stopped"] and state["reading"] < limit and state["next"] < len(wins) and not self.d.aborted():
                win = wins[state["next"]]
                head = win[0]
                task = asyncio.ensure_future(self._read(win, {"cwd": head.get("cwd"), "engine": head["engine"], "title": head["title"]}))
                state["reading"] += 1

                def done(t: asyncio.Task[Any]) -> None:
                    state["reading"] -= 1
                    # One left behind by a pause or a stop is not an unhandled error; the awaited one still raises.
                    if not t.cancelled() and t.exception() is None:
                        launch()

                task.add_done_callback(done)
                extracting.append(task)
                state["next"] += 1

        launch()
        together = max(1, min(limit, self.d.reconcile_concurrency or 1))

        async def settle(i: int) -> str | None:
            """Window i, once read: reconciled, observed, marked done — or 'stop' when the run pauses."""
            try:
                x = await extracting[i]
            except BaseException:
                state["stopped"] = True
                raise
            win = wins[i]
            head = win[0]
            if "unavailable" in x or "busy" in x:
                # A window that failed before fails for itself (its answer too long to finish, say), not because the
                # model is away: it is read in halves from now on, and the windows after it go on — one window must
                # never hold up the rest, and none is marked done unread. A first failure before any window was read
                # pauses the run, as when the model is unreachable; after others were read it is the window's own, and
                # it waits for the next run.
                times = 0 if "busy" in x else self._failed.get(head["uid"], 0) + 1
                if times < 2:
                    if times:
                        self._failed[head["uid"]] = times
                    if r["windows"]:
                        return None
                    state["stopped"] = True
                    r["paused"] = "the fast model is busy" if "busy" in x else x["unavailable"]
                    return "stop"
                self._failed.pop(head["uid"], None)
                if len(win) > 1:
                    self._narrowed[head["uid"]] = math.ceil(len(win) / 2)
                why = x["unavailable"] if "unavailable" in x else "busy"
                then = "reading it in halves" if len(win) > 1 else "it waits for the next run"
                self.d.log(f"[memory] a window of {len(win)} requests failed again ({why}); {then}")
                return None
            self._failed.pop(head["uid"], None)
            now = self.d.time()
            by_uid = {e["uid"]: e for e in win}
            # A request the window's typed memories already came from is not kept again as an episode.
            learned = {u for c in x["candidates"] for u in c["provenance"]}
            kept = [k for k in x["episodes"] if k["uid"] in by_uid and k["uid"] not in learned]
            fresh = [episode_item(by_uid[k["uid"]], k, now) for k in kept]
            for c in x["candidates"]:
                src = by_uid.get(c["provenance"][-1] if c["provenance"] else "", head)
                fresh.append(item_of(c, src, now, [by_uid[u] for u in c["provenance"] if u in by_uid]))
            await self._reconcile(fresh, r)
            await self._observe(
                [observation_item(o, by_uid.get(o["provenance"][-1] if o["provenance"] else "", head), now) for o in x["observations"]],
                r,
            )
            r["dropped"] += x["dropped"]
            r["episodes"] += len(kept)
            # Read: marked done in the queue. The requests themselves stay searchable — the raw record is the truth and
            # the memories learned from them are keys to it. Each as it is now, not as it was read before the model's
            # long answer: one removed meanwhile stays removed, one changed meanwhile is read again.
            marked = []
            for e in win:
                meta = self.d.stm.meta(e["uid"])
                if not meta or meta.get("fp") != e["fingerprint"]:
                    continue
                cur = self.d.stm.get(e["uid"])
                if cur:
                    marked.append({**cur, "done": True})
            self.d.stm.put(marked)
            r["windows"] += 1
            # One at a time, each window's keys are dropped after it; windows at once share the cache until the run ends.
            if together == 1:
                self._vecs.clear()
            return None

        lane_state = {"at": 0, "halt": False}

        async def lane() -> None:
            while not lane_state["halt"] and lane_state["at"] < len(wins) and lane_state["at"] < len(extracting) and not self.d.aborted():
                i = lane_state["at"]
                lane_state["at"] += 1
                try:
                    if await settle(i) == "stop":
                        lane_state["halt"] = True
                except BaseException:
                    lane_state["halt"] = True
                    raise

        try:
            await asyncio.gather(*(lane() for _ in range(together)))
        finally:
            for t in extracting:
                t.add_done_callback(_retrieve_exception)
        if lane_state["halt"] and "paused" in r:
            return r
        self._vecs.clear()
        if not self.d.aborted():
            c = await self.fill_cues(CUE_BACKLOG)
            r["cued"] = c["cued"]
            if "paused" in c:
                r["paused"] = c["paused"]
        await asyncio.to_thread(self.d.ltm.compact)
        await asyncio.to_thread(self.d.stm.compact)
        return r

    async def _read(self, win: list[StmEntry], meta: dict[str, Any]) -> Any:
        """One window's reading, and what the guard (when there is one) lets it keep."""
        x = await self._extract(win, meta)
        return await self.d.guard(win, x) if self.d.guard and "candidates" in x else x

    async def _extract(self, win: list[StmEntry], meta: dict[str, Any]) -> Any:
        """One window's extraction: the one ask — and, when observations are read apart, the window's small windows too,
        each with the one before it as context. Any part not answered leaves the whole window for the next run."""
        chars = self.d.observe_chars
        known = await self._known_for(win) if self.d.resolve else None
        if not chars:
            return await extract_window(win, meta, ask=self.d.ask, known=known)
        subs = windows_of(win, chars)
        x, *seen = await asyncio.gather(
            extract_window(win, meta, ask=self.d.ask, observations=False, known=known),
            *(observe_window(sub, subs[k - 1] if k else [], meta, chars, ask=self.d.ask, known=known) for k, sub in enumerate(subs)),
        )
        if "candidates" not in x:
            return x
        failed = next((o for o in seen if "observations" not in o), None)
        if failed:
            return failed
        return {**x, "observations": [ob for o in seen for ob in o["observations"]]}

    async def _known_for(self, win: list[StmEntry]) -> list[str]:
        """What other conversations say that this window may refer to: a few of its requests searched for, the best
        passages outside this conversation, dated. Nothing found, or a search that fails, reads the window without it."""
        step = max(1, math.ceil(len(win) / _RESOLVE_QUERIES))
        asks = [q for q in (cut(e["request"], 600) for i, e in enumerate(win) if i % step == 0) if q][:_RESOLVE_QUERIES]
        session = win[0]["session"]
        deps = RecallDeps(client=self.d.client, embed=self.d.embed, now=self.d.now)
        try:
            found = [h for hits in await asyncio.gather(*(memory_search(deps, {"query": q, "limit": 6}) for q in asks)) for h in hits]
        except Exception:  # noqa: BLE001 - the window is read without it
            return []
        best: dict[str, Any] = {}
        for h in found:
            hid = h["id"]
            uid = hid[4:] if hid.startswith("stm:") else hid[3:] if hid.startswith("ep:") else None
            meta = self.d.stm.meta(uid) if uid else None
            if uid and meta and meta.get("session") == session:
                continue
            if (best[hid]["score"] if hid in best else -1) < h["score"]:
                best[hid] = h
        ranked = sorted(best.values(), key=lambda h: -h["score"])[:_RESOLVE_KEPT]
        return [f"({h.get('eventAt') or iso_day(h['ts'])}) {cut(h.get('match') or h['summary'], 400)}" for h in ranked]

    def _narrow(self, win: list[StmEntry]) -> list[list[StmEntry]]:
        """A window as it is read now: in parts of at most the size its earlier failures narrowed it to, each part
        narrowed again by its own failures."""
        n = self._narrowed.get(win[0]["uid"])
        if not n or n >= len(win):
            return [win]
        parts: list[list[StmEntry]] = []
        for i in range(0, len(win), n):
            parts += self._narrow(win[i : i + n])
        return parts

    async def fill_cues(self, max_items: int = CUE_BACKLOG) -> dict[str, Any]:
        """Encoding's backlog: active items without cues or concepts — saved notes and what was learned before either
        existed — get them, newest first, a few an ask, several asks at once. Cues an item already has stay."""
        # An observation is its own key, with the concepts it was extracted with; a profile has its subject as its
        # concept, and no cues (as measured).
        candidates = [
            (uid, m)
            for uid, m in self.d.ltm.entries()
            if m.get("status") == "active"
            and m.get("kind") not in ("observation", "profile")
            and (not m.get("cued") or not m.get("mapped"))
        ]
        todo = [it for it in (self.d.ltm.get(uid) for uid, _ in sorted(candidates, key=lambda c: -c[1]["ts"])[:max_items]) if it]
        batches = [todo[i : i + _CUE_BATCH] for i in range(0, len(todo), _CUE_BATCH)]
        limit = max(1, self.d.concurrency or EXTRACT_CONCURRENCY)
        cued = 0
        for i in range(0, len(batches), limit):
            if self.d.aborted():
                break
            answers = await asyncio.gather(*(self._cue_batch(b) for b in batches[i : i + limit]))
            paused = next((a for a in answers if "paused" in a), None)
            got = [it for a in answers for it in a.get("items", [])]
            if got:
                recued = [it for it in got if not (self.d.ltm.meta(it["uid"]) or {}).get("cued")]
                if recued:
                    await write_cues(self._writer, recued)
                await write_concepts(self._writer, got)
                self.d.ltm.put(got)
                cued += len(got)
            if paused:
                return {"cued": cued, "paused": paused["paused"]}
        return {"cued": cued}

    async def _cue_batch(self, items: list[MemoryItem]) -> dict[str, Any]:
        def entry(n: int, it: MemoryItem) -> str:
            needed = f"\nNeeded when: {it['appliesWhen']}" if it.get("appliesWhen") else ""
            return f"[i{n + 1}] ({it['kind']}) {it['title']}\n{it['summary']}{needed}"

        prompt = "\n\n".join(entry(n, it) for n, it in enumerate(items))
        a = await ask_with(
            self.d.ask or default_ask,
            LlmRequest(
                purpose="memory",
                system=CUE_SYSTEM,
                prompt=redact_string(prompt),
                schema=_CUE_SCHEMA,
                name="memory_cues",
                max_tokens=550 * len(items) + 200,
                timeout_ms=120_000,
                consequence="so these memories wait for their cues",
            ),
        )
        if "unavailable" in a:
            return {"paused": a["unavailable"]}
        if "busy" in a:
            return {"paused": "the fast model is busy"}
        out: list[MemoryItem] = []
        answer: dict[str, Any] = a["answer"] if isinstance(a.get("answer"), dict) else {}
        for x in answer.get("items") or []:
            if not isinstance(x, Mapping):
                continue
            n = label_no(x["id"], "i") if isinstance(x.get("id"), str) else 0
            it = items[n - 1] if 1 <= n <= len(items) else None
            if not it:
                continue
            cues = it.get("cues") or clean_cues(x.get("cues"))
            concepts = it.get("concepts") or clean_concepts(x.get("concepts"))
            if cues or concepts:
                out.append({**it, "cues": cues, "concepts": concepts})  # what is missing is asked again next time
        return {"items": out}

    async def _reconcile(self, items: list[MemoryItem], r: dict[str, Any]) -> None:
        """Each item against what long-term memory holds — written as it goes, so a later item sees an earlier one."""
        ask: list[dict[str, Any]] = []
        # Every key of the window in one embedding call (the cache serves each below and again when it is written).
        await self._embed([doc_text(it["title"], it["summary"]) for it in items], embed_budget(len(items)))
        for it in items:
            v = (await self._embed([doc_text(it["title"], it["summary"])]))[0]
            near = await self.d.client.neighbors(v.tolist(), _scope_filter(it), 3)
            top = next((n for n in near if n["item"] != it["uid"]), None)
            known = self.d.ltm.get(top["item"]) if top else None
            same = SAME_EPISODE if it["kind"] == "episode" else SAME
            if known and top and top["score"] >= same:

                async def merge_into(k: MemoryItem = known, i: MemoryItem = it) -> bool:
                    now = self._latest(k)
                    if now:
                        await self.merge(now, i, None)
                    return now is not None

                if await self.locked([known["uid"]], merge_into):
                    r["merged"] += 1
                    continue
            elif known and top and top["score"] >= RELATED and it["kind"] != "episode" and known["kind"] != "episode":
                ask.append({"it": it, "known": known})
                continue
            await self._add(it)
            r["added"] += 1
        # The pairs are fixed before any is asked about: every batch asked at once, the answers applied in order.
        batches = relate_batches(ask)
        answers = await asyncio.gather(*(self.relate(b) for b in batches))
        for b, batch in enumerate(batches):
            got = answers[b]
            for n, pair in enumerate(batch):
                it, known = pair["it"], pair["known"]
                d = got.get(f"p{n + 1}") if "paused" not in got else None
                if not d or d["relation"] == "different":
                    await self._add(it)
                    r["added"] += 1
                    continue

                async def apply(it: MemoryItem = it, known: MemoryItem = known, d: Decision = d) -> None:
                    # Changed by a window beside this one since it was asked about: the answer was about another text —
                    # kept beside it, linked, for the sleep pass to read again.
                    now = self._latest(known)
                    moved = not now or now["fingerprint"] != known["fingerprint"]
                    if d["relation"] == "related" or (moved and d["relation"] != "contradicts"):
                        await self._add(it)
                        r["added"] += 1
                        if now:
                            await self.link(it, now)
                            r["linked"] += 1
                        return
                    if d["relation"] == "contradicts":
                        await self._add(it)
                        r["added"] += 1
                        if now:
                            await self.retire(now, it["uid"])
                            r["superseded"] += 1
                        return
                    assert now is not None
                    text = (
                        {"title": d["title"], "summary": d["summary"], "body": d["body"] or now["body"]}
                        if d["title"] and d["summary"]
                        else None
                    )
                    await self.merge(now, it, text)
                    if d["relation"] == "updates":
                        r["updated"] += 1
                    else:
                        r["merged"] += 1

                await self.locked([it["uid"], known["uid"]], apply)

    async def _observe(self, items: list[MemoryItem], r: dict[str, Any]) -> None:
        """Observations are only ever added: one is the same as a known one only when it says the same
        (SAME_OBSERVATION) about the same date — then the known one is told again (provenance, ``seen``). Never asked
        about, never merged into another text, never superseded: an older one stays, dated, beside the newer."""
        if not items:
            return
        # The window's observations looked up together and written together: one embedding call, one batched query,
        # one write — and each also against the ones before it in this window.
        vecs = await self._embed([doc_text(it["title"], it["summary"]) for it in items], embed_budget(len(items)))
        filter: Filter = {
            "must": [
                {"key": "tier", "match": {"value": "ltm"}},
                {"key": "status", "match": {"value": "active"}},
                {"key": "kind", "match": {"value": "observation"}},
                {"key": "part", "match": {"value": "key"}},
            ]
        }
        near = await self.d.client.nearest_many([v.tolist() for v in vecs], filter, 3, 30_000)
        fresh: list[dict[str, Any]] = []
        #: Told again in this window: how many times, by whom, the newest time.
        told: dict[str, dict[str, Any]] = {}
        for n, it in enumerate(items):
            event = it.get("eventAt") or ""
            from_window = next(
                (
                    f
                    for f in fresh
                    if f["it"]["uid"] != it["uid"]
                    and (f["it"].get("eventAt") or "") == event
                    and _cosine(f["v"], vecs[n]) >= SAME_OBSERVATION
                ),
                None,
            )
            if from_window:
                was = from_window["it"]
                from_window["it"] = {
                    **was,
                    "seen": was["seen"] + 1,
                    "ts": max(was["ts"], it["ts"]),
                    "provenance": _uniq([*was["provenance"], *it["provenance"]]),
                }
                r["merged"] += 1
                continue
            hit = next((h for h in (near[n] if n < len(near) else []) if h["item"] != it["uid"]), None)
            known = self.d.ltm.get(hit["item"]) if hit and hit["score"] >= SAME_OBSERVATION else None
            if known and (known.get("eventAt") or "") == event:
                t = told.get(known["uid"], {"times": 0, "ts": 0, "provenance": []})
                told[known["uid"]] = {
                    "times": t["times"] + 1,
                    "ts": max(t["ts"], it["ts"]),
                    "provenance": _uniq([*t["provenance"], *it["provenance"]]),
                }
                r["merged"] += 1
                continue
            fresh.append({"it": it, "v": vecs[n]})
            r["observed"] += 1
        # Told again: what this window adds (times, sources, the newest time) put onto the item as it is by now — a
        # window beside this one may have told it too.
        for uid, t in told.items():

            async def tell(uid: str = uid, t: dict[str, Any] = t) -> None:
                was = self.d.ltm.get(uid)
                if not was:
                    return
                now = {
                    **was,
                    "ts": max(was["ts"], t["ts"]),
                    "seen": was["seen"] + t["times"],
                    "provenance": _uniq([*was["provenance"], *t["provenance"]]),
                }
                await self.d.client.set_payload([uid], {"ts": now["ts"]})
                self.d.ltm.put([now])

            await self.locked([uid], tell)
        if fresh:
            await write_ltm(self._writer, [f["it"] for f in fresh])
            self.d.ltm.put([f["it"] for f in fresh])

    async def _add(self, it: MemoryItem) -> None:
        await write_ltm(self._writer, [it])
        self.d.ltm.put([it])

    async def merge(self, known: MemoryItem, it: MemoryItem, text: Mapping[str, str] | None) -> None:
        """The known item absorbs the new one: provenance, ``seen``, the newest time, the higher importance — and, when
        the model merged the texts, the merged key (embedded again). Nothing of either is lost: the new item's own text,
        and the known one's before a rewrite, stay in the log as ``merged`` records the item lists (``absorbed``)."""
        kept: list[MemoryItem] = [{**it, "status": "merged", "mergedInto": known["uid"]}]
        if text:
            kept.append({**known, "uid": f"{known['uid']}@{known['fingerprint']}", "status": "merged", "mergedInto": known["uid"]})
        merged: MemoryItem = {**known}
        if text:
            merged.update(title=redact_string(text["title"]), summary=redact_string(text["summary"]), body=redact_string(text["body"]))
        concepts: dict[str, str] = {}
        for c in [*(known.get("concepts") or []), *(it.get("concepts") or [])]:
            concepts[concept_key(c)] = c
        merged.update(
            ts=max(known["ts"], it["ts"]),
            importance=max(known["importance"], it["importance"]),
            seen=known["seen"] + 1,
            provenance=_uniq([*known["provenance"], *it["provenance"]]),
            files=_uniq([*known["files"], *it["files"]]),
            confidence="verified" if known["confidence"] == "verified" or it["confidence"] == "verified" else known["confidence"],
            cues=_uniq([*(known.get("cues") or []), *(it.get("cues") or [])]),
            related=[u for u in _uniq([*(known.get("related") or []), *(it.get("related") or [])]) if u != known["uid"]][-MAX_LINKS:],
            concepts=list(concepts.values()),
            absorbed=_uniq([*(known.get("absorbed") or []), *(k["uid"] for k in kept)]),
        )
        merged["fingerprint"] = fingerprint(merged["kind"], merged["title"], merged["summary"], merged["body"])
        # Changed now: what sleep relates next.
        merged["learnedAt"] = max(learned_at(known), learned_at(it))
        if text:
            await write_ltm(self._writer, [merged])
        else:
            await self.d.client.set_payload([known["uid"]], {"ts": merged["ts"], "importance": merged["importance"]})
            if "\n".join(merged["related"]) != "\n".join(known.get("related") or []):
                await self.d.client.set_payload([known["uid"]], {"related": merged["related"]}, True)
            if len(merged["concepts"]) > len(known.get("concepts") or []):
                await write_concepts(self._writer, [merged])
            # Learned again in other words: its new cues are found from now on.
            if len(merged["cues"]) > len(known.get("cues") or []):
                await write_cues(self._writer, [merged])
        self.d.ltm.put([*kept, merged])

    async def link(self, a: MemoryItem, b: MemoryItem) -> None:
        """Two items about the same subject that say different things: each lists the other (memory_explore walks these)
        — the newest MAX_LINKS, so an item on a busy subject does not grow its list, and its payload, without end."""

        def add(x: MemoryItem, y: MemoryItem) -> MemoryItem:
            return {**x, "related": [*(u for u in (x.get("related") or []) if u != y["uid"]), y["uid"]][-MAX_LINKS:]}

        one, two = add(a, b), add(b, a)
        self.d.ltm.put([one, two])
        await self.d.client.set_payload([one["uid"]], {"related": one["related"]}, True)
        await self.d.client.set_payload([two["uid"]], {"related": two["related"]}, True)

    async def retire(self, known: MemoryItem, by: str) -> None:
        """Superseded: out of the index and of recall, kept in the log with what replaced it."""
        old = {**known, "status": "superseded", "supersededBy": by, "supersededAt": self.d.time()}
        await write_ltm(self._writer, [old])
        self.d.ltm.put([old])

    async def relate(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The model's decision for each pair, by its label (``p1``, …) — or ``{"paused": why}`` when it did not
        answer."""
        # Whole texts: a merge the model writes from a shortened body loses the rest of it.
        prompt = "\n\n".join(
            f"[p{n + 1}]\nNEW ({p['it']['kind']}): {p['it']['title']}\n{p['it']['summary']}\n{p['it']['body']}\n\n"
            f"KNOWN ({p['known']['kind']}): {p['known']['title']}\n{p['known']['summary']}\n{p['known']['body']}"
            for n, p in enumerate(batch)
        )
        # Room for every merged text in full (≈ 2.5 characters a token, the two sides together), with the decisions.
        room = sum(
            math.ceil((_u16(p["it"]["summary"]) + _u16(p["it"]["body"]) + _u16(p["known"]["summary"]) + _u16(p["known"]["body"])) / 2.5)
            for p in batch
        )
        a = await ask_with(
            self.d.ask or default_ask,
            LlmRequest(
                purpose="memory",
                system=RELATE_SYSTEM,
                prompt=redact_string(prompt),
                schema=_RELATE_SCHEMA,
                name="memory_reconcile",
                max_tokens=900 * len(batch) + 200 + room,
                timeout_ms=300_000,
                consequence="so these memories are kept apart for now",
            ),
        )
        if "unavailable" in a:
            return {"paused": a["unavailable"]}
        if "answer" not in a:
            return {"paused": "the fast model is busy"}
        out: dict[str, Any] = {}
        answer = a["answer"] if isinstance(a["answer"], dict) else {}
        for p in answer.get("pairs") or []:
            if not isinstance(p, Mapping):
                continue
            # A small model may quote the label's line rather than the label: the number is looked for.
            n = label_no(p["id"], "p") if isinstance(p.get("id"), str) else 0
            if not n or str(p.get("relation")) not in RELATIONS:
                continue
            out[f"p{n}"] = {
                "relation": p["relation"],
                **{key: js_trim(p[key]) if isinstance(p.get(key), str) else "" for key in ("title", "summary", "body")},
            }
        return out
