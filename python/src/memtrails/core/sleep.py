"""
Sleep — the daily pass that keeps long-term memory organised (LightMem's sleep-time update; MemoryOS's heat):

  1. RELATE what was learned since the last sleep to every near neighbour (cosine >= 0.80, any knowledge kind, the same
     scope) — not only the nearest one consolidation looked at: from 0.93 they merge; below, the fast model merges the
     two, says one replaces the other (superseded), links them (the same subject, different facts) or says they differ;
  2. PRIORITY for every active item = importance × recency (a half-life by kind: preferences, architecture, references
     and notes do not fade; facts and pitfalls fade in a year; episodes in three months) × use (how often recall showed
     it and an agent opened it — MemRL-style utility);
  3. ARCHIVE what stays low and unused for half a year: out of the index, kept in the log. Nothing is forgotten
     outright: everything is remembered, efficiently;
  4. PROFILES (profiles.py): for each subject enough items share, what they say gathered into one dated list — written
     again when those items changed, archived when the subject no longer qualifies.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .._aio import gather_or_cancel
from .._fs import write_file
from .._js import parse, stringify
from .consolidate import RELATED, SAME, ConsolidateDeps, Consolidator, Decision, relate_batches
from .embedder import doc_text
from .profiles import write_profiles
from .qdrant_client import Filter, point_id
from .store import MemoryItem, UsageBook, learned_at
from .writer import write_ltm

_DAY = 24 * 60 * 60_000
#: A fresh item's neighbours looked at, and the pairs a night asks about (a large first sleep goes on the next nights).
_NEIGHBOURS = 8
PAIRS_PER_SLEEP = 1600
_RELATE_IN_FLIGHT = 4
ARCHIVE_BELOW = 0.02
ARCHIVE_UNUSED_MS = 180 * _DAY

_HALF_LIFE: dict[str, float] = {
    "episode": 90 * _DAY,
    "fact": 365 * _DAY,
    "pitfall": 365 * _DAY,
    "decision": 540 * _DAY,
    "procedure": 540 * _DAY,
    "snippet": 540 * _DAY,
    "template": 540 * _DAY,
}


def priority(it: Mapping[str, Any], use: Mapping[str, Any], now: float) -> float:
    half = _HALF_LIFE.get(it["kind"])
    recency = 0.5 ** (max(0, now - it["ts"]) / half) if half else 1
    return (it["importance"] / 5) * recency * (1 + math.log1p(use["opened"]) + 0.3 * math.log1p(use["shown"]))


@dataclass
class SleepDeps(ConsolidateDeps):
    usage: UsageBook | None = None
    state_path: str = ""
    #: Where the last sleep is kept instead of ``state_path`` (a server running many memories: a row per memory).
    read_state: Callable[[], Awaitable[dict[str, Any] | None]] | None = None
    write_state: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    #: A backfill's night: fresh items looked at this many at once (1), and relate asks in flight (RELATE_IN_FLIGHT).
    scan_at_once: int | None = None
    relate_in_flight: int | None = None
    #: Answers applied at once (1, in order).
    apply_at_once: int | None = None


def _knowledge(m: Mapping[str, Any]) -> bool:
    """Observations are add-only (consolidate.py observe): two events are never one, and an older one stays, dated. A
    profile is written again from its items (4), never related to anything."""
    return m.get("status") == "active" and m.get("kind") not in ("episode", "note", "observation", "profile")


class Sleep:
    def __init__(self, d: SleepDeps) -> None:
        self.d = d

    async def _last_sleep(self) -> float:
        if self.d.read_state:
            return ((await self.d.read_state()) or {}).get("last", 0)
        try:
            with open(self.d.state_path, encoding="utf-8") as f:
                return parse(f.read()).get("last", 0) or 0
        except (OSError, ValueError, AttributeError):
            return 0

    async def run(self) -> dict[str, Any]:
        """One offline pass: ``{looked, merged, superseded, linked, archived, profiled, unprofiled, paused?}``."""
        d = self.d
        now = d.time()
        since = await self._last_sleep()
        r: dict[str, Any] = {"looked": 0, "merged": 0, "superseded": 0, "linked": 0, "archived": 0, "profiled": 0, "unprofiled": 0}
        # 1. What was learned since the last sleep, against its near neighbours, oldest first.
        worker = Consolidator(d)
        # Learned since: by when it was written, not by its source's time — history imported after a sleep is new.
        fresh = [
            uid
            for uid, m in sorted(
                ((u, m) for u, m in d.ltm.entries() if _knowledge(m) and learned_at(m) >= since), key=lambda kv: learned_at(kv[1])
            )
        ]
        #: ``at``: when the fresh item of the pair was learned — where the next night starts when it is not asked.
        pairs: list[dict[str, Any]] = []
        seen: set[str] = set()
        resume_at: float | None = None

        async def look(it: MemoryItem) -> list[dict[str, Any]]:
            # The key vector the index already holds; embedded again only when it is not there.
            held = await d.client.retrieve([point_id(it["uid"], 0)], True)
            v = held[0].get("vector") if held else None
            if not v:
                v = (await d.embed([doc_text(it["title"], it["summary"])]))[0].tolist()
            must: list[Any] = [{"key": "tier", "match": {"value": "ltm"}}, {"key": "status", "match": {"value": "active"}}]
            must.append(
                {"key": "project", "match": {"value": it["project"]}}
                if it["scope"] == "project"
                else {"key": "scope", "match": {"any": ["global", "person"]}}
            )
            f: Filter = {"must": must, "must_not": [{"key": "kind", "match": {"any": ["episode", "note", "observation", "profile"]}}]}
            return [{"uid": it["uid"], "item": n["item"], "score": n["score"]} for n in await d.client.neighbors(v, f, _NEIGHBOURS)]

        # Oldest first, a few at once: each group's neighbours become pairs in order, and a full night stops at the next
        # group, which the next night starts from.
        at_once = max(1, d.scan_at_once or 1)
        for i in range(0, len(fresh), at_once):
            if d.aborted():
                return r
            group = [it for it in (d.ltm.get(u) for u in fresh[i : i + at_once]) if it and it.get("status") == "active"]
            if not group:
                continue
            if len(pairs) >= PAIRS_PER_SLEEP:
                resume_at = learned_at(group[0])
                break
            r["looked"] += len(group)
            for k, near in enumerate(await asyncio.gather(*(look(it) for it in group))):
                it = group[k]
                for n in near:
                    known = d.ltm.get(n["item"]) if n["item"] != it["uid"] and n["score"] >= RELATED else None
                    key = " ".join(sorted([it["uid"], n["item"]]))
                    if not known or not _knowledge(known) or key in seen:
                        continue
                    seen.add(key)
                    # The newer of the two is NEW, the older KNOWN: a contradiction retires the older.
                    at = learned_at(it)
                    pairs.append(
                        {"it": it, "known": known, "score": n["score"], "at": at}
                        if it["ts"] >= known["ts"]
                        else {"it": known, "known": it, "score": n["score"], "at": at}
                    )
        # The fast model reads the pairs below 0.93, several asks at once; the answers are applied in order, each to the
        # items as they are by then (one an earlier answer retired is left alone).
        asked = [p for p in pairs if p["score"] < SAME]
        answers: dict[int, Decision] = {}
        batches = relate_batches(asked)
        # As many asks in flight as allowed, the next one starting as soon as any is answered; one the model did not
        # answer stops the asking, and the next night starts from its pairs.
        in_flight = max(1, d.relate_in_flight or _RELATE_IN_FLIGHT)
        queue = list(batches)
        state = {"unanswered": False}

        async def ask_lane() -> None:
            nonlocal resume_at
            while not state["unanswered"] and queue and not d.aborted():
                b = queue.pop(0)
                got = await worker.relate(b)
                if "paused" in got:
                    state["unanswered"] = True
                    r.setdefault("paused", got["paused"])
                    resume_at = min(resume_at if resume_at is not None else math.inf, *(p["at"] for p in b))
                    return
                for n, p in enumerate(b):
                    dec = got.get(f"p{n + 1}")
                    if dec:
                        answers[id(p)] = dec

        await gather_or_cancel(*(ask_lane() for _ in range(min(in_flight, len(batches)))))

        async def apply(p: dict[str, Any]) -> None:
            async def under_lock() -> None:
                it = d.ltm.get(p["it"]["uid"])
                known = d.ltm.get(p["known"]["uid"])
                if not it or not known or it.get("status") != "active" or known.get("status") != "active":
                    return
                dec: Decision | None = (
                    {"relation": "same", "title": "", "summary": "", "body": ""} if p["score"] >= SAME else answers.get(id(p))
                )
                if not dec or dec["relation"] == "different":
                    return
                if dec["relation"] == "related":
                    await worker.link(it, known)
                    r["linked"] += 1
                    return
                if dec["relation"] == "contradicts":
                    await worker.retire(known, it["uid"])
                    r["superseded"] += 1
                    return
                text = (
                    {"title": dec["title"], "summary": dec["summary"], "body": dec["body"] or known["body"]}
                    if dec["title"] and dec["summary"]
                    else None
                )
                await worker.merge(known, it, text)
                await worker.retire(it, known["uid"])
                r["merged"] += 1

            await worker.locked([p["it"]["uid"], p["known"]["uid"]], under_lock)

        order = list(pairs)

        async def apply_lane() -> None:
            while order and not d.aborted():
                await apply(order.pop(0))

        await gather_or_cancel(*(apply_lane() for _ in range(max(1, d.apply_at_once or 1))))
        if d.aborted():
            return r
        # 2–3. Priority, and the archive.
        retire: list[MemoryItem] = []
        for uid, m in d.ltm.entries():
            if m.get("status") != "active" or m.get("kind") == "note" or not _HALF_LIFE.get(m.get("kind") or ""):
                continue
            use = d.usage.of(uid) if d.usage else {"shown": 0, "opened": 0, "lastUsed": 0}
            # Unused for half a year since it was learned, not since its source was written: imported history is not
            # archived the night it arrives.
            if priority(m, use, now) >= ARCHIVE_BELOW or now - max(m["ts"], learned_at(m), use["lastUsed"]) < ARCHIVE_UNUSED_MS:
                continue
            old = d.ltm.get(uid)
            if old and now - learned_at(old) >= ARCHIVE_UNUSED_MS:
                retire.append({**old, "status": "archived"})
        for i in range(0, len(retire), 50):
            if d.aborted():
                break
            part = retire[i : i + 50]
            await write_ltm(d, part)
            d.ltm.put(part)
            r["archived"] += len(part)
        # 4. Profiles of what is known now; one the fast model did not answer waits for the next night.
        if d.aborted():
            return r
        p = await write_profiles(d)
        r["profiled"] = p["profiled"]
        r["unprofiled"] = p["unprofiled"]
        if "paused" in p:
            r.setdefault("paused", p["paused"])
            d.log(f"memory sleep: profiles wait for the next night ({p['paused']})")
        await asyncio.to_thread(d.ltm.compact)
        # A night that stopped early (the pair cap, the fast model away) starts the next one where it stopped.
        state_out = {"last": resume_at if resume_at is not None else now, **r}
        if d.write_state:
            await d.write_state(state_out)
            return r
        try:
            os.makedirs(os.path.dirname(d.state_path) or ".", exist_ok=True)
            write_file(d.state_path, stringify(state_out).encode("utf-8"))
        except OSError:
            pass  # the next sleep looks at a little more
        return r
