"""Stand-ins the TypeScript tests use, ported: an embedder whose vectors are the words of the text (so "the same" and
"nearly the same" are under the test's control) and Qdrant as far as memory uses it, in memory."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

import numpy as np
import regex

from memtrails.core.store import JsonlLog, ltm_log, stm_log
from memtrails.core.text import term_id, tokens

T = TypeVar("T")
DIMS = 512
_PROMPT = regex.compile(r"^title: | \| text: |^task: search result \| query: ")


async def embed(texts: Sequence[str], _timeout_ms: float | None = None) -> list[np.ndarray]:
    """The words of a text as a unit vector (the embedding prompt's wrapper left out)."""
    out = []
    for t in texts:
        v = np.zeros(DIMS, dtype=np.float32)
        for w in set(tokens(_PROMPT.sub(" ", t))):
            v[term_id(w) % DIMS] = 1
        n = math.sqrt(float(np.sum(v.astype(np.float64) ** 2))) or 1
        out.append((v / n).astype(np.float32))
    return out


def cos(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=False)))


def matches(p: dict[str, Any], f: dict[str, Any] | None) -> bool:
    def one(c: dict[str, Any]) -> bool:
        # A nested filter is a condition (Qdrant's rule).
        if "must" in c or "should" in c or "must_not" in c:
            return matches(p, c)
        if "is_empty" in c:
            return p.get(c["is_empty"]["key"]) is None
        v = p.get(c["key"])
        m = c.get("match")
        # An array field matches when any of its values does (Qdrant's rule).
        if m is not None and "value" in m:
            return m["value"] in v if isinstance(v, list) else v == m["value"] and type(v) is type(m["value"])
        if m is not None and "any" in m:
            return any(x in m["any"] for x in v) if isinstance(v, list) else v in m["any"]
        if "range" in c:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False
            r = c["range"]
            return ("lt" not in r or v < r["lt"]) and ("lte" not in r or v <= r["lte"]) and ("gte" not in r or v >= r["gte"])
        return True

    f = f or {}
    return (
        all(one(c) for c in f.get("must") or [])
        and not any(one(c) for c in f.get("must_not") or [])
        and (not f.get("should") or any(one(c) for c in f["should"]))
    )


class FakeQdrant:
    """Qdrant, as far as memory uses it: each item ranked by its best point within the filter, channels fused by rank."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}

    async def ensure_collection(self) -> None:
        return None

    async def upsert(self, ps: Sequence[dict[str, Any]], wait: bool = False) -> None:
        for p in ps:
            self.points[p["id"]] = {"id": p["id"], "vector": p["vector"], "payload": dict(p["payload"])}

    async def delete_items(self, items: Sequence[str]) -> None:
        for pid in [k for k, p in self.points.items() if p["payload"].get("item") in items]:
            del self.points[pid]

    async def delete_parts(self, items: Sequence[str], part: str) -> None:
        for pid in [k for k, p in self.points.items() if p["payload"].get("item") in items and p["payload"].get("part") == part]:
            del self.points[pid]

    async def set_payload(self, items: Sequence[str], payload: dict[str, Any], key_only: bool = False) -> None:
        for p in self.points.values():
            if p["payload"].get("item") in items:
                p["payload"].update(payload)

    async def count(self, f: dict[str, Any] | None = None) -> int:
        return len(self.points)

    async def retrieve(self, ids: Sequence[str], with_vector: bool = False) -> list[dict[str, Any]]:
        out = []
        for pid in ids:
            p = self.points.get(pid)
            if p:
                out.append({"id": pid, "payload": p["payload"], **({"vector": p["vector"]["dense"]} if with_vector else {})})
        return out

    def _near(self, vector: Sequence[float], f: dict[str, Any] | None, limit: int) -> list[dict[str, Any]]:
        hits = [
            {"item": str(p["payload"].get("item")), "score": cos(vector, p["vector"]["dense"]), "payload": p["payload"]}
            for p in self.points.values()
            if matches(p["payload"], f)
        ]
        return sorted(hits, key=lambda h: -h["score"])[:limit]

    async def nearest_many(
        self, vectors: Sequence[Sequence[float]], f: dict[str, Any], limit: int = 10, *_: Any
    ) -> list[list[dict[str, Any]]]:
        return [self._near(v, f, limit) for v in vectors]

    async def scroll(self, f: dict[str, Any], limit: int = 256, with_vector: bool = False) -> list[dict[str, Any]]:
        out = [p for p in self.points.values() if matches(p["payload"], f)][:limit]
        return [{"id": p["id"], "payload": p["payload"], **({"vector": p["vector"]["dense"]} if with_vector else {})} for p in out]

    async def neighbors(self, vector: Sequence[float], f: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        return self._near(vector, {**f, "must": [*(f.get("must") or []), {"key": "part", "match": {"value": "key"}}]}, limit)

    async def fused_items(
        self,
        dense: Sequence[Sequence[float]],
        sparse: Any = None,
        filter: dict[str, Any] | None = None,
        per: int | None = None,
        limit: int | None = None,
        timeout_ms: float = 0,
    ) -> list[dict[str, Any]]:
        score: dict[str, float] = {}
        point: dict[str, str] = {}
        for v in dense:
            ranked = sorted(
                (
                    {"item": str(p["payload"].get("item")), "id": p["id"], "s": cos(v, p["vector"]["dense"])}
                    for p in self.points.values()
                    if matches(p["payload"], filter)
                ),
                key=lambda r: -r["s"],
            )
            best: dict[str, int] = {}
            for r in ranked:
                if r["item"] in best:
                    continue
                best[r["item"]] = len(best)
                point.setdefault(r["item"], r["id"])
            for item, rank in best.items():
                if rank < (per if per is not None else 50):
                    score[item] = score.get(item, 0) + 1 / (60 + rank + 1)
        top = sorted(score.items(), key=lambda kv: -kv[1])[: limit if limit is not None else 40]
        return [{"item": item, "score": s, "point": point.get(item)} for item, s in top]


OLD = 1767225600000  # 2026-01-01T00:00:00Z
NOW = 1792756800000  # 2026-10-23T12:00:00Z
DAY = 86_400_000


def item(uid: str, title: str, summary: str, **over: Any) -> dict[str, Any]:
    return {
        "uid": uid,
        "kind": "fact",
        "scope": "project",
        "project": "-p",
        "engine": "conversation",
        "session": "s0",
        "cwd": "/p",
        "where": "conv",
        "ts": OLD,
        "firstTs": OLD,
        "title": title,
        "summary": summary,
        "body": summary,
        "files": [],
        "appliesWhen": "",
        "confidence": "verified",
        "importance": 3,
        "status": "active",
        "provenance": [],
        "seen": 1,
        "fingerprint": uid,
        **over,
    }


def entry(uid: str, request: str, **over: Any) -> dict[str, Any]:
    return {
        "uid": uid,
        "fingerprint": uid,
        "engine": "conversation",
        "session": "sess-1",
        "turn": 0,
        "cwd": "/p",
        "project": "-p",
        "title": "Deploy work",
        "ts": OLD,
        "synthetic": False,
        "typed": "human",
        "request": request,
        "reply": "ok",
        "said": "ok",
        "steps": [],
        "files": [],
        "done": False,
        **over,
    }


class Setup:
    """Logs in a fresh folder, the stand-in index, and what consolidation and sleep take."""

    def __init__(self, tmp: Any, now: float = NOW) -> None:
        self.dir = str(tmp)
        self.client = FakeQdrant()
        self.points = self.client.points
        self.stm = stm_log(f"{self.dir}/stm.jsonl")
        self.ltm: JsonlLog[dict[str, Any]] = ltm_log(f"{self.dir}/ltm.jsonl")
        self.now = now

    def deps(self, **over: Any) -> Any:
        from memtrails.core.consolidate import ConsolidateDeps

        return ConsolidateDeps(client=self.client, embed=embed, stm=self.stm, ltm=self.ltm, now=lambda: self.now, **over)

    def sleep_deps(self, **over: Any) -> Any:
        from memtrails.core.sleep import SleepDeps
        from memtrails.core.store import UsageBook

        over.setdefault("usage", UsageBook(f"{self.dir}/usage.json"))
        over.setdefault("state_path", f"{self.dir}/sleep.json")
        return SleepDeps(client=self.client, embed=embed, stm=self.stm, ltm=self.ltm, now=lambda: self.now, **over)

    def recall(self, **over: Any) -> Any:
        from memtrails.core.recall import RecallDeps

        return RecallDeps(client=self.client, embed=embed, now=lambda: self.now, **over)


def run(coro: Awaitable[T]) -> T:
    return asyncio.run(_wrap(coro))


async def _wrap(coro: Awaitable[T]) -> T:
    return await coro


Ask = Callable[[Any], Awaitable[dict[str, Any]]]
