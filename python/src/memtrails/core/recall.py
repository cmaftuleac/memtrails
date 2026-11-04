"""
Recall: what a search returns, an item opened whole, and exploring around a topic.

``memory_search`` embeds the query and searches dense vectors (each item's key, its chunks and its cues) and a keyword
vector at once, fused and grouped by item; with dates, what happened within them first. The best 30 can be reordered
(``rerank``: word by word on this machine, words.py, or any cross-encoder) and fused with the search's own order.
``memory_open`` reads one item whole. ``memory_explore`` walks the concept graph (graph.py) from a topic or from items
to what is related but not found directly.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from .._js import cut, flatten_ws, u16len
from ..records import collection_key
from .embedder import Embed, query_text
from .graph import explore
from .qdrant_client import Filter, point_id
from .reranker import RERANK_TOP, reranked_order
from .text import children, query_sparse
from .words import Rerank
from .writer import LONG_SUMMARY_CHUNKS, date_span


@dataclass
class RecallDeps:
    client: Any
    embed: Embed
    #: What was shown / opened: the sleep pass's "use" (store.py UsageBook).
    used: Callable[[list[str], str], None] | None = None
    #: What each search showed, for what, and what was opened (recall_log.py).
    record: Callable[[dict[str, Any]], None] | None = None
    now: Callable[[], float] | None = None
    #: Each document's relevance to the query, or None — the search then keeps its own order. None: no reordering.
    rerank: Rerank | None = None


class RecordRef(TypedDict, total=False):
    """Where a found item came from, when it is a record you added."""

    id: str
    source: str
    author: str
    url: str
    metadata: dict[str, Any]


class _RecalledBase(TypedDict):
    id: str
    title: str
    summary: str
    where: str
    project: str
    engine: str
    ts: float
    kind: str
    score: float


class RecalledItem(_RecalledBase, total=False):
    """One found item, keyed as the reference keys it. ``kind``: a record's own kind (``email``, …, ``record`` when none
    was given), or what consolidation learned (decision, procedure, pitfall, fact, preference, …, observation, profile,
    episode). ``eventAt``/``startAt``/``endAt``: for an observation, when it happened (YYYY-MM-DD, YYYY-MM or YYYY) —
    ``ts`` is when it was said. ``match``: the passage of its text that matched, when that was not its key."""

    record: RecordRef
    eventAt: str
    startAt: str
    endAt: str
    match: str


def _base_filter(kinds: Sequence[str] | None = None) -> Filter:
    """What search may show: active items, never a throwaway run, never superseded or archived knowledge. ``kinds``: a
    record's kind is its ``engine`` (every point of a record has ``kind: episode``), a learned item's its ``kind``."""
    of_kind = (
        [
            {
                "should": [
                    {"must": [{"key": "tier", "match": {"value": "stm"}}, {"key": "engine", "match": {"any": list(kinds)}}]},
                    {"must": [{"key": "tier", "match": {"value": "ltm"}}, {"key": "kind", "match": {"any": list(kinds)}}]},
                ]
            }
        ]
        if kinds
        else []
    )
    return {"must": [{"key": "synthetic", "match": {"value": False}}, {"key": "status", "match": {"value": "active"}}, *of_kind]}


def _s(v: Any) -> str:
    """``String(v ?? '')``."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _n(v: Any) -> Any:
    """``Number(v ?? 0)`` for what a payload holds."""
    if v is None:
        return 0
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


async def _items(d: RecallDeps, uids: Sequence[str]) -> list[RecalledItem]:
    got = await d.client.retrieve([point_id(u, 0) for u in uids])
    by_item = {str(p["payload"].get("item")): p["payload"] for p in got}
    out: list[RecalledItem] = []
    for uid in uids:
        p = by_item.get(uid)
        if p is None:
            continue
        it: dict[str, Any] = {
            "id": uid,
            "title": _s(p.get("title")),
            "summary": _s(p.get("summary")),
            "where": _s(p.get("where")),
            "project": _s(p.get("project")),
            "engine": _s(p.get("engine")),
            "ts": _n(p.get("ts")),
            "kind": _s(p.get("kind")),
            "score": 0,
        }
        for key in ("eventAt", "startAt", "endAt"):
            if p.get(key):
                it[key] = _s(p[key])
        if p.get("record"):
            it["record"] = p["record"]
        if p.get("tier") == "stm":
            it["kind"] = _s(p.get("engine") if p.get("engine") is not None else "record")
        out.append(it)  # type: ignore[arg-type]
    return out


MAX_SEARCH = 200
#: A captured request's title shorter than this says nothing on its own.
_SHORT_TITLE = 24


def _within_dates(f: Filter, start: str | None, end: str | None) -> Filter:
    """What happened within the dates (YYYY-MM-DD, YYYY-MM or YYYY; either end may be open): an observation by when it
    happened — its time overlapping the range — anything else by when it was written. The caller reads the dates out
    of a question; nothing here parses words."""
    lo = (date_span(start) or {}).get("from", 0)
    hi = (date_span(end) or {}).get("to", 9007199254740991)
    return {
        **f,
        "must": [
            *(f.get("must") or []),
            {
                "should": [
                    {"must": [{"key": "eventFrom", "range": {"lte": hi}}, {"key": "eventTo", "range": {"gte": lo}}]},
                    {"must": [{"is_empty": {"key": "eventFrom"}}, {"key": "ts", "range": {"gte": lo, "lte": hi}}]},
                ]
            },
        ],
    }


#: How long a search waits for its reranker before keeping its own order.
_RERANK_WAIT_MS = 3000


async def _reranked(d: RecallDeps, query: str, found: list[RecalledItem], dated: set[str]) -> list[RecalledItem]:
    """The best RERANK_TOP in the reranker's order fused with their own; with dates, the hits within them stay first —
    each block is reordered on its own."""
    if not d.rerank or len(found) < 2:
        return found
    top = found[:RERANK_TOP]
    try:
        scores = await d.rerank(query, [f"{it['title']}\n{it['summary']}" for it in top], _RERANK_WAIT_MS)
    except Exception:  # noqa: BLE001 - a reranker that fails leaves the search its own order
        scores = None
    if not scores:
        return found
    score = {it["id"]: scores[i] for i, it in enumerate(top)}

    def order(xs: list[RecalledItem]) -> list[RecalledItem]:
        return reranked_order(xs, [score[it["id"]] for it in xs])

    return [*order([it for it in top if it["id"] in dated]), *order([it for it in top if it["id"] not in dated]), *found[RERANK_TOP:]]


#: Up to this many asked for, memory_search ranks one fused list of twice as many (the room the two kinds had); past it
#: — a question that gathers (memory_gather asks for MAX_SEARCH, then 30 a follow-up) — the dated observations and
#: everything else are two searches of that many each, interleaved, so neither crowds the other out of the room.
_SINGLE_LIST_UP_TO = 20


async def memory_search(d: RecallDeps, q: Mapping[str, Any]) -> list[RecalledItem]:
    """A search: best first, in every collection or the ones named — and with dates, what happened within them first,
    then the rest. Never only what lies within them: on 64 held-out questions that name a time, the answer reached the
    reader whole for 54 searched without dates, 53 with what lies within them first, 44 with only that.

    ``q``: ``query``; ``collections`` (only these); ``kinds`` (a record's kind, or what consolidation learned); ``limit``
    (default 8, at most 200); ``from``/``to`` (YYYY-MM-DD, YYYY-MM or YYYY)."""
    # A date that is not one would leave its end open: the search would quietly ignore the range it was asked for.
    for end, at in (("from", q.get("from")), ("to", q.get("to"))):
        if at and not date_span(at):
            raise ValueError(f'memory search: {end} "{at}" is not a date (YYYY-MM-DD, YYYY-MM or YYYY)')
    projects = [collection_key(c) for c in q["collections"]] if q.get("collections") else None
    # Up to 200: a question that gathers needs the facts of many threads at once, a dated observation a line.
    limit = max(1, min(MAX_SEARCH, q["limit"] if q.get("limit") is not None else 8))
    vecs = await d.embed([query_text(q["query"])])
    base = _base_filter(q.get("kinds"))
    scoped = {**base, "must": [*base["must"], {"key": "project", "match": {"any": projects}}]} if projects else base
    # Two searches at once: the dated observations, and everything else — up to `limit` of each. In one search an
    # observation (one point) competed with every chunk of every long request for the same places.
    dense = [v.tolist() for v in vecs]
    sparse = query_sparse([q["query"]])
    observations = {"key": "kind", "match": {"value": "observation"}}

    async def ranked(f: Filter) -> list[dict[str, Any]]:
        # A search for a few: one fused list over every kind, twice as deep as asked. The interleave is for gathering.
        if limit <= _SINGLE_LIST_UP_TO:
            return await d.client.fused_items(dense, sparse=sparse, filter=f, limit=2 * limit)
        obs, rest = await asyncio.gather(
            d.client.fused_items(dense, sparse=sparse, filter={**f, "must": [*(f.get("must") or []), observations]}, limit=limit),
            d.client.fused_items(dense, sparse=sparse, filter={**f, "must_not": [*(f.get("must_not") or []), observations]}, limit=limit),
        )
        # Interleaved by rank, so each kind keeps its order and neither crowds out the other.
        out: list[dict[str, Any]] = []
        for i in range(max(len(obs), len(rest))):
            out += [h for h in (rest[i] if i < len(rest) else None, obs[i] if i < len(obs) else None) if h]
        return out

    async def nothing() -> list[dict[str, Any]]:
        return []

    in_dates, open_ = await asyncio.gather(
        ranked(_within_dates(scoped, q.get("from"), q.get("to"))) if q.get("from") or q.get("to") else nothing(), ranked(scoped)
    )
    dated = {h["item"] for h in in_dates}
    merged = [*in_dates, *(h for h in open_ if h["item"] not in dated)][: max(len(open_), len(in_dates))]
    # A kept episode and its captured request hold the same text: the one found first stands for both.
    seen: set[str] = set()
    hits = []
    for h in merged:
        item = h["item"]
        request = item[3:] if item.startswith("ep:") else item[4:] if item.startswith("stm:") else None
        if request is not None:
            if request in seen:
                continue
            seen.add(request)
        hits.append(h)
    found = await _reranked(d, q["query"], await _items(d, [h["item"] for h in hits]), dated)
    score = {h["item"]: h["score"] for h in hits}
    matched = await _passages(d, hits)
    out: list[RecalledItem] = []
    for it in found:
        # A record titled by its first line, when that line is as short as "continue" or "ok", says nothing of what was
        # found: it shows the start of its key instead, which is what matched.
        shown: dict[str, Any] = {**it}
        if it["id"].startswith("stm:") and u16len(it["title"]) < _SHORT_TITLE:
            shown["title"] = cut(flatten_ws(it["summary"]), 100)
        shown["score"] = score.get(it["id"], 0)
        if it["id"] in matched:
            shown["match"] = matched[it["id"]]
        out.append(shown)  # type: ignore[arg-type]
    if d.record:
        event: dict[str, Any] = {"ev": "search"}
        if projects:
            event["project"] = ",".join(projects)
        event.update(query=q["query"], shown=[it["id"] for it in out])
        d.record(event)
    return out


async def _passages(d: RecallDeps, hits: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """For each hit whose best point was a chunk of its text, that chunk — found again from the text (writer.py chunks a
    request's whole body; an item's body, and a summary too long for one key vector)."""
    chunked = [h for h in hits if h.get("point") and h["point"] != point_id(h["item"], 0)]
    if not chunked:
        return {}
    got = await d.client.retrieve([point_id(h["item"], 0) for h in chunked])
    by_item = {str(p["payload"].get("item")): p["payload"] for p in got}
    out: dict[str, str] = {}
    for h in chunked:
        p = by_item.get(h["item"])
        if p is None:
            continue
        body, summary = _s(p.get("body")), _s(p.get("summary"))
        if p.get("tier") == "stm":
            kids = children(body)
        else:
            summary_kids = children(summary)
            kids = [*children(body), *(summary_kids if len(summary_kids) >= LONG_SUMMARY_CHUNKS else [])]
        for i, k in enumerate(kids):
            if point_id(h["item"], i + 1) == h["point"]:
                out[h["item"]] = k
                break
    return out


class Explored(TypedDict):
    #: What search found for the topic, best first.
    found: list[RecalledItem]
    #: Reached through the graph and not found by search, best first, with the way it was reached (``via``).
    related: list[dict[str, Any]]
    #: The concepts around it: where to look next.
    concepts: list[str]


async def memory_explore(d: RecallDeps, q: Mapping[str, Any]) -> Explored:
    """An agent exploring: from a topic (search finds the first items; the topic's concepts start the walk too) or from
    items, what the concept graph reaches that search did not, and the concepts around it. ``q``: ``topic``, ``from``
    (memory ids), ``collections``, ``limit`` (default 10, at most 20)."""
    projects = [collection_key(c) for c in q["collections"]] if q.get("collections") else None
    base = _base_filter()
    f = {**base, "must": [*base["must"], {"key": "project", "match": {"any": projects}}]} if projects else base
    r = await explore(
        d,
        topic=q.get("topic"),
        start=q.get("from"),
        filter=f,
        projects=projects,
        limit=max(1, min(20, q["limit"] if q.get("limit") is not None else 10)),
    )
    found, related = await asyncio.gather(_items(d, r["found"][:8]), _items(d, [x["id"] for x in r["related"]]))
    how = {x["id"]: x for x in r["related"]}
    return {
        "found": found,
        "related": [{**it, "score": how[it["id"]]["score"], "via": how[it["id"]]["via"]} for it in related],
        "concepts": [c["name"] for c in r["concepts"]],
    }


async def memory_open(d: RecallDeps, id: str) -> dict[str, Any] | None:
    """One item whole, or None."""
    got = await d.client.retrieve([point_id(id, 0)])
    p = got[0]["payload"] if got else None
    if p:
        if d.used:
            d.used([id], "opened")
        if d.record:
            d.record({"ev": "open", "item": id})
    return p
