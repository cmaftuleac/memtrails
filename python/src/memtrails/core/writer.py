"""
How memory reaches the index: points built, embedded in batches and written to Qdrant — both tiers.

  - a record (a short-term entry): ONE key point, its start as the key, and its whole text in chunks: no model,
    searchable the moment it is added;
  - a long-term item: a key point (content title + ~150-word summary, BM25 over title, summary, body and files) and its
    ~96-word children (small-to-big: searched, never shown — the item is what comes back), its cues, and the keys of
    its concepts on the key point (the concept graph's links: graph.py).
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import regex

from .._js import cut, date_utc, js_trim, utc_parts
from .embedder import Embed, doc_text
from .episodes import episode_body
from .qdrant_client import PointStruct, point_id
from .text import children, concept_key, doc_sparse, first_words


@dataclass
class WriterDeps:
    """An index and an embedder: what the writers write with."""

    client: Any
    embed: Embed


class _Deps(Protocol):
    @property
    def client(self) -> Any: ...

    @property
    def embed(self) -> Embed: ...


def embed_budget(n: int) -> int:
    """Time for embedding this many texts: a 30 s floor, and more for a long batch (a long record is many chunks)."""
    return max(30_000, n * 150)


def stm_item(uid: str) -> str:
    """A short-term entry's item id in the index (its long-term episode, if consolidation keeps it, is ``ep:<uid>``)."""
    return f"stm:{uid}"


def raw_key(e: Mapping[str, Any]) -> str:
    """The raw key of a captured request: what it asked and the start of what was done — ~150 words."""
    return first_words(f"{e['request']}\n{e['reply']}", 150)


def stm_title(e: Mapping[str, Any]) -> str:
    """A captured request's own title, its first line: the conversation's title says where it was, not what it is
    (one long conversation's requests would all read alike, and match alike)."""
    first = next((x for x in (js_trim(line) for line in e["request"].split("\n")) if x), "")
    return cut(first, 100) or e["title"]


async def write_stm(d: _Deps, entries: Sequence[Mapping[str, Any]]) -> None:
    if not entries:
        return
    # The raw key (its start), and the whole body in chunks: every word of a captured request can be found.
    plan = []
    for e in entries:
        body = episode_body(e)
        plan.append((e, body, children(body)))
    texts = [t for e, _, kids in plan for t in (doc_text(stm_title(e), raw_key(e)), *(doc_text(stm_title(e), k) for k in kids))]
    vecs = await d.embed(texts, embed_budget(len(texts)))
    points: list[PointStruct] = []
    at = 0
    for e, body, kids in plan:
        # The record's kind (`engine`) on every point: a kinds filter must find a record by any of its chunks.
        common = {
            "item": stm_item(e["uid"]),
            "tier": "stm",
            "status": "active",
            "kind": "episode",
            "engine": e["engine"],
            "scope": "project",
            "project": e["project"],
            "session": e["session"],
            "ts": e["ts"],
            "synthetic": e["synthetic"],
        }
        payload: dict[str, Any] = {
            **common,
            "part": "key",
            "cwd": e.get("cwd") or "",
            "where": e["title"],
            "title": stm_title(e),
            "summary": raw_key(e),
            "body": body,
            "files": e["files"],
            "importance": 0,
        }
        if e.get("record"):
            payload["record"] = e["record"]
        points.append(
            {
                "id": point_id(stm_item(e["uid"]), 0),
                "vector": {"dense": vecs[at].tolist(), "bm25": doc_sparse(f"{stm_title(e)}\n{body}")},
                "payload": payload,
            }
        )
        for k in range(len(kids)):
            points.append(
                {
                    "id": point_id(stm_item(e["uid"]), k + 1),
                    "vector": {"dense": vecs[at + 1 + k].tolist()},
                    "payload": {**common, "part": "child"},
                }
            )
        at += 1 + len(kids)
    await d.client.delete_items([stm_item(e["uid"]) for e in entries])
    for i in range(0, len(points), 256):
        await d.client.upsert(points[i : i + 256], True)


def cue_point(k: int) -> int:
    """A cue's point number: the k-th cue is -(k + 1) — below the key (0) and the children (1, 2, …), however many of
    either there are, so the two never meet."""
    return -(k + 1)


_DATE = regex.compile(r"^([0-9]{4})(?:-([0-9]{2})(?:-([0-9]{2}))?)?\Z")


def date_span(d: str | None) -> dict[str, int] | None:
    """The time a date stands for, as the first and last millisecond: a day, a month or a year (an observation's
    ``eventAt``, or a date an agent searches by) — None for anything else, a month or day that does not exist too
    (2024-02-30 is not 1 March)."""
    m = _DATE.match(d or "")
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2)) - 1 if m.group(2) else None
    dd = int(m.group(3)) if m.group(3) else None
    if mo is not None and not 0 <= mo <= 11:
        return None
    if mo is not None and dd is not None and (dd < 1 or utc_parts(date_utc(y, mo, dd))[1] != mo):
        return None
    if mo is None:
        return {"from": date_utc(y, 0, 1), "to": date_utc(y + 1, 0, 1) - 1}
    if dd is None:
        return {"from": date_utc(y, mo, 1), "to": date_utc(y, mo + 1, 1) - 1}
    return {"from": date_utc(y, mo, dd), "to": date_utc(y, mo, dd + 1) - 1}


def _ltm_payload(it: Mapping[str, Any]) -> dict[str, Any]:
    """What every point of a long-term item carries: what recall and the filters select on — for an observation, the
    time it happened too (memory_search's ``from``/``to``): from its date, or from when what it tells began to when it
    ended."""
    spans = [s for s in (date_span(it.get(k)) for k in ("eventAt", "startAt", "endAt")) if s]
    out: dict[str, Any] = {
        "item": it["uid"],
        "tier": "ltm",
        "status": it["status"],
        "kind": it["kind"],
        "scope": it["scope"],
        "project": it["project"],
        "session": it.get("session") or "",
        "ts": it["ts"],
        "synthetic": False,
    }
    if spans:
        out["eventFrom"] = min(s["from"] for s in spans)
        out["eventTo"] = max(s["to"] for s in spans)
    return out


async def write_cues(d: _Deps, items: Sequence[Mapping[str, Any]]) -> None:
    """Only the cues of these items, written again (the key and the children stay as they are)."""
    live = [it for it in items if it.get("status") == "active"]
    texts = [doc_text(it["title"], c) for it in live for c in it.get("cues") or []]
    if not texts:
        await d.client.delete_parts([it["uid"] for it in live], "cue")
        return
    vecs = await d.embed(texts, embed_budget(len(texts)))
    points: list[PointStruct] = []
    n = 0
    for it in live:
        for k in range(len(it.get("cues") or [])):
            points.append(
                {
                    "id": point_id(it["uid"], cue_point(k)),
                    "vector": {"dense": vecs[n].tolist()},
                    "payload": {**_ltm_payload(it), "part": "cue"},
                }
            )
            n += 1
    await d.client.delete_parts([it["uid"] for it in live], "cue")
    for i in range(0, len(points), 256):
        await d.client.upsert(points[i : i + 256], True)


#: A summary this many chunks long (~380 words) is also chunked: its key vector reads only its start.
LONG_SUMMARY_CHUNKS = 5


async def write_ltm(d: _Deps, items: Sequence[Mapping[str, Any]]) -> None:
    """Embed and write long-term items: their old points out, the key point, the children and the cues in. Cues are
    embedded as documents titled by the item (how they were measured); the item, not the cue, is what comes back."""
    live = [it for it in items if it.get("status") == "active"]
    if not live:
        await d.client.delete_items([it["uid"] for it in items])
        return
    texts: list[str] = []
    plan: list[tuple[Mapping[str, Any], list[str], list[str], int]] = []
    for it in live:
        # Every chunk of the body — and of a summary too long for one key vector (the embedder reads its start).
        summary_kids = children(it["summary"])
        kids = [*children(it["body"]), *(summary_kids if len(summary_kids) >= LONG_SUMMARY_CHUNKS else [])]
        cues = list(it.get("cues") or [])
        plan.append((it, kids, cues, len(texts)))
        texts += [
            doc_text(it["title"], it["summary"]),
            *(doc_text(it["title"], k) for k in kids),
            *(doc_text(it["title"], c) for c in cues),
        ]
    vecs = await d.embed(texts, embed_budget(len(texts)))
    points: list[PointStruct] = []
    for it, kids, cues, at in plan:
        common = _ltm_payload(it)
        # Node's basename: a trailing slash is not the name (``docs/page/`` is ``page``).
        files = " ".join(posixpath.basename(f.rstrip("/")) for f in it["files"])
        payload: dict[str, Any] = {
            **common,
            "part": "key",
            "engine": it["engine"],
            "cwd": it.get("cwd") or "",
            "where": it["where"],
            "title": it["title"],
            "summary": it["summary"],
            "body": it["body"],
            "files": it["files"],
            "importance": it["importance"],
            "appliesWhen": it["appliesWhen"],
            "related": it.get("related") or [],
            "concepts": concept_keys(it.get("concepts")),
        }
        for key in ("eventAt", "startAt", "endAt", "phase", "about"):
            if it.get(key):
                payload[key] = it[key]
        points.append(
            {
                "id": point_id(it["uid"], 0),
                "vector": {
                    "dense": vecs[at].tolist(),
                    "bm25": doc_sparse(f"{it['title']}\n{it['summary']}\n{it['body']}\n{files}"),
                },
                "payload": payload,
            }
        )
        for k in range(len(kids)):
            points.append(
                {
                    "id": point_id(it["uid"], k + 1),
                    "vector": {"dense": vecs[at + 1 + k].tolist()},
                    "payload": {**common, "part": "child"},
                }
            )
        for k in range(len(cues)):
            points.append(
                {
                    "id": point_id(it["uid"], cue_point(k)),
                    "vector": {"dense": vecs[at + 1 + len(kids) + k].tolist()},
                    "payload": {**common, "part": "cue"},
                }
            )
    # Remote embedding/validation can fail: keep the old searchable item until every replacement vector exists.
    await d.client.delete_items([it["uid"] for it in items])
    for i in range(0, len(points), 256):
        await d.client.upsert(points[i : i + 256], True)


def concept_keys(names: Sequence[str] | None) -> list[str]:
    """The keys of an item's concepts, one per concept (text.py concept_key)."""
    return list(dict.fromkeys(k for k in (concept_key(n) for n in names or []) if k))


async def write_concepts(d: _Deps, items: Sequence[Mapping[str, Any]]) -> None:
    """Only the concepts of these items, written again (their keys on the key point)."""
    for it in items:
        if it.get("status") == "active":
            await d.client.set_payload([it["uid"]], {"concepts": concept_keys(it.get("concepts"))}, True)
