"""
The concept graph — memory's associative index (HippoRAG's hippocampal index, 2405.14831), for an agent EXPLORING a
topic or what it found, rather than for a search's top few (as a ranker feature it added ~1 point; for reaching what
search does not, it is where the gain is).

Nodes are long-term items and the concepts they are about (consolidation writes 3–12 per item: files, services,
commands, named things), a concept being its spelling key (text.py concept_key). Edges:
  - item ↔ concept, weighted by how specific the concept is (1 / log(2 + the items that share it));
  - item ↔ item: ``related`` (the same subject, different facts — consolidation and sleep link them) and each seed's
    nearest items by key vector, two hops out.
Concepts are NOT embedded: matched and linked by embedding instead of spelling, the walk found the same needed items
and ~0.4 more items a topic, for a point and six embeddings an item more — not worth a laptop's RAM and CPU. It lives
in Qdrant — the concept keys on items' key points, indexed — so it grows with memory without a copy in RAM: ``explore``
reads the neighbourhood of its seeds and walks it with Personalized PageRank (restart 0.5), within recall's walls
(active, real, the scope asked).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .._js import u16len
from .embedder import query_text
from .qdrant_client import Filter, point_id
from .text import concept_key, fold, query_sparse

#: A seed's nearest items by key vector (from cos 0.5: a small memory's "nearest" can be anything), the walk's search
#: hits used as seeds, the items read through concepts.
_KNN = 10
_KNN_NEAR = 0.5
_SEEDS = 5
_FOUND = 20
_POSTINGS = 400
#: A ``related`` link's weight, beside item ↔ concept's 1/log(2 + df).
_RELATED = 1.0
_RESTART = 0.5
_STEPS = 20


def _strings(v: Any) -> list[str]:
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


class _Walk:
    """A small weighted undirected graph, walked by Personalized PageRank."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        self.adj: list[dict[int, float]] = []

    def node(self, name: str) -> int:
        i = self.ids.get(name)
        if i is None:
            i = len(self.adj)
            self.ids[name] = i
            self.adj.append({})
        return i

    def link(self, a: str, b: str, w: float) -> None:
        if a == b or not w > 0:
            return
        i, j = self.node(a), self.node(b)
        self.adj[i][j] = max(self.adj[i].get(j, 0.0), w)
        self.adj[j][i] = max(self.adj[j].get(i, 0.0), w)

    def ppr(self, seeds: dict[str, float]) -> dict[str, float]:
        """Each node's share of a walk that starts again at ``seeds`` (by weight) half the time."""
        n = len(self.adj)
        e = [0.0] * n
        total = 0.0
        for name, w in seeds.items():
            i = self.ids.get(name)
            if i is not None:
                e[i] += w
                total += w
        if not total:
            return {}
        e = [x / total for x in e]
        deg = [sum(m.values()) for m in self.adj]
        x = list(e)
        for _ in range(_STEPS):
            nxt = [0.0] * n
            for i in range(n):
                if not x[i] or not deg[i]:
                    continue
                for j, w in self.adj[i].items():
                    nxt[j] += (1 - _RESTART) * x[i] * (w / deg[i])
            for i in range(n):
                nxt[i] += _RESTART * e[i]
            x = nxt
        return {name: x[i] for name, i in self.ids.items()}


async def explore(
    d: Any,
    topic: str | None = None,
    start: Sequence[str] | None = None,
    filter: Filter | None = None,
    projects: Sequence[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Walk the graph from a topic and/or items (``start``): what is related but not found directly (``related``,
    with the way it was reached), what search found for the topic (``found``), and the concepts around it."""
    base: Filter = filter or {}

    def key_only(extra: Sequence[Any] = ()) -> Filter:
        return {
            **base,
            "must": [
                *(base.get("must") or []),
                {"key": "tier", "match": {"value": "ltm"}},
                {"key": "part", "match": {"value": "key"}},
                *extra,
            ],
        }

    def shown(p: dict[str, Any]) -> bool:
        return (
            p.get("tier") == "ltm"
            and p.get("status") == "active"
            and p.get("synthetic") is False
            and (not projects or str(p.get("project")) in projects)
        )

    seed_items: dict[str, float] = {}
    seed_concepts: dict[str, float] = {}
    found: list[str] = []
    if topic and topic.strip():
        qv = (await d.embed([query_text(topic)]))[0]
        hits = await d.client.fused_items([qv.tolist()], sparse=query_sparse([topic]), filter=filter, limit=_FOUND)
        found = [h["item"] for h in hits]
        for u in found[:_SEEDS]:
            seed_items[u] = 1
        # The topic's own concepts: the topic named as one, and those of what search found that the topic names.
        key = concept_key(topic)
        if key:
            seed_concepts[key] = 1
        said = fold(topic)
        for p in await d.client.retrieve([point_id(u, 0) for u in found[:_SEEDS]]):
            for k in _strings(p["payload"].get("concepts")):
                if u16len(k) > 2 and k in said:
                    seed_concepts[k] = 1
    for u in start or []:
        seed_items[u] = 1
    if not seed_items and not seed_concepts:
        return {"found": found, "related": [], "concepts": []}

    g = _Walk()
    items: dict[str, dict[str, Any]] = {}
    # Hop 1: the seed items, their concepts and links, and their nearest items by key vector.
    seeds = [p for p in await d.client.retrieve([point_id(u, 0) for u in seed_items], True) if p.get("vector")]
    for p in seeds:
        items[str(p["payload"].get("item"))] = p["payload"]
    near = await d.client.nearest_many([p["vector"] for p in seeds], key_only(), _KNN + 1)
    hop1: dict[str, float] = {}
    for i, p in enumerate(seeds):
        for n in near[i] if i < len(near) else []:
            if n["item"] == p["payload"].get("item") or n["score"] < _KNN_NEAR:
                continue
            items.setdefault(n["item"], n["payload"])
            if n["item"] not in seed_items:
                hop1[n["item"]] = max(hop1.get(n["item"], 0), n["score"])
            g.link(f"i:{p['payload'].get('item')}", f"i:{n['item']}", n["score"])
    # A second hop from the nearest of those (measured: the multi-hop item walk found more than one hop).
    nearest = [u for u, _ in sorted(hop1.items(), key=lambda kv: -kv[1])][:_KNN]
    nxt = [p for p in await d.client.retrieve([point_id(u, 0) for u in nearest], True) if p.get("vector")]
    near2 = await d.client.nearest_many([p["vector"] for p in nxt], key_only(), _KNN + 1)
    for i, p in enumerate(nxt):
        for n in near2[i] if i < len(near2) else []:
            if n["item"] == p["payload"].get("item") or n["score"] < _KNN_NEAR:
                continue
            items.setdefault(n["item"], n["payload"])
            g.link(f"i:{p['payload'].get('item')}", f"i:{n['item']}", n["score"])
    # The concepts of what was reached and of the topic.
    concepts: dict[str, None] = dict.fromkeys(seed_concepts)
    for payload in items.values():
        for k in _strings(payload.get("concepts")):
            concepts[k] = None
    # Hop 2: the items holding those concepts — a concept many of them share links each one less — and the items
    # linked to anything reached.
    asked = list(concepts)[:300]
    if asked:
        for p in await d.client.scroll(key_only([{"key": "concepts", "match": {"any": asked}}]), _POSTINGS):
            items.setdefault(str(p["payload"].get("item")), p["payload"])
    linked_out = [u for u in dict.fromkeys(r for payload in items.values() for r in _strings(payload.get("related"))) if u not in items]
    for p in await d.client.retrieve([point_id(u, 0) for u in linked_out]):
        items[str(p["payload"].get("item"))] = p["payload"]
    df: dict[str, int] = {}
    for payload in items.values():
        for k in _strings(payload.get("concepts")):
            if k in concepts:
                df[k] = df.get(k, 0) + 1
    for u, payload in items.items():
        if not shown(payload):
            continue
        for k in _strings(payload.get("concepts")):
            if k in concepts:
                g.link(f"i:{u}", f"c:{k}", 1 / math.log(2 + df.get(k, 1)))
        for r in _strings(payload.get("related")):
            if r in items and shown(items[r]):
                g.link(f"i:{u}", f"i:{r}", _RELATED)

    # The walk starts again half at the seed items and half at the topic's concepts (either alone when only one is).
    begin: dict[str, float] = {}
    for u in seed_items:
        begin[f"i:{u}"] = (0.5 if seed_concepts else 1) / len(seed_items)
    c_total = sum(seed_concepts.values())
    for k, w in seed_concepts.items():
        begin[f"c:{k}"] = (0.5 if seed_items else 1) * (w / c_total)
    x = g.ppr(begin)

    skip = {*found, *seed_items}
    seed_titles = {u: str((items.get(u) or {}).get("title", u)) for u in seed_items}
    seed_keys = {*seed_concepts, *(k for u in seed_items for k in _strings((items.get(u) or {}).get("concepts")))}

    def via(p: dict[str, Any]) -> str:
        """How an item was reached: the concepts it shares with where the walk started, a link, or nearness to a seed."""
        shared = [f'"{k}"' for k in _strings(p.get("concepts")) if k in seed_keys][:3]
        if shared:
            return f"shares {', '.join(shared)}"
        linked = next((r for r in _strings(p.get("related")) if r in seed_titles), None)
        if linked:
            return f'linked to "{seed_titles[linked]}"'
        through = next((k for k in _strings(p.get("concepts")) if x.get(f"c:{k}", 0) > 0), None)
        if through:
            return f'through "{through}"'
        return "near what was found"

    reached: list[dict[str, Any]] = [
        {"id": u, "score": x[f"i:{u}"], "via": via(p)} for u, p in items.items() if u not in skip and shown(p) and x.get(f"i:{u}", 0) > 0
    ]
    related = sorted(reached, key=lambda r: -float(r["score"]))[: limit if limit is not None else 10]
    concept_scores = [
        {"name": n[2:], "score": s}
        for n, s in sorted(((n, s) for n, s in x.items() if n.startswith("c:") and s > 0), key=lambda kv: -kv[1])
    ][:12]
    return {"found": found, "related": related, "concepts": concept_scores}
