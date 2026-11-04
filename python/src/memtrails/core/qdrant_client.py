"""
The few Qdrant REST calls the memory index makes — plain HTTP, no client library, the same bodies the reference sends.
One collection, ``memory``, two tiers (payload ``tier``):

  - ``stm`` — short-term memory: one point per captured request (its raw text as the key), there until consolidation;
  - ``ltm`` — long-term memory: per item a KEY point (``part: key`` — its content title + ~150-word summary as the
    dense vector, a BM25 sparse vector over its title, summary, body and files, the item itself as payload) and CHILD
    points (``part: child``, ~96-word windows of its body, dense only) pointing at it.

A query ranks items once per query vector and once by keywords (each hit grouped by item, so an item found through
several children counts once), then fuses those item rankings. Only ``active`` items are ever returned to recall;
superseded and archived ones stay in the log, out of the index.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict
from urllib.parse import quote

import httpx

from .._http import new_client
from .embedder import MEMORY_EMBEDDER
from .text import SparseVector

COLLECTION = "memory"
DIMS = MEMORY_EMBEDDER["dims"]
#: What the index is made of; a change empties it and it is written in again from the logs (rebuild.py).
#: 2: binary quantisation with rescoring, keyword terms folded and segmented for every language; 3: on-disk binary
#: copy, HNSW and payload indexes; 4: remote EmbeddingGemma vectors (768 dimensions); 5: nothing cut — every chunk of
#: a body, short-term entries chunked too, cues numbered below the key, full payloads; 6: every captured request
#: indexed for good; 7: a record's kind (``engine``) on every point of it and indexed, so a kinds filter finds a record
#: by its chunks.
INDEX_LAYOUT = 7
#: The memory-mapped 1-bit copy finds candidates; full vectors on disk re-score 3× as many as asked.
_RESCORE = {"quantization": {"rescore": True, "oversampling": 3.0}}
#: Two segments: Qdrant's default is one a core, which costs a laptop RAM and per-query work for nothing at memory's
#: size. The write-ahead log keeps its 32 MB: a 4 MB one returned wrong results on 1.20, for page cache the system can
#: take back anyway.
_SEGMENTS = 2
#: Item-level fusion: each channel (a query vector, the keywords) is its own grouped query, its items' scores are
#: min-max normalised over its own list, and an item's score is the weighted sum (CombSUM) — the dense channels share
#: DENSE_SHARE of the weight, keywords the rest. Each channel is its own query, never a prefetch under a fusion: Qdrant
#: 1.20 orders a prefetch by the 1-bit scores whatever its ``rescore``. Measured against gold evidence
#: (docs/RESEARCH.md): dense 0.7 was the best share on both benchmarks, and CombSUM beat rank fusion at the top of both.
_DENSE_SHARE = 0.7
#: Cues — the questions an item answers, in the asker's words — search as their own channel at this share of the dense
#: weight rather than among the item's points (LoCoMo multi-hop 50.6 → 55.2, docs/RESEARCH.md).
_CUE_SHARE = 0.1

Filter = dict[str, Any]


class PointStruct(TypedDict):
    id: str
    vector: dict[str, list[float] | SparseVector]
    payload: dict[str, Any]


@dataclass
class QdrantEndpoint:
    url: str
    api_key: str


EndpointFn = Callable[[], QdrantEndpoint | Awaitable[QdrantEndpoint | None] | None]


def point_id(item: str, n: int) -> str:
    """A point's id from its item and its place (0 = the key, n = the n-th child): the same item writes the same ids.
    SHA-1 of ``item#n`` laid out as a UUID version 5."""
    h = hashlib.sha1(f"{item}#{n}".encode("utf-8", "surrogatepass")).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-5{h[13:16]}-{(int(h[16], 16) & 0x3) | 0x8:x}{h[17:20]}-{h[20:32]}"


class QdrantError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class QdrantClient:
    """``endpoint`` may start Qdrant: every call waits for it. ``http``: the client to call it with (one of its own when
    none is given)."""

    def __init__(
        self,
        endpoint: EndpointFn,
        timeout_ms: float = 10_000,
        collection: str | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        # A name you give is a path segment, never more of the path.
        self._at = f"/collections/{quote(collection or COLLECTION, safe='')}"
        self._given = http
        self._own: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        if self._given is not None:
            return self._given
        if self._own is None:
            self._own = new_client()
        return self._own

    async def aclose(self) -> None:
        """Closes the client it made for itself (a client it was given is its owner's to close)."""
        if self._own is not None:
            await self._own.aclose()
            self._own = None

    async def _call(self, method: str, path: str, body: Any = None, timeout_ms: float | None = None) -> Any:
        got = self._endpoint()
        e = await got if inspect.isawaitable(got) else got
        if not e:
            raise QdrantError("memory index is not running", 503)
        payload = json.dumps(body, allow_nan=False) if body is not None else None
        headers = {"api-key": e.api_key}
        if payload is not None:
            headers["content-type"] = "application/json"
        timeout = (timeout_ms if timeout_ms is not None else self._timeout_ms) / 1000
        res: httpx.Response | None = None
        # Every call here can be sent again (upserts by id, deletes by filter, reads): a connection the server closed
        # while it sat idle (keep-alive) fails before the request reaches it, so it is tried again — twice at most.
        attempt = 0
        while res is None:
            try:
                res = await self._client().request(method, f"{e.url}{path}", headers=headers, content=payload, timeout=timeout)
            except httpx.TimeoutException as err:
                raise QdrantError(f"{method} {path}: timed out ({type(err).__name__})", 0) from err
            except httpx.TransportError as err:
                if attempt < 2:
                    await asyncio.sleep(0.15 * (attempt + 1))
                    attempt += 1
                    continue
                raise QdrantError(f"{method} {path}: {type(err).__name__} ({err})", 0) from err
        text = res.text
        if not res.is_success:
            raise QdrantError(f"{method} {path}: {res.status_code} {text[:300]}", res.status_code)
        data = json.loads(text) if text else {}
        return data.get("result") if isinstance(data, dict) else None

    async def drop_collection(self) -> None:
        """Drop the whole collection: the index is derived, written again from the logs (rebuild.py)."""
        try:
            await self._call("DELETE", self._at)
        except Exception:  # noqa: BLE001 - a collection that is not there is dropped already
            return

    async def ensure_collection(self) -> None:
        """The collection and its payload indexes, made if missing — an index added since a collection was made is
        added to it in place. All of it on disk, memory-mapped: RAM holds what queries touch, and the system may take
        it back."""
        info: dict[str, Any] | None = None
        try:
            info = (await self._call("GET", self._at)) or {}
        except QdrantError as err:
            if err.status != 404:
                raise
        if info is None:
            await self._call(
                "PUT",
                self._at,
                {
                    "vectors": {"dense": {"size": DIMS, "distance": "Cosine", "on_disk": True}},
                    "sparse_vectors": {"bm25": {"modifier": "idf", "index": {"on_disk": True}}},
                    "quantization_config": {"binary": {"always_ram": False}},
                    "hnsw_config": {"m": 16, "ef_construct": 100, "on_disk": True},
                    "on_disk_payload": True,
                    "optimizers_config": {"default_segment_number": _SEGMENTS},
                },
            )
            info = {"payload_schema": {}, "config": {"optimizer_config": {"default_segment_number": _SEGMENTS}}}
        elif ((info.get("config") or {}).get("optimizer_config") or {}).get("default_segment_number") != _SEGMENTS:
            # Made before: two segments from now on (the optimiser merges them in the background).
            await self._call("PATCH", self._at, {"optimizers_config": {"default_segment_number": _SEGMENTS}})
        have = info.get("payload_schema") or {}
        for field_name, field_schema in [
            ("item", "keyword"),
            ("part", "keyword"),
            ("tier", "keyword"),
            ("status", "keyword"),
            ("kind", "keyword"),
            ("engine", "keyword"),
            ("scope", "keyword"),
            ("project", "keyword"),
            ("session", "keyword"),
            ("ts", "integer"),
            ("synthetic", "bool"),
            ("concepts", "keyword"),
            ("eventFrom", "integer"),
            ("eventTo", "integer"),
        ]:
            schema: Any = {"type": field_schema, "on_disk": True} if field_schema in ("keyword", "integer") else field_schema
            if field_name not in have:
                await self._call("PUT", f"{self._at}/index?wait=true", {"field_name": field_name, "field_schema": schema})

    async def upsert(self, points: Sequence[PointStruct], wait: bool = False) -> None:
        if not points:
            return
        await self._call("PUT", f"{self._at}/points?wait={'true' if wait else 'false'}", {"points": list(points)}, 60_000)

    async def delete_items(self, items: Sequence[str]) -> None:
        """Every point of these items — before an item is written again (its children may be fewer now)."""
        if not items:
            return
        await self._call(
            "POST",
            f"{self._at}/points/delete?wait=true",
            {"filter": {"must": [{"key": "item", "match": {"any": list(items)}}]}},
            60_000,
        )

    async def delete_parts(self, items: Sequence[str], part: str) -> None:
        """One kind of point of these items (``cue``), before that part is written again."""
        if not items:
            return
        await self._call(
            "POST",
            f"{self._at}/points/delete?wait=true",
            {"filter": {"must": [{"key": "item", "match": {"any": list(items)}}, {"key": "part", "match": {"value": part}}]}},
            60_000,
        )

    async def set_payload(self, items: Sequence[str], payload: dict[str, Any], key_only: bool = False) -> None:
        """Change some payload fields of these items' points (status, a merged key's display fields)."""
        if not items:
            return
        must: list[Any] = [{"key": "item", "match": {"any": list(items)}}]
        if key_only:
            must.append({"key": "part", "match": {"value": "key"}})
        await self._call("POST", f"{self._at}/points/payload?wait=true", {"payload": payload, "filter": {"must": must}}, 60_000)

    async def neighbors(self, vector: list[float], filter: Filter, limit: int = 5) -> list[dict[str, Any]]:
        """The nearest key points to one vector (consolidation's "do I know this already?"), with their cosine."""
        must = [*(filter.get("must") or []), {"key": "part", "match": {"value": "key"}}]
        r = await self._call(
            "POST",
            f"{self._at}/points/query",
            {
                "query": vector,
                "using": "dense",
                "limit": limit,
                "params": _RESCORE,
                "filter": {**filter, "must": must},
                "with_payload": ["item", "title", "summary", "kind", "scope", "project", "status"],
            },
        )
        return [
            {"item": str((p.get("payload") or {}).get("item", "")), "score": p["score"], "payload": p.get("payload") or {}}
            for p in (r or {}).get("points", [])
        ]

    async def best_per_item(self, vector: list[float], items: Sequence[str], part: str, timeout_ms: float = 2000) -> dict[str, float]:
        """Each of these items' best cosine to one vector among its points of one part (key, child, cue)."""
        if not items:
            return {}
        r = await self._call(
            "POST",
            f"{self._at}/points/query/groups",
            {
                "query": vector,
                "using": "dense",
                "params": _RESCORE,
                "filter": {"must": [{"key": "item", "match": {"any": list(items)}}, {"key": "part", "match": {"value": part}}]},
                "group_by": "item",
                "group_size": 1,
                "limit": len(items),
                "with_payload": False,
            },
            timeout_ms,
        )
        return {str(g["id"]): (g["hits"][0]["score"] if g.get("hits") else 0) for g in (r or {}).get("groups", [])}

    async def nearest_many(
        self,
        vectors: Sequence[list[float]],
        filter: Filter,
        limit: int = 10,
        timeout_ms: float = 5000,
        fields: Sequence[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """The nearest points to several vectors in one call. ``fields``: only these payload fields (and the item)."""
        if not vectors:
            return []
        with_payload: Any = list(dict.fromkeys([*fields, "item"])) if fields else True
        r = await self._call(
            "POST",
            f"{self._at}/points/query/batch",
            {
                "searches": [
                    {
                        "query": v,
                        "using": "dense",
                        "limit": limit,
                        "params": _RESCORE,
                        "filter": filter,
                        "with_payload": with_payload,
                    }
                    for v in vectors
                ]
            },
            timeout_ms,
        )
        return [
            [
                {"item": str((p.get("payload") or {}).get("item", "")), "score": p["score"], "payload": p.get("payload") or {}}
                for p in (x or {}).get("points", [])
            ]
            for x in (r or [])
        ]

    async def scroll(self, filter: Filter, limit: int = 256, with_vector: bool = False, timeout_ms: float = 5000) -> list[dict[str, Any]]:
        """Points within a filter, with their payload (and dense vector when asked) — at most ``limit``."""
        r = await self._call(
            "POST",
            f"{self._at}/points/scroll",
            {
                "filter": filter,
                "limit": limit,
                "with_payload": True,
                "with_vector": ["dense"] if with_vector else False,
            },
            timeout_ms,
        )
        return [_point(p) for p in (r or {}).get("points", [])]

    async def count(self, filter: Filter | None = None) -> int:
        body: dict[str, Any] = {"exact": False}
        if filter:
            body["filter"] = filter
        r = await self._call("POST", f"{self._at}/points/count", body)
        return int((r or {}).get("count", 0))

    async def retrieve(self, ids: Sequence[str], with_vector: bool = False) -> list[dict[str, Any]]:
        """The items behind ``ids`` (key points), with their payload — and their dense vector when asked."""
        if not ids:
            return []
        r = await self._call(
            "POST",
            f"{self._at}/points",
            {"ids": list(ids), "with_payload": True, "with_vector": ["dense"] if with_vector else False},
        )
        return [_point(p) for p in (r or [])]

    async def fused_items(
        self,
        dense: Sequence[list[float]],
        sparse: SparseVector | None = None,
        filter: Filter | None = None,
        per: int | None = None,
        limit: int | None = None,
        timeout_ms: float = 3000,
    ) -> list[dict[str, Any]]:
        """The best items for the query, fused — each with the point that matched best (``point``: its key, a chunk of
        its text, or a cue), so a caller can show the passage that matched. Each channel ranks up to ``per`` items
        (default: ``limit``, at least 50); their scores are then fused (CombSUM, above)."""
        lim = 40 if limit is None else limit
        each = per if per is not None else max(lim, 50)
        channels: list[tuple[Any, str, bool, float, Filter | None]] = []
        sp = sparse if sparse and sparse["indices"] else None
        dense_weight = (_DENSE_SHARE if sp else 1) / max(1, len(dense))
        base = filter or {}
        cue = {"key": "part", "match": {"value": "cue"}}
        cue_off = {**base, "must_not": [*(base.get("must_not") or []), cue]}
        cue_only = {**base, "must": [*(base.get("must") or []), cue]}
        for v in dense:
            channels.append((v, "dense", True, dense_weight * (1 - _CUE_SHARE), cue_off))
            channels.append((v, "dense", True, dense_weight * _CUE_SHARE, cue_only))
        if sp:
            channels.append((sp, "bm25", False, 1 - _DENSE_SHARE, None))
        if not channels:
            return []

        async def ask(c: tuple[Any, str, bool, float, Filter | None]) -> Any:
            body: dict[str, Any] = {"query": c[0], "using": c[1]}
            if c[2]:
                body["params"] = _RESCORE
            f = c[4] if c[4] is not None else filter
            if f is not None:
                body["filter"] = f
            body.update({"group_by": "item", "group_size": 1, "limit": each, "with_payload": False})
            return await self._call("POST", f"{self._at}/points/query/groups", body, timeout_ms)

        lists = await asyncio.gather(*(ask(c) for c in channels))
        fused: dict[str, dict[str, Any]] = {}
        for c, r in zip(channels, lists, strict=False):
            groups = (r or {}).get("groups", [])
            scores = [(g["hits"][0]["score"] if g.get("hits") else 0) for g in groups]
            lo = min(scores) if scores else math.inf
            hi = max(scores) if scores else -math.inf
            for rank, g in enumerate(groups):
                item = str(g["id"])
                raw = g["hits"][0].get("id") if g.get("hits") else None
                at = fused.setdefault(item, {"score": 0.0, "rank": math.inf})
                # A channel whose items all score alike (or that found one) counts each of them fully.
                at["score"] += c[3] * ((scores[rank] - lo) / (hi - lo) if hi > lo else 1)
                # The matched passage: the point from the channel that ranked the item highest.
                if rank < at["rank"] and raw:
                    at["rank"] = rank
                    at["point"] = str(raw)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:lim]
        return [{"item": item, "score": f["score"], **({"point": f["point"]} if f.get("point") else {})} for item, f in ranked]


def _point(p: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": str(p.get("id")), "payload": p.get("payload") or {}}
    vec = p.get("vector")
    if isinstance(vec, dict) and vec.get("dense"):
        out["vector"] = vec["dense"]
    return out
