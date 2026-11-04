"""
An optional cross-encoder reranker on a server, for search: any service that speaks the common rerank API
(``POST {url}/rerank``, ``{ model, query, documents }`` → ``results[{ index, relevance_score }]``, as
text-embeddings-inference, Infinity, Jina and Cohere do). Measured with BAAI/bge-reranker-v2-m3 over the best 30
(docs/BENCHMARKS.md): on little text it matched word by word (words.py), which needs no server — so it is off unless you
give it one.

Optional by design: no route, a refusal, an error or a late answer — and the search keeps its own order.
"""

from __future__ import annotations

import gzip
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict, TypeVar

import httpx

from .._js import js_trim, stringify, u16slice
from .words import Rerank

MEMORY_RERANKER = {"model": "bge-reranker-v2-m3"}
#: How many of the best candidates it reads (20 lost 4.6 points of LoCoMo multi-hop against 30).
RERANK_TOP = 30
#: What it reads of each: the title and summary, cut here.
RERANK_CHARS = 1500

T = TypeVar("T")


class _RouteBase(TypedDict):
    #: The API's base, e.g. ``http://localhost:8081/v1`` (``/rerank`` is added).
    url: str


class RerankRoute(_RouteBase, total=False):
    api_key: str
    #: Default ``bge-reranker-v2-m3``.
    model: str
    gzip: bool


def resolve_rerank_route(env: Mapping[str, str] | None = None) -> RerankRoute | None:
    """The route the environment names: MEMTRAILS_RERANK_URL, MEMTRAILS_RERANK_KEY, MEMTRAILS_RERANK_MODEL; None
    without one."""
    e = os.environ if env is None else env
    url = js_trim(e.get("MEMTRAILS_RERANK_URL") or "")
    if not url:
        return None
    r: RerankRoute = {"url": url}
    key, model = js_trim(e.get("MEMTRAILS_RERANK_KEY") or ""), js_trim(e.get("MEMTRAILS_RERANK_MODEL") or "")
    if key:
        r["api_key"] = key
    if model:
        r["model"] = model
    return r


def remote_reranker(route: Callable[[], RerankRoute | None] | None = None, transport: httpx.AsyncBaseTransport | None = None) -> Rerank:
    """Each document's relevance to the query, in the documents' order — or None (no route, refused, failed, late)."""
    get_route = route or resolve_rerank_route

    async def rerank(query: str, documents: list[str], timeout_ms: float) -> list[float] | None:
        r = get_route()
        if not r or not documents or timeout_ms < 100:
            return None
        try:
            body = stringify(
                {
                    "model": r.get("model") or MEMORY_RERANKER["model"],
                    "query": query,
                    "documents": [u16slice(t, 0, RERANK_CHARS) or " " for t in documents],
                }
            ).encode("utf-8")
            headers = {"content-type": "application/json"}
            if r.get("gzip"):
                body = gzip.compress(body)
                headers["content-encoding"] = "gzip"
            if r.get("api_key"):
                headers["authorization"] = f"Bearer {r['api_key']}"
            url = f"{r['url'].rstrip('/').removesuffix('/rerank')}/rerank"
            async with httpx.AsyncClient(transport=transport, trust_env=False) as http:
                res = await http.post(url, content=body, headers=headers, timeout=timeout_ms / 1000)
            if not res.is_success:
                return None
            reply = res.json()
            scores = [math.nan] * len(documents)
            for x in reply.get("results") or []:
                i, s = x.get("index"), x.get("relevance_score")
                if isinstance(i, int) and 0 <= i < len(documents) and isinstance(s, (int, float)) and not isinstance(s, bool):
                    scores[i] = float(s)
            return None if any(math.isnan(s) for s in scores) else scores
        except Exception:  # noqa: BLE001 - optional by design: the search keeps its own order
            return None

    return rerank


def reranked_order(items: Sequence[T], scores: Sequence[float] | None) -> list[T]:
    """The candidates in the reranker's order, fused with their own (by rank, k = 10: 0.7 to the reranker, 0.3 to the
    search's own order — the measured mix); no scores leave the order as it was."""
    if not scores or len(scores) != len(items):
        return list(items)
    by_score = sorted(range(len(items)), key=lambda i: (-scores[i], i))
    fused = [0.0] * len(items)
    for r, i in enumerate(by_score):
        fused[i] += 0.7 / (10 + r + 1)
    for i in range(len(items)):
        fused[i] += 0.3 / (10 + i + 1)
    return [items[i] for i in sorted(range(len(items)), key=lambda i: (-fused[i], i))]
