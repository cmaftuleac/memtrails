"""
Embedding over HTTP, for a memory that embeds on a server: any service that speaks the OpenAI embeddings API
(``POST {url}/embeddings``, ``{ model, input }``) and serves EmbeddingGemma-300m — text-embeddings-inference, Infinity,
Ollama (``embeddinggemma``), vLLM, your own. Vectors may come back as numbers or as little-endian float32 base64.

Memory's prompts, indexes and measured thresholds are EmbeddingGemma's: a different model needs its own index, and
its results are not the measured ones.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import gzip
import math
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypedDict

import httpx
import numpy as np
import numpy.typing as npt

from .._http import using
from .._js import stringify, u16slice, utf8

MEMORY_EMBEDDER = {"id": "embeddinggemma-300m-768", "model": "embeddinggemma-300m", "dims": 768}

Vector = npt.NDArray[np.float32]
#: Prompted texts → unit vectors, in order; the second argument is the time it may take, in milliseconds.
Embed = Callable[..., Awaitable[list[Vector]]]


def doc_text(title: str, text: str) -> str:
    """EmbeddingGemma's document prompt (a query and a document are embedded with different prefixes)."""
    return f"title: {title or 'none'} | text: {text}"


def query_text(q: str) -> str:
    """EmbeddingGemma's query prompt."""
    return f"task: search result | query: {q}"


class _RouteBase(TypedDict):
    #: The API's base, e.g. ``http://localhost:8080/v1`` (``/embeddings`` is added).
    url: str


class EmbedRoute(_RouteBase, total=False):
    api_key: str
    #: The model name the service knows it by. Default ``embeddinggemma-300m``.
    model: str
    #: Send bodies gzipped (a service that accepts ``content-encoding: gzip``).
    gzip: bool


def _env(env: Mapping[str, str], key: str) -> str | None:
    v = (env.get(key) or "").strip()
    return v or None


def resolve_embed_route(env: Mapping[str, str] | None = None) -> EmbedRoute | None:
    """The route the environment names: MEMTRAILS_EMBED_URL, MEMTRAILS_EMBED_KEY, MEMTRAILS_EMBED_MODEL; None without
    one."""
    e = os.environ if env is None else env
    url = _env(e, "MEMTRAILS_EMBED_URL")
    if not url:
        return None
    r: EmbedRoute = {"url": url}
    key, model = _env(e, "MEMTRAILS_EMBED_KEY"), _env(e, "MEMTRAILS_EMBED_MODEL")
    if key:
        r["api_key"] = key
    if model:
        r["model"] = model
    return r


class EmbeddingUnavailable(Exception):
    """No way to embed: neither a model directory nor a server was given."""


_MAX_INPUTS = 256
_MAX_BYTES = 600_000  # JSON-escaped UTF-8 bytes in one request.
_MAX_TEXT = 24_000
_IN_FLIGHT = 4


def _vectors(body: Any, count: int) -> list[Vector]:
    """Validate the complete response before any vector reaches the index. Never mix dimensions or vector spaces."""
    data = sorted((body or {}).get("data") or [], key=lambda d: d.get("index", 0))
    if len(data) != count:
        raise ValueError(f"embeddings: {len(data)} vectors for {count} texts")
    dims = int(MEMORY_EMBEDDER["dims"])  # type: ignore[call-overload]
    out: list[Vector] = []
    for n, d in enumerate(data):
        if d.get("index") != n:
            raise ValueError("embeddings: missing or duplicate response index")
        emb = d.get("embedding")
        if isinstance(emb, str):
            try:
                raw = base64.b64decode(emb, validate=True)
            except (binascii.Error, ValueError):
                raw = b""
            if len(raw) != dims * 4 or base64.b64encode(raw).decode() != emb:
                raise ValueError("embeddings: invalid base64 vector")
            v = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        else:
            if not isinstance(emb, list) or len(emb) != dims or any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in emb):
                raise ValueError("embeddings: wrong vector dimensions")
            v = np.asarray(emb, dtype=np.float32)
        norm = math.sqrt(float(np.sum(v.astype(np.float64) ** 2)))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("embeddings: non-finite or zero vector")
        out.append((v.astype(np.float64) / norm).astype(np.float32))
    return out


def remote_embedder(
    route: Callable[[], EmbedRoute | None] | None = None,
    timeout_ms: float = 30_000,
    http: httpx.AsyncClient | None = None,
) -> Embed:
    """``embed(texts)`` for memory: unit vectors in the order of ``texts``, however many requests a batch needs.
    ``timeout_ms`` is the whole operation's budget, retries and every batch included; ``http``: the client to call it
    with (one for the call when none is given)."""
    get_route = route or resolve_embed_route

    async def embed(texts: Sequence[str], budget_ms: float | None = None) -> list[Vector]:
        if not texts:
            return []
        r = get_route()
        if not r:
            raise EmbeddingUnavailable("no embedder: give open_memory a model_dir, an embedding server, or set MEMTRAILS_EMBED_URL")
        until = time.monotonic() + (timeout_ms if budget_ms is None else budget_ms) / 1000

        def remaining() -> float:
            ms = math.floor((until - time.monotonic()) * 1000)
            if ms < 1:
                raise TimeoutError("embeddings: operation deadline exceeded")
            return ms

        inputs = [u16slice(t, 0, _MAX_TEXT) or " " for t in texts]
        parts: list[tuple[int, list[str]]] = []
        cur: list[str] = []
        size_sum = 0
        for i, t in enumerate(inputs):
            size = len(utf8(stringify(t))) + 1
            if cur and (len(cur) >= _MAX_INPUTS or size_sum + size > _MAX_BYTES):
                parts.append((i - len(cur), cur))
                cur, size_sum = [], 0
            cur.append(t)
            size_sum += size
        if cur:
            parts.append((len(inputs) - len(cur), cur))
        out: list[Vector | None] = [None] * len(inputs)
        url = f"{r['url'].rstrip('/').removesuffix('/embeddings')}/embeddings"

        async def call(client: httpx.AsyncClient, at: int, batch: list[str]) -> None:
            remaining()
            body = stringify({"model": r.get("model") or MEMORY_EMBEDDER["model"], "input": batch}).encode("utf-8")
            if r.get("gzip"):
                body = gzip.compress(body)
            headers = {"content-type": "application/json"}
            if r.get("gzip"):
                headers["content-encoding"] = "gzip"
            if r.get("api_key"):
                headers["authorization"] = f"Bearer {r['api_key']}"
            attempt = 0
            while True:
                # The whole call in what is left of the budget, the answer read too: httpx's own timeout is per phase.
                left = remaining() / 1000
                try:
                    res = await asyncio.wait_for(client.post(url, content=body, headers=headers, timeout=left), left)
                except asyncio.TimeoutError:
                    raise TimeoutError("embeddings: operation deadline exceeded") from None
                if res.is_success:
                    payload = res.json()
                    remaining()
                    for k, v in enumerate(_vectors(payload, len(batch))):
                        out[at + k] = v
                    return
                message = res.text[:200]
                if (res.status_code == 429 or res.status_code >= 500) and attempt < 3:
                    delay = 500 * 2**attempt
                    if remaining() <= delay:
                        raise TimeoutError("embeddings: operation deadline exceeded")
                    await asyncio.sleep(delay / 1000)
                    attempt += 1
                    continue
                raise RuntimeError(f"embeddings {res.status_code}: {message}")

        queue = list(parts)

        async with using(http) as client:

            async def lane() -> None:
                while queue:
                    at, batch = queue.pop(0)
                    await call(client, at, batch)

            lanes = [asyncio.ensure_future(lane()) for _ in range(min(_IN_FLIGHT, len(parts)))]
            try:
                await asyncio.gather(*lanes)
            except BaseException:
                queue.clear()
                for task in lanes:
                    task.cancel()
                await asyncio.gather(*lanes, return_exceptions=True)
                raise
        return [v for v in out if v is not None]

    return embed
