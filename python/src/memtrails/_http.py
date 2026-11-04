"""One HTTP client for everything a memory talks to (Qdrant, the embedding and rerank servers, the language model, the
downloads): the caller's, or one made for the call and closed after it when there is none. Never the environment's
proxies — memory's own Qdrant is on loopback, and a key meant for a server must reach that server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx


def new_client() -> httpx.AsyncClient:
    """The client a memory makes for itself when it is given none."""
    return httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(30.0))


@asynccontextmanager
async def using(http: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """``http`` when given; else a client of its own for this call."""
    if http is not None:
        yield http
        return
    async with new_client() as own:
        yield own
