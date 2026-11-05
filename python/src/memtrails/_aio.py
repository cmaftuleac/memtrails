"""asyncio pieces the port needs on every Python it supports (3.10 has neither TaskGroup nor asyncio.timeout)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def gather_or_cancel(*aws: Awaitable[T]) -> list[T]:
    """``asyncio.gather``, but the first that raises cancels the others before it is raised, and they have ended when it
    is: asks the model is paid for are not left running for an answer no one reads (a cancelled caller cancels them
    too, as gather does)."""
    tasks: list[asyncio.Future[T]] = [asyncio.ensure_future(a) for a in aws]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
