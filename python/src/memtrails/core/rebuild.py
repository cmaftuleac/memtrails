"""A derived index rebuild survives unavailable embeddings, a failed batch and a restart. Logs stay truth.

``rebuild.json`` holds the checkpoint (``{"phase": "ltm" | "stm", "after": uid | null}``), the reference's file: a
rebuild a crash left half-done in either library is finished by the next open of either."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .._fs import write_file
from .._js import parse, stringify
from .store import LtmStore, StmStore


@dataclass
class RebuildDeps:
    path: str
    ltm: LtmStore
    stm: StmStore
    write_ltm: Callable[[list[dict[str, Any]]], Awaitable[None]]
    write_stm: Callable[[list[dict[str, Any]]], Awaitable[None]]
    aborted: Callable[[], bool]
    #: Batches written at once (1 when not given). The checkpoint only ever passes batches already written: it moves
    #: over the written ones at the front, so a crash never skips one still in flight.
    concurrency: int | None = None


class MemoryRebuild:
    def __init__(self, d: RebuildDeps) -> None:
        self.d = d
        self._running: asyncio.Future[None] | None = None
        #: The generation ``_running`` was started for: a run() of the same one joins it, a later one waits for it.
        self._running_for = -1
        self._generation = 0

    def cancel(self) -> None:
        self._generation += 1

    def pending(self) -> bool:
        return os.path.exists(self.d.path)

    def begin(self) -> None:
        """Written BEFORE clearing an old index. A crash at any later point leaves a resumable rebuild. A run in flight
        is cancelled: it wrote into the index about to be cleared, and must neither move nor delete the new
        checkpoint."""
        self._generation += 1
        self._save({"phase": "ltm", "after": None})

    def run(self) -> asyncio.Future[None]:
        """Joins the run in flight for the same begin(); after a newer begin(), waits for the cancelled one to stop
        first."""
        generation = self._generation
        if self._running is not None and self._running_for == generation:
            return self._running
        before = self._running

        async def work() -> None:
            if before is not None:
                # Its end, whatever it was — failed, or cancelled by its own caller — is its own: waited for, never
                # raised here.
                await asyncio.wait({before})
            await self._resume(generation)

        task = asyncio.ensure_future(work())

        def clear(t: asyncio.Future[None]) -> None:
            if self._running is t:
                self._running = None
            if not t.cancelled():
                t.exception()

        task.add_done_callback(clear)
        self._running = task
        self._running_for = generation
        return task

    def _save(self, checkpoint: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.d.path) or ".", exist_ok=True)
        tmp = f"{self.d.path}.tmp"
        write_file(tmp, stringify(checkpoint).encode("utf-8"))
        os.replace(tmp, self.d.path)

    async def _phase(
        self,
        every: Iterable[dict[str, Any]],
        after: str | None,
        keep: Callable[[dict[str, Any]], bool],
        size: int,
        write: Callable[[list[dict[str, Any]]], Awaitable[None]],
        phase: str,
        cancelled: Callable[[], bool],
    ) -> bool:
        """One phase from its checkpoint: batches of ``size``, up to ``concurrency`` written at once, the checkpoint
        moved over the written batches at the front. False when cancelled. A failed batch fails the run, the checkpoint
        before it."""
        limit = max(1, self.d.concurrency or 1)
        queue: list[dict[str, Any]] = []

        def advance() -> None:
            last = None
            while queue and queue[0]["task"].done() and not queue[0]["task"].cancelled() and queue[0]["task"].exception() is None:
                last = queue.pop(0)["last"]
            if last is not None and not cancelled():
                self._save({"phase": phase, "after": last})

        async def fail(err: BaseException) -> None:
            # On a failure: the batches still in flight settle first, and the checkpoint takes every written one at the
            # front.
            await asyncio.gather(*(q["task"] for q in queue), return_exceptions=True)
            advance()
            raise err

        async def send(items: list[dict[str, Any]]) -> None:
            task = asyncio.ensure_future(write(items))
            task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
            queue.append({"last": items[-1]["uid"], "task": task})
            while True:
                busy = [q["task"] for q in queue if not q["task"].done()]
                if len(busy) < limit:
                    break
                done, _ = await asyncio.wait(busy, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is not None:
                        await fail(t.exception())
                advance()

        skip = after is not None
        batch: list[dict[str, Any]] = []
        try:
            for it in every:
                if cancelled():
                    # Stopped (a newer begin, a close): what is being written ends before this returns.
                    await asyncio.gather(*(q["task"] for q in queue), return_exceptions=True)
                    return False
                if skip:
                    if it["uid"] == after:
                        skip = False
                    continue
                if not keep(it):
                    continue
                batch.append(it)
                if len(batch) >= size:
                    await send(batch)
                    batch = []
            if batch:
                await send(batch)
            results = await asyncio.gather(*(q["task"] for q in queue), return_exceptions=True)
        except asyncio.CancelledError:
            # The run itself cancelled: no batch is left writing into the index after it.
            for q in queue:
                q["task"].cancel()
            await asyncio.gather(*(q["task"] for q in queue), return_exceptions=True)
            raise
        failed = next((x for x in results if isinstance(x, BaseException)), None)
        if failed is not None:
            await fail(failed)
        advance()
        return not cancelled()

    async def _resume(self, generation: int) -> None:
        def cancelled() -> bool:
            return generation != self._generation or self.d.aborted()

        if not self.pending() or cancelled():
            return
        try:
            with open(self.d.path, encoding="utf-8") as f:
                checkpoint = parse(f.read())
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("phase") not in ("ltm", "stm")
                or (checkpoint.get("after") is not None and not isinstance(checkpoint.get("after"), str))
            ):
                raise ValueError("invalid checkpoint")
        except (OSError, ValueError):
            checkpoint = {"phase": "ltm", "after": None}
        # While rebuilding, nothing else writes to the logs. Live log order therefore stays stable; if a checkpoint's
        # item is absent (a repaired/compacted log), replay that phase instead of silently skipping records.
        log = self.d.ltm if checkpoint["phase"] == "ltm" else self.d.stm
        if checkpoint.get("after") and log.meta(checkpoint["after"]) is None:
            checkpoint["after"] = None
        if checkpoint["phase"] == "ltm":
            done = await self._phase(
                self.d.ltm.all(), checkpoint.get("after"), lambda it: it.get("status") == "active", 20, self.d.write_ltm, "ltm", cancelled
            )
            if not done:
                return
            checkpoint = {"phase": "stm", "after": None}
            self._save(checkpoint)
        # Every request the person made stays searchable, consolidated or not: the raw record is the truth, a memory a
        # key to it (LongMemEval 2410.10813). Only throwaway runs (synthetic) stay out.
        if not await self._phase(
            self.d.stm.all(), checkpoint.get("after"), lambda it: not it.get("synthetic"), 32, self.d.write_stm, "stm", cancelled
        ):
            return
        if not cancelled():
            os.remove(self.d.path)
