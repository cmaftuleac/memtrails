"""Checkpoints survive a failed batch and a restart; a rebuild begun again while one runs writes everything
(rebuild.test.ts and rebuild-overlap.test.ts, ported)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from fakes import run
from memtrails.core.rebuild import MemoryRebuild, RebuildDeps
from memtrails.core.store import ltm_log, stm_log

NOW = 1_800_000_000_000


def item(uid: str, status: str = "active") -> dict[str, Any]:
    return {
        "uid": uid,
        "kind": "fact",
        "scope": "project",
        "project": "-p",
        "engine": "conversation",
        "session": "s",
        "cwd": "/p",
        "where": "c",
        "ts": 1,
        "firstTs": 1,
        "title": uid,
        "summary": uid,
        "body": uid,
        "files": [],
        "appliesWhen": "",
        "confidence": "verified",
        "importance": 3,
        "status": status,
        "provenance": [],
        "seen": 1,
        "fingerprint": uid,
    }


def entry(uid: str, **over: Any) -> dict[str, Any]:
    return {
        "uid": uid,
        "fingerprint": uid,
        "engine": "conversation",
        "session": "s",
        "turn": 1,
        "cwd": "/p",
        "project": "-p",
        "title": uid,
        "ts": NOW - 1000,
        "synthetic": False,
        "typed": "human",
        "request": uid,
        "reply": "",
        "said": "",
        "steps": [],
        "files": [],
        "done": False,
        **over,
    }


async def rebuild(d: RebuildDeps) -> None:
    await MemoryRebuild(d).run()


class Rig:
    def __init__(self, tmp: Path) -> None:
        self.path = str(tmp / "rebuild.json")
        self.ltm = ltm_log(str(tmp / "ltm.jsonl"))
        self.stm = stm_log(str(tmp / "stm.jsonl"))
        self.written: list[list[str]] = []
        self.stopped = False

    async def write(self, xs: list[dict[str, Any]]) -> None:
        self.written.append([x["uid"] for x in xs])

    def deps(self, **over: Any) -> RebuildDeps:
        base: dict[str, Any] = {
            "path": self.path,
            "ltm": self.ltm,
            "stm": self.stm,
            "write_ltm": self.write,
            "write_stm": self.write,
            "aborted": lambda: self.stopped,
        }
        return RebuildDeps(**{**base, **over})

    def checkpoint(self) -> Any:
        return json.loads(Path(self.path).read_text())


def test_resumes_after_a_failed_batch_and_a_restart(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item(f"mem:{i}") for i in range(41)])
    calls = {"n": 0}

    async def flaky(xs: list[dict[str, Any]]) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("GPU service temporarily unavailable")
        await r.write(xs)

    async def first() -> None:
        job = MemoryRebuild(r.deps(write_ltm=flaky))
        job.begin()
        await job.run()

    with pytest.raises(RuntimeError, match="unavailable"):
        run(first())
    assert r.checkpoint() == {"phase": "ltm", "after": "mem:19"}
    r.ltm.load()

    async def resume() -> bool:
        job = MemoryRebuild(r.deps())
        await job.run()
        return job.pending()

    assert run(resume()) is False
    assert [u for w in r.written for u in w] == [f"mem:{i}" for i in range(41)]


def test_several_batches_at_once_never_pass_one_in_flight(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.stm.put([entry(f"stm:{i:03d}") for i in range(32 * 6)])
    n = {"k": 0}
    delays = [30, 10, 40, 1, 1, 1]

    async def slow(xs: list[dict[str, Any]]) -> None:
        k = n["k"]
        n["k"] += 1
        await asyncio.sleep(delays[k] / 1000)
        if k == 2:
            raise RuntimeError("GPU worker reset")
        await r.write(xs)

    async def first() -> None:
        job = MemoryRebuild(r.deps(concurrency=4, write_stm=slow))
        job.begin()
        await job.run()

    with pytest.raises(RuntimeError, match="reset"):
        run(first())
    assert r.checkpoint() == {"phase": "stm", "after": "stm:063"}
    r.stm.load()
    r.written.clear()
    run(rebuild(r.deps(concurrency=4)))
    assert [u for w in r.written for u in w] == [f"stm:{i + 64:03d}" for i in range(32 * 4)]


def test_no_work_when_stopped_and_the_marker_stays(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item("mem:a")])

    async def go() -> bool:
        job = MemoryRebuild(r.deps())
        job.begin()
        r.stopped = True
        await job.run()
        return job.pending()

    assert run(go()) is True
    assert r.written == []


def test_indexes_active_knowledge_and_every_real_request(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item("mem:active"), item("mem:archived", "archived"), item("mem:old", "superseded")])
    r.stm.put([entry("recent"), entry("expired", ts=NOW - 7 * 86400_000 - 1), entry("done", done=True), entry("synthetic", synthetic=True)])

    async def go() -> bool:
        job = MemoryRebuild(r.deps())
        job.begin()
        await job.run()
        return job.pending()

    assert run(go()) is False
    assert r.written == [["mem:active"], ["recent", "expired", "done"]]


def test_simultaneous_retries_are_one_rebuild(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item("mem:a")])

    async def go() -> None:
        job = MemoryRebuild(r.deps())
        job.begin()
        await asyncio.gather(job.run(), job.run(), job.run())

    run(go())
    assert r.written == [["mem:a"]]


@pytest.mark.parametrize("checkpoint", ["broken", json.dumps({"phase": "ltm", "after": "not-here"})])
def test_a_damaged_checkpoint_replays_the_phase(tmp_path: Path, checkpoint: str) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item("mem:a")])
    Path(r.path).write_text(checkpoint)

    async def go() -> bool:
        job = MemoryRebuild(r.deps())
        await job.run()
        return job.pending()

    assert run(go()) is False
    assert r.written == [["mem:a"]]


def test_a_cancelled_run_does_not_move_the_checkpoint(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item("mem:a")])

    async def go() -> None:
        release = asyncio.Event()

        async def blocked(xs: list[dict[str, Any]]) -> None:
            await release.wait()
            await r.write(xs)

        job = MemoryRebuild(r.deps(write_ltm=blocked))
        job.begin()
        running = job.run()
        await asyncio.sleep(0)
        job.cancel()
        release.set()
        await running
        assert r.checkpoint() == {"phase": "ltm", "after": None}
        again = MemoryRebuild(r.deps())
        await again.run()
        assert not again.pending()

    run(go())
    assert r.written == [["mem:a"], ["mem:a"]]


def test_resumes_requests_after_all_knowledge(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item("mem:a")])
    r.stm.put([entry(f"stm:{i}") for i in range(33)])
    calls = {"n": 0}

    async def flaky(xs: list[dict[str, Any]]) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("retry")
        await r.write(xs)

    async def first() -> None:
        job = MemoryRebuild(r.deps(write_stm=flaky))
        job.begin()
        await job.run()

    with pytest.raises(RuntimeError, match="retry"):
        run(first())
    assert r.checkpoint() == {"phase": "stm", "after": "stm:31"}
    run(rebuild(r.deps()))
    assert [u for w in r.written for u in w] == ["mem:a", *(f"stm:{i}" for i in range(33))]


def test_a_rebuild_begun_again_while_one_runs(tmp_path: Path) -> None:
    stm = stm_log(str(tmp_path / "stm.jsonl"))
    stm.put([entry(f"r{i:03d}", ts=1) for i in range(32 * 3)])
    index: set[str] = set()

    async def write_stm(xs: list[dict[str, Any]]) -> None:
        await asyncio.sleep(0.02)
        index.update(x["uid"] for x in xs)

    async def nothing(_: list[dict[str, Any]]) -> None:
        return None

    async def go() -> None:
        job = MemoryRebuild(
            RebuildDeps(
                path=str(tmp_path / "rebuild.json"),
                stm=stm,
                ltm=ltm_log(str(tmp_path / "ltm.jsonl")),
                write_ltm=nothing,
                write_stm=write_stm,
                aborted=lambda: False,
            )
        )
        job.begin()
        first = job.run()
        while not index:
            await asyncio.sleep(0.005)
        # memory.rebuild(): the checkpoint first, then the index cleared, then run.
        job.begin()
        index.clear()
        second = job.run()
        await asyncio.gather(first, second)

    run(go())
    assert len(index) == 32 * 3
    assert not (tmp_path / "rebuild.json").exists()


class Writing:
    """A batch write that takes its time, counting the writes still going on."""

    def __init__(self, r: Rig, seconds: float = 0.05) -> None:
        self.r = r
        self.seconds = seconds
        self.now = 0
        self.started = asyncio.Event()

    async def __call__(self, xs: list[dict[str, Any]]) -> None:
        self.now += 1
        self.started.set()
        try:
            await asyncio.sleep(self.seconds)
            await self.r.write(xs)
        finally:
            self.now -= 1


async def test_a_run_after_one_its_caller_cancelled_is_not_cancelled_with_it(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item(f"mem:{i}") for i in range(30)])
    slow = Writing(r, seconds=10)
    job = MemoryRebuild(r.deps(write_ltm=slow))
    job.begin()
    first = job.run()
    await slow.started.wait()
    first.cancel()
    job.begin()
    job.d.write_ltm = r.write
    await job.run()  # waits for the cancelled one, and is not cancelled by it
    assert not job.pending()


async def test_a_cancelled_run_leaves_no_batch_writing(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item(f"mem:{i}") for i in range(30)])
    slow = Writing(r, seconds=10)
    job = MemoryRebuild(r.deps(write_ltm=slow, concurrency=2))
    job.begin()
    running = job.run()
    await slow.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert slow.now == 0


async def test_a_stopped_run_returns_once_its_writes_are_done(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.ltm.put([item(f"mem:{i}") for i in range(60)])
    slow = Writing(r)
    took = iter([0.01, 0.2])  # the second batch written slower than the first

    async def write_then_stop(xs: list[dict[str, Any]]) -> None:
        r.stopped = True  # a close, while the first batches are being written
        slow.seconds = next(took)
        await slow(xs)

    job = MemoryRebuild(r.deps(write_ltm=write_then_stop, concurrency=2))
    job.begin()
    await job.run()
    assert slow.now == 0
    # The two batches in flight when it stopped were written; the third was never started.
    assert r.written == [[f"mem:{i}" for i in range(20)], [f"mem:{i}" for i in range(20, 40)]]
