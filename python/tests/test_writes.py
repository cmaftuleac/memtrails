"""Every file memory writes lands whole, or the write raises and nothing is indexed past what is on disk: ``os.write``
may write fewer bytes than it was given (a full disk, a signal)."""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from memtrails.core.rebuild import MemoryRebuild, RebuildDeps
from memtrails.core.recall_log import RecallBook
from memtrails.core.store import UsageBook, stm_log

_real_write = os.write


def short_writes(limit: int = 7) -> Callable[[int, Any], int]:
    """An os.write that writes at most ``limit`` bytes a call, as a busy or nearly full disk may."""

    def write(fd: int, data: Any) -> int:
        return _real_write(fd, bytes(data)[:limit])

    return write


def failing_after(n: int) -> Callable[[int, Any], int]:
    """An os.write that writes ``n`` bytes, then fails as a full disk does."""
    left = {"n": n}

    def write(fd: int, data: Any) -> int:
        if left["n"] <= 0:
            raise OSError(errno.ENOSPC, "No space left on device")
        k = min(left["n"], len(bytes(data)))
        left["n"] -= k
        return _real_write(fd, bytes(data)[:k])

    return write


def entry(uid: str, text: str = "x") -> dict[str, Any]:
    return {"uid": uid, "fingerprint": uid, "session": "s", "ts": 1, "request": text * 40, "reply": "", "done": False}


async def noop(_: list[dict[str, Any]]) -> None:
    return None


def test_short_writes_still_write_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "write", short_writes())
    log = stm_log(str(tmp_path / "records.jsonl"))
    log.put([entry("a"), entry("b", "y")])
    log.remove(["a"])
    assert log.get("b") == entry("b", "y")

    book = RecallBook(str(tmp_path / "recall.jsonl"))
    book.record({"ev": "open", "item": "stm:b"})
    assert json.loads((tmp_path / "recall.jsonl").read_text())["item"] == "stm:b"

    usage = UsageBook(str(tmp_path / "usage.json"))
    usage.note(["mem:a"], "opened")
    usage.close()
    assert json.loads((tmp_path / "usage.json").read_text())["mem:a"]["opened"] == 1

    job = MemoryRebuild(
        RebuildDeps(path=str(tmp_path / "rebuild.json"), ltm=log, stm=log, write_ltm=noop, write_stm=noop, aborted=lambda: False)
    )
    job.begin()
    assert json.loads((tmp_path / "rebuild.json").read_text()) == {"phase": "ltm", "after": None}

    monkeypatch.setattr(os, "write", _real_write)
    again = stm_log(str(tmp_path / "records.jsonl"))
    again.load()
    assert again.size() == 1
    assert again.get("b") == entry("b", "y")


def test_a_failed_append_indexes_nothing_and_leaves_no_torn_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "records.jsonl"
    log = stm_log(str(path))
    log.put([entry("a")])
    monkeypatch.setattr(os, "write", failing_after(10))
    with pytest.raises(OSError):
        log.put([entry("b")])
    monkeypatch.setattr(os, "write", _real_write)
    assert log.meta("b") is None
    # What is put next is where the index says it is: no torn half-line before it.
    log.put([entry("c", "z")])
    assert log.get("c") == entry("c", "z")
    again = stm_log(str(path))
    again.load()
    assert sorted(uid for uid, _ in again.entries()) == ["a", "c"]
