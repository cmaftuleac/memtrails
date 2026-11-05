"""A log's compaction and the recall record's trim read and write a whole file on a thread: what the event loop asks
meanwhile (a get, a put, a remove, a recorded event) never waits for them, and is kept by them."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import memtrails.core.recall_log as recall_log
from memtrails.core.recall_log import RecallBook
from memtrails.core.store import JsonlLog

DAY = 86_400_000


def log(path: Path) -> JsonlLog[dict[str, Any]]:
    return JsonlLog(str(path), lambda r: {"n": r["n"]})


def soon(work: Callable[[], Any]) -> bool:
    """``work`` done on a thread of its own within a second (the event loop's turn, standing in)."""
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(1)
    return not t.is_alive()


class Paused:
    """Stops the first call made on the thread being watched until ``go`` is set."""

    def __init__(self, real: Callable[..., Any]) -> None:
        self.real = real
        self.on: threading.Thread | None = None
        self.inside = threading.Event()
        self.go = threading.Event()

    def __call__(self, *args: Any) -> Any:
        if threading.current_thread() is self.on and not self.inside.is_set():
            self.inside.set()
            self.go.wait(5)
        return self.real(*args)

    def start(self, work: Callable[[], Any]) -> threading.Thread:
        t = threading.Thread(target=work, daemon=True)
        self.on = t
        t.start()
        assert self.inside.wait(5)
        return t


def test_a_compaction_never_holds_up_a_get_or_a_put(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.jsonl"
    lg = log(path)
    for i in range(1200):
        lg.put([{"uid": f"u{i % 4}", "text": f"v{i}", "n": i}])
    paused = Paused(JsonlLog._read)
    monkeypatch.setattr(JsonlLog, "_read", staticmethod(paused))
    compaction = paused.start(lg.compact)
    # Meanwhile: read, replaced, added, removed — none of them waits for the compaction.
    assert soon(lambda: lg.get("u1"))
    assert soon(lambda: lg.put([{"uid": "u1", "text": "changed meanwhile", "n": -1}, {"uid": "new", "text": "added meanwhile", "n": -2}]))
    assert soon(lambda: lg.remove(["u2"]))
    paused.go.set()
    compaction.join(5)
    assert not compaction.is_alive()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) < 20
    assert lg.get("u1")["text"] == "changed meanwhile"
    assert lg.get("new")["text"] == "added meanwhile"
    assert lg.get("u2") is None
    assert lg.get("u3")["text"] == "v1199"
    lg.put([{"uid": "after", "text": "after it", "n": 0}])
    again = log(path)
    again.load()
    assert {uid: again.get(uid)["text"] for uid, _ in again.entries()} == {
        "u0": "v1196",
        "u1": "changed meanwhile",
        "u3": "v1199",
        "new": "added meanwhile",
        "after": "after it",
    }


def test_a_trim_never_holds_up_what_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "recall.jsonl"
    clock = {"now": 1_000 * DAY}
    book = RecallBook(str(path), lambda: clock["now"], keep_ms=10 * DAY)
    book.record({"ev": "open", "item": "stm:ancient"})
    clock["now"] += 30 * DAY
    book.record({"ev": "open", "item": "stm:recent"})
    paused = Paused(recall_log.parse)
    monkeypatch.setattr(recall_log, "parse", paused)
    dropped: list[int] = []
    trim = paused.start(lambda: dropped.append(book.trim()))
    assert soon(lambda: book.record({"ev": "open", "item": "stm:meanwhile"}))
    paused.go.set()
    trim.join(5)
    assert dropped == [1]
    items = [json.loads(x)["item"] for x in path.read_text(encoding="utf-8").strip().split("\n")]
    assert items == ["stm:recent", "stm:meanwhile"]
