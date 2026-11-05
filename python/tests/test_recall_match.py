"""The passage search shows as ``match`` is one found beyond the key's start: a short record's only chunk is its whole
key, the two points tie, and whichever Qdrant returns first must not repeat the key as a "match". (recall-match.test.ts.)"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fakes import FakeQdrant, Setup, item, run
from memtrails._js import date_parse
from memtrails.core.qdrant_client import point_id
from memtrails.core.recall import memory_search
from memtrails.core.writer import write_ltm, write_stm


class ChunkFirst(FakeQdrant):
    """Qdrant whose best point for every item is its first chunk — the tie broken the way that showed the quirk."""

    async def fused_items(self, dense: Sequence[Sequence[float]], *_: Any, **__: Any) -> list[dict[str, Any]]:
        out = []
        for p in self.points.values():
            if p["payload"].get("part") != "key":
                continue
            it = str(p["payload"]["item"])
            chunk = point_id(it, 1)
            out.append({"item": it, "score": 1.0, "point": chunk if chunk in self.points else point_id(it, 0)})
        return out


def setup(tmp_path: Path) -> Setup:
    s = Setup(tmp_path)
    s.client = ChunkFirst()
    s.points = s.client.points
    return s


def record(uid: str, request: str, reply: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "fingerprint": uid,
        "engine": "email",
        "session": uid,
        "turn": 0,
        "cwd": None,
        "project": "-p",
        "title": "Mail",
        "ts": int(date_parse("2026-04-02T00:00:00Z")),
        "synthetic": False,
        "typed": "human",
        "request": request,
        "reply": reply,
        "said": reply,
        "steps": [],
        "files": [],
        "done": False,
    }


def test_never_the_key_again(tmp_path: Path) -> None:
    s = setup(tmp_path)
    run(write_stm(s.deps(), [record("mail:1", "Router password", "It is on the sticker under the router.")]))
    hits = run(memory_search(s.recall(), {"query": "router password", "limit": 5}))
    assert hits[0]["id"] == "stm:mail:1"
    assert hits[0]["summary"] == "Router password It is on the sticker under the router."
    assert "match" not in hits[0]


def test_never_the_start_of_the_key(tmp_path: Path) -> None:
    s = setup(tmp_path)
    summary = "The backup server moved to rack seven.\nIt was on Monday."
    run(write_ltm(s.deps(), [item("mem:rack", summary, summary, body="The backup server moved to rack seven.")]))
    hits = run(memory_search(s.recall(), {"query": "backup server", "limit": 5}))
    assert hits[0]["id"] == "mem:rack"
    assert "match" not in hits[0]


def test_still_a_passage_beyond_the_start(tmp_path: Path) -> None:
    s = setup(tmp_path)
    body = "It now sits in rack seven, beside the switch."
    run(write_ltm(s.deps(), [item("mem:rack", "The backup server moved.", "The backup server moved.", body=body)]))
    hits = run(memory_search(s.recall(), {"query": "rack seven", "limit": 5}))
    assert hits[0]["match"] == body
