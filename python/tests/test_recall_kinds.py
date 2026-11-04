"""Search by kind and by date: a record's kind is on every point of it (a match deep in its text counts), a learned
item's kind is its own; a date that is not one is refused rather than read as an open end. (recall-kinds.test.ts.)"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from fakes import Setup, item, run
from memtrails._js import date_parse, date_utc
from memtrails.core.qdrant_client import QdrantClient, QdrantEndpoint
from memtrails.core.recall import memory_search
from memtrails.core.writer import date_span, write_ltm, write_stm


def entry(uid: str, engine: str, reply: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "fingerprint": uid,
        "engine": engine,
        "session": uid,
        "turn": 0,
        "cwd": None,
        "project": "-p",
        "title": "Weekly notes",
        "ts": int(date_parse("2026-04-02T00:00:00Z")),
        "synthetic": False,
        "typed": "human",
        "request": "Weekly notes",
        "reply": reply,
        "said": reply,
        "steps": [],
        "files": [],
        "done": False,
    }


FILLER = " ".join(f"agenda{i % 41} minutes{i % 29}" for i in range(300))


def seeded(tmp_path: Path) -> Setup:
    s = Setup(tmp_path)
    # The fact the search is for sits deep in each record's text: only a chunk of it matches, never its key.
    said = f"{FILLER} the backup server moved to rack seven {FILLER}"
    run(write_stm(s.deps(), [entry("mail:1", "email", said), entry("chat:1", "message", said)]))
    run(
        write_ltm(
            s.deps(),
            [
                item(
                    "mem:rack",
                    "the backup server moved to rack seven",
                    "the backup server moved to rack seven",
                    engine="email",
                    ts=1,
                    firstTs=1,
                    confidence="stated",
                )
            ],
        )
    )
    return s


def test_finds_a_record_of_that_kind_by_a_chunk(tmp_path: Path) -> None:
    s = seeded(tmp_path)
    hits = run(memory_search(s.recall(), {"query": "backup server rack seven", "kinds": ["email"]}))
    assert [h["id"] for h in hits] == ["stm:mail:1"]
    assert hits[0]["kind"] == "email"
    assert "backup server moved to rack seven" in hits[0]["match"]


def test_finds_learned_items_by_their_own_kind(tmp_path: Path) -> None:
    s = seeded(tmp_path)

    def ids(kinds: list[str]) -> list[str]:
        return [h["id"] for h in run(memory_search(s.recall(), {"query": "backup server rack seven", "kinds": kinds}))]

    assert ids(["fact"]) == ["mem:rack"]
    assert "mem:rack" not in ids(["email"])
    assert sorted(ids(["email", "fact"])) == ["mem:rack", "stm:mail:1"]


def test_the_records_kind_is_on_every_point(tmp_path: Path) -> None:
    s = seeded(tmp_path)
    mail = [p for p in s.points.values() if p["payload"]["item"] == "stm:mail:1"]
    assert len(mail) > 1
    assert all(p["payload"]["engine"] == "email" for p in mail)


def test_refuses_a_date_that_is_not_one(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    for q, why in (
        ({"from": "April 2026"}, 'from "April 2026" is not a date'),
        ({"to": "2026-13"}, 'to "2026-13"'),
        ({"from": "2026-02-30"}, "is not a date"),
    ):
        with pytest.raises(ValueError, match=why):
            run(memory_search(s.recall(), {"query": "x", **q}))
    assert run(memory_search(s.recall(), {"query": "x", "from": "2026-04", "to": "2026-04-30"})) == []


def test_reads_only_days_and_months_that_exist() -> None:
    for d in ("2024-02-30", "2024-02-00", "2024-13", "2024-00", "2023-02-29"):
        assert date_span(d) is None
    assert date_span("2024-02-29") == {"from": date_utc(2024, 1, 29), "to": date_utc(2024, 2, 1) - 1}
    assert date_span("2024-12") == {"from": date_utc(2024, 11, 1), "to": date_utc(2025, 0, 1) - 1}


def test_the_index_has_the_kind_and_keeps_the_collection_in_its_segment() -> None:
    calls: list[tuple[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((str(req.url), json.loads(req.content) if req.content else None))
        if req.method == "GET":
            return httpx.Response(404, text='{"status":{"error":"not found"}}')
        return httpx.Response(200, text='{"result":true}')

    client = QdrantClient(lambda: QdrantEndpoint("http://q", "k"), 1000, "a/b?c", transport=httpx.MockTransport(handler))
    run(client.ensure_collection())
    assert all(url.startswith("http://q/collections/a%2Fb%3Fc") for url, _ in calls)
    assert "engine" in [(b or {}).get("field_name") for _, b in calls]
