"""The second Python answer key (conformance/generate-python-2.ts): where JavaScript and Python differ underneath — a lone
surrogate in what is hashed (Node writes it as U+FFFD), a JSON Lines id that is a number (String(1.0) is "1"), a
record whose metadata is an empty object — the port gives what the reference gives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conftest import fixture
from fakes import FakeQdrant, embed, item
from memtrails.core.consolidate import ConsolidateDeps
from memtrails.core.profiles import profile_subjects
from memtrails.core.qdrant_client import point_id
from memtrails.core.store import fingerprint, ltm_log, stm_log
from memtrails.core.text import doc_sparse
from memtrails.core.writer import write_ltm
from memtrails.importers.jsonl import from_jsonl
from memtrails.records import entry_of, record_uid

KEY = fixture("python-2.json")
NOW = 1775131200000


def test_record_uids() -> None:
    for x in KEY["recordUid"]:
        r = {"id": x["id"], **({"source": x["source"]} if x["source"] is not None else {})}
        assert record_uid(r) == x["uid"]


def test_fingerprints() -> None:
    for x in KEY["fingerprint"]:
        assert fingerprint(*x["parts"]) == x["fingerprint"]


def test_point_ids() -> None:
    for x in KEY["pointId"]:
        assert point_id(x["item"], x["n"]) == x["id"]


def test_profile_subjects() -> None:
    lone = "note \ud83d end"
    items: list[dict[str, Any]] = [
        item(
            uid,
            f"title {uid} {lone}",
            "summary \ude00 first",
            body="",
            ts=NOW - int(uid[-1]) * 1000,
            firstTs=NOW,
            session="s",
            concepts=[f"gate{lone}"],
        )
        for uid in ("mem:1", "mem:2", "mem:3")
    ]
    got = [{"uid": s["uid"], "concept": s["concept"], "fingerprint": s["fingerprint"]} for s in profile_subjects(items, {"-p": 15})]
    assert got == KEY["profileSubjects"]


def test_entries() -> None:
    for x in KEY["entries"]:
        assert entry_of(x["record"], NOW) == x["entry"]


async def test_jsonl_ids_that_are_numbers(tmp_path: Path) -> None:
    path = tmp_path / "ids.jsonl"
    path.write_text("".join(f'{{"id":{id},"text":"a record"}}\n' for id in KEY["jsonlIds"]["written"]))
    assert [r["id"] async for r in from_jsonl(path)] == KEY["jsonlIds"]["read"]


async def test_a_file_named_with_a_trailing_slash_is_indexed_by_its_name(tmp_path: Path) -> None:
    client = FakeQdrant()
    deps = ConsolidateDeps(client=client, embed=embed, stm=stm_log(str(tmp_path / "s")), ltm=ltm_log(str(tmp_path / "l")))
    it = item("mem:f", "the page", "where it is", files=["docs/page/", "a/b.md"])
    await write_ltm(deps, [it])
    key = next(p for p in client.points.values() if p["payload"].get("part") == "key")
    assert key["vector"]["bm25"] == doc_sparse("the page\nwhere it is\nwhere it is\npage b.md")
