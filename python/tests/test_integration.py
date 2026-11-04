"""The whole library on the real embedding model and a real Qdrant, with no language model and no server
(memory.integration.test.ts, ported). Both are fetched into the cache on first use; MEMTRAILS_MODEL_DIR and
QDRANT_BINARY point at copies instead. Run with MEMTRAILS_INTEGRATION=1."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from memtrails import Memory, open_memory
from memtrails.llm import LlmRequest

pytestmark = pytest.mark.skipif(
    os.environ.get("MEMTRAILS_INTEGRATION") != "1", reason="needs the real model and Qdrant (MEMTRAILS_INTEGRATION=1)"
)


def opts() -> dict[str, Any]:
    o: dict[str, Any] = {}
    if os.environ.get("MEMTRAILS_MODEL_DIR"):
        o["embedding"] = {"model_dir": os.environ["MEMTRAILS_MODEL_DIR"], "threads": 2}
    if os.environ.get("QDRANT_BINARY"):
        o["qdrant"] = {"binary": os.environ["QDRANT_BINARY"]}
    return o


MAIL = [
    {
        "id": "<a1@ops>",
        "source": "mail",
        "kind": "email",
        "collection": "ops",
        "thread": "cert",
        "author": "Ana Popescu",
        "title": "Proxy certificate expired",
        "text": "The TLS certificate of the proxy expired on the 1st. Renewal stopped because the cron job was removed in March.",
        "at": "2026-04-02",
    },
    {
        "id": "<a2@ops>",
        "source": "mail",
        "kind": "email",
        "collection": "ops",
        "thread": "cert",
        "author": "Ion",
        "title": "Re: Proxy certificate expired",
        "text": "I added the cron job back and renewed it by hand. It is valid until July.",
        "at": "2026-04-03",
    },
    {
        "id": "n1",
        "source": "notes",
        "kind": "note",
        "collection": "home",
        "title": "Router",
        "text": "Parola routerului este în caietul albastru din sertarul de la birou.",
        "at": "2026-04-04",
    },
    {
        "id": "c9f1e2",
        "source": "repo",
        "kind": "commit",
        "collection": "ops",
        "author": "Ion",
        "title": "Restore certificate renewal cron",
        "text": "Files: deploy/cron/renew.sh",
        "at": "2026-04-03",
    },
]


async def test_runs_from_a_relative_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    async with await open_memory("./memory", **opts()) as m:
        await m.add({"id": "r", "text": "A relative home works."})
        assert "relative" in (await m.search({"query": "relative home", "limit": 1}))[0]["summary"]


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[Memory]:
    m = await open_memory(tmp_path / "memory", **opts())
    assert await m.add(MAIL) == {"added": 4, "unchanged": 0}
    await m.add_conversation(
        {
            "id": "lunch",
            "collection": "home",
            "at": "2026-04-05",
            "turns": [
                {"role": "user", "text": "Where did we say we would have lunch on Friday with Ana?"},
                {"role": "assistant", "text": "At the small Italian place near the office, at one o'clock."},
            ],
        }
    )
    yield m
    await m.close()


async def test_finds_searches_reads_and_explains(memory: Memory, tmp_path: Path) -> None:
    hit = (await memory.search({"query": "why did the certificate stop renewing?", "limit": 3}))[0]
    assert "certificate" in hit["title"]
    assert hit["kind"] == "email"
    assert hit["record"]["source"] == "mail" and isinstance(hit["record"]["author"], str)

    commits = await memory.search({"query": "certificate renewal", "kinds": ["commit"], "limit": 5})
    assert [h["kind"] for h in commits] == ["commit"]
    assert all(h["project"] == "home" for h in await memory.search({"query": "certificate renewal", "collections": ["home"], "limit": 5}))

    # Another language than it was asked in, and as an agent reads it.
    text = await memory.search_text({"query": "unde am scris parola de la router?", "limit": 3})
    assert "Router" in text.split("\n")[0]

    room = await memory.room({"query": "what made the renewal stop?", "chars": 300})
    assert len(room) <= 300
    assert "cron job was removed" in room

    lunch = (await memory.search({"query": "lunch with Ana", "limit": 1}))[0]
    assert "Italian" in str((await memory.open(lunch["id"]) or {}).get("body"))
    search = next(t for t in memory.tools() if t.name == "memory_search")
    assert search.json_schema["type"] == "object"
    assert "certificate" in (await search.handler({"query": "proxy certificate"}))["content"][0]["text"]

    assert await memory.add(MAIL) == {"added": 0, "unchanged": 4}
    assert await memory.add({**MAIL[1], "text": "Renewed by hand; valid until August."}) == {"added": 1, "unchanged": 0}
    assert "August" in (await memory.search({"query": "valid until August", "limit": 3}))[0]["summary"]
    assert await memory.remove([{"id": "<a2@ops>", "source": "mail"}]) == 1
    assert not any("August" in h["summary"] for h in await memory.search({"query": "valid until August", "limit": 10}))

    with pytest.raises(TypeError, match="id"):
        await memory.add({"id": "", "text": "x"})
    with pytest.raises(TypeError, match="time"):
        await memory.add({"id": "a", "text": "x", "at": "yesterday-ish"})


async def test_streams_in_and_comes_back_after_a_restart_and_a_lost_index(tmp_path: Path) -> None:
    home = tmp_path / "memory"

    async def many() -> AsyncIterator[dict[str, Any]]:
        for i in range(40):
            yield {"id": f"msg-{i}", "source": "chat", "kind": "message", "text": f"Standup {i}: worked on ticket {1000 + i}."}

    async with await open_memory(home, **opts()) as m:
        await m.add(MAIL)
        assert await m.import_(many(), batch=16) == {"added": 40, "unchanged": 0}
        before = m.stats()
    async with await open_memory(home, **opts()) as m:
        assert m.stats() == before
        assert await m.index_words() == 0
        await m.rebuild()
        assert "1017" in (await m.search({"query": "ticket 1017", "limit": 3}))[0]["summary"]
        assert "certificate" in (await m.search({"query": "expired TLS certificate", "limit": 1}))[0]["summary"]


async def test_gather_reads_as_many_as_asked_and_shows_what_was_kept(tmp_path: Path) -> None:
    prompts: list[str] = []

    async def llm(r: LlmRequest) -> dict[str, Any]:
        prompts.append(r.prompt)
        lines = [x for x in r.prompt.split("\n") if x.startswith("[m")]
        return {"answer": {"keep": [x[1 : x.index("]")] for x in lines if "Restore certificate renewal cron" in x], "searches": []}}

    async with await open_memory(tmp_path / "memory", llm=llm, **opts()) as m:
        await m.add(MAIL)
        g = await m.gather({"query": "the commit that restored the certificate renewal cron", "limit": 1})
        assert len(prompts) == 1
        # A search for a few reads twice as deep as asked: two of five, not all of them.
        assert len([x for x in prompts[0].split("\n") if x.startswith("[m")]) <= 2
        assert len(g["kept"]) == 1
        assert "Restore certificate renewal cron" in g["found"][0]["title"]
