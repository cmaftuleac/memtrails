"""The agent tools at their edges: arguments past the limits, an index that fails (not a memory that is missing), dates
that do not exist (agent-tools-limits.test.ts, ported), and the schemas the reference gives, key for key."""

from __future__ import annotations

import json
from typing import Any

from memtrails.tools.agent_tools import AgentTool, build_agent_tools, readable_date


class Provider:
    def __init__(self, **over: Any) -> None:
        self.searched = 0
        self._open = over.get("open")

    def search(self, q: Any) -> list[dict[str, Any]]:
        self.searched += 1
        return []

    def open(self, id: str) -> dict[str, Any] | None:
        return self._open(id) if self._open else None

    def explore(self, q: Any) -> dict[str, Any]:
        return {"found": [], "related": [], "concepts": []}


def by_name(p: Any) -> dict[str, AgentTool]:
    return {t.name: t for t in build_agent_tools(p)}


def said(r: dict[str, Any]) -> str:
    return str(r["content"][0]["text"])


def test_refuses_what_is_past_the_limits() -> None:
    p = Provider()
    tools = by_name(p)
    long = "x" * 2001
    for name, args in (
        ("memory_search", {"query": long}),
        ("memory_search", {"query": "ok", "collections": [f"c{n}" for n in range(51)]}),
        ("memory_open", {"id": long}),
        ("memory_explore", {"topic": long}),
    ):
        r = tools[name].handler(args)
        assert r.get("isError") is True, name
        assert said(r).startswith("Invalid arguments")
    assert p.searched == 0
    schema = json.dumps(tools["memory_search"].json_schema, separators=(",", ":"))
    assert '"maxLength":2000' in schema and '"maxItems":50' in schema
    assert "isError" not in tools["memory_search"].handler({"query": "x" * 2000})


def test_the_messages_zod_gives() -> None:
    tools = by_name(Provider())
    bad = {"query": "x" * 2001, "id": "x" * 2001, "topic": "x" * 2001, "collections": ["c"] * 51, "limit": 0, "from": "bad"}
    assert said(tools["memory_search"].handler(bad)) == (
        "Invalid arguments: query: Too big: expected string to have <=2000 characters; collections: Too big: expected array to have <=50 items; "
        "limit: Too small: expected number to be >=1; from: Invalid string: must match pattern /^\\d{4}(-\\d{2}(-\\d{2})?)?$/"
    )
    assert said(tools["memory_explore"].handler(bad)) == (
        "Invalid arguments: topic: Too big: expected string to have <=2000 characters; from: Invalid input: expected array, received string; "
        "collections: Too big: expected array to have <=50 items; limit: Too small: expected number to be >=1"
    )


def test_the_schemas_are_the_references() -> None:
    tools = by_name(Provider())
    assert json.dumps(tools["memory_open"].json_schema, separators=(",", ":"), ensure_ascii=False) == (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"id":{"type":"string","maxLength":2000,'
        '"description":"The memory id, as memory_search or memory_explore gave it."}},"required":["id"],"additionalProperties":false}'
    )


def test_an_index_that_failed_is_not_a_missing_memory() -> None:
    def fail(_id: str) -> None:
        raise RuntimeError("POST /collections/memory/points: 503 unavailable")

    r = by_name(Provider(open=fail))["memory_open"].handler({"id": "mem:1"})
    assert r["isError"] is True
    assert "503 unavailable" in said(r)
    assert "There is no memory" not in said(r)
    assert 'There is no memory "mem:1"' in said(by_name(Provider())["memory_open"].handler({"id": "mem:1"}))


def test_readable_dates() -> None:
    assert readable_date("2023-13") == "2023-13"
    assert readable_date("2023-02-30") == "2023-02-30"
    assert readable_date("2023-07-14") == "Fri 14 Jul 2023"
    assert readable_date("2023-07") == "Jul 2023"
    assert readable_date("2023") == "2023"
    assert readable_date("") == ""
