"""The MCP server (mcp.py; mcp.test.ts, ported): its answers are the reference's (conformance/fixtures/mcp.json), the
tools are listed before a memory is open, the memory is opened at the first call and closed when idle or when the input
ends, a memory that cannot be opened is a tool error, and memory_save adds a note. Over a stand-in memory."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from conftest import fixture
from memtrails.mcp import MCP_PROTOCOL_VERSIONS, serve_mcp

KEY = fixture("mcp.json")


class Stand:
    """A stand-in memory: its tools answer with what they were asked; what is added and closed is kept."""

    def __init__(self, log: dict[str, Any]) -> None:
        self.log = log

    def tools(self) -> list[Any]:
        class Tool:
            def __init__(self, name: str) -> None:
                self.name = name

            async def handler(self, args: Any) -> dict[str, Any]:
                return {"content": [{"type": "text", "text": f"{self.name} {json.dumps(args, separators=(',', ':'))}"}]}

        return [Tool(n) for n in ("memory_search", "memory_open", "memory_explore")]

    async def add(self, r: Any) -> dict[str, int]:
        self.log["added"].append(r)
        return {"added": 1, "unchanged": 0}

    async def close(self) -> None:
        self.log["closed"] += 1


class Server:
    """A server over in-memory lines, and a way to talk to it one JSON line at a time."""

    def __init__(self, **o: Any) -> None:
        self.log: dict[str, Any] = {"added": [], "closed": 0, "opened": 0}
        self.lines: asyncio.Queue[bytes] = asyncio.Queue()
        self.answers: asyncio.Queue[str] = asyncio.Queue()
        self.id = 0

        async def open_() -> Any:
            self.log["opened"] += 1
            return Stand(self.log)

        args: dict[str, Any] = {"open": open_, "version": "9.9.9", "log": lambda _l: None, **o}
        self.done = asyncio.ensure_future(serve_mcp(read_line=self.lines.get, write=lambda b: self.answers.put_nowait(b.decode()), **args))

    def send(self, line: str) -> None:
        self.lines.put_nowait(line.encode() + b"\n")

    async def next(self) -> Any:
        return json.loads(await asyncio.wait_for(self.answers.get(), 5))

    async def ask(self, method: str, params: Any = None) -> Any:
        self.id += 1
        self.send(json.dumps({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params}))
        return await self.next()

    async def end(self) -> None:
        self.lines.put_nowait(b"")
        await asyncio.wait_for(self.done, 5)


async def test_answers_every_message_as_the_reference_does() -> None:
    s = Server(version=KEY["version"])
    for x in KEY["session"]:
        s.send(x["send"])
        if x["answer"] is not None:
            assert await s.next() == x["answer"], x["send"]
    await s.end()
    assert s.answers.empty()


async def test_initialize_in_the_clients_version_when_it_speaks_it_else_its_newest() -> None:
    s = Server()
    a = await s.ask("initialize", {"protocolVersion": "2025-03-26"})
    assert a["result"]["protocolVersion"] == "2025-03-26"
    assert a["result"]["serverInfo"] == {"name": "memtrails", "version": "9.9.9"}
    b = await s.ask("initialize", {"protocolVersion": "2030-01-01"})
    assert b["result"]["protocolVersion"] == MCP_PROTOCOL_VERSIONS[0]
    await s.end()


async def test_lists_tools_without_opening_the_memory_and_read_only_has_no_save() -> None:
    s = Server()
    names = [t["name"] for t in (await s.ask("tools/list"))["result"]["tools"]]
    assert names == ["memory_search", "memory_open", "memory_explore", "memory_save"]
    assert s.log["opened"] == 0
    await s.end()
    r = Server(read_only=True)
    assert "memory_save" not in [t["name"] for t in (await r.ask("tools/list"))["result"]["tools"]]
    await r.end()


async def test_opens_once_at_the_first_call_and_closes_when_the_input_ends() -> None:
    s = Server()
    a, b = await asyncio.gather(
        s.ask("tools/call", {"name": "memory_search", "arguments": {"query": "renewal"}}),
        s.ask("tools/call", {"name": "memory_open", "arguments": {"id": "k:1"}}),
    )
    texts = sorted(x["result"]["content"][0]["text"] for x in (a, b))
    assert texts == ['memory_open {"id":"k:1"}', 'memory_search {"query":"renewal"}']
    assert s.log["opened"] == 1
    await s.end()
    assert s.log["closed"] == 1


async def test_closes_when_idle_and_opens_again_at_the_next_call() -> None:
    s = Server(idle_s=0.03)
    await s.ask("tools/call", {"name": "memory_search", "arguments": {"query": "a"}})
    await asyncio.sleep(0.15)
    assert s.log["closed"] == 1
    await s.ask("tools/call", {"name": "memory_search", "arguments": {"query": "b"}})
    assert s.log["opened"] == 2
    await s.end()
    assert s.log["closed"] == 2


async def test_says_why_the_memory_cannot_be_opened_and_tries_again() -> None:
    tries = {"n": 0}

    async def open_() -> Any:
        tries["n"] += 1
        if tries["n"] == 1:
            raise RuntimeError("memory /m is open in process 4242: close it there first")
        return Stand({"added": [], "closed": 0})

    s = Server(open=open_)
    a = await s.ask("tools/call", {"name": "memory_search", "arguments": {"query": "a"}})
    assert a["result"] == {
        "content": [{"type": "text", "text": "memory /m is open in process 4242: close it there first"}],
        "isError": True,
    }
    b = await s.ask("tools/call", {"name": "memory_search", "arguments": {"query": "a"}})
    assert "isError" not in b["result"]
    await s.end()


async def test_saves_a_note() -> None:
    s = Server()
    a = await s.ask(
        "tools/call", {"name": "memory_save", "arguments": {"text": "Ana prefers the Friday deploy window.", "title": "Deploy window"}}
    )
    assert a["result"]["content"][0]["text"].startswith("Saved (")
    [r] = s.log["added"]
    assert {k: r[k] for k in ("source", "kind", "title", "text")} == {
        "source": "mcp",
        "kind": "note",
        "title": "Deploy window",
        "text": "Ana prefers the Friday deploy window.",
    }
    assert r["at"].endswith("Z") and r["id"].startswith(r["at"])
    await s.end()
