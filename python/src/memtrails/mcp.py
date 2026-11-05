"""A memory as an MCP server over stdio (newline-delimited JSON-RPC), for Claude Desktop, Claude Code, Cursor and any
other MCP client: the agent tools (memory_search, memory_open, memory_explore) and, unless read-only, memory_save
(mcp.ts, ported; its answers are held to conformance/fixtures/mcp.json).

The memory is opened at the first tool call and closed again after ``idle_s`` seconds without one, so the home is free in
between: for ``memtrails add``, or another client on the same memory. A memory another process holds is a tool error
that says which process, not a server that will not start."""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from .memory import Memory
from .tools.agent_tools import AgentTool, _validate, build_agent_tools

#: The protocol versions this server speaks, newest first; a client asking for another gets the newest.
MCP_PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]

ToolResult = dict[str, Any]

_INSTRUCTIONS = (
    "Long-term memory: what this person or team worked on, wrote, decided and said, and what was learned "
    "from it. Search it before answering about past work, people, decisions or dates, and open a result to read it whole. "
    "Save what is worth remembering later: a decision and why, a preference, a fact about a person or a system."
)

_SAVE_DESCRIPTION = (
    "Save something to memory, to be found later by memory_search: a decision and why, a preference, "
    "a fact about a person, a project or a system, what was learned. Write it to be understood on its own, later, by "
    "someone who was not here: name who and what, and the date when it matters."
)

_SAVE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1, "description": "What to remember, written to be understood on its own."},
        "title": {"description": "A short title (default: the start of the text).", "type": "string", "maxLength": 200},
    },
    "required": ["text"],
    "additionalProperties": False,
}

_READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}
_SAVE_ANNOTATIONS = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}


def _text(text: str, error: bool = False) -> ToolResult:
    return {"content": [{"type": "text", "text": text}], **({"isError": True} if error else {})}


class _Definition:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        annotations: dict[str, bool],
        run: Callable[[Memory, Any], Awaitable[ToolResult]],
    ) -> None:
        self.name, self.description, self.input_schema, self.annotations, self.run = name, description, input_schema, annotations, run


def _shapes() -> list[AgentTool]:
    """The agent tools' names, texts and schemas, from a memory that is never searched: listing needs no open memory."""

    class Never:
        async def search(self, q: Any) -> Any:
            raise RuntimeError("not a memory")

        async def open(self, id: str) -> Any:
            raise RuntimeError("not a memory")

        async def explore(self, q: Any) -> Any:
            raise RuntimeError("not a memory")

    return build_agent_tools(Never())


def _definitions(read_only: bool) -> list[_Definition]:
    def live(name: str) -> Callable[[Memory, Any], Awaitable[ToolResult]]:
        async def run(m: Memory, args: Any) -> ToolResult:
            tool = next((t for t in m.tools() if t.name == name), None)
            if tool is None:
                return _text(f"Unknown tool: {name}", True)
            return await tool.handler(args)

        return run

    defs = [_Definition(t.name, t.description, t.json_schema, _READ_ONLY, live(t.name)) for t in _shapes()]
    if not read_only:

        async def save(m: Memory, args: Any) -> ToolResult:
            got, why = _validate(_SAVE_SCHEMA, args)
            if got is None:
                return _text(f"Invalid arguments: {why}", True)
            now = datetime.now(timezone.utc)
            stamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
            uid = f"{stamp}-{secrets.token_hex(4)}"
            record: dict[str, Any] = {"id": uid, "source": "mcp", "kind": "note", "text": got["text"], "at": stamp}
            if got.get("title") is not None:
                record["title"] = got["title"]
            await m.add(record)
            return _text(f"Saved ({stamp[:10]}, id {uid}).")

        defs.append(_Definition("memory_save", _SAVE_DESCRIPTION, _SAVE_SCHEMA, _SAVE_ANNOTATIONS, save))
    return defs


async def _stdin_line() -> bytes:
    return await asyncio.to_thread(sys.stdin.buffer.readline)


def _stdout_write(data: bytes) -> None:
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


async def serve_mcp(
    open: Callable[[], Awaitable[Memory]],
    *,
    version: str,
    read_only: bool = False,
    idle_s: float = 300.0,
    read_line: Callable[[], Awaitable[bytes]] | None = None,
    write: Callable[[bytes], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """Serves until the input ends (``read_line`` gives ``b""``), then closes the memory. ``open`` opens it: at the first
    tool call, and again after an idle close (``idle_s`` 0 keeps it open)."""
    read_line = read_line or _stdin_line
    write = write or _stdout_write
    say = log or (lambda line: print(line, file=sys.stderr, flush=True))
    memory: asyncio.Task[Memory] | None = None
    busy = 0
    idle: asyncio.TimerHandle | None = None
    closing: set[asyncio.Task[None]] = set()

    def send(message: Any) -> None:
        write((json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))

    async def close_memory() -> None:
        nonlocal memory
        m, memory = memory, None
        if m is None:
            return
        try:
            opened = await m
        except Exception:  # noqa: BLE001 - it never opened: nothing to close
            return
        await opened.close()

    def idle_close() -> None:
        if busy == 0:
            task = asyncio.ensure_future(close_memory())
            closing.add(task)
            task.add_done_callback(closing.discard)

    def touch() -> None:
        nonlocal idle
        if idle is not None:
            idle.cancel()
        idle = None
        if idle_s > 0 and busy == 0 and memory is not None:
            idle = asyncio.get_running_loop().call_later(idle_s, idle_close)

    async def with_memory(f: Callable[[Memory], Awaitable[ToolResult]]) -> ToolResult:
        nonlocal busy, memory, idle
        busy += 1
        if idle is not None:
            idle.cancel()
            idle = None
        try:
            if memory is None:
                memory = asyncio.ensure_future(open())
            task = memory
            try:
                m = await task
            except BaseException:
                if memory is task:
                    memory = None  # the next call tries again
                raise
            return await f(m)
        finally:
            busy -= 1
            touch()

    definitions = _definitions(read_only)

    async def call(name: str, args: Any) -> ToolResult:
        d = next((x for x in definitions if x.name == name), None)
        if d is None:
            return _text(f"Unknown tool: {name}", True)
        try:
            return await with_memory(lambda m: d.run(m, args))
        except Exception as e:  # noqa: BLE001 - every failure is the tool's answer
            return _text(str(e), True)

    async def handle(r: Any) -> Any:
        r = r if isinstance(r, dict) else {}
        is_call = "id" in r
        method = r.get("method")
        if r.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return {"jsonrpc": "2.0", "id": r.get("id"), "error": {"code": -32600, "message": "Invalid request"}} if is_call else None
        params = r.get("params") if isinstance(r.get("params"), dict) else {}

        def ok(result: Any) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": r["id"], "result": result}

        if method == "initialize":
            asked = params.get("protocolVersion")
            chosen = asked if isinstance(asked, str) and asked in MCP_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSIONS[0]
            return ok(
                {
                    "protocolVersion": chosen,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "memtrails", "version": version},
                    "instructions": _INSTRUCTIONS,
                }
            )
        if method == "ping":
            return ok({}) if is_call else None
        if method == "tools/list":
            return ok(
                {
                    "tools": [
                        {"name": d.name, "description": d.description, "inputSchema": d.input_schema, "annotations": d.annotations}
                        for d in definitions
                    ]
                }
            )
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                return {"jsonrpc": "2.0", "id": r["id"], "error": {"code": -32602, "message": "tools/call needs a tool name"}}
            args = params.get("arguments")
            return ok(await call(name, {} if args is None else args))
        # Notifications (initialized, cancelled, …) need no answer; an unknown request does.
        return {"jsonrpc": "2.0", "id": r["id"], "error": {"code": -32601, "message": f"Method not found: {method}"}} if is_call else None

    async def answer(message: Any) -> None:
        try:
            if isinstance(message, list):
                answers = [a for a in await asyncio.gather(*(handle(m) for m in message)) if a is not None]
                if answers:
                    send(answers)
            else:
                a = await handle(message)
                if a is not None:
                    send(a)
        except Exception as e:  # noqa: BLE001 - one message's failure is not the server's
            say(f"[memtrails] mcp: {e}")

    pending: set[asyncio.Task[None]] = set()
    while True:
        line = await read_line()
        if not line:
            break
        text = line.decode("utf-8", "replace").strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except ValueError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue
        # Answered as each finishes: a slow search does not hold up a ping.
        task = asyncio.ensure_future(answer(message))
        pending.add(task)
        task.add_done_callback(pending.discard)
    if pending:
        await asyncio.gather(*pending)
    if idle is not None:
        idle.cancel()
    if closing:
        await asyncio.gather(*closing)
    await close_memory()
