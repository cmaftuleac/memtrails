"""
An agent's tools over a memory: ``memory_search`` (find), ``memory_open`` (read one whole, with the records it was
learned from) and ``memory_explore`` (from a topic or items to what is related but not found by search). Each is a plain
dict-shaped tool — a name, a description, the input's JSON Schema (the reference's, key for key) and a handler — as an
MCP server or any function-calling API declares tools:

    for t in memory.tools():
        server.add_tool(t.handler, name=t.name, description=t.description)   # an async handler; or t.json_schema to a model

The handler checks its arguments itself (a caller doing function calling hands the model's arguments straight to it)
and always returns (awaited) ``{"content": [{"type": "text", "text": …}], "isError"?: True}``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import regex

from .._js import cut, date_utc, flatten_ws, iso_day, u16len, u16slice, utc_weekday
from ..core.profiles import PROFILE_SHOWN
from ..core.writer import date_span

ToolResult = dict[str, Any]


@dataclass
class AgentTool:
    name: str
    description: str
    #: The input as JSON Schema, for function calling.
    json_schema: dict[str, Any]
    #: Takes the arguments (a dict) and returns the tool's result: ``await tool.handler({...})``.
    handler: Callable[[Any], Awaitable[ToolResult]]


class ToolProvider(Protocol):
    """What the tools read: a memory (open_memory) gives these."""

    async def search(self, q: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...

    async def open(self, id: str) -> Mapping[str, Any] | None: ...


_DESCRIPTIONS = {
    "memory_search": "Search memory: everything it was given (e-mails, messages, notes, documents, conversations, commits, …) and "
    "what was learned from them — facts, decisions, dated events, profiles of people and things. Best first; each result "
    "with its date and id. Ask in plain words, as you would ask a colleague who read it all. Name dates when the question "
    "names a time. Open a result with memory_open for all of it.",
    "memory_open": "Read one memory item whole, by the id memory_search or memory_explore gave: everything it holds, when it "
    "happened, and for something learned, the records it was learned from.",
    "memory_explore": "Explore memory around a topic, or around items you already found: it follows what they share — the same "
    "people, places, systems, names — and links between memories, to what a search for the words does not find. Returns "
    "what search found, what is related, and the concepts around it to explore next.",
}

#: What a model may send: a question, a topic or an id is a line, not a document; a few dozen collections at most.
TEXT_MAX = 2000
COLLECTIONS_MAX = 50
_DATE_PATTERN = r"^\d{4}(-\d{2}(-\d{2})?)?$"
_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _day(ts: float) -> str:
    return iso_day(ts) if ts else ""


def _hit_line(h: Mapping[str, Any], n: int) -> str:
    kind = "earlier work" if h.get("kind") == "episode" else h.get("kind", "")
    author = (h.get("record") or {}).get("author")
    return f"{n + 1}. {h.get('title', '')} ({kind} · {_day(h.get('ts', 0))}{f' · {author}' if author else ''} · id {h['id']})"


_WEEKDAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
_MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def readable_date(d: str) -> str:
    """A date as a reader reckons with it — the weekday too ("the Friday before …"): 2023-07-14 → "Fri 14 Jul 2023"."""
    # Not a date (2023-13, 2023-02-30): shown as written rather than as "undefined 2023" or a day that does not exist.
    if not date_span(d):
        return d
    parts = [int(x) for x in d.split("-")]
    y = parts[0]
    m = parts[1] if len(parts) > 1 else 0
    dd = parts[2] if len(parts) > 2 else 0
    if not y:
        return ""
    if not m:
        return str(y)
    if not dd:
        return f"{_MONTH[m - 1]} {y}"
    return f"{_WEEKDAY[utc_weekday(date_utc(y, m - 1, dd))]} {dd} {_MONTH[m - 1]} {y}"


def lasted_text(h: Mapping[str, Any]) -> str:
    """How long something lasted, as a reader reckons with it: "since Feb 2022", "Nov 2021 – May 2022"."""
    start, end = (readable_date(v) if isinstance(v, str) and v else "" for v in (h.get("startAt"), h.get("endAt")))
    if start and end:
        return f"{start} – {end}"
    return f"since {start}" if start else f"until {end}" if end else ""


def _when(h: Mapping[str, Any]) -> str:
    """When a hit happened, for ordering by time: an observation's own date, else when it was learned."""
    return str(h.get("eventAt") or _day(h.get("ts", 0)))


def search_text(hits: Sequence[Mapping[str, Any]], order: str = "relevance") -> str:
    """What memory_search shows: an observation as one dated line (when it happened, and when it was told when that
    differs), a profile whole (up to PROFILE_SHOWN), anything else as its title line and summary — shorter when many
    are asked for. Best first, or by time."""
    listed = sorted(hits, key=_when) if order == "time" else list(hits)
    # Many hits: one line each — the date, what it is, its summary — so a list question sees every one of them clearly.
    compact = len(listed) > 20
    lines = []
    for n, h in enumerate(listed):
        summary = str(h.get("summary", ""))
        kind = h.get("kind")
        if kind == "profile":
            lines.append(f"{_hit_line(h, n)}\n   {cut(summary, PROFILE_SHOWN).replace(chr(10), chr(10) + '   ')}")
            continue
        if kind != "observation":
            # The passage that matched, when it was not the start: a fact said deep in a long request is what was found.
            match = h.get("match")
            matched = f" … {flatten_ws(match)}" if match else ""
            if not compact:
                lines.append(f"{_hit_line(h, n)}\n   {u16slice(summary, 0, 200 if match else 500)}{matched}")
                continue
            what = "earlier work" if kind == "episode" else kind
            lines.append(
                f"{n + 1}. ({readable_date(_day(h.get('ts', 0)))} · {what}) "
                f"{u16slice(flatten_ws(summary), 0, 150 if match else 300)}{matched} (id {h['id']})"
            )
            continue
        told = _day(h.get("ts", 0))
        lasted = lasted_text(h)
        event = h.get("eventAt")
        happened = ", ".join(x for x in (readable_date(event) if event else "", lasted) if x)
        at = (
            f"{happened}{f'; told {readable_date(told)}' if event != told or lasted else ''}" if happened else f"told {readable_date(told)}"
        )
        lines.append(f"{n + 1}. ({at}) {summary} (id {h['id']})")
    head = "Oldest first — the date is when it happened (and when it was told, when that differs).\n" if order == "time" else ""
    return head + "\n".join(lines)


def _text(t: str, is_error: bool = False) -> ToolResult:
    r: ToolResult = {"content": [{"type": "text", "text": t}]}
    if is_error:
        r["isError"] = True
    return r


# --- the input schemas, and checking against them (zod's rules and words) ------------------------------------------


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (list, tuple)):
        return "array"
    return "object"


def _check(schema: Mapping[str, Any], v: Any, path: str, issues: list[str]) -> None:
    where = path or "input"
    t = schema.get("type")
    if t == "string":
        if not isinstance(v, str):
            issues.append(f"{where}: Invalid input: expected string, received {_type_name(v)}")
            return
        if "enum" in schema and v not in schema["enum"]:
            options = "|".join(f'"{x}"' for x in schema["enum"])
            issues.append(f"{where}: Invalid option: expected one of {options}")
        if "maxLength" in schema and u16len(v) > schema["maxLength"]:
            issues.append(f"{where}: Too big: expected string to have <={schema['maxLength']} characters")
        if "pattern" in schema and not regex.search(schema["pattern"], v, regex.ASCII):
            issues.append(f"{where}: Invalid string: must match pattern /{schema['pattern']}/")
    elif t == "integer":
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            issues.append(f"{where}: Invalid input: expected number, received {_type_name(v)}")
            return
        if not float(v).is_integer():
            issues.append(f"{where}: Invalid input: expected int, received number")
            return
        if v < schema["minimum"]:
            issues.append(f"{where}: Too small: expected number to be >={schema['minimum']}")
        if v > schema["maximum"]:
            issues.append(f"{where}: Too big: expected number to be <={schema['maximum']}")
    elif t == "array":
        if not isinstance(v, (list, tuple)):
            issues.append(f"{where}: Invalid input: expected array, received {_type_name(v)}")
            return
        for i, x in enumerate(v):
            _check(schema["items"], x, f"{path}.{i}" if path else str(i), issues)
        if len(v) > schema["maxItems"]:
            issues.append(f"{where}: Too big: expected array to have <={schema['maxItems']} items")


def _validate(schema: Mapping[str, Any], args: Any) -> tuple[dict[str, Any] | None, str]:
    """The arguments the handler may use, or why they are refused. Unknown keys are left out, as zod leaves them."""
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        return None, f"input: Invalid input: expected object, received {_type_name(args)}"
    issues: list[str] = []
    got: dict[str, Any] = {}
    for key, prop in schema["properties"].items():
        if key not in args or (args[key] is None and key not in schema.get("required", [])):
            if key in schema.get("required", []):
                issues.append(f"{key}: Invalid input: expected {prop['type']}, received undefined")
            continue
        _check(prop, args[key], key, issues)
        got[key] = args[key]
    return (None, "; ".join(issues)) if issues else (got, "")


def _tool(name: str, schema: dict[str, Any], run: Callable[[dict[str, Any]], Awaitable[ToolResult]]) -> AgentTool:
    async def handler(args: Any) -> ToolResult:
        got, why = _validate(schema, args)
        if got is None:
            return _text(f"Invalid arguments: {why}", True)
        return await run(got)

    return AgentTool(name=name, description=_DESCRIPTIONS[name], json_schema=schema, handler=handler)


def _string(
    description: str | None = None, pattern: str | None = None, max_length: int | None = TEXT_MAX, described_first: bool = True
) -> dict[str, Any]:
    s: dict[str, Any] = {}
    if description and described_first:
        s["description"] = description
    s["type"] = "string"
    if max_length is not None:
        s["maxLength"] = max_length
    if pattern:
        s["pattern"] = pattern
    if description and not described_first:
        s["description"] = description
    return s


def _strings(description: str, max_items: int) -> dict[str, Any]:
    return {
        "description": description,
        "maxItems": max_items,
        "type": "array",
        "items": {"type": "string", "maxLength": TEXT_MAX},
    }


SEARCH_SCHEMA: dict[str, Any] = {
    "$schema": _SCHEMA,
    "type": "object",
    "properties": {
        "query": _string("What you need to know, in plain words.", described_first=False),
        "collections": _strings("Only these collections (default: all).", COLLECTIONS_MAX),
        "limit": {
            "description": "How many of each to return — dated observations, and other memories (default 20; 100 to 200 "
            "when you gather everything on a topic, list or count things, or need their dates).",
            "type": "integer",
            "minimum": 1,
            "maximum": 200,
        },
        "order": {
            "description": "Best first (default), or oldest first — for what happened when, and in what order.",
            "type": "string",
            "enum": ["relevance", "time"],
        },
        "from": _string(
            "Only what happened on or after this date (YYYY-MM-DD, YYYY-MM or YYYY) — when the request names a "
            'time ("in June 2023", "last week").',
            _DATE_PATTERN,
            None,
        ),
        "to": _string("Only what happened on or before this date (YYYY-MM-DD, YYYY-MM or YYYY).", _DATE_PATTERN, None),
    },
    "required": ["query"],
    "additionalProperties": False,
}
OPEN_SCHEMA: dict[str, Any] = {
    "$schema": _SCHEMA,
    "type": "object",
    "properties": {"id": _string("The memory id, as memory_search or memory_explore gave it.", described_first=False)},
    "required": ["id"],
    "additionalProperties": False,
}
EXPLORE_SCHEMA: dict[str, Any] = {
    "$schema": _SCHEMA,
    "type": "object",
    "properties": {
        "topic": _string("What to explore, in plain words — a system, a project, a problem, a name."),
        "from": _strings("Memory ids to explore from, as memory_search or memory_explore gave them.", 5),
        "collections": _strings("Only these collections (default: all).", COLLECTIONS_MAX),
        "limit": {
            "description": "How many related items to return (default 10).",
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
        },
    },
    "additionalProperties": False,
}


def build_agent_tools(p: ToolProvider) -> list[AgentTool]:
    """memory_search, memory_open and memory_explore (when the provider explores) over a memory."""

    async def search(a: dict[str, Any]) -> ToolResult:
        try:
            q = {
                "query": a["query"],
                "collections": a.get("collections"),
                "limit": a.get("limit") or 20,
                "from": a.get("from"),
                "to": a.get("to"),
            }
            hits = await p.search({k: v for k, v in q.items() if v is not None})
            if not hits:
                return _text("Nothing in memory matches that.")
            return _text(
                f"{search_text(hits, a.get('order') or 'relevance')}\n\nOpen one with memory_open and its id. It can "
                "be stale: check it against the current state before relying on it."
            )
        except Exception as err:  # noqa: BLE001 - the agent is told, not the host
            return _text(str(err), True)

    async def open_(a: dict[str, Any]) -> ToolResult:
        # An index that failed is not an id that is missing: the agent should not conclude the memory is gone.
        try:
            it = await p.open(a["id"])
        except Exception as err:  # noqa: BLE001
            return _text(f"Memory could not be read: {err}", True)
        if not it:
            return _text(f'There is no memory "{a["id"]}" (it may have been merged or cleared).', True)
        files = f"\nFiles: {', '.join(it['files'])}" if isinstance(it.get("files"), list) and it["files"] else ""
        told = _day(float(it.get("ts") or 0))
        event = it.get("eventAt")
        happened = ", ".join(x for x in (readable_date(event) if isinstance(event, str) and event else "", lasted_text(it)) if x)
        when = f"happened {happened}; told {readable_date(told)}" if happened else told
        sources = ""
        if isinstance(it.get("sources"), list) and it["sources"]:
            # The raw requests it was learned from, whole (the truth it is a key to).
            sources = "\n\n## Where it came from\n" + "\n\n".join(
                f"### {_day(x['when'])} · {x['title']} (id {x['id']})\n{x['body']}" for x in it["sources"]
            )
        return _text(
            f"# {_str(it.get('title'))}\n({_str(it.get('kind'))} · {when} · {_str(it.get('project'))})\n\n"
            f"{_str(it.get('summary'))}\n\n{_str(it.get('body'))}{files}{sources}"
        )

    tools = [_tool("memory_search", SEARCH_SCHEMA, search), _tool("memory_open", OPEN_SCHEMA, open_)]
    explore = getattr(p, "explore", None)
    if explore is not None:

        async def explore_(a: dict[str, Any]) -> ToolResult:
            topic = a.get("topic")
            if not (topic and topic.strip()) and not a.get("from"):
                return _text("Give a topic, or memory ids to explore from.", True)
            try:
                q = {"topic": topic, "from": a.get("from"), "collections": a.get("collections"), "limit": a.get("limit") or 10}
                r = await explore({k: v for k, v in q.items() if v is not None})
            except Exception as err:  # noqa: BLE001
                return _text(str(err), True)
            parts = []
            if r["found"]:
                parts.append("Found by search:\n" + "\n".join(_hit_line(h, n) for n, h in enumerate(r["found"])))
            if r["related"]:
                parts.append(
                    "Related — not found by search, reached through what they share:\n"
                    + "\n".join(
                        f"{_hit_line(h, n)} — {h['via']}\n   {u16slice(h.get('summary', ''), 0, 300)}" for n, h in enumerate(r["related"])
                    )
                )
            if r["concepts"]:
                parts.append(f"Concepts around it: {', '.join(r['concepts'])}")
            if not parts:
                return _text("Nothing in memory reaches that.")
            return _text(
                "\n\n".join(parts) + "\n\nOpen one with memory_open; explore further from an item (from: [its id]) or "
                "with one of the concepts as the topic. It can be stale: check it against the current state before "
                "relying on it."
            )

        tools.append(_tool("memory_explore", EXPLORE_SCHEMA, explore_))
    return tools


def _str(v: Any) -> str:
    """``String(v ?? '')``."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
