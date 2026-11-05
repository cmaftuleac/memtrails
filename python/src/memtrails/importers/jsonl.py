"""
JSON Lines: one JSON object per line, read a line at a time. With no ``map``, each line is a record as it is
(``{"id": "…", "text": "…", "title": …, "at": …}``); with one, ``map(obj, line_number)`` turns each object into a
record, or into None to leave it out. Lines that are not JSON or not a record are skipped, counted and reported through
``on_skip``; so is a line longer than 64 MiB, which is never held whole.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from typing import Any

from .._js import js_trim, num_str, parse
from ..records import invalid_record
from .text import LineTooLong, read_lines, record, undef

JsonlMap = Callable[[Any, int], "dict[str, Any] | None"]


async def from_jsonl(
    path: str | os.PathLike[str],
    map: JsonlMap | None = None,
    on_skip: Callable[[str, str, int], None] | None = None,
    stats: dict[str, int] | None = None,
    source: str | None = None,
    collection: str | None = None,
    kind: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Records from a JSON Lines file. ``stats``, when given, is filled in as the file is read: ``lines``, ``records``,
    ``skipped``. ``source``, ``collection``, ``kind``: for records that have none."""
    count = stats if stats is not None else {}
    for k in ("lines", "records", "skipped"):
        count.setdefault(k, 0)

    def skip(line: str, reason: str, n: int) -> None:
        count["skipped"] += 1
        if on_skip:
            on_skip(line, reason, n)

    n = 0
    async for raw in read_lines(os.fspath(path), "utf-8"):
        n += 1
        if isinstance(raw, LineTooLong):
            count["lines"] += 1
            skip("", raw.reason, n)
            continue
        line = raw.removeprefix("\ufeff") if n == 1 else raw
        if not js_trim(line):
            continue
        count["lines"] += 1
        try:
            obj = parse(line)
        except ValueError as err:
            skip(line, f"not JSON: {err}", n)
            continue
        except RecursionError:
            # Nested deeper than Python's parser goes (JavaScript's has no such limit): the line is skipped, said why,
            # and the file read on.
            skip(line, "not JSON: nested too deeply", n)
            continue
        if map is not None:
            try:
                r = map(obj, n)
            except Exception as err:  # noqa: BLE001 - a map that fails skips its line, said why
                skip(line, f"map failed: {err}", n)
                continue
            if not r:
                skip(line, "left out by map", n)
                continue
        else:
            if not isinstance(obj, dict):
                skip(line, "not an object", n)
                continue
            # String(id), as JavaScript writes a number: 1.0 is "1", 1e21 is "1e+21".
            r = (
                {**obj, "id": num_str(obj["id"])}
                if isinstance(obj.get("id"), (int, float)) and not isinstance(obj.get("id"), bool)
                else obj
            )
        filled = record(
            {
                **r,
                "source": r.get("source") if r.get("source") is not None else undef(source),
                "collection": r.get("collection") if r.get("collection") is not None else undef(collection),
                "kind": r.get("kind") if r.get("kind") is not None else undef(kind),
            }
        )
        wrong = invalid_record(filled)
        if wrong:
            skip(line, wrong, n)
            continue
        count["records"] += 1
        yield filled
