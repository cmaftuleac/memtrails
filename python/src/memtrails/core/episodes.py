"""
How a record reads: to the index (its body, every word of which is searched in chunks) and to the consolidation model.

A conversation's exchange keeps the format consolidation was measured on: what was asked, the steps, what was said.
Any other record reads as itself: its title, then its text.

What is read is a short-term entry (store.py StmEntry), or anything with its ``request``, ``reply``, ``said``,
``files``, ``ts`` and, for a record, ``engine`` (its kind).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .._js import date_utc, iso, iso_day, js_trim, utc_parts, utc_weekday, weekday_short

_DAY = 86_400_000


def is_conversation(e: Mapping[str, Any]) -> bool:
    return e.get("engine") == "conversation"


def _is_record(e: Mapping[str, Any]) -> bool:
    """A record that is not an exchange of a conversation (``e.engine !== undefined && !isConversation(e)``)."""
    return "engine" in e and not is_conversation(e)


def episode_body(e: Mapping[str, Any]) -> str:
    """A record's body: what its key summarises, its chunks are cut from, and opening it shows — whole, never
    shortened."""
    if _is_record(e):
        return f"{e['request']}\n{e['reply']}" if js_trim(e["reply"]) else e["request"]
    said = e.get("said")
    parts = [f"Asked: {e['request']}", f"Done: {said if said and js_trim(said) else e['reply']}"]
    if e.get("files"):
        parts.append(f"Files: {', '.join(e['files'])}")
    return "\n".join(parts)


def calendar_of(tss: Iterable[float]) -> str:
    """For each day a request was made, the days a time said relatively that day means — yesterday, the week before,
    the last weekend, each of the six days before by name, this week, last month, last year (UTC as the requests'
    times) — so the model reads "last week" or "on Friday" off a line instead of counting. "Last week" is the seven
    days before, as people mean it, not the Monday-to-Sunday week before this one: measured on LoCoMo, the calendar
    week put events said on a Sunday thirteen days back, a week early."""
    days = sorted({math.floor(t / _DAY) * _DAY for t in tss if t > 0})
    lines = []
    for d in days:
        wd = utc_weekday(d)
        monday = d - ((wd + 6) % 7) * _DAY
        sunday = d - (7 if wd == 0 else wd) * _DAY
        y, m, _ = utc_parts(d)
        last_month = iso(date_utc(y, m - 1, 1))[:7]
        before = ", ".join(f"{weekday_short(d - n * _DAY)} {iso_day(d - n * _DAY)}" for n in (2, 3, 4, 5, 6, 7))
        lines.append(
            f"- {weekday_short(d)} {iso_day(d)}: yesterday {weekday_short(d - _DAY)} {iso_day(d - _DAY)} · the days before "
            f"{before} · last week (the week before) {iso_day(d - 7 * _DAY)} to {iso_day(d - _DAY)} · last weekend "
            f"{iso_day(sunday - _DAY)} to {iso_day(sunday)} · this week {iso_day(monday)} to {iso_day(monday + 6 * _DAY)} "
            f"· last month {last_month} · last year {y - 1}"
        )
    return "\n".join(lines)


def render_for_extraction(e: Mapping[str, Any], n: int) -> str:
    ts = e.get("ts")
    when = iso(ts)[:16].replace("T", " ") if ts else ""
    # Whole: a record too long for one window is read in parts and its parts joined (extract.py), never shortened.
    if _is_record(e):
        return "\n".join(x for x in (f"## Record {n} · {when} · {e['engine']}", e["request"], e["reply"]) if x)
    parts = [f"## Request {n} · {when} · typed by the person", e["request"]]
    if e.get("said"):
        parts.append(f"Agent said:\n{e['said']}")
    if e.get("files"):
        parts.append(f"Files changed: {', '.join(e['files'])}")
    return "\n".join(parts)
