"""
Gathering: a SET of memories for a question, not a top-k.

A question that counts, lists, totals or compares ("how many plumbers did we call", "what did I spend on camera gear")
needs every instance, and one ranked below the cut is a wrong answer however good the rest are. Here the fast model
reads them:
  1. a wide search (memory_search at its limit: up to 200 observations and 200 other memories);
  2. the fast model reads every candidate, in parts at once, and keeps each one the answer needs — each instance of
     what is counted, what changed or cancelled one, the dates — and names what it did NOT see that could also count;
  3. those searches run, and what they find that is new is read the same way.
The kept set comes first, in rank order; the rest follows by rank (a reader still has them). Nothing is dropped: a part
the model did not answer keeps all of its candidates.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, TypedDict

import regex

from .._js import cut, iso_day, js_trim
from ..llm import Ask, LlmRequest, ask_with, default_ask
from .profiles import PROFILE_SHOWN
from .recall import MAX_SEARCH, RecallDeps, RecalledItem, memory_search
from .redact import redact_string

#: Candidates the model reads in one ask.
_PART = 40
#: With notes, all of them in one ask, so a question's notes are written once: on LoCoMo the same accuracy as two parts
#: of 40 (80.9 against 80.6), at 58% of the cost and 2.9 s against 7.5 s median.
_NOTES_PART = 80
#: Searches it may ask for.
MORE = 6

SYSTEM = f"""You pick, from someone's memories, every one needed to answer their question.

Keep a memory when it states something the answer needs. When the question counts, lists, totals, compares or orders
things, keep EVERY memory that is one instance of what is asked about — each purchase, visit, trip, item, person or
event — even one mentioned in passing inside another topic, and every memory that changes an instance (returned,
cancelled, sold, replaced, ended, a new amount) or dates it. When unsure, keep it: a missing instance makes the answer
wrong, an extra one does not.

In "searches", write up to {MORE} short searches for instances the question could count that none of these memories
shows — other kinds, other words for the same thing (for "camera gear": "lens", "tripod", "memory card") —
in the language of the question. Leave it empty when nothing is missing. Never translate."""
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["keep", "searches"],
    "properties": {"keep": {"type": "array", "items": {"type": "string"}}, "searches": {"type": "array", "items": {"type": "string"}}},
}

#: With notes: the same reading also writes down what the kept memories say that answers the question, so a list spread
#: over many memories reaches the reader as one line.
NOTES_RULE = """

In "notes", write what the memories you kept say that answers the question, in a few short lines: every instance with
its date as the memories give it ("Concerts Lena went to: a jazz trio at the Blue Room (2024-03-02), the orchestra's Mahler night (2024-04-19)"),
names, numbers and words as written. Only what they state, as plain statements: never write that something is not said,
not given, unclear or unknown — leave it out instead, since other memories may hold it — and never a guess. Leave notes
empty when the memories you kept say nothing about it. In the language of the question."""
_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["keep", "searches", "notes"],
    "properties": {**_SCHEMA["properties"], "notes": {"type": "array", "items": {"type": "string"}}},
}


def _day(ts: float) -> str:
    return iso_day(ts) if ts else ""


def _line(h: Mapping[str, Any], n: int) -> str:
    """One candidate as the model reads it: when it happened (from when to when, for what lasted) and when it was told,
    its text, and the passage that matched."""
    happened = " ".join(
        x for x in (h.get("eventAt"), h.get("startAt") and f"from {h['startAt']}", h.get("endAt") and f"to {h['endAt']}") if x
    )
    when = f"{happened}; told {_day(h['ts'])}" if happened else _day(h["ts"])
    text = (
        h["summary"]
        if h["kind"] == "observation"
        else f"{h['title']} — {cut(h['summary'], PROFILE_SHOWN if h['kind'] == 'profile' else 600)}"
    )
    match = f"\n    … {cut(h['match'], 600)}" if h.get("match") else ""
    return f"[m{n}] ({when}) {text}{match}"


_LABEL = regex.compile(r"\bm([0-9]+)", regex.ASCII)


async def _pick(ask: Ask, question: str, found: list[RecalledItem], notes: bool = False) -> dict[str, Any]:
    size = _NOTES_PART if notes else _PART
    parts = [found[i : i + size] for i in range(0, len(found), size)]

    async def read(part: list[RecalledItem]) -> dict[str, Any]:
        prompt = f"Question: {question}\n\nMemories:\n" + "\n".join(_line(h, n + 1) for n, h in enumerate(part))
        a = await ask_with(
            ask,
            LlmRequest(
                purpose="memory",
                system=SYSTEM + NOTES_RULE if notes else SYSTEM,
                prompt=redact_string(prompt),
                schema=_NOTES_SCHEMA if notes else _SCHEMA,
                name="memory_gather",
                max_tokens=40 * len(part) + (1400 if notes else 600),
                timeout_ms=120_000,
                consequence="so these memories are kept unread",
            ),
        )
        if "answer" not in a:
            return {
                "keep": [h["id"] for h in part],
                "searches": [],
                "notes": [],
                "paused": a["unavailable"] if "unavailable" in a else "the fast model is busy",
            }
        got = a["answer"] if isinstance(a["answer"], dict) else {}
        ids = [x for x in got.get("keep") or [] if isinstance(x, str)] if isinstance(got.get("keep"), list) else []
        # A small model can quote the line it means ("[m3] (2023-06-03) …"), so the label is looked for, not assumed.
        keep = []
        for x in ids:
            m = _LABEL.search(x)
            n = int(m.group(1)) if m else 0
            if 1 <= n <= len(part):
                keep.append(part[n - 1]["id"])
        raw_searches = got.get("searches")
        searches = [x for x in raw_searches if isinstance(x, str) and js_trim(x)] if isinstance(raw_searches, list) else []
        raw_notes = got.get("notes")
        said = (
            [redact_string(js_trim(x)) for x in raw_notes if isinstance(x, str) and js_trim(x)]
            if notes and isinstance(raw_notes, list)
            else []
        )
        return {"keep": keep, "searches": searches, "notes": said, "paused": None}

    answers = await asyncio.gather(*(read(p) for p in parts))
    paused = next((a["paused"] for a in answers if a["paused"] is not None), None)
    out: dict[str, Any] = {
        "keep": {k for a in answers for k in a["keep"]},
        "searches": list(dict.fromkeys(js_trim(s) for a in answers for s in a["searches"])),
        "notes": [n for a in answers for n in a["notes"]],
    }
    if paused is not None:
        out["paused"] = paused
    return out


class Gathered(TypedDict, total=False):
    #: What the answer needs, best first, then the rest of what was found, by rank.
    found: list[RecalledItem]
    #: The ids kept.
    kept: list[str]
    #: The searches the model asked for.
    searches: list[str]
    #: With ``notes``: what the kept memories say that answers the question, a few short lines (empty otherwise). Show
    #: them before the memories, as notes on them: on LoCoMo this took 79.5 to 80.9 (docs/BENCHMARKS.md).
    notes: list[str]
    #: Set when the model did not read some candidates (none configured, unreachable, busy): why. They are all kept, by
    #: rank — the result is then a wide search, not a gathered set.
    paused: str


async def memory_gather(
    d: RecallDeps, q: Mapping[str, Any], ask: Ask | None = None, limit: int | None = None, notes: bool = False
) -> Gathered:
    """``from``/``to``: the time the question asks about. What happened then is searched for besides everything else
    and read first — not instead: a question also names dates it only counts from ("as of November 2023"), and a date
    filter alone would hide the event it counts."""
    asker = ask or default_ask
    lim = limit if limit is not None else MAX_SEARCH
    open_q = {k: v for k, v in q.items() if k not in ("from", "to")}

    async def nothing() -> list[RecalledItem]:
        return []

    in_dates, wide = await asyncio.gather(
        memory_search(d, {**q, "limit": lim}) if q.get("from") or q.get("to") else nothing(),
        memory_search(d, {**open_q, "limit": lim}),
    )
    dated = {h["id"] for h in in_dates}
    first = [*in_dates, *(h for h in wide if h["id"] not in dated)]
    one = await _pick(asker, q["query"], first, notes)
    searches = one["searches"][:MORE]
    have = {h["id"] for h in first}
    more: list[RecalledItem] = []
    for found in await asyncio.gather(*(memory_search(d, {**open_q, "query": s, "limit": 30}) for s in searches)):
        for h in found:
            if h["id"] not in have:
                have.add(h["id"])
                more.append(h)
    two = await _pick(asker, q["query"], more, notes) if more else {"keep": set(), "searches": [], "notes": []}
    every = [*first, *more]
    kept = [h for h in every if h["id"] in one["keep"] or h["id"] in two["keep"]]
    kept_ids = {h["id"] for h in kept}
    out: Gathered = {
        "found": [*kept, *(h for h in every if h["id"] not in kept_ids)],
        "kept": [h["id"] for h in kept],
        "searches": searches,
        "notes": [*one["notes"], *two["notes"]],
    }
    paused = one.get("paused") if one.get("paused") is not None else two.get("paused")
    if paused is not None:
        out["paused"] = paused
    return out
