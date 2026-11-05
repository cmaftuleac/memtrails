"""
Profiles — what memory knows about one subject, gathered in one place (sleep's last step).

A question about a subject whose facts were told over many conversations needs the dated facts of all of them, and a
top 10 holds a few. For every concept enough long-term items share — at least PROFILE_MIN_ITEMS of a project's items,
and on at most PROFILE_BROAD of them (not the person themselves, not the project's own name) — the fast model gathers
what they say into one dated list of facts, word for word: a ``profile`` item, keyed and searched like any other, whose
provenance is every source it read.

Measured on all of LoCoMo (docs/BENCHMARKS.md): the answer fully shown in the search's top 10 59.1% → 61.7%, dates
66.6% → 73.1%, shown whole.

  - written again only when what it read changed (a hash of those items is its fingerprint), never from a profile;
  - a subject that no longer qualifies has its profile archived;
  - never related, merged or superseded by sleep or consolidation: it is written again from its items.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .._aio import gather_or_cancel
from .._js import cut, day_month_year, flatten_ws, js_trim, locale_compare
from ..llm import LlmRequest, ask_with, default_ask
from .consolidate import stamped
from .redact import redact_string
from .text import concept_key
from .writer import write_ltm

PROFILE_MIN_ITEMS = 3
PROFILE_BROAD = 0.2
#: The items a profile reads: the newest this many, oldest first, each line cut to PROFILE_LINE.
PROFILE_READ = 40
_PROFILE_LINE = 300
#: Profiles a night writes (a large first sleep goes on the next nights), and asks in flight.
PROFILES_PER_SLEEP = 400
_PROFILE_IN_FLIGHT = 8
#: What memory_search and gathering show of a profile: all of it, up to this (measured p90 1,082 characters).
PROFILE_SHOWN = 1500

PROFILE_SYSTEM = """You write a profile: everything a person's memories say about one subject, gathered in one place, so that a
later question about the subject finds all of it at once. You get the subject and the memories that mention it, each
with its date. Write:
- title: the subject and what the profile covers, at most 12 words
- facts: one line per distinct fact about the subject, oldest first, each starting with the date it happened or was
  said, as given. Keep every name, number, place, title and quote exactly as written; say who did what. Merge
  repeats into one line, keep facts that differ apart. Nothing that is not in the memories. Never include secret
  values. Never translate."""
_PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "facts"],
    "properties": {"title": {"type": "string"}, "facts": {"type": "array", "items": {"type": "string"}}},
}


def _when(it: Mapping[str, Any]) -> str:
    return str(it.get("eventAt") or day_month_year(it["ts"]))


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def profile_subjects(items: Sequence[Mapping[str, Any]], sizes: Mapping[str, int]) -> list[dict[str, Any]]:
    """The subjects worth a profile, per project: each concept (as concept_key folds it, named as most items spell it)
    on at least PROFILE_MIN_ITEMS items and on at most PROFILE_BROAD of the project's ``sizes`` (its active items). A
    subject: uid, concept, scope, project, read (the items it reads), fingerprint."""
    on: dict[str, dict[str, Any]] = {}
    for it in items:
        if it.get("status") != "active" or it.get("kind") == "profile":
            continue
        named: dict[str, str] = {}
        for c in it.get("concepts") or []:
            key = concept_key(c)
            if key and key not in named:
                named[key] = js_trim(c)
        for key, name in named.items():
            sid = f"{it['project']}\n{key}"
            s = on.setdefault(sid, {"project": it["project"], "key": key, "names": {}, "items": []})
            s["names"][name] = s["names"].get(name, 0) + 1
            s["items"].append(it)
    out: list[dict[str, Any]] = []

    def newest_first(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
        return (b["ts"] > a["ts"]) - (b["ts"] < a["ts"]) or locale_compare(a["uid"], b["uid"])

    def most_spelled(a: tuple[str, int], b: tuple[str, int]) -> int:
        return b[1] - a[1] or locale_compare(a[0], b[0])

    for sid, s in on.items():
        if len(s["items"]) < PROFILE_MIN_ITEMS or len(s["items"]) > PROFILE_BROAD * sizes.get(s["project"], 0):
            continue
        read = list(reversed(sorted(s["items"], key=functools.cmp_to_key(newest_first))[:PROFILE_READ]))
        scopes: dict[str, int] = {}
        for it in read:
            scopes[it["scope"]] = scopes.get(it["scope"], 0) + 1
        out.append(
            {
                "uid": f"pf:{_hash(sid)[:24]}",
                "concept": sorted(s["names"].items(), key=functools.cmp_to_key(most_spelled))[0][0],
                "scope": sorted(scopes.items(), key=lambda kv: -kv[1])[0][0],
                "project": s["project"],
                "read": read,
                "fingerprint": _hash(
                    "\n\n".join([s["key"], *(f"{it['uid']}\n{_when(it)}\n{it['title']}\n{it['summary']}" for it in read)])
                ),
            }
        )
    return out


def _prompt_of(s: Mapping[str, Any]) -> str:
    lines = "\n".join(f"- ({_when(it)}) {it['title']}: {cut(flatten_ws(it['summary']), _PROFILE_LINE)}" for it in s["read"])
    return f"SUBJECT: {s['concept']}\n\nMEMORIES:\n{lines}"


def _profile_item(s: Mapping[str, Any], title: str, facts: list[str], now: float) -> dict[str, Any]:
    newest = s["read"][-1]
    return stamped(
        {
            "uid": s["uid"],
            "kind": "profile",
            "scope": s["scope"],
            "project": s["project"],
            "engine": newest["engine"],
            "session": None,
            "cwd": newest.get("cwd"),
            "where": s["concept"],
            "ts": max(it["ts"] for it in s["read"]),
            "firstTs": min(it.get("firstTs") or it["ts"] for it in s["read"]),
            "title": cut(js_trim(title) or s["concept"], 120),
            "summary": "\n".join(facts),
            "body": "",
            "files": [],
            "appliesWhen": "",
            "confidence": "curated",
            "importance": 3,
            "status": "active",
            "provenance": list(dict.fromkeys(p for it in s["read"] for p in it["provenance"])),
            "seen": 1,
            "fingerprint": s["fingerprint"],
            "concepts": [s["concept"]],
        },
        now,
    )


async def write_profiles(d: Any, max_profiles: int = PROFILES_PER_SLEEP) -> dict[str, Any]:
    """Profiles written for the subjects whose items changed (at most ``max_profiles``, several asks at once), and
    archived for the subjects that no longer qualify. An ask the model did not answer stops the night's profiles; the
    next night asks again, as their fingerprints still differ. ``d``: consolidate.py ConsolidateDeps."""
    r: dict[str, Any] = {"profiled": 0, "unprofiled": 0}
    sizes: dict[str, int] = {}
    mapped: list[dict[str, Any]] = []
    for uid, m in d.ltm.entries():
        if m.get("status") != "active" or m.get("kind") == "profile":
            continue
        sizes[m["project"]] = sizes.get(m["project"], 0) + 1
        it = d.ltm.get(uid) if m.get("mapped") else None
        if it:
            mapped.append(it)
    subjects = profile_subjects(mapped, sizes)
    wanted = {s["uid"] for s in subjects}
    gone = [
        {**it, "status": "archived"}
        for it in (
            d.ltm.get(uid) for uid, m in d.ltm.entries() if m.get("kind") == "profile" and m.get("status") == "active" and uid not in wanted
        )
        if it
    ]
    if gone:
        await write_ltm(d, gone)
        d.ltm.put(gone)
        r["unprofiled"] = len(gone)

    def changed(s: Mapping[str, Any]) -> bool:
        m = d.ltm.meta(s["uid"])
        return not m or m.get("status") != "active" or m.get("fp") != s["fingerprint"]

    todo = [s for s in subjects if changed(s)][:max_profiles]
    queue = list(todo)

    async def lane() -> None:
        while "paused" not in r and queue and not d.aborted():
            s = queue.pop(0)
            a = await ask_with(
                d.ask or default_ask,
                LlmRequest(
                    purpose="memory",
                    system=PROFILE_SYSTEM,
                    prompt=redact_string(_prompt_of(s)),
                    schema=_PROFILE_SCHEMA,
                    name="memory_profile",
                    max_tokens=3000,
                    timeout_ms=180_000,
                    consequence="so this subject's profile waits for the next night",
                ),
            )
            if "answer" not in a:
                r["paused"] = a["unavailable"] if "unavailable" in a else "busy"
                return
            answer = a["answer"] if isinstance(a["answer"], dict) else {}
            raw: list[Any] = answer["facts"] if isinstance(answer.get("facts"), list) else []
            facts = [js_trim(f) for f in raw if isinstance(f, str) and js_trim(f)]
            if not facts:
                continue
            title = answer["title"] if isinstance(answer.get("title"), str) else ""
            it = _profile_item(s, title, facts, d.time())
            await write_ltm(d, [it])
            d.ltm.put([it])
            r["profiled"] += 1

    await gather_or_cancel(*(lane() for _ in range(min(_PROFILE_IN_FLIGHT, len(todo)))))
    return r
