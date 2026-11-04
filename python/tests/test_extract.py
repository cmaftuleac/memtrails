"""Consolidation's reading of a conversation (extract.py; extract.test.ts, ported) — the gate is the rule in code."""

from __future__ import annotations

from typing import Any

from fakes import run
from memtrails._js import u16len
from memtrails.core.episodes import render_for_extraction
from memtrails.core.extract import EXTRACT_MAX_TOKENS, WINDOW_CHARS, clean_observations, extract_window, gate, parts_of, windows_of
from memtrails.llm import LlmRequest

KEY = "sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"


def req(uid: str, request: str, **extra: Any) -> dict[str, Any]:
    return {
        "uid": uid,
        "typed": "human",
        "request": request,
        "reply": "done",
        "said": "done",
        "steps": [],
        "files": [],
        "ts": 1_700_000_000_000,
        **extra,
    }


def item(**over: Any) -> dict[str, Any]:
    return {
        "kind": "procedure",
        "title": "Deploy the API",
        "summary": "Push to main; Jenkins deploys.",
        "body": "git push origin main",
        "applies_when": "Deploying the API",
        "scope": "project",
        "requests": [1],
        "confidence": "verified",
        "importance": 4,
        "contested": False,
        "ambiguous": False,
        **over,
    }


def uid_of(n: Any) -> str | None:
    return "conv:s:0" if n == 1 else None


def test_keeps_verified_and_stated_knowledge() -> None:
    kept, dropped = gate([item(), item(confidence="stated", title="Prefers Lingui", kind="preference")], uid_of)
    assert dropped == 0
    assert [k["title"] for k in kept] == ["Deploy the API", "Prefers Lingui"]
    assert kept[0]["provenance"] == ["conv:s:0"]


def test_drops_the_inferred_contested_ambiguous_trivial_malformed() -> None:
    kept, dropped = gate(
        [
            item(confidence="inferred"),
            item(contested=True),
            item(ambiguous=True),
            item(importance=1),
            item(kind="gossip"),
            item(summary=""),
        ],
        uid_of,
    )
    assert kept == [] and dropped == 6


def test_keeps_every_cue() -> None:
    cues = ["", "how do we deploy the api", "how do we deploy the api", *(f"cum fac deploy {i}" for i in range(10)), f"token {KEY}"]
    kept, _ = gate([item(cues=cues)], uid_of)
    assert len(kept[0]["cues"]) == 12
    assert kept[0]["cues"][0] == "how do we deploy the api"
    assert len(set(kept[0]["cues"])) == 12
    assert KEY not in " ".join(kept[0]["cues"])
    assert gate([item(cues="not a list")], uid_of)[0][0]["cues"] == []


def test_observation_cues() -> None:
    cues = ["What instruments does Priya play?", "When did Priya start the cello?", f"key {KEY}", "a", "b"]
    o = clean_observations([{"text": "Priya started the cello in May.", "requests": [1], "cues": cues}], uid_of)[0]
    assert len(o["cues"]) == 4
    assert o["cues"][:2] == cues[:2]
    assert KEY not in " ".join(o["cues"])
    assert clean_observations([{"text": "Priya has a brother.", "requests": [1]}], uid_of)[0]["cues"] == []


def test_this_works_kinds_need_a_request_seen_working() -> None:
    kinds = ["procedure", "snippet", "template", "pitfall", "decision", "fact", "preference"]
    for o in ("fail", "uncertain"):
        kept, _ = gate([item(kind=k, title=k) for k in kinds], uid_of, lambda _n, o=o: o)
        assert [k["kind"] for k in kept] == ["decision", "fact", "preference"]
    for o in ("success", "partial"):
        assert len(gate([item(kind=k, title=k) for k in kinds], uid_of, lambda _n, o=o: o)[0]) == 7
    assert len(gate([item()], uid_of)[0]) == 1


def test_scrubs_a_secret() -> None:
    kept, _ = gate([item(body=f"export OPENAI_API_KEY={KEY}")], uid_of)
    assert KEY not in kept[0]["body"]


def test_windows_of_whole_requests() -> None:
    eps = [req(f"u{i}", "x" * 2500) for i in range(5)]
    wins = windows_of(eps, 6000)
    assert [e["uid"] for w in wins for e in w] == [e["uid"] for e in eps]
    assert len(wins) > 1


def test_asks_once_and_maps_the_answer() -> None:
    asks: list[LlmRequest] = []

    async def ask(a: LlmRequest) -> dict[str, Any]:
        asks.append(a)
        return {
            "answer": {
                "segment_summary": "Set up the deploy.",
                "episodes": [
                    {"request": 1, "keep": True, "title": "Deploy set up", "summary": "The API deploys through Jenkins on push."},
                    {"request": 2, "keep": False, "title": "", "summary": ""},
                ],
                "items": [item(), item(confidence="inferred")],
            }
        }

    x = run(
        extract_window(
            [req("conv:s:0", "set up the deploy"), req("conv:s:1", "yes")],
            {"cwd": "/p", "engine": "conversation", "title": "Deploy"},
            ask=ask,
        )
    )
    assert len(asks) == 1
    assert asks[0].purpose == "memory"
    assert asks[0].max_tokens == EXTRACT_MAX_TOKENS
    assert "## Request 1" in asks[0].prompt
    assert [c["title"] for c in x["candidates"]] == ["Deploy the API"]
    assert [e["uid"] for e in x["episodes"]] == ["conv:s:0"]
    assert x["dropped"] == 1


def test_passes_on_busy_or_unavailable() -> None:
    async def busy(_: LlmRequest) -> dict[str, Any]:
        return {"busy": True}

    async def off(_: LlmRequest) -> dict[str, Any]:
        return {"unavailable": "no route"}

    meta = {"cwd": None, "engine": "e", "title": "t"}
    assert run(extract_window([req("u", "x")], meta, ask=busy)) == {"busy": True}
    assert run(extract_window([req("u", "x")], meta, ask=off)) == {"unavailable": "no route"}


def test_parts_hold_every_character_once() -> None:
    text = "\n".join(f"line {i} {'x' * (i % 97)}" for i in range(3000)) + "\n" + "y" * 25_000
    parts = parts_of(text, 10_000)
    assert len(parts) > 5
    assert all(u16len(p) <= 10_000 for p in parts)
    assert "\n".join(parts).replace("\n", "") == text.replace("\n", "")


def test_a_long_request_is_read_part_by_part() -> None:
    steps = [f"Bash: step {i} ran the migration batch {i}" for i in range(900)]
    said = " ".join(f"table{i} moved" for i in range(9000))
    long = req("conv:s:0", "migrate every table of the warehouse", steps=steps, said=said)
    rendered = render_for_extraction(long, 1)
    assert u16len(rendered) > WINDOW_CHARS
    prompts: list[str] = []

    async def ask(a: LlmRequest) -> dict[str, Any]:
        prompts.append(a.prompt)
        n = len(prompts)
        return {
            "answer": {
                "segment_summary": f"part {n}",
                "episodes": [
                    {
                        "request": 1,
                        "keep": True,
                        "title": f"Warehouse migration, parts 1-{n}",
                        "summary": f"what parts 1 to {n} show",
                        "cues": [f"cue {n}"],
                        "concepts": [f"c{n}"],
                    }
                ],
                "items": [item(title=f"Item from part {n}", summary=f"knowledge {n}")],
            }
        }

    x = run(extract_window([long], {"cwd": "/p", "engine": "conversation", "title": "Migration"}, ask=ask))
    parts = parts_of(rendered, WINDOW_CHARS - 2000)
    assert len(parts) > 1
    assert len(prompts) == len(parts)
    assert all(p.endswith(parts[k]) for k, p in enumerate(prompts))
    assert "What the earlier parts showed" not in prompts[0]
    assert all(f"Summary: what parts 1 to {k + 1} show" in p for k, p in enumerate(prompts[1:]))
    assert [c["title"] for c in x["candidates"]] == [f"Item from part {k + 1}" for k in range(len(prompts))]
    assert len(x["episodes"]) == 1 and x["episodes"][0]["summary"] == f"what parts 1 to {len(parts)} show"
    assert x["episodes"][0]["cues"] == [f"cue {k + 1}" for k in range(len(prompts))]


def test_a_long_request_waits_whole_when_a_part_is_not_answered() -> None:
    long = req("conv:s:0", "big", said=" ".join(f"row{i} copied" for i in range(12_000)))
    n = {"v": 0}

    async def ask(_: LlmRequest) -> dict[str, Any]:
        n["v"] += 1
        return {"unavailable": "no route"} if n["v"] == 2 else {"answer": {"segment_summary": "", "episodes": [], "items": []}}

    assert run(extract_window([long], {"cwd": None, "engine": "e", "title": "t"}, ask=ask)) == {"unavailable": "no route"}
