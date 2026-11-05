"""Consolidation and sleep against what changes around them: records removed or changed while the model reads them, a
thread that never goes quiet, a model that quotes its labels, a subject linked without end, history imported after a
sleep, and a model that is not there. (consolidate-races.test.ts, ported.)"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from fakes import DAY, NOW, OLD, Setup, entry, item, run
from memtrails._js import stringify
from memtrails.core.consolidate import MAX_LINKS, Consolidator, label_no
from memtrails.core.gather import memory_gather
from memtrails.core.sleep import Sleep
from memtrails.core.writer import write_ltm
from memtrails.llm import LlmRequest

NOTHING = {"answer": {"segment_summary": "x", "episodes": [], "items": [], "observations": []}}


def with_cwd_none(e: dict[str, Any]) -> dict[str, Any]:
    return {**e, "cwd": None}


def test_a_record_removed_or_changed_while_read_is_left_so(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.stm.put([entry("c:1", "rotate the staging certificate", turn=0), entry("c:2", "renew the token", turn=1)])

    async def ask(a: LlmRequest) -> dict[str, Any]:
        if a.name == "memory_consolidation":
            # Meanwhile: one record removed, the other changed (added again with new text).
            s.stm.remove(["c:1"])
            s.stm.put([{**s.stm.get("c:2"), "fingerprint": "c:2-new", "request": "renew the token, and the key too"}])
        return NOTHING

    r = run(Consolidator(s.deps(ask=ask)).run())
    assert r["windows"] == 1
    assert s.stm.get("c:1") is None
    assert s.stm.get("c:2")["request"] == "renew the token, and the key too"
    assert s.stm.get("c:2")["done"] is False


def test_marks_done_a_record_left_as_it_was(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.stm.put([entry("c:1", "rotate the staging certificate")])

    async def ask(_: LlmRequest) -> dict[str, Any]:
        return NOTHING

    run(Consolidator(s.deps(ask=ask)).run())
    assert s.stm.get("c:1")["done"] is True


def test_a_thread_that_never_goes_quiet_is_read_after_a_day(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.stm.put([entry(f"busy:{n}", f"update {n}", session="busy", turn=n, ts=NOW - 25 * 3600_000 + n * 30 * 60_000) for n in range(51)])
    s.stm.put([entry("fresh:0", "just now", session="fresh", ts=NOW - 10 * 60_000)])
    assert [p["session"] for p in Consolidator(s.deps()).pending(NOW)] == ["busy"]


def test_labels_given_alone_or_with_their_line() -> None:
    assert label_no("p3", "p") == 3
    assert label_no("[p12]\nNEW (fact): x", "p") == 12
    assert label_no("[i2] (fact) deploy the api", "i") == 2
    assert label_no("item", "i") == 0


def test_a_pair_labelled_with_its_whole_line(tmp_path: Path) -> None:
    s = Setup(tmp_path)

    async def ask(a: LlmRequest) -> dict[str, Any]:
        label = "\n".join(a.prompt.split("\n\n")[0].split("\n")[:2])
        return {"answer": {"pairs": [{"id": label, "relation": "related", "title": "", "summary": "", "body": ""}]}}

    got = run(
        Consolidator(s.deps(ask=ask)).relate(
            [{"it": item("mem:new", "new", "deploy through jenkins"), "known": item("mem:old", "old", "deploy by hand")}]
        )
    )
    assert got["p1"]["relation"] == "related"


def test_says_why_the_model_did_not_answer(tmp_path: Path) -> None:
    s = Setup(tmp_path)

    async def ask(_: LlmRequest) -> dict[str, Any]:
        return {"unavailable": "no model is set"}

    got = run(Consolidator(s.deps(ask=ask)).relate([{"it": item("a", "a", "a"), "known": item("b", "b", "b")}]))
    assert got == {"paused": "no model is set"}


def test_cues_for_a_quoted_label(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    it = item("mem:api", "deploy the api", "push main and jenkins deploys")
    run(write_ltm(s.deps(), [it]))
    s.ltm.put([it])

    async def ask(a: LlmRequest) -> dict[str, Any]:
        return {"answer": {"items": [{"id": a.prompt.split("\n")[0], "cues": ["how do I ship the api"], "concepts": ["jenkins"]}]}}

    assert run(Consolidator(s.deps(ask=ask)).fill_cues()) == {"cued": 1}
    got = s.ltm.get("mem:api")
    assert got["cues"] == ["how do I ship the api"] and got["concepts"] == ["jenkins"]


def test_keeps_the_newest_links(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    busy = item("mem:busy", "busy", "busy", related=[f"mem:r{n}" for n in range(MAX_LINKS)])
    other = item("mem:new", "new", "new")
    run(write_ltm(s.deps(), [busy, other]))
    s.ltm.put([busy, other])
    run(Consolidator(s.deps()).link(busy, other))
    related = s.ltm.get("mem:busy")["related"]
    assert len(related) == MAX_LINKS
    assert related[-1] == "mem:new"
    assert "mem:r0" not in related
    run(Consolidator(s.deps()).link(s.ltm.get("mem:busy"), other))
    assert s.ltm.get("mem:busy")["related"].count("mem:new") == 1


def test_stamps_what_consolidation_writes_with_when_it_was_learned(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.stm.put([with_cwd_none(entry("c:1", "deploy the api through jenkins", ts=OLD))])

    async def ask(_: LlmRequest) -> dict[str, Any]:
        return {
            "answer": {
                "segment_summary": "x",
                "episodes": [],
                "observations": [],
                "items": [
                    {
                        "kind": "procedure",
                        "title": "deploy through jenkins",
                        "summary": "push main and jenkins deploys",
                        "body": "b",
                        "applies_when": "",
                        "scope": "project",
                        "requests": [1],
                        "confidence": "verified",
                        "importance": 4,
                        "contested": False,
                        "ambiguous": False,
                        "cues": [],
                        "concepts": [],
                    }
                ],
            }
        }

    run(Consolidator(s.deps(ask=ask)).run())
    made = next(x for x in s.ltm.all() if x["title"] == "deploy through jenkins")
    assert made["ts"] == OLD
    assert made["learnedAt"] == NOW


def test_history_imported_after_a_sleep_is_merged_and_not_archived(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    learned = NOW - 3600_000
    a = {**item("mem:a", "ports", "the proxy listens on port 8080 behind caddy", ts=OLD - 30 * DAY), "learnedAt": learned}
    b = {**item("mem:b", "ports", "the proxy listens on port 8080 behind caddy", ts=OLD), "learnedAt": learned}
    old = {
        **item("ep:old", "one-off fix", "fixed a typo in a readme long ago", kind="episode", importance=2, ts=NOW - 400 * DAY),
        "learnedAt": learned,
    }
    run(write_ltm(s.deps(), [a, b, old]))
    s.ltm.put([a, b, old])
    (tmp_path / "sleep.json").write_text(stringify({"last": NOW - 2 * 3600_000}))
    r = run(Sleep(s.sleep_deps()).run())
    assert r["merged"] == 1
    assert r["archived"] == 0
    assert s.ltm.get("ep:old")["status"] == "active"


def test_archives_an_item_without_learned_at_by_its_own_time(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    old = item("ep:old", "one-off fix", "fixed a typo in a readme long ago", kind="episode", importance=2, ts=NOW - 400 * DAY)
    run(write_ltm(s.deps(), [old]))
    s.ltm.put([old])
    assert run(Sleep(s.sleep_deps()).run())["archived"] == 1


def test_sleep_says_the_model_is_not_there(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    words = "deploy jenkins pipeline main branch build docker image registry"
    items = [item("mem:alpha", "alpha", f"{words} one"), item("mem:bravo", "bravo", f"{words} three", ts=OLD + DAY)]
    run(write_ltm(s.deps(), items))
    s.ltm.put(items)

    async def ask(_: LlmRequest) -> dict[str, Any]:
        return {"unavailable": "no model is set"}

    assert run(Sleep(s.sleep_deps(ask=ask)).run())["paused"] == "no model is set"


def test_gathering_without_the_model_keeps_every_candidate(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    run(
        write_ltm(
            s.deps(),
            [
                item(f"obs:{n}", f"Nadia {t}", f"Nadia {t}", kind="observation", scope="person")
                for n, t in enumerate(["bought an evening gown", "went camping"])
            ],
        )
    )

    async def away(_: LlmRequest) -> dict[str, Any]:
        return {"unavailable": "no model is set"}

    g = run(memory_gather(s.recall(), {"query": "what did Nadia buy"}, ask=away))
    assert g["paused"] == "no model is set"
    assert sorted(g["kept"]) == ["obs:0", "obs:1"]

    async def reads(_: LlmRequest) -> dict[str, Any]:
        return {"answer": {"keep": ["m1"], "searches": []}}

    assert "paused" not in run(memory_gather(s.recall(), {"query": "what did Nadia buy"}, ask=reads))


def test_a_plain_function_works_as_the_model(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    s.stm.put([entry("c:1", "rotate the staging certificate")])
    asked: list[str] = []

    def ask(a: LlmRequest) -> dict[str, Any]:
        asked.append(a.name or "")
        return NOTHING

    run(Consolidator(s.deps(ask=ask)).run())
    assert asked == ["memory_consolidation"]


class Reading:
    """A model whose first answer comes at once and every later one only after a long while: counts what is asked
    and what is still being asked."""

    def __init__(self) -> None:
        self.asked = 0
        self.now = 0

    async def __call__(self, _: LlmRequest) -> dict[str, Any]:
        self.asked += 1
        if self.asked == 1:
            return NOTHING
        self.now += 1
        try:
            await asyncio.sleep(30)
            return NOTHING
        finally:
            self.now -= 1


def windows(s: Setup, n: int) -> None:
    s.stm.put([entry(f"w{k}:0", f"window {k} says something", session=f"w{k}", ts=OLD + k) for k in range(n)])


async def test_a_failed_window_stops_the_reading_of_the_others(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    windows(s, 6)
    model = Reading()
    worker = Consolidator(s.deps(ask=model, concurrency=2))

    async def broken(*_: Any) -> None:
        raise RuntimeError("the index is gone")

    worker._reconcile = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="the index is gone"):
        await worker.run(6)
    # Nothing is still being asked once run() raised, and nothing is asked after it.
    assert model.now == 0
    asked = model.asked
    await asyncio.sleep(0.05)
    assert model.asked == asked


async def test_a_cancelled_run_leaves_nothing_being_asked(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    windows(s, 6)
    model = Reading()
    model.asked = 1  # every answer is slow
    running = asyncio.ensure_future(Consolidator(s.deps(ask=model, concurrency=3)).run(6))
    while model.now < 3:
        await asyncio.sleep(0.01)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert model.now == 0


async def test_a_window_cancelled_while_it_waits_for_an_item_lets_go_of_it(tmp_path: Path) -> None:
    worker = Consolidator(Setup(tmp_path).deps())
    release = asyncio.Event()

    async def hold() -> None:
        await release.wait()

    async def nothing() -> None:
        return None

    holder = asyncio.ensure_future(worker.locked(["mem:a", "mem:b"], hold))
    await asyncio.sleep(0)
    waiting = asyncio.ensure_future(worker.locked(["mem:a", "mem:b"], nothing))
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    release.set()
    await holder
    await asyncio.wait_for(worker.locked(["mem:a", "mem:b"], nothing), 1)
    assert not worker._locks


async def test_an_ask_that_raises_cancels_the_asks_beside_it(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    items = [item(f"mem:{k}", f"thing {k}", f"what thing {k} is") for k in range(30)]
    s.ltm.put(items)
    state = {"asked": 0, "now": 0}

    async def ask(_: LlmRequest) -> dict[str, Any]:
        state["asked"] += 1
        if state["asked"] == 1:
            await asyncio.sleep(0.02)
            raise RuntimeError("the model's client broke")
        state["now"] += 1
        try:
            await asyncio.sleep(30)
            return {"answer": {"items": []}}
        finally:
            state["now"] -= 1

    with pytest.raises(RuntimeError, match="client broke"):
        await Consolidator(s.deps(ask=ask)).fill_cues()
    assert state["asked"] == 3
    assert state["now"] == 0


async def test_a_sleep_state_without_a_time_is_no_sleep_yet_and_written_after_the_log(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    it = item("mem:a", "deploy the api", "push main")
    await write_ltm(s.deps(), [it])
    s.ltm.put([it])
    order: list[str] = []
    real_flush = s.ltm.flush

    def flush() -> None:
        order.append("log")
        real_flush()

    s.ltm.flush = flush  # type: ignore[method-assign]

    async def read_state() -> dict[str, Any]:
        return {"last": None}

    async def write_state(_: dict[str, Any]) -> None:
        order.append("state")

    async def away(_: LlmRequest) -> dict[str, Any]:
        return {"unavailable": "test: away"}

    r = await Sleep(s.sleep_deps(ask=away, read_state=read_state, write_state=write_state)).run()
    assert r["looked"] == 1
    assert order == ["log", "state"]
