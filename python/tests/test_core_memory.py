"""
Memory end to end — records (short-term), consolidation (long-term), sleep and recall — over an in-memory stand-in for
Qdrant and an embedder whose vectors are the words of the text, with a stand-in for the fast model. No model, no
Qdrant, no network. (memory.test.ts, ported.)

Each case is a way memory would turn into chaos or mislead an agent: a request missing from short-term memory, the
same knowledge stored twice, a changed decision kept beside the new one, junk kept, an old unused item never leaving,
a superseded or throwaway item shown, a turn held up by a slow or missing judge.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from fakes import DAY, NOW, OLD, Setup, embed, entry, item, run
from memtrails._js import date_parse
from memtrails.core.consolidate import Consolidator, relate_batches
from memtrails.core.gather import memory_gather
from memtrails.core.qdrant_client import point_id
from memtrails.core.recall import memory_explore, memory_open, memory_search
from memtrails.core.recall_log import RECALL_QUERY_CHARS, RecallBook
from memtrails.core.sleep import Sleep
from memtrails.core.text import children
from memtrails.core.writer import WriterDeps, cue_point, write_cues, write_ltm, write_stm
from memtrails.llm import LlmRequest

BASE = {
    "body": "b",
    "applies_when": "",
    "scope": "project",
    "requests": [1],
    "confidence": "verified",
    "importance": 4,
    "contested": False,
    "ambiguous": False,
}


def active(s: Setup) -> list[dict[str, Any]]:
    return [x for x in s.ltm.all() if x["status"] == "active"]


class TestNothingLost:
    def test_files_come_from_the_requests_steps(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put([entry("conv:f:0", "wire the deploy script", session="f", files=["/p/deploy.sh", "/p/Buildfile"])])

        async def ask(_: LlmRequest) -> dict[str, Any]:
            return {
                "answer": {
                    "segment_summary": "x",
                    "episodes": [
                        {"request": 1, "outcome": "success", "keep": False, "title": "", "summary": "", "cues": [], "concepts": []}
                    ],
                    "items": [
                        {
                            **BASE,
                            "kind": "procedure",
                            "title": "deploy through the script",
                            "summary": "run deploy sh after pushing",
                            "cues": [],
                            "concepts": [],
                        }
                    ],
                }
            }

        run(Consolidator(s.deps(ask=ask)).run())
        made = next(x for x in s.ltm.all() if x["title"] == "deploy through the script")
        assert made["files"] == ["/p/deploy.sh", "/p/Buildfile"]

    def test_a_merge_keeps_both_texts(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        known = item("mem:k", "deploy api buildbot", "deploy the api by pushing main buildbot builds and deploys it", body="the first text")
        run(write_ltm(s.deps(), [known]))
        s.ltm.put([known])
        w = Consolidator(s.deps())
        fresh = item("mem:n", "deploy api buildbot", "deploy the api with buildbot", body="the second text, with a detail")
        run(w.merge(s.ltm.get("mem:k"), fresh, {"title": "Deploy the API through Buildbot", "summary": "merged", "body": "merged text"}))
        merged = s.ltm.get("mem:k")
        assert merged["body"] == "merged text"
        assert sorted(merged["absorbed"]) == sorted([f"mem:k@{known['fingerprint']}", "mem:n"])
        assert s.ltm.get("mem:n")["status"] == "merged" and s.ltm.get("mem:n")["mergedInto"] == "mem:k"
        assert s.ltm.get("mem:n")["body"] == "the second text, with a detail"
        assert s.ltm.get(f"mem:k@{known['fingerprint']}")["body"] == "the first text"
        assert "mem:n" not in [p["payload"]["item"] for p in s.points.values()]

    def test_every_chunk_of_a_long_body_is_indexed(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        body = " ".join(f"detail{i}" for i in range(3000))
        it = item("mem:long", "long", "a long one", body=body, cues=[f"cue question {i}" for i in range(10)])
        run(write_ltm(s.deps(), [it]))

        def parts(part: str) -> list[dict[str, Any]]:
            return [p for p in s.points.values() if p["payload"]["item"] == "mem:long" and p["payload"]["part"] == part]

        assert len(parts("child")) == len(children(body))
        assert len(parts("cue")) == 10
        assert point_id("mem:long", cue_point(9)) in s.points
        assert parts("key")[0]["payload"]["body"] == body
        e = entry("conv:x:0", " ".join(f"ask{i}" for i in range(1500)))
        run(write_stm(s.deps(), [e]))
        assert len([p for p in s.points.values() if p["payload"]["item"] == "stm:conv:x:0" and p["payload"]["part"] == "child"]) > 10

    def test_a_long_batch_gets_time_to_embed(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        budgets: list[tuple[int, Any]] = []

        async def timed(texts: list[str], ms: Any = None) -> Any:
            budgets.append((len(texts), ms))
            return await embed(texts)

        d = WriterDeps(client=s.client, embed=timed)
        run(write_stm(d, [entry("conv:big:0", " ".join(f"step{i}" for i in range(30_000)))]))
        n, ms = budgets[0]
        assert n > 300
        assert ms == max(30_000, n * 150)
        run(write_stm(d, [entry("conv:small:0", "short")]))
        assert budgets[1][1] == 30_000

    def test_pairs_are_asked_in_batches(self) -> None:
        def pair(n: int) -> dict[str, Any]:
            return {"it": item(f"a{n}", "t", "s", body="x" * n), "known": item(f"b{n}", "t", "s", body="")}

        assert [len(b) for b in relate_batches([pair(10) for _ in range(20)])] == [8, 8, 4]
        assert [len(b) for b in relate_batches([pair(10), pair(70_000), pair(10)])] == [1, 1, 1]


def test_remote_embedding_failure_keeps_the_old_item(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    known = item("mem:known", "deploy", "push main", cues=["deploy the api"])
    run(write_ltm(s.deps(), [known]))
    before = json.dumps(s.points, sort_keys=True, default=str)

    async def unavailable(*_: Any) -> Any:
        raise RuntimeError("GPU unavailable")

    d = WriterDeps(client=s.client, embed=unavailable)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        run(write_ltm(d, [{**known, "summary": "new deployment procedure"}]))
    assert json.dumps(s.points, sort_keys=True, default=str) == before
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        run(write_cues(d, [{**known, "cues": ["new deployment question"]}]))
    assert json.dumps(s.points, sort_keys=True, default=str) == before
    run(write_cues(s.deps(), [{**known, "cues": []}]))
    assert not any(p["payload"]["part"] == "cue" for p in s.points.values())
    run(write_ltm(s.deps(), [{**known, "status": "archived"}]))
    assert not s.points


class TestConsolidation:
    def test_keeps_what_matters_once(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        known = item("mem:known", "deploy api buildbot", "deploy the api by pushing main buildbot builds and deploys it")
        old = item("mem:old", "database choice postgres", "alpha beta gamma delta epsilon zeta eta theta", kind="decision")
        run(write_ltm(s.deps(), [known, old]))
        s.ltm.put([known, old])
        pending = [entry("conv:sess-1:0", "how do we deploy"), entry("conv:sess-1:1", "switch the database", turn=1)]
        run(write_stm(s.deps(), pending))
        s.stm.put(pending)

        async def ask(a: LlmRequest) -> dict[str, Any]:
            if a.name == "memory_consolidation":
                return {
                    "answer": {
                        "segment_summary": "Deploy and database work.",
                        "episodes": [
                            {"request": 1, "keep": False, "title": "", "summary": "", "cues": []},
                            {
                                "request": 2,
                                "keep": True,
                                "title": "database switched",
                                "summary": "the database was switched to sqlite after load tests",
                                "cues": ["de ce am trecut pe sqlite", "which database do we use"],
                            },
                        ],
                        "items": [
                            {
                                **BASE,
                                "kind": "fact",
                                "title": "deploy api buildbot",
                                "summary": "deploy the api by pushing main buildbot builds and deploys it",
                            },
                            {
                                **BASE,
                                "kind": "decision",
                                "title": "database choice postgres",
                                "summary": "alpha beta gamma delta epsilon zeta eta iota",
                            },
                            {
                                **BASE,
                                "kind": "pitfall",
                                "title": "flaky e2e on wifi",
                                "summary": "the e2e suite times out on hotel wifi run it wired",
                                "cues": ["the e2e tests keep timing out", "testele e2e cad pe wifi", "run the suite wired"],
                            },
                            {**BASE, "kind": "fact", "title": "a guess", "summary": "probably redis", "confidence": "inferred"},
                        ],
                    }
                }
            return {"answer": {"pairs": [{"id": "p1", "relation": "contradicts", "title": "", "summary": "", "body": ""}]}}

        r = run(Consolidator(s.deps(ask=ask)).run())
        assert {k: r[k] for k in ("windows", "merged", "superseded", "dropped", "episodes")} == {
            "windows": 1,
            "merged": 1,
            "superseded": 1,
            "dropped": 1,
            "episodes": 1,
        }
        assert s.ltm.get("mem:known")["seen"] == 2 and s.ltm.get("mem:known")["status"] == "active"
        assert "conv:sess-1:0" in s.ltm.get("mem:known")["provenance"]
        assert s.ltm.get("mem:old")["status"] == "superseded"
        assert not any(p["payload"]["item"] == "mem:old" for p in s.points.values())
        fresh = [x["title"] for x in active(s)]
        assert {"database choice postgres", "flaky e2e on wifi", "database switched"} <= set(fresh)
        assert "a guess" not in fresh

        def cues_of(title: str) -> int:
            uid = next(x for x in s.ltm.all() if x["title"] == title)["uid"]
            return len([p for p in s.points.values() if p["payload"]["item"] == uid and p["payload"]["part"] == "cue"])

        assert cues_of("flaky e2e on wifi") == 3
        assert cues_of("database switched") == 2
        assert all(m["done"] for _, m in s.stm.entries())
        assert all(point_id(f"stm:{e['uid']}", 0) in s.points for e in pending)

    def test_no_episode_when_typed_memories_came_from_it(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put([entry("conv:sess-1:0", "fix the flaky e2e"), entry("conv:sess-1:1", "rename the job", turn=1)])

        async def ask(_: LlmRequest) -> dict[str, Any]:
            return {
                "answer": {
                    "segment_summary": "e2e and job work.",
                    "episodes": [
                        {
                            "request": 1,
                            "keep": True,
                            "title": "e2e fixed",
                            "summary": "the e2e suite was fixed by running it wired",
                            "cues": [],
                        },
                        {"request": 2, "keep": True, "title": "job renamed", "summary": "the deploy job was renamed to ship", "cues": []},
                    ],
                    "items": [
                        {
                            **BASE,
                            "kind": "pitfall",
                            "title": "flaky e2e on wifi",
                            "summary": "the e2e suite times out on hotel wifi run it wired",
                        }
                    ],
                }
            }

        r = run(Consolidator(s.deps(ask=ask)).run())
        assert r["episodes"] == 1
        assert sorted(x["title"] for x in active(s)) == ["flaky e2e on wifi", "job renamed"]
        assert s.ltm.get("ep:conv:sess-1:0") is None
        assert all(m["done"] for _, m in s.stm.entries())

    def test_waits_for_a_conversation_to_settle(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put(
            [
                entry("e:busy:0", "still going", session="busy", ts=NOW - 60_000),
                entry("e:calm:0", "done", session="calm", ts=NOW - 2 * 3600_000),
            ]
        )
        assert [p["session"] for p in Consolidator(s.deps()).pending()] == ["calm"]

    def test_several_windows_at_once_reconciled_in_order(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        words = ["kangaroo", "volcano", "trumpet", "glacier", "saffron"]
        s.stm.put([entry(f"e:s{i}:0", f"ask about {w}", session=f"s{i}", ts=NOW - (10 - i) * 3600_000) for i, w in enumerate(words)])
        state = {"in": 0, "most": 0}

        async def ask(a: LlmRequest) -> dict[str, Any]:
            state["in"] += 1
            state["most"] = max(state["most"], state["in"])
            await asyncio.sleep(0.02)
            state["in"] -= 1
            w = next(x for x in words if x in a.prompt)
            return {
                "answer": {
                    "segment_summary": w,
                    "episodes": [{"request": 1, "keep": False, "title": "", "summary": ""}],
                    "items": [{**BASE, "kind": "fact", "title": w, "summary": f"{w} {w} {w}"}],
                }
            }

        r = run(Consolidator(s.deps(ask=ask, concurrency=3)).run())
        assert r["windows"] == 5
        assert state["most"] == 3
        assert [x["title"] for x in s.ltm.all()] == list(reversed(words))
        assert all(m["done"] for _, m in s.stm.entries())

    def test_a_saved_note_gets_cues_and_concepts(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        note = item("note:/m/deploy.md", "deploy rule", "deploys go through ci never rsync", kind="note")
        known = item(
            "mem:k",
            "deploy api buildbot",
            "deploy the api by pushing main buildbot builds and deploys it",
            cues=["how is the api deployed"],
        )
        run(write_ltm(s.deps(), [note, known]))
        s.ltm.put([note, known])
        s.stm.put([entry("conv:s9:0", "how do we deploy", session="s9")])

        async def ask(a: LlmRequest) -> dict[str, Any]:
            if a.name == "memory_cues":
                ids = re.findall(r"\[(i\d+)\]", a.prompt)
                return {
                    "answer": {
                        "items": [
                            {
                                "id": i,
                                "cues": ["how do deploys run here", "cum facem deploy"],
                                "concepts": ["Buildbot", "deploy.sh", " buildbot "],
                            }
                            for i in ids
                        ]
                    }
                }
            return {
                "answer": {
                    "segment_summary": "deploy",
                    "episodes": [{"request": 1, "keep": False, "title": "", "summary": "", "cues": []}],
                    "items": [
                        {
                            **BASE,
                            "kind": "fact",
                            "title": "deploy api buildbot",
                            "summary": "deploy the api by pushing main buildbot builds and deploys it",
                            "cues": ["push to main to ship the api"],
                        }
                    ],
                }
            }

        r = run(Consolidator(s.deps(ask=ask)).run())
        assert r["merged"] == 1
        assert s.ltm.get("mem:k")["cues"] == ["how is the api deployed", "push to main to ship the api"]
        assert s.ltm.get("note:/m/deploy.md")["cues"] == ["how do deploys run here", "cum facem deploy"]
        assert s.ltm.meta("note:/m/deploy.md")["cued"] is True

        def cue_points(uid: str) -> int:
            return len([p for p in s.points.values() if p["payload"]["item"] == uid and p["payload"]["part"] == "cue"])

        assert cue_points("mem:k") == 2
        assert cue_points("note:/m/deploy.md") == 2
        assert r["cued"] >= 1
        assert s.ltm.get("note:/m/deploy.md")["concepts"] == ["Buildbot", "deploy.sh"]
        assert s.ltm.meta("note:/m/deploy.md")["mapped"] is True
        assert s.points[point_id("note:/m/deploy.md", 0)]["payload"]["concepts"] == ["buildbot", "deploy.sh"]

    def test_a_failing_window_is_read_in_halves(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put(
            [
                entry("e:long:0", "first long request", session="long"),
                entry("e:long:1", "second long request", session="long", turn=1),
                entry("e:old:0", "older request", session="old", ts=OLD - 3600_000),
            ]
        )
        answered: list[list[str]] = []

        async def ask(a: LlmRequest) -> dict[str, Any]:
            if a.name != "memory_consolidation":
                return {"answer": {"items": []}}
            asked = [t for t in ("first long request", "second long request", "older request") if t in a.prompt]
            if len(asked) > 1:
                return {"unavailable": "Unterminated string in JSON"}
            answered.append(asked)
            return {"answer": {"segment_summary": "x", "episodes": [], "items": []}}

        w = Consolidator(s.deps(ask=ask))
        assert "Unterminated" in run(w.run())["paused"]
        assert not [1 for _, m in s.stm.entries() if m["done"]]
        second = run(w.run())
        assert "paused" not in second
        assert s.stm.meta("e:old:0")["done"] is True
        assert s.stm.meta("e:long:0")["done"] is False
        run(w.run())
        assert all(m["done"] for _, m in s.stm.entries())
        assert {",".join(x) for x in answered} == {"older request", "first long request", "second long request"}

    def test_goes_on_past_a_window_that_fails_once(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put(
            [
                entry("e:a:0", "newest request", session="a"),
                entry("e:b:0", "middle request", session="b", ts=OLD - 3600_000),
                entry("e:c:0", "oldest request", session="c", ts=OLD - 7200_000),
            ]
        )
        failures = {"n": 0}

        async def ask(a: LlmRequest) -> dict[str, Any]:
            if a.name != "memory_consolidation":
                return {"answer": {"items": []}}
            if "middle request" in a.prompt and not failures["n"]:
                failures["n"] += 1
                return {"unavailable": "Unterminated string in JSON"}
            return {"answer": {"segment_summary": "x", "episodes": [], "items": []}}

        w = Consolidator(s.deps(ask=ask))
        first = run(w.run())
        assert first["windows"] == 2 and "paused" not in first
        assert [s.stm.meta(u)["done"] for u in ("e:a:0", "e:b:0", "e:c:0")] == [True, False, True]
        run(w.run())
        assert s.stm.meta("e:b:0")["done"] is True

    def test_pauses_when_the_model_cannot_be_asked(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put([entry("e:s:0", "x")])

        async def ask(_: LlmRequest) -> dict[str, Any]:
            return {"unavailable": "no route"}

        r = run(Consolidator(s.deps(ask=ask)).run())
        assert r["paused"] == "no route"
        assert s.stm.meta("e:s:0")["done"] is False


class TestSleep:
    def test_merges_a_fresh_duplicate_and_archives_the_old(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        a = item("mem:a", "ports", "the proxy listens on port 8080 behind caddy", ts=NOW - 10 * DAY)
        b = item("mem:b", "ports", "the proxy listens on port 8080 behind caddy", ts=NOW - 3600_000)
        stale = item("ep:old", "one-off fix", "fixed a typo in a readme long ago", kind="episode", importance=2, ts=NOW - 400 * DAY)
        lasting = item(
            "mem:arch", "how sync works", "sync runs in the background worker not the client", kind="architecture", ts=NOW - 400 * DAY
        )
        run(write_ltm(s.deps(), [a, b, stale, lasting]))
        s.ltm.put([a, b, stale, lasting])
        r = run(Sleep(s.sleep_deps()).run())
        assert r["merged"] == 1
        statuses = {u: m["status"] for u, m in s.ltm.entries()}
        assert sorted([statuses["mem:a"], statuses["mem:b"]]) == ["active", "superseded"]
        assert statuses["ep:old"] == "archived"
        assert statuses["mem:arch"] == "active"

    def test_relates_a_fresh_item_to_every_neighbour(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        deploy = "deploy buildbot pipeline main branch build docker image registry"
        db = "database postgres cluster replica failover backup nightly snapshot restore"
        pdf = "invoice pdf render service template header footer locale font"
        items = [
            item("mem:alpha", "alpha", f"{deploy} one", ts=NOW - 5 * DAY),
            item("mem:golf", "golf", f"{deploy} two", ts=NOW - 4 * DAY),
            item("mem:bravo", "bravo", f"{deploy} three", ts=NOW - DAY),
            item("mem:delta", "delta", f"{db} four", kind="decision", ts=NOW - 5 * DAY),
            item("mem:echo", "echo", f"{db} five", kind="decision", ts=NOW - DAY),
            item("mem:foxtrot", "foxtrot", f"{pdf} six", ts=NOW - 5 * DAY),
            item("mem:hotel", "hotel", f"{pdf} seven", ts=NOW - DAY),
        ]
        run(write_ltm(s.deps(), items))
        s.ltm.put(items)
        said = {
            "bravo alpha": "related",
            "bravo golf": "related",
            "golf alpha": "different",
            "echo delta": "contradicts",
            "hotel foxtrot": "same",
        }
        asked: list[str] = []

        async def ask(a: LlmRequest) -> dict[str, Any]:
            pairs = []
            for pid, new, known in re.findall(r"\[(p\d+)\]\nNEW \([^)]*\): (\S+)[\s\S]*?KNOWN \([^)]*\): (\S+)", a.prompt):
                asked.append(f"{new} {known}")
                relation = said.get(f"{new} {known}", "different")
                same = relation == "same"
                pairs.append(
                    {
                        "id": pid,
                        "relation": relation,
                        "title": "invoice pdf rendering" if same else "",
                        "summary": f"{pdf} six seven" if same else "",
                        "body": "",
                    }
                )
            return {"answer": {"pairs": pairs}}

        r = run(Sleep(s.sleep_deps(ask=ask)).run())
        assert sorted(asked) == sorted(said)
        assert (r["linked"], r["superseded"], r["merged"]) == (2, 1, 1)
        assert sorted(s.ltm.get("mem:bravo")["related"]) == ["mem:alpha", "mem:golf"]
        assert s.ltm.get("mem:alpha")["related"] == ["mem:bravo"]
        assert s.ltm.get("mem:delta")["status"] == "superseded" and s.ltm.get("mem:delta")["supersededBy"] == "mem:echo"
        assert s.ltm.get("mem:foxtrot")["status"] == "active" and s.ltm.get("mem:foxtrot")["title"] == "invoice pdf rendering"
        assert s.ltm.get("mem:hotel")["status"] == "superseded"


def project() -> list[dict[str, Any]]:
    """A project of 15 items: three about the club (on 3 of 15, under a fifth), Marco on all of them (too broad)."""
    return [
        item(
            "obs:join",
            "Marco joined the club",
            "Marco joined the harbour rowing club",
            kind="observation",
            eventAt="2023-05-08",
            provenance=["c:s1:0"],
            concepts=["Marco", "harbour rowing club"],
            ts=NOW - 30 * DAY,
        ),
        item(
            "mem:drill",
            "club drill",
            "the club drills every Tuesday at 7pm",
            provenance=["c:s2:3"],
            concepts=["Marco", "Harbour rowing club "],
            ts=NOW - 20 * DAY,
        ),
        item(
            "obs:rescue",
            "first regatta",
            "Marco won the autumn regatta",
            kind="observation",
            eventAt="2023-06-02",
            provenance=["c:s3:1", "c:s3:2"],
            concepts=["Marco", "harbour rowing club", "flood"],
            ts=NOW - 10 * DAY,
        ),
        *(item(f"mem:other{k}", f"topic{k}", f"w{k}a w{k}b w{k}c w{k}d", concepts=["Marco"], ts=NOW - (40 + k) * DAY) for k in range(12)),
    ]


class TestProfiles:
    def test_a_subject_gets_one_dated_profile(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        items = project()
        run(write_ltm(s.deps(), items))
        s.ltm.put(items)
        asked: list[str] = []

        async def ask(a: LlmRequest) -> dict[str, Any]:
            if a.name != "memory_profile":
                return {"answer": {"pairs": []}}
            asked.append(a.prompt)
            return {
                "answer": {
                    "title": "Marco and the harbour rowing club",
                    "facts": ["2023-05-08: Marco joined the harbour rowing club", "2023-06-02: Marco won the autumn regatta"],
                }
            }

        def sleep() -> dict[str, Any]:
            return run(Sleep(s.sleep_deps(ask=ask)).run())

        r = sleep()
        assert (r["profiled"], r["unprofiled"]) == (1, 0)
        assert len(asked) == 1
        assert asked[0].startswith(
            "SUBJECT: harbour rowing club\n\nMEMORIES:\n- (2023-05-08) Marco joined the club: Marco joined the harbour rowing club\n"
        )
        assert "- (2023-06-02) first regatta: Marco won the autumn regatta" in asked[0]
        profiles = [u for u, m in s.ltm.entries() if m["kind"] == "profile"]
        assert len(profiles) == 1
        profile = s.ltm.get(profiles[0])
        assert {k: profile[k] for k in ("kind", "status", "title", "concepts", "project")} == {
            "kind": "profile",
            "status": "active",
            "title": "Marco and the harbour rowing club",
            "concepts": ["harbour rowing club"],
            "project": "-p",
        }
        assert profile["summary"] == "2023-05-08: Marco joined the harbour rowing club\n2023-06-02: Marco won the autumn regatta"
        assert sorted(profile["provenance"]) == ["c:s1:0", "c:s2:3", "c:s3:1", "c:s3:2"]
        hits = run(memory_search(s.recall(), {"query": "harbour rowing club regatta", "limit": 5}))
        assert profile["uid"] in [h["id"] for h in hits]

        assert sleep()["profiled"] == 0
        assert len(asked) == 1
        drill = {**s.ltm.get("mem:drill"), "summary": "the club drills every Wednesday at 7pm now"}
        run(write_ltm(s.deps(), [drill]))
        s.ltm.put([drill])
        assert sleep()["profiled"] == 1
        assert len(asked) == 2
        assert "drills every Wednesday at 7pm now" in asked[1]

        retired = [{**s.ltm.get(u), "status": "superseded"} for u in ("obs:rescue", "mem:drill")]
        run(write_ltm(s.deps(), retired))
        s.ltm.put(retired)
        r = sleep()
        assert (r["profiled"], r["unprofiled"]) == (0, 1)
        assert s.ltm.get(profile["uid"])["status"] == "archived"
        assert len(asked) == 2

    def test_a_night_without_the_model_writes_nothing(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        items = project()
        run(write_ltm(s.deps(), items))
        s.ltm.put(items)
        up = {"v": False}

        async def ask(a: LlmRequest) -> dict[str, Any]:
            if a.name != "memory_profile":
                return {"answer": {"pairs": []}}
            return {"answer": {"title": "the club", "facts": ["2023-05-08: Marco joined"]}} if up["v"] else {"unavailable": "no route"}

        assert run(Sleep(s.sleep_deps(ask=ask)).run())["profiled"] == 0
        up["v"] = True
        assert run(Sleep(s.sleep_deps(ask=ask)).run())["profiled"] == 1


def test_explore_reaches_what_shares_concepts(tmp_path: Path) -> None:
    s = Setup(tmp_path)
    items = [
        item(
            "mem:image",
            "proxy config baked into the image",
            "the proxy config is baked into the image rebuild it after edits",
            concepts=["proxy.conf", "acme-proxy:dev"],
            related=["mem:recreate"],
        ),
        item(
            "mem:reload",
            "a reload ignores host edits",
            "reloading inside the container keeps the old file",
            concepts=["Proxy.conf", "caddy"],
        ),
        item(
            "mem:recreate",
            "recreate after settings change",
            "a restart keeps the old settings, recreate it",
            concepts=["docker compose"],
            related=["mem:image"],
        ),
        item("mem:invoice", "invoice pdf", "invoices render with the pdf service", concepts=["pdf service"]),
        item("mem:alias", "proxy file location", "it lives in the project folder", concepts=["proxy.conf file"]),
    ]
    run(write_ltm(s.deps(), items))
    s.ltm.put(items)
    r = run(memory_explore(s.recall(), {"from": ["mem:image"]}))
    ids = [x["id"] for x in r["related"]]
    assert {"mem:reload", "mem:recreate"} <= set(ids)
    assert not {"mem:image", "mem:invoice", "mem:alias"} & set(ids)
    assert next(x for x in r["related"] if x["id"] == "mem:reload")["via"] == 'shares "proxy.conf"'
    assert next(x for x in r["related"] if x["id"] == "mem:recreate")["via"] == 'linked to "proxy config baked into the image"'
    assert "proxy.conf" in r["concepts"]


def seeded(tmp_path: Path) -> Setup:
    s = Setup(tmp_path)
    items = [
        item("mem:deploy", "deploy api buildbot", "deploy the api by pushing main buildbot builds it"),
        item("mem:old-deploy", "deploy by rsync", "deploy the api by rsync to the server", status="superseded"),
        item("mem:other", "invoice pdf", "invoices render with the pdf service"),
    ]
    run(write_ltm(s.deps(), items))
    run(write_stm(s.deps(), [entry("conv:here:0", "deploy the api now", session="here", ts=NOW - 3600_000)]))
    return s


class TestRecall:
    def test_records_what_a_search_showed_and_what_was_opened(self, tmp_path: Path) -> None:
        s = seeded(tmp_path)
        path = str(tmp_path / "recall.jsonl")
        book = RecallBook(path, lambda: NOW)
        deps = s.recall(record=book.record)
        found = run(memory_search(deps, {"query": "invoice pdf"}))
        run(memory_open(deps, "mem:other"))
        run(memory_open(deps, "mem:nothing-here"))
        lines = [json.loads(x) for x in Path(path).read_text().strip().split("\n")]
        assert [x["ev"] for x in lines] == ["search", "open"]
        assert lines[0]["t"] == NOW and lines[0]["query"] == "invoice pdf" and lines[0]["shown"] == [h["id"] for h in found]
        assert lines[1] == {"t": NOW, "ev": "open", "item": "mem:other"}
        assert os.stat(path).st_mode & 0o777 == 0o600
        book.record({"ev": "search", "query": "x" * (RECALL_QUERY_CHARS + 100), "shown": []})
        assert len(json.loads(Path(path).read_text().strip().split("\n")[-1])["query"]) == RECALL_QUERY_CHARS

    def test_trims_the_record(self, tmp_path: Path) -> None:
        path = str(tmp_path / "recall.jsonl")
        clock = {"now": NOW - 200 * DAY}
        book = RecallBook(path, lambda: clock["now"], keep_ms=180 * DAY, keep_bytes=400)
        book.record({"ev": "open", "item": "mem:ancient"})
        clock["now"] = NOW
        for i in range(6):
            book.record({"ev": "search", "query": f"q{i} {'x' * 60}", "shown": [f"mem:{i}"]})
        assert book.trim() >= 2
        book.record({"ev": "open", "item": "mem:after"})
        lines = [json.loads(x) for x in Path(path).read_text().strip().split("\n")]
        assert not any(x.get("item") == "mem:ancient" for x in lines)
        assert lines[-1]["item"] == "mem:after"
        assert lines[-2]["query"].startswith("q5 ")
        assert os.stat(path).st_size <= 600
        assert os.stat(path).st_mode & 0o777 == 0o600
        book.trim()
        assert os.stat(path).st_size <= 400
        assert book.trim() == 0

    def test_an_agent_searches_in_its_own_words(self, tmp_path: Path) -> None:
        s = seeded(tmp_path)
        assert run(memory_search(s.recall(), {"query": "invoice pdf"}))[0]["id"] == "mem:other"

    def test_reorders_its_best_by_the_reranker(self, tmp_path: Path) -> None:
        s = seeded(tmp_path)
        plain = run(memory_search(s.recall(), {"query": "deploy the api"}))

        async def rerank(_q: str, docs: list[str], _t: float) -> list[float]:
            return [10 if "invoice" in t else 0 for t in docs]

        reranked = run(memory_search(s.recall(rerank=rerank), {"query": "deploy the api"}))

        def at(xs: list[Any]) -> int:
            return [h["id"] for h in xs].index("mem:other")

        assert at(reranked) < at(plain)
        assert sorted(h["id"] for h in reranked) == sorted(h["id"] for h in plain)

        async def nothing(*_: Any) -> None:
            return None

        assert [h["id"] for h in run(memory_search(s.recall(rerank=nothing), {"query": "deploy the api"}))] == [h["id"] for h in plain]

    def test_a_short_title_shows_what_matched(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        run(
            write_stm(
                s.deps(), [entry("conv:x:0", "continue", reply="Signed the Windows binaries before publishing them.", ts=NOW - 3600_000)]
            )
        )
        hit = run(memory_search(s.recall(), {"query": "sign the windows binaries"}))[0]
        assert hit["id"] == "stm:conv:x:0"
        assert hit["title"] == "continue Signed the Windows binaries before publishing them."


def obs(text: str, event_date: str, **over: Any) -> dict[str, Any]:
    return {
        "text": text,
        "about": "Nadia",
        "event_date": event_date,
        "phase": "done",
        "requests": [1],
        "concepts": ["Nadia", "beach"],
        **over,
    }


def answer(observations: list[Any], items: list[Any] | None = None) -> dict[str, Any]:
    return {"answer": {"segment_summary": "s", "episodes": [], "items": items or [], "observations": observations}}


CAMPING = "Nadia went camping with her family at the beach and roasted marshmallows"


def asker(result: Any) -> Any:
    async def ask(_: LlmRequest) -> dict[str, Any]:
        return result

    return ask


class TestObservations:
    def test_each_kept_dated(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put([entry("conv:sess-1:0", "we went camping")])
        fact = {
            "kind": "fact",
            "title": "camping at the beach",
            "summary": CAMPING,
            "body": CAMPING,
            "applies_when": "",
            "scope": "person",
            "requests": [1],
            "confidence": "stated",
            "importance": 3,
            "contested": False,
            "ambiguous": False,
            "cues": [],
            "concepts": [],
        }
        ask = asker(
            answer(
                [
                    obs(CAMPING, "2023-06-20"),
                    obs(CAMPING, "2023-07-10"),
                    obs("Nadia painted a sunset over the lake", "last summer"),
                    obs("", "2023-06-20"),
                    obs("Nadia read a book", "2023-06-01", requests=[9]),
                ],
                [fact],
            )
        )
        r = run(Consolidator(s.deps(ask=ask)).run())
        assert r["observed"] == 3

        def observations() -> list[dict[str, Any]]:
            return [x for x in active(s) if x["kind"] == "observation"]

        assert sorted(o.get("eventAt", "") for o in observations()) == ["", "2023-06-20", "2023-07-10"]
        assert all(o["about"] == "Nadia" and o["provenance"][0] == "conv:sess-1:0" for o in observations())
        assert len([x for x in active(s) if x["kind"] == "fact"]) == 1
        s.stm.put([entry("conv:sess-2:0", "remember the camping", session="sess-2")])
        again = run(Consolidator(s.deps(ask=asker(answer([obs(CAMPING, "2023-06-20")])))).run())
        assert (again["observed"], again["merged"]) == (0, 1)
        told = next(o for o in observations() if o.get("eventAt") == "2023-06-20")
        assert told["seen"] == 2
        assert told["provenance"] == ["conv:sess-1:0", "conv:sess-2:0"]
        run(Sleep(s.sleep_deps(ask=asker({"answer": {"pairs": []}}))).run())
        assert len(observations()) == 3
        hits = run(memory_search(s.recall(), {"query": "where did Nadia camp", "limit": 50}))
        assert sorted(h.get("eventAt", "") for h in hits if h["kind"] == "observation") == ["", "2023-06-20", "2023-07-10"]

    def test_what_lasts_keeps_its_start(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put([entry("conv:sess-1:0", "I have been playing cello again for about a month")])
        cello = "Marco has been playing cello again for about a month"
        ask = asker(
            answer(
                [
                    obs(cello, "2022-03-27", about="Marco", phase="ongoing", start_date="2022-02", end_date=""),
                    obs("Marco played cello in a recording studio", "2022-03-20", about="Marco", start_date="a while ago", end_date=""),
                ]
            )
        )
        run(Consolidator(s.deps(ask=ask)).run())
        kept = [x for x in s.ltm.all() if x["kind"] == "observation"]
        first = next(o for o in kept if o["summary"] == cello)
        assert first["eventAt"] == "2022-03-27" and first["startAt"] == "2022-02"
        assert "startAt" not in next(o for o in kept if o["summary"] != cello)
        hits = run(memory_search(s.recall(), {"query": "Marco cello", "from": "2022-02-10", "to": "2022-02-20", "limit": 10}))
        assert [(h["summary"], h.get("startAt", "")) for h in hits if h["kind"] == "observation"] == [
            (cello, "2022-02"),
            ("Marco played cello in a recording studio", ""),
        ]

    def test_told_twice_in_one_window(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        s.stm.put([entry("conv:sess-1:0", "we went camping"), entry("conv:sess-1:1", "the camping again", turn=1)])
        r = run(
            Consolidator(
                s.deps(
                    ask=asker(answer([obs(CAMPING, "2023-06-20"), obs(CAMPING, "2023-06-20", requests=[2]), obs(CAMPING, "2023-07-10")]))
                )
            ).run()
        )
        assert (r["observed"], r["merged"]) == (2, 1)
        told = next(o for o in active(s) if o["kind"] == "observation" and o.get("eventAt") == "2023-06-20")
        assert told["seen"] == 2
        assert told["provenance"] == ["conv:sess-1:0", "conv:sess-1:1"]

    def test_windows_reconciled_at_once_lose_nothing(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        known = item("obs:known", CAMPING, CAMPING, kind="observation", eventAt="2023-06-20", scope="person")
        run(write_ltm(s.deps(), [known]))
        s.ltm.put([known])
        s.stm.put(
            [entry("conv:sess-1:0", "we went camping", session="sess-1"), entry("conv:sess-2:0", "the camping again", session="sess-2")]
        )
        r = run(Consolidator(s.deps(ask=asker(answer([obs(CAMPING, "2023-06-20")])), reconcile_concurrency=2)).run())
        assert (r["windows"], r["merged"], r["observed"]) == (2, 2, 0)
        told = s.ltm.get("obs:known")
        assert told["seen"] == known["seen"] + 2
        assert {"conv:sess-1:0", "conv:sess-2:0"} <= set(told["provenance"])

    def test_read_with_its_calendar_and_what_other_conversations_say(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        older = entry(
            "conv:sess-0:0",
            "I just finished The Overstory by Richard Powers, I recommend it to you",
            session="sess-0",
            ts=int(date_parse("2023-07-06T10:00:00Z")),
        )
        later = entry("conv:sess-1:0", "I am reading the book you recommended about Nicole", ts=int(date_parse("2023-10-13T10:00:00Z")))
        s.stm.put([older, later])
        run(write_stm(s.deps(), [older, later]))
        prompts: list[str] = []

        async def ask(a: LlmRequest) -> dict[str, Any]:
            prompts.append(a.prompt)
            return answer([])

        run(Consolidator(s.deps(ask=ask, resolve=True)).run())
        read = next(p for p in prompts if "the book you recommended" in p)
        assert (
            "- Fri 2023-10-13: yesterday Thu 2023-10-12 · the days before Wed 2023-10-11, Tue 2023-10-10, Mon 2023-10-09, Sun 2023-10-08, "
            "Sat 2023-10-07, Fri 2023-10-06 · last week (the week before) 2023-10-06 to 2023-10-12 · last weekend 2023-10-07 to 2023-10-08 · "
            "this week 2023-10-09 to 2023-10-15 · last month 2023-09 · last year 2022"
        ) in read
        known = read[read.index("KNOWN FROM OTHER CONVERSATIONS") : read.index("## Request 1")]
        assert "(2023-07-06) I just finished The Overstory" in known
        assert "reading the book you recommended" not in known

    def test_searches_within_dates_first(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)

        def at(uid: str, event: str, summary: str) -> dict[str, Any]:
            return item(
                uid, summary, summary, kind="observation", eventAt=event, scope="person", ts=int(date_parse("2023-08-01T00:00:00Z"))
            )

        run(
            write_ltm(
                s.deps(),
                [
                    at("obs:june", "2023-06-19", "Nadia went camping with her family in the mountains"),
                    at("obs:july", "2023-07", "Nadia went camping with her family at the beach"),
                    at("obs:year", "2022", "Nadia went camping with her family in the forest"),
                ],
            )
        )
        req = entry("conv:sess-1:0", "Nadia went camping with her family again", ts=int(date_parse("2023-06-28T10:00:00Z")))
        run(write_stm(s.deps(), [req]))

        def lead(n: int, start: str | None = None, end: str | None = None) -> list[str]:
            hits = run(
                memory_search(
                    s.recall(),
                    {"query": "Nadia camping family", "limit": 20, **({"from": start} if start else {}), **({"to": end} if end else {})},
                )
            )
            assert len(hits) == 4
            return sorted(h["id"] for h in hits[:n])

        assert lead(2, "2023-06", "2023-06") == ["obs:june", f"stm:{req['uid']}"]
        assert lead(1, "2023-07-15") == ["obs:july"]
        assert lead(1, None, "2022-12-31") == ["obs:year"]

    def test_reranks_within_dates_and_after(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        june = int(date_parse("2023-06-15T10:00:00Z"))
        rows = [
            entry(f"conv:j{i}:0", t, session=f"j{i}", ts=june)
            for i, t in enumerate(["deploy the api now", "deploy the api today", "deploy the api again"])
        ]
        older = entry("conv:o:0", "deploy the api through buildbot safely", session="o", ts=int(date_parse("2022-03-01T00:00:00Z")))
        run(write_stm(s.deps(), [*rows, older]))

        async def rerank(_q: str, docs: list[str], _t: float) -> list[float]:
            return [10 if "buildbot" in t else 0 for t in docs]

        hits = [
            h["id"]
            for h in run(
                memory_search(s.recall(rerank=rerank), {"query": "deploy the api", "limit": 8, "from": "2023-06", "to": "2023-06"})
            )
        ]
        assert hits.index("stm:conv:o:0") == 3

    def test_shows_the_passage_that_matched(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        filler = " ".join(f"plan{i % 37} schedule{i % 23}" for i in range(400))
        said = f"{filler} we celebrated my daughter birthday last night with a concert under the stars {filler}"
        e = entry("conv:sess-1:0", f"what should we do this weekend {filler}", reply=said, said=said)
        run(write_stm(s.deps(), [e]))
        hit = run(memory_search(s.recall(), {"query": "daughter birthday concert stars", "limit": 5}))[0]
        assert hit["id"] == f"stm:{e['uid']}"
        assert "daughter birthday last night with a concert" in hit["match"]

    def test_observations_beside_long_requests(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        filler = " ".join(f"hiking{i % 7} camping trail{i % 11}" for i in range(300))
        run(write_stm(s.deps(), [entry(f"conv:sess-1:{n}", f"camping hiking trail plans {filler}", turn=n, said=filler) for n in range(4)]))
        run(
            write_ltm(
                s.deps(),
                [
                    item(f"obs:{n}", f"Nadia {t}", f"Nadia {t}", kind="observation", eventAt=f"2023-0{n + 6}", scope="person")
                    for n, t in enumerate(["went camping in the forest", "went hiking on a mountain trail", "went camping at the beach"])
                ],
            )
        )
        hits = run(memory_search(s.recall(), {"query": "Nadia camping hiking trail", "limit": 30}))
        assert sorted(h["id"] for h in hits if h["kind"] == "observation") == ["obs:0", "obs:1", "obs:2"]
        assert any(h["id"].startswith("stm:") for h in hits)

    def test_gathers_a_set(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        said = [
            "bought an evening gown for $800",
            "went camping at the beach",
            "bought a Gucci handbag for $1,200",
            "bought leather boots for $500",
        ]
        run(
            write_ltm(
                s.deps(),
                [
                    item(f"obs:{n}", f"Nadia {t}", f"Nadia {t}", kind="observation", eventAt="2023-05", scope="person")
                    for n, t in enumerate(said)
                ],
            )
        )
        asked: list[str] = []

        async def ask(a: LlmRequest) -> dict[str, Any]:
            asked.append(a.prompt)
            lines = [x for x in a.prompt.split("\n") if x.startswith("[m")]
            return {
                "answer": {
                    "keep": [x[1 : x.index("]")] for x in lines if "bought" in x],
                    "searches": ["Nadia leather boots"] if len(asked) == 1 else [],
                }
            }

        g = run(memory_gather(s.recall(), {"query": "how much did Nadia spend on designer goods"}, ask=ask, limit=1))
        assert len(asked) == 2
        assert g["searches"] == ["Nadia leather boots"]
        assert "obs:3" in g["kept"]
        assert all("bought" in h["summary"] for h in g["found"][: len(g["kept"])])
        assert "obs:1" in [h["id"] for h in g["found"]]

    def test_notes_only_when_asked(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        run(
            write_ltm(
                s.deps(),
                [item("obs:0", "Leo read The Hobbit", "Leo read The Hobbit", kind="observation", eventAt="2023-12", scope="person")],
            )
        )
        asked: list[LlmRequest] = []
        key = "sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"

        async def ask(a: LlmRequest) -> dict[str, Any]:
            asked.append(a)
            return {"answer": {"keep": ["m1"], "searches": [], "notes": ["Books Leo has read: The Hobbit (2023-12)", " ", f"key {key}"]}}

        g = run(memory_gather(s.recall(), {"query": "what books has Leo read"}, ask=ask, notes=True))
        assert g["notes"][0] == "Books Leo has read: The Hobbit (2023-12)"
        assert len(g["notes"]) == 2
        assert key not in " ".join(g["notes"])
        assert "never write that something is not said" in asked[0].system
        assert '"notes"' in json.dumps(asked[0].schema)
        plain = run(memory_gather(s.recall(), {"query": "what books has Leo read"}, ask=ask))
        assert plain["notes"] == []
        assert '"notes"' not in json.dumps(asked[-1].schema)

    def test_a_quoted_label(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        run(
            write_ltm(
                s.deps(),
                [
                    item(f"obs:{n}", f"Nadia {t}", f"Nadia {t}", kind="observation", eventAt="2023-05", scope="person")
                    for n, t in enumerate(["bought an evening gown", "went camping"])
                ],
            )
        )

        async def ask(a: LlmRequest) -> dict[str, Any]:
            return {"answer": {"keep": [x for x in a.prompt.split("\n") if x.startswith("[m") and "gown" in x], "searches": []}}

        assert run(memory_gather(s.recall(), {"query": "what did Nadia buy"}, ask=ask))["kept"] == ["obs:0"]

    def test_gathers_within_dates_first(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        said = [
            ("adopted a puppy named Pixel", "2023-07"),
            ("adopted a second dog named Mochi", "2023-10"),
            ("adopted Bean, a third dog", "2023-11"),
        ]
        run(
            write_ltm(
                s.deps(),
                [
                    item(f"obs:{n}", f"Pavel {t}", f"Pavel {t}", kind="observation", eventAt=d, scope="person")
                    for n, (t, d) in enumerate(said)
                ],
            )
        )
        read: list[list[str]] = []

        async def ask(a: LlmRequest) -> dict[str, Any]:
            read.append([x for x in a.prompt.split("\n") if x.startswith("[m")])
            return {"answer": {"keep": [], "searches": []}}

        g = run(
            memory_gather(
                s.recall(),
                {"query": "how many dogs did Pavel adopt, as of November 2023", "from": "2023-11", "to": "2023-11"},
                ask=ask,
                limit=10,
            )
        )
        assert "Bean" in read[0][0]
        assert sorted(h["id"] for h in g["found"]) == ["obs:0", "obs:1", "obs:2"]

    def test_an_episode_and_its_request_once(self, tmp_path: Path) -> None:
        s = Setup(tmp_path)
        e = entry("conv:sess-1:0", "rotate the proxy certificate on staging")
        run(write_stm(s.deps(), [e]))
        run(
            write_ltm(
                s.deps(),
                [
                    item(
                        f"ep:{e['uid']}",
                        "rotate the proxy certificate on staging",
                        "rotate the proxy certificate on staging",
                        kind="episode",
                        ts=OLD,
                    )
                ],
            )
        )
        hits = run(memory_search(s.recall(), {"query": "rotate the proxy certificate", "limit": 10}))
        assert len([h for h in hits if h["id"].endswith(e["uid"])]) == 1


def test_the_recall_record_left_open_keeps_to_its_cap(tmp_path: Path) -> None:
    from memtrails.core.recall_log import RECALL_TRIM_EVERY

    path = tmp_path / "recall.jsonl"

    async def fill() -> None:
        book = RecallBook(str(path), keep_ms=DAY, keep_bytes=10_000)
        for i in range(RECALL_TRIM_EVERY):
            book.record({"ev": "open", "item": f"stm:{i}"})
        await book.settled()

    run(fill())
    assert os.stat(path).st_size <= 10_000
