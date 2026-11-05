"""The Python answer key (conformance/generate-python.ts): every request the reference makes of a language model for fixed
inputs — system text, prompt and JSON Schema byte for byte, with its token budget — the fused search's arithmetic, more
of Date.parse, and (with the model at hand) the local embedder's vectors — the same to 1e-6 on the same ONNX Runtime (the reference's
pinned 1.23.2)."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from conftest import FIXTURES, fixture
from memtrails._js import date_parse, stringify
from memtrails.core.consolidate import ConsolidateDeps, Consolidator
from memtrails.core.extract import extract_window, observe_window
from memtrails.core.gather import memory_gather
from memtrails.core.profiles import write_profiles
from memtrails.core.qdrant_client import QdrantClient, QdrantEndpoint, point_id
from memtrails.core.recall import RecallDeps
from memtrails.core.store import ltm_log, stm_log
from memtrails.llm import LlmRequest

PROMPTS = fixture("python-prompts.json")
INPUTS = PROMPTS["inputs"]


def as_dict(req: LlmRequest) -> dict[str, Any]:
    return {
        "purpose": req.purpose,
        "name": req.name,
        "system": req.system,
        "prompt": req.prompt,
        "schema": req.schema,
        "maxTokens": req.max_tokens,
        "timeoutMs": req.timeout_ms,
        "consequence": req.consequence,
    }


def same(got: list[dict[str, Any]], want: list[dict[str, Any]]) -> None:
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        for k in ("purpose", "name", "system", "prompt", "maxTokens", "timeoutMs", "consequence"):
            assert g[k] == w[k], (w["name"], k)
        assert stringify(g["schema"]) == stringify(w["schema"]), w["name"]


class Capture:
    def __init__(self, answer: Any) -> None:
        self.asked: list[dict[str, Any]] = []
        self._answer = answer

    async def __call__(self, req: LlmRequest) -> dict[str, Any]:
        self.asked.append(as_dict(req))
        return self._answer(req) if callable(self._answer) else self._answer


NOTHING = {"answer": {"segment_summary": "", "episodes": [], "items": [], "observations": []}}


async def test_reading_a_window_asks_the_same() -> None:
    conversation, mail, meta, known, long = INPUTS["conversation"], INPUTS["mail"], INPUTS["meta"], INPUTS["known"], INPUTS["long"]
    ask = Capture(NOTHING)
    await extract_window(conversation, meta, ask=ask, known=known)
    await extract_window([mail], {"cwd": "/srv/ops", "engine": "email", "title": "cert"}, ask=ask, observations=False)
    observe = Capture({"answer": {"observations": []}})
    await observe_window(conversation[1:], conversation[:1], meta, 300, ask=observe, known=known)
    part = {"n": 0}

    def folded(_req: LlmRequest) -> dict[str, Any]:
        part["n"] += 1
        n = part["n"]
        return {
            "answer": {
                "segment_summary": f"part {n}",
                "items": [],
                "observations": [],
                "episodes": [
                    {
                        "request": 1,
                        "outcome": "success",
                        "keep": True,
                        "title": f"Migration, part {n}",
                        "summary": f"what part {n} shows",
                        "cues": [],
                        "concepts": [],
                    }
                ],
            }
        }

    long_ask = Capture(folded)
    await extract_window([long], meta, ask=long_ask)
    same(ask.asked[:2] + observe.asked + long_ask.asked, PROMPTS["extraction"])


class NoIndex:
    async def upsert(self, *_: Any, **__: Any) -> None:
        return None

    async def delete_items(self, *_: Any) -> None:
        return None

    async def delete_parts(self, *_: Any) -> None:
        return None

    async def set_payload(self, *_: Any, **__: Any) -> None:
        return None


async def embed(texts: list[str], _ms: Any = None) -> list[np.ndarray]:
    return [np.eye(768, dtype=np.float32)[0] for _ in texts]


def deps(tmp: Path, name: str, ask: Any) -> ConsolidateDeps:
    return ConsolidateDeps(
        client=NoIndex(), embed=embed, stm=stm_log(str(tmp / "stm.jsonl")), ltm=ltm_log(str(tmp / name)), ask=ask, now=lambda: INPUTS["now"]
    )


async def test_reconciling_cues_and_profiles_ask_the_same(tmp_path: Path) -> None:
    asked: list[dict[str, Any]] = []
    relate = Capture({"answer": {"pairs": []}})
    await Consolidator(deps(tmp_path, "a.jsonl", relate)).relate(INPUTS["pairs"])
    cues = Capture({"answer": {"items": []}})
    d = deps(tmp_path, "b.jsonl", cues)
    d.ltm.put(INPUTS["cueItems"])
    await Consolidator(d).fill_cues()
    profiles = Capture({"unavailable": "enough"})
    p = deps(tmp_path, "c.jsonl", profiles)
    p.ltm.put(INPUTS["profileItems"])
    await write_profiles(p)
    asked += relate.asked + cues.asked + profiles.asked
    same(asked, PROMPTS["knowledge"])


class Found:
    def __init__(self, found: list[dict[str, Any]]) -> None:
        self.found = found

    async def fused_items(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return [{"item": str(p["item"]), "score": 1 - i / 10} for i, p in enumerate(self.found)]

    async def retrieve(self, ids: list[str], _with_vector: bool = False) -> list[dict[str, Any]]:
        return [{"id": point_id(str(p["item"]), 0), "payload": p} for p in self.found if point_id(str(p["item"]), 0) in ids]


async def test_gathering_asks_the_same() -> None:
    d = RecallDeps(client=Found(INPUTS["found"]), embed=embed)
    plain = Capture({"answer": {"keep": [], "searches": []}})
    await memory_gather(d, {"query": "Since when does Dana play the cello?"}, ask=plain)
    noted = Capture({"answer": {"keep": [], "searches": [], "notes": []}})
    await memory_gather(d, {"query": "Since when does Dana play the cello?"}, ask=noted, notes=True)
    same(plain.asked + noted.asked, PROMPTS["gathering"])


FUSION = fixture("python-fusion.json")


async def test_the_fused_search_adds_up_the_same() -> None:
    bodies: list[Any] = []

    def handler(req: httpx.Request) -> httpx.Response:
        b = json.loads(req.content)
        bodies.append(b)
        cue = '"cue"' in json.dumps(b.get("filter", {}).get("must", []))
        key = "bm25" if b["using"] == "bm25" else f"{'cue' if cue else 'dense'}-{0 if b['query'][0] == 1 else 1}"
        return httpx.Response(
            200,
            json={
                "result": {"groups": [{"id": g["id"], "hits": [{"id": g["point"], "score": g["score"]}]} for g in FUSION["groups"][key]]}
            },
        )

    client = QdrantClient(lambda: QdrantEndpoint("http://qdrant.test", "k"), http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    q = FUSION["query"]
    fused = await client.fused_items(q["dense"], sparse=q["sparse"], filter=q["filter"], limit=q["limit"])
    dense_only = await client.fused_items([[1, 0]], limit=10)

    def close(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
        return [(x["item"], x.get("point")) for x in a] == [(x["item"], x.get("point")) for x in b] and all(
            abs(x["score"] - y["score"]) < 1e-12 for x, y in zip(a, b, strict=True)
        )

    assert close(fused, FUSION["fused"])
    assert close(dense_only, FUSION["denseOnly"])
    assert bodies == FUSION["bodies"]


@pytest.fixture
def utc() -> Iterator[None]:
    if not hasattr(time, "tzset"):  # Windows: the zone cannot be changed here, only used when it is UTC already
        if time.timezone != 0 or time.daylight:
            pytest.skip("local time is not UTC and cannot be set to it in this process")
        yield
        return
    before = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    yield
    if before is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = before
    time.tzset()


@pytest.mark.parametrize("case", fixture("python-dates.json")["cases"], ids=lambda c: repr(c["s"]))
def test_date_parse_reads_as_v8(utc: None, case: dict[str, Any]) -> None:
    got = date_parse(case["s"])
    assert (None if math.isnan(got) else got) == case["ms"]


@pytest.mark.skipif(os.environ.get("MEMTRAILS_INTEGRATION") != "1", reason="needs the real model (MEMTRAILS_INTEGRATION=1)")
async def test_the_local_model_gives_the_same_vectors() -> None:
    from memtrails.core.local_embedder import LocalEmbedder
    from memtrails.core.model import ensure_model

    key = json.loads((FIXTURES / "python-embeddings.json").read_text(encoding="utf-8"))
    e = LocalEmbedder(os.environ.get("MEMTRAILS_MODEL_DIR") or await ensure_model(), threads=1)
    try:
        for got, want in zip(await e.embed(key["texts"]), key["vectors"], strict=True):
            assert np.max(np.abs(got - np.asarray(want, dtype=np.float32))) <= 1e-6
        words = await e.words(key["words"]["texts"], "query")
        for got_w, want_w in zip(words, key["words"]["vectors"], strict=True):
            assert got_w.words == want_w["words"]
            assert got_w.dims == want_w["dims"]
            for k, head in enumerate(want_w["head"]):
                assert np.max(np.abs(got_w.data[k * got_w.dims : k * got_w.dims + 32] - np.asarray(head))) <= 1e-6
    finally:
        await e.close()
