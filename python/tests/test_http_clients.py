"""The servers memory talks to over HTTP, stood in for by httpx's mock transport: the embeddings server, the rerank server,
Qdrant's fused search, and the OpenAI-compatible model (embedder.test.ts, reranker.test.ts, qdrant-client.test.ts,
llm.test.ts, ported)."""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any

import httpx
import numpy as np
import pytest

from fakes import run
from memtrails.core.embedder import EmbeddingUnavailable, doc_text, query_text, remote_embedder, resolve_embed_route
from memtrails.core.qdrant_client import QdrantClient, QdrantEndpoint
from memtrails.core.reranker import remote_reranker, reranked_order, resolve_rerank_route
from memtrails.llm import LlmRequest, OpenAiAsk

DIMS = 768


def vec(k: int) -> list[float]:
    return [3.0 if i == k % DIMS else 0.0 for i in range(DIMS)]


class Server:
    def __init__(self, reply: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reply = reply

    def __call__(self, req: httpx.Request) -> httpx.Response:
        raw = req.content
        body = json.loads(gzip.decompress(raw) if req.headers.get("content-encoding") == "gzip" else raw)
        self.calls.append({"url": str(req.url), "headers": dict(req.headers), "body": body})
        return self._reply(body, len(self.calls))  # type: ignore[no-any-return]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def ok(body: dict[str, Any], _n: int = 0, b64: bool = False) -> httpx.Response:
    data = [
        {"index": i, "embedding": base64.b64encode(np.asarray(vec(i), dtype="<f4").tobytes()).decode() if b64 else vec(i)}
        for i in range(len(body["input"]))
    ]
    return httpx.Response(200, json={"data": data})


def route(**over: Any) -> Any:
    r = {"url": "http://embed.test/v1", "api_key": "k", **over}
    return lambda: r


def test_embeddinggemma_prompts() -> None:
    assert query_text("why") == "task: search result | query: why"
    assert doc_text("", "text") == "title: none | text: text"
    assert doc_text("T", "text") == "title: T | text: text"


def test_posts_model_and_input_and_returns_unit_vectors_in_order() -> None:
    s = Server(ok)
    out = run(remote_embedder(route(), transport=s.transport)(["a", "b", "c"]))
    assert s.calls[0]["url"] == "http://embed.test/v1/embeddings"
    assert s.calls[0]["headers"]["authorization"] == "Bearer k"
    assert s.calls[0]["body"] == {"model": "embeddinggemma-300m", "input": ["a", "b", "c"]}
    assert [int(np.argmax(v)) for v in out] == [0, 1, 2]
    assert all(abs(float(np.linalg.norm(v)) - 1) < 1e-6 for v in out)


def test_base64_a_model_name_gzip_and_no_key() -> None:
    s = Server(lambda b, n: ok(b, n, b64=True))
    r = {"url": "http://e.test/v1/embeddings", "model": "embeddinggemma", "gzip": True}
    v = run(remote_embedder(lambda: r, transport=s.transport)(["a"]))[0]
    assert v[0] == 1
    assert s.calls[0]["url"] == "http://e.test/v1/embeddings"
    assert s.calls[0]["headers"]["content-encoding"] == "gzip"
    assert "authorization" not in s.calls[0]["headers"]
    assert s.calls[0]["body"]["model"] == "embeddinggemma"


def test_splits_a_large_batch_and_keeps_order() -> None:
    s = Server(ok)
    texts = [f"text {i}" for i in range(600)]
    out = run(remote_embedder(route(), transport=s.transport)(texts))
    assert len(s.calls) > 1
    assert len(out) == 600
    assert [t for c in s.calls for t in c["body"]["input"]] == texts


def test_retries_a_busy_server_and_refuses_bad_vectors() -> None:
    busy = Server(lambda b, n: httpx.Response(503, text="busy") if n == 1 else ok(b))
    assert len(run(remote_embedder(route(), transport=busy.transport)(["a"]))) == 1

    def bad(body: Any) -> httpx.MockTransport:
        return httpx.MockTransport(lambda _r: httpx.Response(200, json=body))

    with pytest.raises(ValueError, match="dimensions"):
        run(remote_embedder(route(), transport=bad({"data": [{"index": 0, "embedding": [1, 2]}]}))(["a"]))
    with pytest.raises(ValueError, match="zero"):
        run(remote_embedder(route(), transport=bad({"data": [{"index": 0, "embedding": [0.0] * DIMS}]}))(["a"]))
    with pytest.raises(ValueError, match="0 vectors for 1"):
        run(remote_embedder(route(), transport=bad({"data": []}))(["a"]))


def test_says_what_to_configure() -> None:
    with pytest.raises(EmbeddingUnavailable):
        run(remote_embedder(lambda: None)(["a"]))
    assert run(remote_embedder(lambda: None)([])) == []
    assert resolve_embed_route({}) is None
    assert resolve_embed_route({"MEMTRAILS_EMBED_URL": "http://x/v1", "MEMTRAILS_EMBED_KEY": "k", "MEMTRAILS_EMBED_MODEL": "m"}) == {
        "url": "http://x/v1",
        "api_key": "k",
        "model": "m",
    }


RERANK_OK = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": -2}]}


def test_rerank_posts_and_returns_scores_in_order() -> None:
    s = Server(lambda _b, _n: httpx.Response(200, json=RERANK_OK))
    r = {"url": "http://rerank.test/v1", "api_key": "k"}
    assert run(remote_reranker(lambda: r, transport=s.transport)("deploy the api", ["invoices", "buildbot deploys it"], 1000)) == [-2, 0.9]
    assert s.calls[0]["url"] == "http://rerank.test/v1/rerank"
    assert s.calls[0]["body"] == {
        "model": "bge-reranker-v2-m3",
        "query": "deploy the api",
        "documents": ["invoices", "buildbot deploys it"],
    }
    assert s.calls[0]["headers"]["authorization"] == "Bearer k"


def test_rerank_is_nothing_when_it_cannot_help() -> None:
    r = {"url": "http://rerank.test/v1"}

    def answer(status: int, body: Any) -> httpx.MockTransport:
        return httpx.MockTransport(lambda _r: httpx.Response(status, json=body))

    def down(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert run(remote_reranker(lambda: r, transport=answer(503, {}))("q", ["a", "b"], 1000)) is None
    assert run(remote_reranker(lambda: r, transport=answer(200, {"results": RERANK_OK["results"][:1]}))("q", ["a", "b"], 1000)) is None
    assert run(remote_reranker(lambda: r, transport=httpx.MockTransport(down))("q", ["a", "b"], 1000)) is None
    assert run(remote_reranker(lambda: None, transport=answer(200, RERANK_OK))("q", ["a", "b"], 1000)) is None
    assert run(remote_reranker(lambda: r, transport=answer(200, RERANK_OK))("q", ["a", "b"], 50)) is None


def test_reranked_order() -> None:
    assert reranked_order(["a", "b", "c"], [0.1, 0.2, 0.9]) == ["c", "b", "a"]
    assert reranked_order(["a", "b", "c"], None) == ["a", "b", "c"]
    assert resolve_rerank_route({}) is None
    assert resolve_rerank_route({"MEMTRAILS_RERANK_URL": "http://r/v1"}) == {"url": "http://r/v1"}


CUE = {"key": "part", "match": {"value": "cue"}}


def qdrant(groups: Any) -> tuple[QdrantClient, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        b = json.loads(req.content)
        bodies.append(b)
        return httpx.Response(
            200, json={"result": {"groups": [{"id": g["id"], "hits": [{"id": f"{g['id']}-p", "score": g["score"]}]} for g in groups(b)]}}
        )

    return QdrantClient(lambda: QdrantEndpoint("http://qdrant", "k"), transport=httpx.MockTransport(handler)), bodies


def test_cues_are_a_channel_of_their_own() -> None:
    client, bodies = qdrant(lambda _b: [])
    run(
        client.fused_items([[1, 0]], sparse={"indices": [1], "values": [1]}, filter={"must": [{"key": "kind", "match": {"value": "fact"}}]})
    )
    assert [b["using"] for b in bodies] == ["dense", "dense", "bm25"]
    assert CUE in bodies[0]["filter"]["must_not"]
    assert bodies[0]["filter"]["must"] == [{"key": "kind", "match": {"value": "fact"}}]
    assert bodies[1]["filter"]["must"] == [{"key": "kind", "match": {"value": "fact"}}, CUE]
    assert bodies[2]["filter"] == {"must": [{"key": "kind", "match": {"value": "fact"}}]}


def test_a_cue_match_weighs_a_tenth() -> None:
    def groups(b: dict[str, Any]) -> list[dict[str, Any]]:
        if b["using"] != "dense":
            return []
        if '"cue"' in json.dumps(b.get("filter", {}).get("must", [])):
            return [{"id": "b", "score": 0.9}, {"id": "a", "score": 0.1}]
        return [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.1}]

    client, _ = qdrant(groups)
    got = run(client.fused_items([[1, 0]]))
    assert [g["item"] for g in got] == ["a", "b"]
    assert abs(got[0]["score"] - 0.9) < 1e-9
    assert abs(got[1]["score"] - 0.1) < 1e-9


REQ = LlmRequest(purpose="memory", system="s", prompt="p", schema={"type": "object"})


def capture(status: int = 200, body: str = '{"output":[{"content":[{"text":"{}"}]}]}') -> tuple[list[dict[str, str]], httpx.MockTransport]:
    seen: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append({"url": str(req.url), "auth": req.headers.get("authorization", "")})
        return httpx.Response(status, text=body)

    return seen, httpx.MockTransport(handler)


def test_openai_key_goes_to_openai_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("MEMTRAILS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("MEMTRAILS_LLM_BASE_URL", raising=False)
    seen, t = capture()
    assert "answer" in run(OpenAiAsk(model="m", transport=t)(REQ))
    assert seen[0] == {"url": "https://api.openai.com/v1/responses", "auth": "Bearer openai-key"}
    seen_b, tb = capture()
    other = run(OpenAiAsk(model="m", base_url="https://llm.example/v1", transport=tb)(REQ))
    assert "no API key" in other["unavailable"]
    assert seen_b == []
    monkeypatch.setenv("MEMTRAILS_LLM_API_KEY", "own-key")
    run(OpenAiAsk(model="m", base_url="https://llm.example/v1", transport=tb)(REQ))
    assert seen_b[0]["auth"] == "Bearer own-key"


def test_says_what_the_provider_said_secrets_taken_out() -> None:
    key = "sk-" + "abcdefghijklmnop1234"
    _, t = capture(400, json.dumps({"error": {"message": f"model gpt-x does not exist; key {key}"}}))
    r = run(OpenAiAsk(model="m", api_key="k", transport=t)(REQ))
    assert "model gpt-x does not exist" in r["unavailable"]
    assert key not in json.dumps(r)


def test_retries_then_says_it_did_not_answer() -> None:
    seen, t = capture(503, "busy")
    r = run(OpenAiAsk(model="m", api_key="k", retries=0, transport=t)(REQ))
    assert r == {"unavailable": "the model did not answer (HTTP 503)"}
    assert len(seen) == 1
