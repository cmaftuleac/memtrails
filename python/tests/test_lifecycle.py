"""open_memory's life (lifecycle.test.ts, ported): what an open that fails leaves behind, one process a home, the index
written again when its layout is not the logs', jobs one at a time, close() stopping what runs, forgetting a record
whole — over a stand-in Qdrant and a stand-in embedding server (httpx's mock transport), and a stand-in Qdrant binary.
No model, no network."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import memtrails.memory as memory_module
from memtrails import open_memory
from memtrails._js import stringify
from memtrails.core.qdrant_client import INDEX_LAYOUT
from memtrails.core.qdrant_process import alive, boot_time
from memtrails.core.store import ltm_log
from memtrails.records import record_uid

DIMS = 768
Q = "http://qdrant.test"
E = "http://embed.test/v1"
R = "http://rerank.test/v1"


class Servers:
    """Qdrant's REST API as far as memory uses it, in memory (points by id, deletes by item, every key point a search
    hit), an OpenAI-style embeddings server, and a rerank server."""

    def __init__(self) -> None:
        self.exists = False
        self.points: dict[str, dict[str, Any]] = {}
        self.delay = 0.0
        #: An upsert applied at once and answered this much later (its answer slow on the way back).
        self.answer_delay = 0.0
        self.upserts = 0
        self.drops = 0
        self.embeds = 0
        self.reranks = 0

    async def __call__(self, req: httpx.Request) -> httpx.Response:
        b = json.loads(req.content) if req.content else {}
        url = str(req.url)
        if url.startswith(E):
            self.embeds += 1
            data = []
            for i, t in enumerate(b["input"]):
                h = 7
                for ch in t.encode("utf-16-le"):
                    h = (h * 31 + ch) % 100_003
                data.append({"index": i, "embedding": [((h + k * 7919) % 1000) / 1000 + 0.001 for k in range(DIMS)]})
            return httpx.Response(200, json={"data": data})
        if url.startswith(R):
            self.reranks += 1
            return httpx.Response(200, json={"results": [{"index": i, "relevance_score": 1 / (i + 1)} for i in range(len(b["documents"]))]})
        return await self.qdrant(req, b)

    async def qdrant(self, req: httpx.Request, b: dict[str, Any]) -> httpx.Response:
        path = req.url.path

        def send(status: int, result: Any) -> httpx.Response:
            return httpx.Response(status, json={"result": result})

        if not path.startswith("/collections/"):
            return send(200, {"collections": []})
        rest = path.split("/", 3)[3] if path.count("/") >= 3 else ""
        rest = f"/{rest}" if rest else ""
        if not rest:
            if req.method == "GET":
                return (
                    send(200, {"config": {"optimizer_config": {"default_segment_number": 2}}, "payload_schema": {}})
                    if self.exists
                    else send(404, None)
                )
            if req.method == "PUT":
                self.exists = True
                return send(200, True)
            if req.method == "DELETE":
                self.exists = False
                self.points.clear()
                self.drops += 1
            return send(200, True)
        if not self.exists:
            return send(404, None)
        if rest == "/index":
            return send(200, True)
        if rest == "/points" and req.method == "PUT":
            self.upserts += 1
            await asyncio.sleep(self.delay)
            if not self.exists:
                return send(404, None)
            for p in b["points"]:
                self.points[p["id"]] = {"id": p["id"], "payload": p["payload"]}
            await asyncio.sleep(self.answer_delay)
            return send(200, True)
        if rest == "/points":
            return send(200, [self.points[i] for i in b["ids"] if i in self.points])
        if rest == "/points/delete":
            must = b["filter"]["must"]
            items = next((c["match"].get("any", []) for c in must if c["key"] == "item"), [])
            part = next((c["match"].get("value") for c in must if c["key"] == "part"), None)
            for pid in [
                k for k, p in self.points.items() if p["payload"].get("item") in items and (not part or p["payload"].get("part") == part)
            ]:
                del self.points[pid]
            return send(200, True)
        if rest == "/points/count":
            return send(200, {"count": len(self.points)})
        if rest in ("/points/payload",):
            return send(200, True)
        if rest == "/points/scroll":
            return send(200, {"points": []})
        if rest == "/points/query":
            return send(200, {"points": []})
        if rest == "/points/query/batch":
            return send(200, [{"points": []} for _ in b["searches"]])
        if rest == "/points/query/groups":
            groups: dict[str, dict[str, Any]] = {}
            for p in self.points.values():
                if p["payload"].get("part") == "key" or p["payload"].get("tier") == "stm":
                    groups.setdefault(str(p["payload"].get("item")), p)
            return send(
                200,
                {
                    "groups": [
                        {"id": item, "hits": [{"id": p["id"], "score": 0.9 - i * 0.01}]}
                        for i, (item, p) in enumerate(list(groups.items())[: int(b.get("limit", 10))])
                    ]
                },
            )
        return send(200, True)

    def stm_items(self) -> list[str]:
        return list(dict.fromkeys(str(p["payload"]["item"]) for p in self.points.values() if p["payload"].get("tier") == "stm"))


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch) -> Servers:
    # Nothing is fetched: a test that reaches for the model or the Qdrant binary fails instead of downloading them.
    async def no_model(**_: Any) -> str:
        raise RuntimeError("test: no model download")

    async def no_qdrant(**_: Any) -> str:
        raise RuntimeError("test: no Qdrant download")

    monkeypatch.setattr(memory_module, "ensure_model", no_model)
    monkeypatch.setattr(memory_module, "ensure_qdrant", no_qdrant)
    for k in ("MEMTRAILS_EMBED_URL", "MEMTRAILS_RERANK_URL"):
        monkeypatch.delenv(k, raising=False)
    return Servers()


def opts(s: Servers, **more: Any) -> dict[str, Any]:
    return {
        "embedding": {"server": {"url": E}},
        "qdrant": {"url": Q},
        "rerank": False,
        "http": httpx.AsyncClient(transport=httpx.MockTransport(s)),
        **more,
    }


def records(n: int, start: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "id": f"r{start + i}",
            "text": f"Record number {start + i}: the gateway certificate expired on day {start + i}.",
            "at": "2025-01-02",
        }
        for i in range(n)
    ]


def broken_qdrant_binary(tmp: Path) -> str:
    """A Qdrant stand-in binary that comes up, proves its key, and fails the collection call."""
    path = tmp / "fake-qdrant.py"
    path.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""
        import http.server, os, re, signal, socketserver, sys
        args = sys.argv[1:]
        port = int(re.search(r"http_port: (\\d+)", open(args[args.index("--config-path") + 1]).read()).group(1))
        with open("spawns.log", "a") as f:
            f.write(f"{os.getpid()}\\n")
        key = os.environ.get("QDRANT__SERVICE__API_KEY")
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def answer(self):
                if self.path != "/readyz" and self.headers.get("api-key") != key:
                    self.send_response(401); self.end_headers(); self.wfile.write(b"{}"); return
                if self.path in ("/readyz", "/collections"):
                    self.send_response(200); self.end_headers(); self.wfile.write(b'{"result":{"collections":[]}}'); return
                self.send_response(500); self.end_headers(); self.wfile.write(b'{"status":{"error":"broken"}}')
            do_GET = do_PUT = do_POST = do_DELETE = do_PATCH = answer
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        socketserver.TCPServer(("127.0.0.1", port), H).serve_forever()  # not HTTPServer: its reverse DNS lookup hangs on some hosts
    """)
    )
    path.chmod(0o755)
    return str(path)


async def until(ok: Any, s: float = 5) -> None:
    end = time.monotonic() + s
    while not ok():
        assert time.monotonic() < end, "timed out"
        await asyncio.sleep(0.01)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX")
async def test_a_failed_open_stops_its_qdrant_and_frees_the_home(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    binary = broken_qdrant_binary(tmp_path)
    # The embedding server stood in for; memory's own Qdrant reached over loopback.
    http = httpx.AsyncClient(mounts={E: httpx.MockTransport(servers)}, trust_env=False)
    with pytest.raises(Exception, match="500"):
        await open_memory(home, **{**opts(servers, qdrant={"binary": binary}), "http": http})
    pid = int((home / "qdrant" / "spawns.log").read_text().split()[0])
    await until(lambda: not alive(pid))
    assert not (home / "memory.lock").exists()
    with pytest.raises(Exception, match="500"):
        await open_memory(home, **{**opts(servers, qdrant={"binary": binary}), "http": http})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX")
async def test_its_own_qdrant_is_never_called_through_a_proxy(tmp_path: Path, servers: Servers, monkeypatch: pytest.MonkeyPatch) -> None:
    # A proxy the environment names, listening: it must hear nothing — its key and documents go to Qdrant directly.
    heard: list[bytes] = []

    async def listen(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        heard.append(await reader.read(200))
        writer.close()

    proxy = await asyncio.start_server(listen, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(k, url)
    for k in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(k, raising=False)
    binary = broken_qdrant_binary(tmp_path)
    try:
        # Your client, which takes the environment's proxies (httpx's default), and memory's own.
        for http in (httpx.AsyncClient(mounts={E: httpx.MockTransport(servers)}), None):
            home = tmp_path / ("yours" if http else "own")
            given = {**opts(servers, qdrant={"binary": binary}), "http": http}
            # Up, proven ours, and refusing the collection call (the stand-in's 500): reached directly.
            with pytest.raises(Exception, match="500"):
                await open_memory(home, **given)
    finally:
        proxy.close()
        await proxy.wait_closed()
    assert heard == []


async def test_a_binary_that_is_not_there(tmp_path: Path, servers: Servers) -> None:
    with pytest.raises(RuntimeError, match=r"could not be started.*ENOENT"):
        await open_memory(tmp_path / "home", **opts(servers, qdrant={"binary": "memtrails-no-such-qdrant"}))
    assert not (tmp_path / "home" / "memory.lock").exists()


async def test_one_process_a_home(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    a = await open_memory(home, **opts(servers))
    with pytest.raises(RuntimeError, match=f"is open in process {os.getpid()}"):
        await open_memory(home, **opts(servers))
    await a.close()
    assert not (home / "memory.lock").exists()
    await (await open_memory(home, **opts(servers))).close()


async def test_takes_over_a_dead_or_pre_reboot_lock(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    home.mkdir()
    dead = int(subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True).stdout)
    assert not alive(dead)
    (home / "memory.lock").write_text(stringify({"pid": dead, "boot": boot_time()}))
    await (await open_memory(home, **opts(servers))).close()
    (home / "memory.lock").write_text(stringify({"pid": os.getpid(), "boot": boot_time() - 3_600_000}))
    await (await open_memory(home, **opts(servers))).close()


async def test_keeps_the_home_private(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o755)
    await (await open_memory(home, **opts(servers))).close()
    assert home.stat().st_mode & 0o777 == 0o700
    fresh = tmp_path / "nested" / "mem"
    await (await open_memory(fresh, **opts(servers))).close()
    assert fresh.stat().st_mode & 0o777 == 0o700


async def test_the_index_written_again_in_another_layout(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    async with await open_memory(home, **opts(servers)) as m:
        await m.add(records(3))
    assert json.loads((home / "index.json").read_text()) == {"layout": INDEX_LAYOUT}
    (home / "index.json").unlink()
    servers.points["junk"] = {"id": "junk", "payload": {"item": "stm:junk", "tier": "stm"}}
    await (await open_memory(home, **opts(servers))).close()
    assert "junk" not in servers.points
    assert len(servers.stm_items()) == 3
    drops = servers.drops
    await (await open_memory(home, **opts(servers))).close()
    assert servers.drops == drops
    (home / "index.json").write_text(stringify({"layout": INDEX_LAYOUT - 1}))
    servers.points["junk"] = {"id": "junk", "payload": {"item": "stm:junk", "tier": "stm"}}
    await (await open_memory(home, **opts(servers))).close()
    assert servers.drops == drops + 1
    assert "junk" not in servers.points
    assert len(servers.stm_items()) == 3


async def test_forgets_a_record_whole(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    async with await open_memory(home, **opts(servers)) as m:
        await m.add({"id": "secret", "text": "The door code is 4711.", "at": "2025-01-02"})
    uid = record_uid({"id": "secret"})
    ep = {
        "uid": f"ep:{uid}",
        "kind": "episode",
        "scope": "project",
        "project": "",
        "engine": "record",
        "session": "",
        "cwd": None,
        "where": "",
        "ts": 1,
        "firstTs": 1,
        "title": "door code",
        "summary": "The door code is 4711.",
        "body": "The door code is 4711.",
        "files": [],
        "appliesWhen": "",
        "confidence": "observed",
        "importance": 2,
        "status": "active",
        "provenance": [uid],
        "seen": 1,
        "fingerprint": "x",
    }
    log = ltm_log(str(home / "knowledge.jsonl"))
    log.load()
    log.put([ep])

    def items() -> list[str]:
        return [str(p["payload"]["item"]) for p in servers.points.values()]

    async with await open_memory(home, **opts(servers)) as m:
        await m.rebuild()
        assert f"ep:{uid}" in items()
        assert await m.remove(["secret"]) == 1
        assert f"ep:{uid}" not in items()
        await m.rebuild()
        assert f"ep:{uid}" not in items()
    log.load()
    assert log.get(f"ep:{uid}") is None
    assert '"removed":true' in (home / "knowledge.jsonl").read_text()


async def test_a_rebuild_asked_for_while_one_runs_waits(tmp_path: Path, servers: Servers) -> None:
    async with await open_memory(tmp_path / "home", **opts(servers)) as m:
        await m.add(records(70))
        servers.delay = 0.02
        drops = servers.drops
        first = asyncio.ensure_future(m.rebuild())
        await until(lambda: servers.drops > drops and bool(servers.stm_items()))
        await asyncio.gather(first, m.rebuild())
        assert len(servers.stm_items()) == 70
        assert not (tmp_path / "home" / "rebuild.json").exists()


async def test_two_consolidations_never_ask_at_once(tmp_path: Path, servers: Servers) -> None:
    state = {"now": 0, "most": 0}

    async def ask(_: Any) -> dict[str, Any]:
        state["now"] += 1
        state["most"] = max(state["most"], state["now"])
        await asyncio.sleep(0.05)
        state["now"] -= 1
        return {"unavailable": "test: the model is away"}

    async with await open_memory(tmp_path / "home", **opts(servers, llm=ask, concurrency=1)) as m:
        await m.add({"id": "r", "text": "Something to read.", "at": "2025-01-02"})
        await asyncio.gather(m.consolidate(), m.consolidate())
    assert state["most"] == 1


async def test_close_stops_a_running_rebuild_and_the_next_open_finishes_it(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    m = await open_memory(home, **opts(servers))
    await m.add(records(200))
    servers.delay = 0.03
    upserts = servers.upserts
    running = asyncio.ensure_future(m.rebuild())
    await until(lambda: servers.upserts > upserts)
    await m.close()
    await running  # stopped, not failed
    after = servers.upserts
    await asyncio.sleep(0.3)
    assert servers.upserts == after
    assert (home / "rebuild.json").exists()
    assert len(servers.stm_items()) < 200
    with pytest.raises(RuntimeError, match="closed"):
        await m.rebuild()
    servers.delay = 0
    async with await open_memory(home, **opts(servers)) as again:
        assert not (home / "rebuild.json").exists()
        assert len(servers.stm_items()) == 200
        assert again.stats()["records"] == 200


async def test_the_recall_record_is_trimmed_when_memory_opens(tmp_path: Path, servers: Servers) -> None:
    home = tmp_path / "home"
    home.mkdir()
    now = int(time.time() * 1000)
    (home / "recall.jsonl").write_text(
        stringify({"t": now - 200 * 86_400_000, "ev": "open", "item": "stm:old"})
        + "\n"
        + stringify({"t": now, "ev": "open", "item": "stm:new"})
        + "\n"
    )
    async with await open_memory(home, **opts(servers)) as m:
        await m.add(records(2))
        await m.search({"query": "certificate"})
    lines = [json.loads(x) for x in (home / "recall.jsonl").read_text().strip().split("\n")]
    assert [x.get("item", x["ev"]) for x in lines] == ["stm:new", "search"]


async def test_servers_the_environment_names(tmp_path: Path, servers: Servers, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMTRAILS_EMBED_URL", E)
    monkeypatch.setenv("MEMTRAILS_RERANK_URL", R)
    async with await open_memory(tmp_path / "home", qdrant={"url": Q}, http=httpx.AsyncClient(transport=httpx.MockTransport(servers))) as m:
        await m.add(records(3))
        assert servers.embeds > 0
        await m.search({"query": "certificate"})
        assert servers.reranks == 1


async def test_adds_nothing_twice_and_refuses_what_it_cannot_keep(tmp_path: Path, servers: Servers) -> None:
    async def stream() -> AsyncIterator[dict[str, Any]]:
        for r in records(10, 4):
            yield r

    async with await open_memory(tmp_path / "home", **opts(servers)) as m:
        assert await m.add(records(4)) == {"added": 4, "unchanged": 0}
        assert await m.add(records(4)) == {"added": 0, "unchanged": 4}
        assert await m.import_(stream(), batch=3) == {"added": 10, "unchanged": 0}
        assert await m.import_(iter(records(2, 20))) == {"added": 2, "unchanged": 0}
        assert m.stats()["records"] == 16
        with pytest.raises(TypeError, match="id"):
            await m.add({"id": "", "text": "x"})
        with pytest.raises(TypeError, match="time"):
            await m.add({"id": "a", "text": "x", "at": "yesterday-ish"})
        tools = {t.name: t for t in m.tools()}
        assert set(tools) == {"memory_search", "memory_open", "memory_explore"}
        assert "certificate" in (await tools["memory_search"].handler({"query": "certificate"}))["content"][0]["text"]


async def test_close_waits_for_the_recall_records_trim(tmp_path: Path, servers: Servers, monkeypatch: pytest.MonkeyPatch) -> None:
    from memtrails.core.recall_log import RECALL_TRIM_EVERY, RecallBook

    go = threading.Event()
    trims: list[str] = []
    real = RecallBook.trim

    def slow(book: RecallBook) -> int:
        if trims:  # the open's own trim is let through
            go.wait(5)
        trims.append("done")
        return real(book)

    monkeypatch.setattr(RecallBook, "trim", slow)
    m = await open_memory(tmp_path / "home", **opts(servers))
    assert m._book is not None
    for i in range(RECALL_TRIM_EVERY):
        m._book.record({"ev": "open", "item": f"stm:{i}"})
    closing = asyncio.ensure_future(m.close())
    await asyncio.sleep(0.05)
    assert not closing.done()
    go.set()
    await closing
    assert trims == ["done", "done"]


async def test_an_add_beside_a_rebuild_is_indexed(tmp_path: Path, servers: Servers) -> None:
    async with await open_memory(tmp_path / "home", **opts(servers)) as m:
        await m.add(records(40))
        servers.answer_delay = 0.1
        upserts = servers.upserts
        adding = asyncio.ensure_future(m.add(records(1, 100)))
        await until(lambda: servers.upserts > upserts)  # its points in the index, the answer on its way back
        # A rebuild asked for meanwhile drops the index only once the add is done and logged, and indexes it with the
        # rest — not the add's points dropped and its record logged after the rebuild read the log.
        await asyncio.gather(adding, m.rebuild())
        assert len(servers.stm_items()) == 41
        assert f"stm:{record_uid({'id': 'r100'})}" in servers.stm_items()


async def test_close_waits_for_the_calls_in_flight_and_refuses_the_next(tmp_path: Path, servers: Servers) -> None:
    m = await open_memory(tmp_path / "home", **opts(servers))
    await m.add(records(3))
    servers.delay = 0.1
    adding = asyncio.ensure_future(m.add(records(2, 10)))
    await asyncio.sleep(0.02)
    await m.close()
    assert adding.done()
    assert (await adding)["added"] == 2
    for call in (lambda: m.search({"query": "certificate"}), lambda: m.add(records(1, 20)), lambda: m.open("stm:x")):
        with pytest.raises(RuntimeError, match="memory is closed"):
            await call()
