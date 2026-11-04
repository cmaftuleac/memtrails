"""Memory's own Qdrant, run safely (qdrant-process-safety.test.ts, ported): a binary that is not there fails the start, a
start that fails leaves nothing running or scheduled, the child sees none of the host's secrets, what answers on the
port must prove it is ours before it gets the key, its log is private, and a Qdrant a dead host left on the storage is
stopped — an unrelated process with its pid never. Over a stand-in Qdrant (a Python script) on loopback."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from memtrails._js import stringify
from memtrails.core.qdrant_process import QdrantProcess, alive, boot_time

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and permissions")


def fake_qdrant(root: Path, mode: str) -> str:
    """A stand-in Qdrant: ``ok`` checks its key like Qdrant; ``anykey`` takes any key; ``exit`` dies at once. Each start
    is a line in spawns.log, and its environment is env.json — both in the folder it runs in."""
    path = root / f"fake-qdrant-{mode}.py"
    path.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(f"""
        import http.server, json, os, re, signal, sys
        args = sys.argv[1:]
        port = int(re.search(r"http_port: (\\d+)", open(args[args.index("--config-path") + 1]).read()).group(1))
        with open("spawns.log", "a") as f:
            f.write(f"{{os.getpid()}}\\n")
        json.dump(dict(os.environ), open("env.json", "w"))
        if {mode!r} == "exit":
            sys.exit(1)
        key = os.environ.get("QDRANT__SERVICE__API_KEY")
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path != "/readyz" and {mode!r} == "ok" and self.headers.get("api-key") != key:
                    self.send_response(401); self.end_headers(); self.wfile.write(b"{{}}"); return
                self.send_response(200); self.end_headers(); self.wfile.write(b'{{"result":{{"collections":[]}}}}')
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
    """)
    )
    path.chmod(0o755)
    return str(path)


def spawns(d: str) -> list[int]:
    p = os.path.join(d, "spawns.log")
    return [int(x) for x in Path(p).read_text().split()] if os.path.exists(p) else []


async def until(ok: Any, s: float = 5) -> bool:
    end = time.monotonic() + s
    while not ok():
        if time.monotonic() > end:
            return False
        await asyncio.sleep(0.02)
    return True


def gone(pid: int) -> bool:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return not alive(pid)


def test_a_binary_that_is_not_there(tmp_path: Path) -> None:
    async def go() -> None:
        p = QdrantProcess("memtrails-no-such-qdrant", str(tmp_path / "q"))
        with pytest.raises(RuntimeError, match=r"could not be started.*ENOENT"):
            await p.start()
        assert p.current() is None

    asyncio.run(go())


def test_no_restart_when_it_dies_while_starting(tmp_path: Path) -> None:
    async def go() -> None:
        p = QdrantProcess(fake_qdrant(tmp_path, "exit"), str(tmp_path / "q"))
        with pytest.raises(RuntimeError, match="exited"):
            await p.start()
        await asyncio.sleep(1.5)
        assert len(spawns(p.directory)) == 1

    asyncio.run(go())


def test_refuses_what_takes_any_key(tmp_path: Path) -> None:
    async def go() -> None:
        p = QdrantProcess(fake_qdrant(tmp_path, "anykey"), str(tmp_path / "q"))
        with pytest.raises(RuntimeError, match="not this memory's Qdrant: it took a wrong key"):
            await p.start()
        pid = spawns(p.directory)[0]
        assert await until(lambda: gone(pid))
        assert p.current() is None

    asyncio.run(go())


def test_started_again_when_it_dies_after_it_was_up(tmp_path: Path) -> None:
    async def go() -> None:
        p = QdrantProcess(fake_qdrant(tmp_path, "ok"), str(tmp_path / "q"))
        await p.start()
        os.kill(spawns(p.directory)[0], signal.SIGKILL)
        assert await until(lambda: len(spawns(p.directory)) == 2 and p.current() is not None, 8)
        await p.stop_and_wait()

    asyncio.run(go())


def test_none_of_the_hosts_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMTRAILS_TEST_SECRET", "hunter2")
    monkeypatch.setenv("QDRANT__SERVICE__HOST", "0.0.0.0")

    async def go() -> None:
        p = QdrantProcess(fake_qdrant(tmp_path, "ok"), str(tmp_path / "q"))
        e = await p.start()
        env = json.loads((Path(p.directory) / "env.json").read_text())
        assert "MEMTRAILS_TEST_SECRET" not in env
        assert "QDRANT__SERVICE__HOST" not in env
        assert env["QDRANT__SERVICE__API_KEY"] == e.api_key
        assert env["PATH"] == os.environ["PATH"]
        await p.stop_and_wait()

    asyncio.run(go())


def test_its_folder_and_log_are_private(tmp_path: Path) -> None:
    async def go() -> None:
        d = tmp_path / "q"
        d.mkdir()
        (d / "qdrant.log").write_text("old\n")
        (d / "qdrant.log").chmod(0o644)
        p = QdrantProcess(fake_qdrant(tmp_path, "ok"), str(d))
        await p.start()
        assert (d / "qdrant.log").stat().st_mode & 0o777 == 0o600
        fresh = QdrantProcess(fake_qdrant(tmp_path, "ok"), str(tmp_path / "fresh"))
        await fresh.start()
        assert (tmp_path / "fresh").stat().st_mode & 0o777 == 0o700
        assert (tmp_path / "fresh" / "qdrant.log").stat().st_mode & 0o777 == 0o600
        await p.stop_and_wait()
        await fresh.stop_and_wait()

    asyncio.run(go())


HOST = """
import asyncio, sys
sys.path.insert(0, {src!r})
from memtrails.core.qdrant_process import QdrantProcess
async def main():
    p = QdrantProcess(sys.argv[1], sys.argv[2])
    await p.start()
    print("up", flush=True)
    if sys.argv[3] == "exit":
        return
    await asyncio.sleep(3600)
asyncio.run(main())
"""


def host(tmp_path: Path, d: str, binary: str, how: str) -> tuple[subprocess.Popen[str], int]:
    """A host process that starts a Qdrant on ``d``, says so, then exits without stopping it, or waits."""
    script = tmp_path / "host.py"
    script.write_text(HOST.format(src=str(Path(__file__).resolve().parents[1] / "src")))
    child = subprocess.Popen([sys.executable, str(script), binary, d, how], stdout=subprocess.PIPE, text=True)
    assert child.stdout is not None
    assert "up" in child.stdout.readline()
    return child, spawns(d)[0]


def test_a_host_that_exits_stops_its_qdrant(tmp_path: Path) -> None:
    d = str(tmp_path / "q")
    child, pid = host(tmp_path, d, fake_qdrant(tmp_path, "ok"), "exit")
    child.wait(20)
    assert asyncio.run(until(lambda: not alive(pid)))


def test_a_host_killed_outright_leaves_a_qdrant_the_next_start_stops(tmp_path: Path) -> None:
    d = str(tmp_path / "q")
    binary = fake_qdrant(tmp_path, "ok")
    child, pid = host(tmp_path, d, binary, "hang")
    child.kill()
    child.wait(10)
    time.sleep(0.2)
    assert alive(pid)

    async def go() -> None:
        nxt = QdrantProcess(binary, d)
        await nxt.start()
        assert not alive(pid)
        assert len(spawns(d)) == 2
        await nxt.stop_and_wait()

    asyncio.run(go())


def test_never_stops_an_unrelated_process_with_the_pid(tmp_path: Path) -> None:
    d = tmp_path / "q"
    d.mkdir()
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
    try:
        (d / "qdrant.pid").write_text(stringify({"pid": stranger.pid, "config": str(d / "config.yaml"), "boot": boot_time()}))

        async def go() -> None:
            p = QdrantProcess(fake_qdrant(tmp_path, "ok"), str(d))
            await p.start()
            await p.stop_and_wait()

        asyncio.run(go())
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait()


def test_alive() -> None:
    done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True)
    assert alive(int(done.stdout)) is False
    assert alive(os.getpid()) is True
    assert alive(0) is False
