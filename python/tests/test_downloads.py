"""The pinned downloads (qdrant_binary.py, model.py): every platform has a pinned build, a download that does not match
is never kept, two at once never take each other's file, a stall is given up on, and the lock is held by one at a time
(qdrant-binary.test.ts, ported)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import socket
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from fakes import run
from memtrails.core.model import ensure_model, model_ready
from memtrails.core.qdrant_binary import (
    QDRANT_ASSETS,
    QdrantAsset,
    download_verified,
    ensure_qdrant,
    qdrant_asset,
    qdrant_binary_path,
    qdrant_dir,
    qdrant_ready,
    with_lock,
)


def test_every_platform_has_a_pinned_build() -> None:
    for p in ("darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"):
        a = QDRANT_ASSETS[p]
        assert len(a.sha256) == 64 and int(a.sha256, 16) >= 0
        assert a.size > 10_000_000
    win = qdrant_asset("win32", "x64")
    assert win is not None and win.exe == "qdrant.exe"
    assert qdrant_asset("freebsd", "x64") is None


def answer(status: int, body: bytes) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _r: httpx.Response(status, content=body))


def test_a_download_that_does_not_match_is_not_kept() -> None:
    with pytest.raises(RuntimeError, match="expected"):
        run(ensure_qdrant(http=httpx.AsyncClient(transport=answer(200, bytes([1, 2, 3])))))
    assert not qdrant_ready()
    leftovers = [n for n in os.listdir(os.path.dirname(qdrant_dir())) if n.startswith(".download-")]
    assert leftovers == []


def test_says_why_a_download_failed() -> None:
    with pytest.raises(RuntimeError, match="HTTP 404"):
        run(ensure_qdrant(http=httpx.AsyncClient(transport=answer(404, b"nope"))))


def test_two_downloads_at_once_never_take_each_others_file(tmp_path: Path) -> None:
    data = bytes(i % 251 for i in range(200_000))
    sha = hashlib.sha256(data).hexdigest()

    async def both() -> list[tuple[str, str]]:
        t = answer(200, data)
        return list(
            await asyncio.gather(
                download_verified("https://x.test/f.bin", str(tmp_path), len(data), sha, http=httpx.AsyncClient(transport=t)),
                download_verified("https://x.test/f.bin", str(tmp_path), len(data), sha, http=httpx.AsyncClient(transport=t)),
            )
        )

    (a, _), (b, _) = run(both())
    assert a != b
    assert os.path.exists(a) and os.path.exists(b)


class Stall(httpx.AsyncByteStream):
    async def __aiter__(self) -> Any:
        yield b"\x01"
        await asyncio.sleep(30)


def test_gives_up_on_a_download_that_stops_sending(tmp_path: Path) -> None:
    t = httpx.MockTransport(lambda _r: httpx.Response(200, stream=Stall()))
    with pytest.raises(TimeoutError, match="no data"):
        run(download_verified("https://x.test/f.bin", str(tmp_path), 10, "0" * 64, stall_ms=50, http=httpx.AsyncClient(transport=t)))
    assert os.listdir(tmp_path) == []


def test_one_holder_of_a_lock_at_a_time_and_a_dead_ones_lock_is_taken(tmp_path: Path) -> None:
    path = str(tmp_path / "x")
    order: list[str] = []

    async def hold(n: str) -> None:
        async def job() -> None:
            order.append(f"{n}+")
            await asyncio.sleep(0.03)
            order.append(f"{n}-")

        await with_lock(path, job, poll_ms=5)

    async def both() -> None:
        await asyncio.gather(hold("a"), hold("b"))

    run(both())
    assert " ".join(order) in ("a+ a- b+ b-", "b+ b- a+ a-")
    Path(f"{path}.lock").write_text(f"999999999 {socket.gethostname()} dead")

    async def taken() -> str:
        return "taken"

    assert run(with_lock(path, taken, poll_ms=5)) == "taken"


def test_the_model_is_fetched_checked_and_marked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import memtrails.core.model as model

    files = {rel: f"bytes of {rel}".encode() for rel, _, _ in model.MODEL_FILES}
    monkeypatch.setattr(model, "MODEL_FILES", [(rel, len(b), hashlib.sha256(b).hexdigest()) for rel, b in files.items()])
    monkeypatch.setattr(model, "_STAMP", "\n".join(f"{rel}:{len(b)}:{hashlib.sha256(b).hexdigest()}" for rel, b in files.items()))

    def serve(req: httpx.Request) -> httpx.Response:
        rel = str(req.url).split(f"/{model.MODEL_REVISION}/", 1)[1]
        return httpx.Response(200, content=files[rel])

    d = str(tmp_path / "model")
    assert run(ensure_model(d, http=httpx.AsyncClient(transport=httpx.MockTransport(serve)))) == d
    assert model_ready(d)
    assert (tmp_path / "model" / "onnx" / "model_q4.onnx").read_bytes() == files["onnx/model_q4.onnx"]
    os.remove(tmp_path / "model" / "tokenizer.json")
    assert not model_ready(d)


unix = pytest.mark.skipif(not hasattr(os, "getuid"), reason="no owner bits")


def counting_404(count: list[int]) -> httpx.AsyncClient:
    def serve(_r: httpx.Request) -> httpx.Response:
        count.append(1)
        return httpx.Response(404, content=b"nope")

    return httpx.AsyncClient(transport=httpx.MockTransport(serve))


@unix
def test_a_cache_others_could_change_is_made_private_or_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened = tmp_path / "open"
    opened.mkdir()
    os.chmod(opened, 0o777)
    monkeypatch.setenv("MEMTRAILS_CACHE", str(opened))
    count: list[int] = []
    with pytest.raises(RuntimeError, match="HTTP 404"):
        run(ensure_qdrant(http=counting_404(count)))
    assert os.stat(opened).st_mode & 0o777 == 0o700
    # A link to a folder: whoever can change the link can swap what it leads to — refused, nothing fetched.
    link = tmp_path / "link"
    link.symlink_to(opened)
    monkeypatch.setenv("MEMTRAILS_CACHE", str(link))
    before = len(count)
    with pytest.raises(RuntimeError, match="not this user's alone"):
        run(ensure_qdrant(http=counting_404(count)))
    assert len(count) == before


@unix
def test_runs_a_binary_unpacked_as_root_whoever_owned_it_in_the_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"#!/bin/sh\n"
    entry = tarfile.TarInfo("qdrant")
    entry.size, entry.mode, entry.uid, entry.gid = len(body), 0o755, 4242, 4242
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        tar.addfile(entry, io.BytesIO(body))
    archive = packed.getvalue()
    asset = QdrantAsset(asset="qdrant-test.tar", size=len(archive), sha256=hashlib.sha256(archive).hexdigest(), exe="qdrant")
    monkeypatch.setenv("MEMTRAILS_CACHE", str(tmp_path / "cache"))
    binary = run(ensure_qdrant(asset=asset, http=httpx.AsyncClient(transport=answer(200, archive))))
    assert os.stat(binary).st_uid == os.getuid()


@unix
def test_no_binary_is_ready_while_others_could_write_to_the_cache_above_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMTRAILS_CACHE", str(tmp_path / "cache"))
    a = qdrant_asset()
    assert a is not None
    os.makedirs(qdrant_dir(), mode=0o755)
    binary = qdrant_binary_path(a)
    assert binary is not None
    Path(binary).write_text("binary")
    os.chmod(binary, 0o755)
    Path(qdrant_dir(), ".verified").write_text(f"{a.size}:{a.sha256}\n")
    os.chmod(Path(qdrant_dir(), ".verified"), 0o644)
    os.chmod(tmp_path / "cache", 0o700)
    assert qdrant_ready(a)
    os.chmod(tmp_path / "cache", 0o777)
    assert not qdrant_ready(a)


@unix
def test_the_model_folder_is_made_private_or_refused_and_not_ready_while_others_could_write(tmp_path: Path) -> None:
    import memtrails.core.model as model

    d = tmp_path / "model"
    d.mkdir()
    os.chmod(d, 0o777)
    count: list[int] = []
    with pytest.raises(RuntimeError, match="HTTP 404"):
        run(ensure_model(str(d), http=counting_404(count)))
    assert os.stat(d).st_mode & 0o777 == 0o700
    link = tmp_path / "link"
    link.symlink_to(d)
    before = len(count)
    with pytest.raises(RuntimeError, match="not this user's alone"):
        run(ensure_model(str(link), http=counting_404(count)))
    assert len(count) == before
    # A whole set in place (sparse files of the pinned sizes): ready while private, not once a folder is open to others.
    for rel, size, _ in model.MODEL_FILES:
        f = d / rel
        f.parent.mkdir(mode=0o700, exist_ok=True)
        with open(f, "wb") as out:
            out.truncate(size)
        os.chmod(f, 0o600)
    (d / ".verified").write_text("\n".join(f"{rel}:{size}:{sha}" for rel, size, sha in model.MODEL_FILES))
    os.chmod(d / ".verified", 0o644)
    assert model_ready(str(d))
    os.chmod(d / "onnx", 0o777)
    assert not model_ready(str(d))
