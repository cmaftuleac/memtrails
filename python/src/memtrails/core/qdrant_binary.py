"""
The Qdrant binary memory runs when you give it no Qdrant server of your own: one pinned release, fetched on first use
from Qdrant's GitHub release, verified against the size and SHA-256 pinned below, unpacked next to it (the cache,
paths.py), and trusted after that through a ``.verified`` marker while the folder is this user's alone. The same
release, the same pins and the same folder as the reference library: the two share one download.

Qdrant publishes no checksums, so the hashes are pinned here from its release. Upgrading means a new version and new
hashes; Qdrant reads storage only from the minor version before it, so after a version change the index is written
again from the logs (rebuild.py) rather than migrated: it is derived data.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import secrets
import shutil
import socket
import stat
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from .._fs import write_all
from .._http import using
from .paths import cache_dir

T = TypeVar("T")

QDRANT_VERSION = "1.20.0"


@dataclass(frozen=True)
class QdrantAsset:
    asset: str
    size: int
    sha256: str
    #: The executable's name inside the archive.
    exe: str


#: Per ``{platform}-{arch}`` (Node's names). Linux takes the static musl build: it runs on any distribution.
QDRANT_ASSETS: dict[str, QdrantAsset] = {
    "darwin-arm64": QdrantAsset(
        "qdrant-aarch64-apple-darwin.tar.gz", 29_070_255, "531904e822050790bb8c1da6705f801ee48b91eb4c229d0d605ff06f6331b20a", "qdrant"
    ),
    "darwin-x64": QdrantAsset(
        "qdrant-x86_64-apple-darwin.tar.gz", 31_917_921, "94a2e6f6b8ef3a3b5756a11172fd66554ac753e0a71c1d8d8b5349356c084f4d", "qdrant"
    ),
    "linux-arm64": QdrantAsset(
        "qdrant-aarch64-unknown-linux-musl.tar.gz", 31_475_234, "bd8a458b8ef8006ce2c603c5bd6d1d973e39d787f794008dd959112e747c3bb7", "qdrant"
    ),
    "linux-x64": QdrantAsset(
        "qdrant-x86_64-unknown-linux-musl.tar.gz", 33_773_335, "b03a64616a1f41ddab78b480a318466541765f19835318113b1821cbea8df52c", "qdrant"
    ),
    "win32-x64": QdrantAsset(
        "qdrant-x86_64-pc-windows-msvc.zip", 30_801_664, "6d13553584fd0411f00637519e293331ff0aa1fb79bd9e30db45839bb75de2be", "qdrant.exe"
    ),
}


def node_platform() -> str:
    """This machine as Node names it: ``darwin``, ``linux``, ``win32``, …"""
    return {"darwin": "darwin", "linux": "linux", "win32": "win32", "cygwin": "win32"}.get(sys.platform, sys.platform)


def node_arch() -> str:
    m = platform.machine().lower()
    return {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(m, m)


def qdrant_asset(plat: str | None = None, arch: str | None = None) -> QdrantAsset | None:
    return QDRANT_ASSETS.get(f"{plat or node_platform()}-{arch or node_arch()}")


def qdrant_dir() -> str:
    return os.path.join(cache_dir(), f"qdrant-{QDRANT_VERSION}")


def qdrant_binary_path(a: QdrantAsset | None = None) -> str | None:
    asset = a or qdrant_asset()
    return os.path.join(qdrant_dir(), asset.exe) if asset else None


QDRANT_RELEASE_BASE = f"https://github.com/qdrant/qdrant/releases/download/v{QDRANT_VERSION}"


def _owned_safely(path: str) -> bool:
    """Run only what this user alone could have put there: owned by this user, no symlink, writable by no one else.
    The archive's hash is pinned but not the unpacked binary's, so on a shared machine this is what stops a binary
    swapped in after unpacking. Windows has no owner bits; there the user's profile ACLs keep others out."""
    if not hasattr(os, "getuid"):
        return True
    st = os.lstat(path)
    return not stat.S_ISLNK(st.st_mode) and st.st_uid == os.getuid() and (st.st_mode & 0o022) == 0


def qdrant_ready(a: QdrantAsset | None = None) -> bool:
    """The binary is in place, its archive was verified when it was unpacked, and no one else could have changed it
    since."""
    asset = a or qdrant_asset()
    binary = qdrant_binary_path(asset)
    if not asset or not binary:
        return False
    marker = os.path.join(qdrant_dir(), ".verified")
    try:
        with open(marker, encoding="utf-8") as f:
            ok = f.read().strip() == f"{asset.size}:{asset.sha256}"
        return ok and os.stat(binary).st_size > 0 and all(_owned_safely(p) for p in (qdrant_dir(), binary, marker))
    except OSError:
        return False


def _tar_program() -> str:
    """The system's tar; on Windows by full path, so a ``tar.exe`` in the working folder or on a planted PATH is not
    run."""
    if node_platform() == "win32":
        return os.path.join(os.environ.get("SystemRoot") or "C:\\Windows", "System32", "tar.exe")
    return "tar"


async def _untar(archive: str, into: str) -> None:
    """Unpack one archive with the system's tar (bsdtar on macOS and Windows reads zip too)."""
    child = await asyncio.create_subprocess_exec(
        _tar_program(),
        "-xf",
        archive,
        "-C",
        into,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    code = await child.wait()
    if code != 0:
        raise RuntimeError(f"tar exited {code}")


#: A download given up after this long without a byte (the answer's headers included).
DOWNLOAD_STALL_MS = 60_000


def _sha256_of(path: str) -> tuple[int, str]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return os.stat(path).st_size, h.hexdigest()


async def download_verified(
    url: str, directory: str, size: int, sha256: str, stall_ms: float = DOWNLOAD_STALL_MS, http: httpx.AsyncClient | None = None
) -> tuple[str, str]:
    """``url`` fetched into a fresh temp folder inside ``directory``, never to a fixed name: two first runs at once
    wrote the same ``.part`` and could rename each other's half-written file. The file is then checked as it is on
    disk, not as it streamed past, against its pinned size and SHA-256 (hashed off the event loop). Returns the file
    and its folder; the caller renames the file into place and removes the folder. Raises (leaving nothing) on a
    failure, a mismatch, or ``stall_ms`` without a byte."""
    os.makedirs(directory, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".download-", dir=directory)
    file = os.path.join(tmp, "file")
    name = url[url.rfind("/") + 1 :]
    stall = stall_ms / 1000

    def stalled() -> TimeoutError:
        return TimeoutError(f"{name}: no data for {round(stall_ms / 1000)} s")

    try:
        async with using(http) as client:
            try:
                res = await asyncio.wait_for(client.send(client.build_request("GET", url), stream=True, follow_redirects=True), stall)
            except (TimeoutError, asyncio.TimeoutError) as err:
                raise stalled() from err
            try:
                if not res.is_success:
                    raise RuntimeError(f"HTTP {res.status_code} for {name}")
                got = 0
                with open(file, "xb") as out:
                    os.chmod(file, 0o600)
                    chunks = res.aiter_bytes().__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(chunks.__anext__(), stall)
                        except StopAsyncIteration:
                            break
                        except (TimeoutError, asyncio.TimeoutError) as err:
                            raise stalled() from err
                        got += len(chunk)
                        if got > size:
                            raise RuntimeError(f"{name}: more than the expected {size} bytes")
                        out.write(chunk)
            finally:
                await res.aclose()
        on_disk, digest = await asyncio.to_thread(_sha256_of, file)
        if on_disk != size or digest != sha256:
            raise RuntimeError(f"{name}: got {on_disk} bytes sha256 {digest}, expected {size} {sha256}")
        return file, tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


#: A lock not touched for this long is a dead holder's (the holder touches it every quarter of this).
LOCK_STALE_MS = 120_000


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _lock_stale(lock: str, stale_ms: float) -> bool:
    try:
        with open(lock, encoding="utf-8") as f:
            parts = f.read().split(" ")
        pid, host = parts[0], parts[1] if len(parts) > 1 else ""
        if host == socket.gethostname() and pid.isdigit() and int(pid) != os.getpid() and int(pid) > 0 and not _alive(int(pid)):
            return True  # its process is gone
        return time.time() * 1000 - os.stat(lock).st_mtime * 1000 > stale_ms
    except OSError:
        return False  # gone meanwhile: just try again


async def with_lock(path: str, run: Callable[[], Awaitable[T]], stale_ms: float = LOCK_STALE_MS, poll_ms: float = 250) -> T:
    """``run`` run holding ``<path>.lock``, created exclusively: a second first run, in another process or this one,
    waits and then finds the first one's result. A lock whose process is gone, or untouched for ``stale_ms``, is taken
    over. Nothing rests on the lock for correctness (every file is verified in a private temp and renamed into place);
    it keeps two runs from fetching the same hundreds of MB, and from swapping folders under each other. The lock file
    is the reference's (``pid host token``): the two libraries wait for each other."""
    lock = f"{path}.lock"
    mine = f"{os.getpid()} {socket.gethostname()} {secrets.token_hex(8)}"
    os.makedirs(os.path.dirname(lock) or ".", exist_ok=True)
    while True:
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                write_all(fd, mine.encode("utf-8"))
            except BaseException:
                os.close(fd)
                try:
                    os.remove(lock)  # a torn lock names no one: the next run would wait it out as stale
                except OSError:
                    pass
                raise
            os.close(fd)
            break
        except FileExistsError:
            pass
        if _lock_stale(lock, stale_ms):
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
            continue
        await asyncio.sleep(poll_ms / 1000)

    async def beat() -> None:
        while True:
            await asyncio.sleep(max(10, stale_ms / 4) / 1000)
            try:
                os.utime(lock)
            except OSError:
                pass  # taken over: nothing to keep alive

    heart = asyncio.ensure_future(beat())
    try:
        return await run()
    finally:
        heart.cancel()
        try:
            with open(lock, encoding="utf-8") as f:
                if f.read() == mine:
                    os.remove(lock)  # not one taken over meanwhile
        except OSError:
            pass


async def ensure_qdrant(
    log: Callable[[str], None] | None = None, asset: QdrantAsset | None = None, http: httpx.AsyncClient | None = None
) -> str:
    """Fetch, verify and unpack the binary unless it is already there; its path. Raises on a network failure or stall,
    a size or hash mismatch (the download is deleted), a platform Qdrant does not ship for, or a folder others could
    change. ``asset`` is for tests: another archive with its own pins."""
    a = asset or qdrant_asset()
    if not a:
        raise RuntimeError(f"no Qdrant build for {node_platform()}-{node_arch()}")
    binary = qdrant_binary_path(a)
    assert binary is not None
    if qdrant_ready(a):
        return binary
    directory = qdrant_dir()
    cache = os.path.dirname(directory)
    os.makedirs(cache, mode=0o700, exist_ok=True)

    async def fetch() -> str:
        if qdrant_ready(a):
            return binary  # another run finished it while this one waited
        if log:
            log(f"[memtrails] fetching Qdrant {QDRANT_VERSION} ({round(a.size / 1e6)} MB)")
        file, tmp = await download_verified(f"{QDRANT_RELEASE_BASE}/{a.asset}", cache, a.size, a.sha256, http=http)
        try:
            # Unpacked into a fresh folder and moved into place whole: never a half-unpacked binary under the real name.
            archive = os.path.join(tmp, a.asset)
            os.rename(file, archive)
            unpacked = os.path.join(tmp, "unpacked")
            os.mkdir(unpacked)
            await _untar(archive, unpacked)
            exe = os.path.join(unpacked, a.exe)
            if not os.path.isfile(exe) or os.path.islink(exe) or os.stat(exe).st_size == 0:
                raise RuntimeError(f"{a.asset} did not contain {a.exe}")
            os.chmod(exe, 0o755)
            marker = os.path.join(unpacked, ".verified")
            with open(marker, "w", encoding="utf-8") as f:
                f.write(f"{a.size}:{a.sha256}\n")
            os.chmod(marker, 0o644)
            os.chmod(unpacked, 0o755)
            if os.path.exists(directory):
                os.rename(directory, os.path.join(tmp, "old"))  # unverified or not this user's alone; removed with tmp
            os.rename(unpacked, directory)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not qdrant_ready(a):
            raise RuntimeError(f"{directory} is not this user's alone (its owner or permissions): not running a binary others could change")
        if log:
            log(f"[memtrails] Qdrant {QDRANT_VERSION} ready")
        return binary

    return await with_lock(directory, fetch)
