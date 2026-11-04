"""
Qdrant as memory runs it when you give it no server of your own: one process for this memory, on a loopback port picked
at start, behind an API key made for that run, gRPC off, telemetry off, stopped with ``close()``.

Tuned for a laptop: the index on disk, memory-mapped (qdrant_client.py), one optimisation thread, two search threads
and ONE web worker: Qdrant's default is one a core, each waking ~4 times a second, so an idle index woke ~83 times a
second and spent 2.7–4 ms/s of CPU with 69 threads; with one it is 6.5 wake-ups a second and ~0.2 ms/s.

A process that dies by itself is started again with a growing pause; five deaths in ten minutes stop the attempts until
the next ``start()``. A start that fails leaves nothing running and nothing scheduled. It gets a minimal environment,
proves it is ours before it is handed anything, is stopped when the host exits, and one a crashed host left on the same
storage is stopped before the next starts. The pid file and the config are the reference's: either library reaps what
the other left.
"""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import errno
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import httpx

from .._js import now_ms, parse, stringify
from .qdrant_client import QdrantEndpoint


def _free_port() -> int:
    """A port the OS says is free on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def qdrant_config(storage: str, port: int) -> str:
    """The config file: what environment variables cannot say (gRPC off is ``grpc_port: null``)."""
    return "\n".join(
        [
            "log_level: WARN",
            "telemetry_disabled: true",
            "service:",
            "  host: 127.0.0.1",
            f"  http_port: {port}",
            "  grpc_port: null",
            "  enable_cors: false",
            "  max_workers: 1",
            "storage:",
            f"  storage_path: {stringify(os.path.join(storage, 'storage'))}",
            f"  snapshots_path: {stringify(os.path.join(storage, 'snapshots'))}",
            "  on_disk_payload: true",
            "  performance:",
            "    max_search_threads: 2",
            "  optimizers:",
            "    max_optimization_threads: 1",
            "",
        ]
    )


_LOG_MAX = 4 * 1024 * 1024
#: Beside the storage: the pid of the Qdrant running on it, with its config and the machine's boot (_reap_leftover).
_PID_FILE = "qdrant.pid"


def _open_log(directory: str) -> int:
    """The process's log beside its data, rotated once at start when it has grown past 4 MB. Private: it names paths."""
    path = os.path.join(directory, "qdrant.log")
    try:
        if os.stat(path).st_size > _LOG_MAX:
            os.replace(path, f"{path}.1")
    except OSError:
        pass  # none yet
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # not ours to change: written all the same
    return fd


def _uptime_s() -> float:
    """Seconds since this machine started, from where Node reads them (os.uptime): /proc/uptime on Linux, the kernel's
    boot time on macOS."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/uptime", encoding="ascii") as f:
                return float(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            pass
    if sys.platform == "darwin":

        class _Timeval(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_int32)]

        tv = _Timeval()
        size = ctypes.c_size_t(ctypes.sizeof(tv))
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.sysctlbyname(b"kern.boottime", ctypes.byref(tv), ctypes.byref(size), None, 0) == 0:
            return time.time() - (tv.tv_sec + tv.tv_usec / 1e6)
    return time.monotonic()


def boot_time() -> float:
    """When this machine started, in ms: a pid written down before a reboot names some other process now."""
    return now_ms() - _uptime_s() * 1000


def same_boot(boot: object) -> bool:
    """The same boot as ``boot`` (boot_time() of then): a minute apart at most, uptime being in whole seconds on some
    systems."""
    return isinstance(boot, (int, float)) and not isinstance(boot, bool) and abs(boot_time() - boot) < 60_000


def alive(pid: object) -> bool:
    """A process with this pid exists — one of another user's too."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _command_of(pid: int) -> str | None:
    """A running process's command line, or None when it cannot be read (no /proc and no ps, Windows)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        pass
    if sys.platform == "win32":
        return None
    try:
        return subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _child_env(api_key: str) -> dict[str, str]:
    """What Qdrant gets of the environment: what a process needs to run and the key. None of the host's secrets, and
    none of its QDRANT__* variables — they override the config (QDRANT__SERVICE__HOST=0.0.0.0 would open it to the
    network)."""
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SystemRoot", "USERPROFILE") if k in os.environ}
    return {**env, "QDRANT__SERVICE__API_KEY": api_key, "QDRANT__TELEMETRY_DISABLED": "true"}


#: Every Qdrant this process started and has not seen exit: stopped when the host exits without close() (a host killed
#: outright leaves its Qdrant to the next start's _reap_leftover).
_running: set[subprocess.Popen[bytes]] = set()
_running_lock = threading.Lock()


def _stop_all() -> None:
    with _running_lock:
        for c in list(_running):
            try:
                c.send_signal(signal.SIGTERM)
            except OSError:
                pass  # gone


atexit.register(_stop_all)


def _errno_text(err: OSError) -> str:
    code = errno.errorcode.get(err.errno or 0, "")
    return f"{code}: {err.strerror}" if code else str(err)


class QdrantProcess:
    """``dir`` may be relative: Qdrant runs inside it, so every path it is given is made absolute first."""

    def __init__(
        self, binary: str, directory: str, log: Callable[[str], None] = lambda _line: None, on_down: Callable[[str], None] | None = None
    ) -> None:
        self._binary = os.path.abspath(binary) if ("/" in binary or "\\" in binary) else binary
        self._dir = os.path.abspath(directory)
        self._log = log
        self._on_down = on_down
        self._child: subprocess.Popen[bytes] | None = None
        self._endpoint: QdrantEndpoint | None = None
        self._stopping = False
        self._deaths: list[float] = []
        self._restart: asyncio.TimerHandle | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._exits: dict[int, asyncio.Event] = {}
        self._task: asyncio.Future[None] | None = None

    def current(self) -> QdrantEndpoint | None:
        """Where it answers, once it does; None while it starts or after it stopped."""
        return self._endpoint

    @property
    def directory(self) -> str:
        return self._dir

    async def start(self) -> QdrantEndpoint:
        """Up and proven ours (owned), or it raises — and then nothing of this start keeps running or is started again:
        the caller was told, and may not be there to stop it."""
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        if self._endpoint and self._child and self._child.poll() is None:
            return self._endpoint
        os.makedirs(self._dir, mode=0o700, exist_ok=True)
        await self._reap_leftover()
        port = _free_port()
        api_key = secrets.token_hex(24)
        config = os.path.join(self._dir, "config.yaml")
        fd = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, qdrant_config(self._dir, port).encode("utf-8"))
        finally:
            os.close(fd)
        log_fd = _open_log(self._dir)
        try:
            # A binary that is missing or cannot run fails here, in the caller, never in the host later.
            child = subprocess.Popen(
                [self._binary, "--config-path", config, "--disable-telemetry"],
                cwd=self._dir,
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=log_fd,
                env=_child_env(api_key),
            )
        except OSError as err:
            raise RuntimeError(f"Qdrant could not be started ({self._binary}): {_errno_text(err)}") from err
        finally:
            os.close(log_fd)
        self._child = child
        self._watch(child, config)
        endpoint = QdrantEndpoint(url=f"http://127.0.0.1:{port}", api_key=api_key)
        try:
            await self._ready(endpoint, child)
            await self._owned(endpoint, child)
            if self._child is not child:
                raise RuntimeError("Qdrant was stopped while it started")
        except BaseException:
            if self._child is child:
                self._child = None  # _exited then schedules nothing
            if child.poll() is None:
                child.terminate()
            raise
        self._endpoint = endpoint
        self._log(f"[memory] Qdrant up on 127.0.0.1:{port}")
        return endpoint

    def _watch(self, child: subprocess.Popen[bytes], config: str) -> None:
        """The pid file, the child tracked for the host's exit, and a thread that hears it exit; the file goes when
        that child does."""
        path = os.path.join(self._dir, _PID_FILE)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, stringify({"pid": child.pid, "config": config, "boot": boot_time()}).encode("utf-8"))
        finally:
            os.close(fd)
        with _running_lock:
            _running.add(child)
        loop = self._loop
        done = asyncio.Event()
        self._exits[child.pid] = done

        def wait() -> None:
            child.wait()
            with _running_lock:
                _running.discard(child)
            try:
                with open(path, encoding="utf-8") as f:
                    if parse(f.read()).get("pid") == child.pid:
                        os.remove(path)
            except (OSError, ValueError, AttributeError):
                pass  # none
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(self._exited, child, done)
                except RuntimeError:
                    pass  # the loop has stopped: nothing to restart

        threading.Thread(target=wait, name=f"memtrails-qdrant-{child.pid}", daemon=True).start()

    async def _reap_leftover(self) -> None:
        """A Qdrant left running on this storage by a host that died without close() (a crash, a kill): stopped before
        another starts on it. Only when the pid is still the process written down — this boot, its command naming this
        folder's config — never an unrelated process that has the pid since."""
        path = os.path.join(self._dir, _PID_FILE)
        try:
            with open(path, encoding="utf-8") as f:
                rec = parse(f.read())
            if not isinstance(rec, dict):
                raise ValueError("not a record")
        except (OSError, ValueError):
            return
        pid = rec.get("pid")
        config = os.path.join(self._dir, "config.yaml")
        mine = self._child.pid if self._child else None
        command = _command_of(pid) if isinstance(pid, int) and alive(pid) else None
        if rec.get("config") == config and same_boot(rec.get("boot")) and pid != mine and command and config in command:
            assert isinstance(pid, int)
            self._log(f"[memory] stopping the Qdrant left running on {self._dir} (pid {pid})")
            for sig, ms in ((signal.SIGTERM, 10_000), (signal.SIGKILL, 2000)):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass  # gone
                until = time.monotonic() + ms / 1000
                while alive(pid) and _not_zombie(pid) and time.monotonic() < until:
                    await asyncio.sleep(0.05)
                if not alive(pid) or not _not_zombie(pid):
                    break
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    async def _ready(self, e: QdrantEndpoint, child: subprocess.Popen[bytes]) -> None:
        """Answering ``/readyz`` within 30 s, or the start failed. /readyz asks no key, so none is sent: what answers may
        not be ours (_owned)."""
        until = time.monotonic() + 30
        async with httpx.AsyncClient(trust_env=False) as http:
            while time.monotonic() < until:
                code = child.poll()
                if code is not None:
                    raise RuntimeError(f"Qdrant exited ({_exit_text(code)}) while starting — see {os.path.join(self._dir, 'qdrant.log')}")
                try:
                    r = await http.get(f"{e.url}/readyz", timeout=1)
                    if r.is_success:
                        return
                except httpx.HTTPError:
                    pass  # not listening yet
                await asyncio.sleep(0.1)
        raise RuntimeError("Qdrant did not answer within 30 s")

    async def _owned(self, e: QdrantEndpoint, child: subprocess.Popen[bytes]) -> None:
        """The port was free when picked, not when Qdrant bound it: another local process may have taken it, answered
        /readyz, and would be handed the key and every document. Ours is still running, refuses a wrong key and takes
        this run's — the wrong key asked first, so a process that takes any key never sees the real one."""
        not_ours = f"what answers on {e.url} is not this memory's Qdrant"
        log = os.path.join(self._dir, "qdrant.log")
        async with httpx.AsyncClient(trust_env=False) as http:

            async def status(key: str) -> int:
                return (await http.get(f"{e.url}/collections", headers={"api-key": key}, timeout=5)).status_code

            if child.poll() is not None:
                raise RuntimeError(f"{not_ours}: ours exited — see {log}")
            wrong = await status(secrets.token_hex(24))
            if wrong not in (401, 403):
                raise RuntimeError(f"{not_ours}: it took a wrong key ({wrong})")
            right = await status(e.api_key)
            if right != 200:
                raise RuntimeError(f"{not_ours}: it refused this run's key ({right})")
            if child.poll() is not None:
                raise RuntimeError(f"{not_ours}: ours exited — see {log}")

    def _exited(self, child: subprocess.Popen[bytes], done: asyncio.Event) -> None:
        done.set()
        if self._child is not child:
            return
        # One that never came up is its start()'s to report: that start raises, and schedules nothing.
        was_up = self._endpoint is not None
        self._child = None
        self._endpoint = None
        if self._stopping or not was_up:
            return
        self._restart_later(f"Qdrant stopped ({_exit_text(child.returncode)})")

    def _restart_later(self, why: str) -> None:
        """Started again after a growing pause; a restart that fails counts as another death."""
        self._log(f"[memory] {why}")
        now = time.monotonic()
        self._deaths = [*(t for t in self._deaths if now - t < 600), now]
        if len(self._deaths) >= 5:
            if self._on_down:
                self._on_down(f"{why}; five times in ten minutes, so it is left stopped")
            return
        wait = min(60.0, 2 ** (len(self._deaths) - 1))
        loop = self._loop
        if loop is None:
            return

        def again() -> None:
            self._restart = None
            if self._stopping:
                return

            async def restart() -> None:
                try:
                    await self.start()
                except Exception as err:  # noqa: BLE001 - told through on_down, and tried again
                    if self._on_down:
                        self._on_down(str(err))
                    if not self._stopping:
                        self._restart_later(str(err))

            self._task = asyncio.ensure_future(restart())

        self._restart = loop.call_later(wait, again)

    def stop(self) -> None:
        self._stopping = True
        if self._restart is not None:
            self._restart.cancel()
            self._restart = None
        c = self._child
        self._child = None
        self._endpoint = None
        if c is not None and c.poll() is None:
            c.terminate()

    async def stop_and_wait(self, ms: float = 10_000) -> None:
        """Stop and wait for the process to be gone — before its storage is deleted."""
        c = self._child
        self.stop()
        if c is None:
            return
        done = self._exits.get(c.pid)
        if done is None or (c.poll() is not None and done.is_set()):
            return
        try:
            await asyncio.wait_for(done.wait(), ms / 1000)
        except (TimeoutError, asyncio.TimeoutError):
            return


def _exit_text(code: int | None) -> str:
    """How a process ended, as Node says it: a signal's name, else the exit code."""
    if code is not None and code < 0:
        try:
            return signal.Signals(-code).name
        except ValueError:
            return str(code)
    return str(code)


def _not_zombie(pid: int) -> bool:
    """A child of this process that exited and was not waited for still answers to its pid: it is gone all the same."""
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True
    except OSError:
        return True


__all__ = ["QdrantProcess", "alive", "boot_time", "qdrant_config", "same_boot"]
