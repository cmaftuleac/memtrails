"""
memtrails: open a memory, add anything with text, ask it things.

    with open_memory(home="./memory") as memory:
        memory.add({"id": "mail-1", "kind": "email", "title": "Gateway certificate", "author": "Ana",
                    "text": "It expired on the 1st; the renewal job had been removed.", "at": "2026-04-02"})
        memory.search({"query": "why did the certificate stop renewing?"})

What is added is searchable at once, with no language model and no server: the embedding model runs here (fetched on
first use) and orders results word by word (words.py). A language model is needed only for ``consolidate``, ``sleep``
and ``gather``, which turn records into typed knowledge, dated observations and profiles — and memory works without
them.

The JSONL logs in ``home`` are the truth. The Qdrant index and the word vectors are derived: lost or out of date, they
are written again from the logs. A home is the reference library's too: either opens what the other wrote.

The API is synchronous: a memory runs its work (the index, the model, the language model's asks, several at once) on
an event loop of its own, on a thread of its own, and each method waits for its answer.
"""

from __future__ import annotations

import asyncio
import atexit
import inspect
import os
import threading
from collections.abc import AsyncIterable, Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, TypedDict, TypeVar

import httpx

from ._js import date_parse, iso_day, now_ms, parse, stringify, u16len, u16slice
from .core.consolidate import ConsolidateDeps, Consolidator, new_report
from .core.embedder import EmbedRoute, remote_embedder, resolve_embed_route
from .core.episodes import episode_body
from .core.gather import Gathered, memory_gather
from .core.local_embedder import LocalEmbedder
from .core.model import ensure_model_async
from .core.qdrant_binary import ensure_qdrant_async
from .core.qdrant_client import COLLECTION, INDEX_LAYOUT, QdrantClient, QdrantEndpoint
from .core.qdrant_process import QdrantProcess, alive, boot_time, same_boot
from .core.rebuild import MemoryRebuild, RebuildDeps
from .core.recall import Explored, RecallDeps, RecalledItem, memory_explore, memory_open, memory_search
from .core.recall_log import RecallBook
from .core.reranker import RerankRoute, remote_reranker, resolve_rerank_route
from .core.room import RoomItem, sentence_room, sentence_spans
from .core.sleep import Sleep, SleepDeps
from .core.store import StmEntry, UsageBook, ltm_log, stm_log
from .core.words import Rerank, WordEncoder, WordStore, pack_words, word_reranker
from .core.writer import raw_key, stm_item, stm_title, write_ltm, write_stm
from .llm import Ask, OpenAiAsk, default_ask
from .records import Conversation, MemoryRecord, conversation_entries, entry_of, invalid_record, record_uid
from .tools.agent_tools import AgentTool, build_agent_tools, lasted_text, readable_date, search_text

T = TypeVar("T")

_BATCH = 32
_ROOM_TEXT = 1500
_ROOM_CHARS = 6000
_RECORD_LIMIT = 2_000_000

#: In the home: who has it open (_lock_home), and the index layout its Qdrant collection was written in.
_LOCK_FILE = "memory.lock"
_LAYOUT_FILE = "index.json"


class EmbeddingOptions(TypedDict, total=False):
    """How texts are embedded (EmbeddingGemma-300m in every case: the index, prompts and measured thresholds are its):
    default, on this machine, the model fetched on first use into the cache (218 MB); ``model_dir``: on this machine,
    from a directory you provide; ``server``: a server that serves it over the OpenAI embeddings API. Word by word then
    needs ``model_dir`` too, or it is off. With neither, MEMTRAILS_EMBED_URL (+ _KEY, _MODEL) names a server when set."""

    model_dir: str
    server: EmbedRoute
    #: CPU threads for the local model (2).
    threads: int


class QdrantOptions(TypedDict, total=False):
    """A Qdrant server of your own (``url``, ``api_key``, ``collection``), or the binary to run (``binary``)."""

    url: str
    api_key: str
    collection: str
    binary: str


class SearchOptions(TypedDict, total=False):
    """``query`` (required); ``collections``; ``kinds``; ``limit`` (8, at most 200); ``from``/``to`` (YYYY-MM-DD,
    YYYY-MM or YYYY); ``order`` (``relevance`` or ``time``, for search_text); ``chars`` (for room)."""

    query: str
    collections: list[str]
    kinds: list[str]
    limit: int
    order: Literal["relevance", "time"]
    chars: int


# --- one process a home -------------------------------------------------------------------------------------------

#: The homes this process holds, released if it exits without close().
_holding: dict[str, Callable[[], None]] = {}
_holding_lock = threading.Lock()


def _release_all() -> None:
    with _holding_lock:
        releases = list(_holding.values())
    for r in releases:
        r()


atexit.register(_release_all)


def _lock_holder(path: str) -> tuple[str, int | None]:
    """Who holds a lock: its text and live pid, 0 when it is being written this instant, None when nobody."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return "", None
    try:
        lock = parse(text)
        pid = lock.get("pid")
        return text, pid if same_boot(lock.get("boot")) and alive(pid) else None
    except (ValueError, AttributeError):
        # Not written yet by the opener that just made it — or torn by a crash, and then old.
        try:
            return text, 0 if now_ms() - os.stat(path).st_mtime * 1000 < 10_000 else None
        except OSError:
            return text, None


def _lock_home(directory: str) -> Callable[[], None]:
    """One process at a time in a home: two would run two Qdrants on one storage and append to the logs at offsets the
    other does not know. The lock is made exclusively with its pid and the boot; one whose process is gone, or ran
    before a reboot (its pid may be another's now), is taken over. Returns its release."""
    path = os.path.join(directory, _LOCK_FILE)
    attempt = 0
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            text, pid = _lock_holder(path)
            if pid is not None or attempt > 0:
                who = f" in process {pid}" if pid else ""
                raise RuntimeError(f"memory {directory} is open{who}: close it there first (if no process has it, delete {path})") from None
            # Stale: taken over — unless another opener took it over since it was read.
            try:
                with open(path, encoding="utf-8") as f:
                    if f.read() == text:
                        os.remove(path)
            except OSError:
                pass  # gone already
            attempt += 1
            continue
        try:
            os.write(fd, stringify({"pid": os.getpid(), "boot": boot_time()}).encode("utf-8"))
        finally:
            os.close(fd)
        break

    def release() -> None:
        with _holding_lock:
            if _holding.get(path) is not release:
                return
            del _holding[path]
        try:
            with open(path, encoding="utf-8") as f:
                if parse(f.read()).get("pid") == os.getpid():
                    os.remove(path)
        except (OSError, ValueError, AttributeError):
            pass  # gone

    with _holding_lock:
        _holding[path] = release
    return release


def _index_layout(directory: str) -> float | None:
    """The layout the index was written in (qdrant_client.py INDEX_LAYOUT); None when not written down."""
    try:
        with open(os.path.join(directory, _LAYOUT_FILE), encoding="utf-8") as f:
            v = parse(f.read()).get("layout")
        return float(v) if isinstance(v, (int, float)) else None
    except (OSError, ValueError, AttributeError):
        return None


def _save_index_layout(directory: str) -> None:
    path = os.path.join(directory, _LAYOUT_FILE)
    fd = os.open(f"{path}.tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, stringify({"layout": INDEX_LAYOUT}).encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(f"{path}.tmp", path)


# --- the engine: everything async, on the memory's own loop --------------------------------------------------------


def _as_rerank(f: Callable[..., Any]) -> Rerank:
    """Your own reranker, plain or ``async``."""

    async def rerank(query: str, documents: list[str], timeout_ms: float) -> list[float] | None:
        got = (
            f(query, documents, timeout_ms) if inspect.iscoroutinefunction(f) else await asyncio.to_thread(f, query, documents, timeout_ms)
        )
        if inspect.isawaitable(got):
            got = await got
        return list(got) if got is not None else None

    return rerank


class _Engine:
    def __init__(self, directory: str, log: Callable[[str], None]) -> None:
        self.dir = directory
        self.log = log
        self.proc: QdrantProcess | None = None
        self.local: LocalEmbedder | None = None
        self.client: QdrantClient | None = None
        self.closing = False
        self._queue: asyncio.Future[Any] | None = None

    async def start(
        self,
        embedding: EmbeddingOptions | None,
        rerank: Any,
        llm: Any,
        qdrant: QdrantOptions | None,
        concurrency: int | None,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        e = embedding or {}
        # Embedding: here, or on a server — the options' own, else the one the environment names.
        server = e.get("server") or (None if e.get("model_dir") else resolve_embed_route())
        if not server or e.get("model_dir"):
            model_dir = e.get("model_dir") or await ensure_model_async(log=self.log)
            self.local = await asyncio.to_thread(LocalEmbedder, model_dir, e.get("threads", 2))
        if server:
            fixed = server
            self.embed = remote_embedder(lambda: fixed, transport=transport)
        else:
            assert self.local is not None
            self.embed = self.local.embed
        self.encode: WordEncoder | None = self.local.words if self.local else None
        self.words = WordStore(os.path.join(self.dir, "words.bin"))
        self.words.load()

        # The index: a Qdrant of its own, or yours.
        q = qdrant or {}
        if q.get("url"):
            fixed_endpoint = QdrantEndpoint(url=q["url"].rstrip("/"), api_key=q.get("api_key", ""))
            endpoint: Callable[[], QdrantEndpoint | None] = lambda: fixed_endpoint  # noqa: E731
        else:
            binary = q.get("binary") or await ensure_qdrant_async(log=self.log)
            self.proc = QdrantProcess(binary, os.path.join(self.dir, "qdrant"), self.log)
            await self.proc.start()
            proc = self.proc
            endpoint = proc.current
        # A server you name is reached through ``transport``; memory's own Qdrant always over loopback.
        own = not q.get("url")
        self.client = QdrantClient(
            endpoint, 10_000, COLLECTION if own else q.get("collection") or COLLECTION, transport=None if own else transport
        )
        await self.client.ensure_collection()

        self.stm = stm_log(os.path.join(self.dir, "records.jsonl"))
        self.ltm = ltm_log(os.path.join(self.dir, "knowledge.jsonl"))
        self.stm.load()
        self.ltm.load()
        self.usage = UsageBook(os.path.join(self.dir, "usage.json"))
        self.book = RecallBook(os.path.join(self.dir, "recall.jsonl"))
        self.book.trim()  # the documented cap: once an open, and every so many events while open (recall_log.py)
        self.ask: Ask = llm if callable(llm) else OpenAiAsk(**llm) if isinstance(llm, Mapping) else default_ask
        self.concurrency = concurrency

        rr: Rerank | None
        if rerank is False:
            rr = None
        elif callable(rerank):
            rr = _as_rerank(rerank)
        elif isinstance(rerank, Mapping):
            route: RerankRoute = dict(rerank)  # type: ignore[assignment]
            rr = remote_reranker(lambda: route, transport=transport)
        else:
            env = resolve_rerank_route() if rerank is None else None
            rr = (
                remote_reranker(lambda: env, transport=transport)
                if env
                else word_reranker(self.encode, self.words)
                if self.encode
                else None
            )
        self.deps = RecallDeps(
            client=self.client, embed=self.embed, rerank=rr, used=lambda ids, what: self.usage.note(ids, what), record=self.book.record
        )

        # The index is derived: written again from the logs when it is empty while they are not (a new Qdrant, a deleted
        # folder), or was written in another layout — or in one not written down (an index from before the layouts).
        self.rebuilder = MemoryRebuild(
            RebuildDeps(
                path=os.path.join(self.dir, "rebuild.json"),
                stm=self.stm,
                ltm=self.ltm,
                write_stm=lambda xs: write_stm(self.base(), xs),
                write_ltm=lambda xs: write_ltm(self.base(), xs),
                aborted=lambda: self.closing,
            )
        )
        indexed = await self.client.count()
        layout = _index_layout(self.dir)
        if layout != INDEX_LAYOUT:
            if indexed:
                shown = "unknown" if layout is None else f"{layout:g}"
                self.log(f"[memtrails] the index was written in layout {shown}, not {INDEX_LAYOUT}: writing it again from the logs")
                self.rebuilder.begin()  # before the drop: a crash after it resumes
                await self.client.drop_collection()
                await self.client.ensure_collection()
            _save_index_layout(self.dir)
        if self.rebuilder.pending() or (not indexed and (self.stm.size() or self.ltm.size())):
            self.log("[memtrails] writing the index from the logs")
            if not self.rebuilder.pending():
                self.rebuilder.begin()
            await self.rebuilder.run()

    def base(self) -> ConsolidateDeps:
        return ConsolidateDeps(
            client=self.client, embed=self.embed, stm=self.stm, ltm=self.ltm, log=self.log, aborted=lambda: self.closing, ask=self.ask
        )

    async def exclusive(self, job: Callable[[], Awaitable[T]]) -> T:
        """consolidate, sleep, rebuild, index_words and remove one at a time: each reads the logs and writes the index
        from them — two consolidations pay the model twice and write stale copies over each other, a rebuild clears what
        another is writing, a remove during a consolidation would see its record's episode written back after it."""
        before = self._queue

        async def run() -> T:
            if before is not None:
                try:
                    await asyncio.shield(before)
                except Exception:  # noqa: BLE001 - the job before failed for its caller
                    pass
            if self.closing:
                raise RuntimeError("memory is closed")
            return await job()

        task = asyncio.ensure_future(run())
        self._queue = task
        return await task

    # --- what is written ---

    def _words_of_entries(self, es: Sequence[StmEntry]) -> list[str]:
        out: list[str] = []
        for x in es:
            room = u16slice(f"{x['request']}\n{x['reply']}", 0, _ROOM_TEXT)
            out += [f"{stm_title(x)}\n{raw_key(x)}", *(_span(room, a, b) for a, b in sentence_spans(room))]
        return out

    async def encode_missing(self, texts: Sequence[str]) -> int:
        if not self.encode:
            return 0
        todo = [t for t in dict.fromkeys(texts) if t.strip() and not self.words.get(t)]
        for i in range(0, len(todo), _BATCH):
            if self.closing:
                break
            part = todo[i : i + _BATCH]
            encoded = await self.encode(part, "document")
            self.words.put([(t, pack_words(encoded[n])) for n, t in enumerate(part)])
        return len(todo)

    async def write(self, entries: Sequence[StmEntry], total: int) -> dict[str, int]:
        fresh = [x for x in entries if (self.stm.meta(x["uid"]) or {}).get("fp") != x["fingerprint"]]
        for i in range(0, len(fresh), _BATCH):
            part = fresh[i : i + _BATCH]
            await write_stm(self.base(), part)
            self.stm.put(part)
            await self.encode_missing(self._words_of_entries(part))
        self.stm.compact()
        return {"added": len(fresh), "unchanged": total - len(fresh)}

    async def index_words(self) -> int:
        """Unlocked: consolidate and sleep run it inside their own turn."""
        if not self.encode:
            return 0
        texts = self._words_of_entries(list(self.stm.all()))
        for uid, m in self.ltm.entries():
            if m.get("status") != "active":
                continue
            it = self.ltm.get(uid)
            if not it:
                continue
            room = u16slice(f"{it['title']}\n{it['summary']}\n{it['body']}", 0, _ROOM_TEXT)
            texts += [f"{it['title']}\n{it['summary']}", *(_span(room, a, b) for a, b in sentence_spans(room))]
        return await self.encode_missing(texts)

    async def add(self, records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        now = now_ms()
        for r in records:
            bad = invalid_record(r)
            if bad:
                raise TypeError(bad)
            if u16len(r["text"]) > _RECORD_LIMIT:
                raise TypeError(f"record {r['id']}: longer than {_RECORD_LIMIT} characters; split it")
        return await self.write([entry_of(r, now) for r in records], len(records))

    async def add_conversation(self, c: Mapping[str, Any]) -> dict[str, int]:
        if not isinstance(c, Mapping) or not c.get("id") or not isinstance(c.get("turns"), list):
            raise TypeError("a conversation needs an id and turns")
        entries = conversation_entries(c, now_ms())
        return await self.write(entries, len(entries))

    async def remove(self, ids: Sequence[str | Mapping[str, Any]]) -> int:
        async def job() -> int:
            uids = [u for u in (record_uid({"id": x} if isinstance(x, str) else x) for x in ids) if self.stm.meta(u)]
            if not uids:
                return 0
            episodes = [f"ep:{u}" for u in uids]
            assert self.client is not None
            await self.client.delete_items([*(stm_item(u) for u in uids), *episodes])
            self.stm.remove(uids)
            # A kept episode is the record's own text: gone from the long-term log too, or a rebuild indexes it again and
            # the profiles go on reading it.
            self.ltm.remove(episodes)
            return len(uids)

        return await self.exclusive(job)

    # --- what is found ---

    def _room_item(self, h: RecalledItem) -> RoomItem:
        """A found item's whole text for the room: a record's own, or what consolidation wrote."""
        r = self.stm.get(h["id"][4:]) if h["id"].startswith("stm:") else None
        it = None if r else self.ltm.get(h["id"])
        text = (
            f"{r['request']}\n{r['reply']}"
            if r
            else f"{it['title']}\n{it['summary']}\n{it['body']}"
            if it
            else f"{h['title']}\n{h['summary']}"
        )
        told = iso_day(h["ts"]) if h["ts"] else ""
        lasted = lasted_text(h)
        event = h.get("eventAt")
        happened = ", ".join(x for x in (readable_date(event) if event else "", lasted) if x)
        when = f"{happened}{f'; told {readable_date(told)}' if event != told or lasted else ''}" if happened else readable_date(told)
        return RoomItem(
            id=h["id"], head=f"({when} · {h['kind']})", text=u16slice(text, 0, _ROOM_TEXT), ts=date_parse(event) if event else h["ts"]
        )

    async def room(self, q: Mapping[str, Any]) -> str:
        if not self.encode:
            raise RuntimeError("the room needs the local embedding model (embedding model_dir, or no embedding server)")
        found = await memory_search(self.deps, {**q, "limit": q.get("limit") or 20})
        return await sentence_room(q["query"], [self._room_item(h) for h in found], self.encode, self.words, q.get("chars") or _ROOM_CHARS)

    async def open(self, id: str) -> dict[str, Any] | None:
        p = await memory_open(self.deps, id)
        if not p:
            return None
        # A record: its kind as given. Something learned: the records it was learned from, whole.
        if p.get("tier") == "stm":
            return {**p, "kind": p.get("engine")}
        it = self.ltm.get(id)
        sources = []
        for u in (it or {}).get("provenance") or []:
            x = self.stm.get(u)
            if x:
                src = {"id": stm_item(x["uid"]), "when": x["ts"], "title": x["title"], "body": episode_body(x)}
                if x.get("record"):
                    src["record"] = x["record"]
                sources.append(src)
        return {**p, "sources": sources} if sources else p

    async def consolidate(self) -> dict[str, Any]:
        async def job() -> dict[str, Any]:
            worker = Consolidator(ConsolidateDeps(**{**self.base().__dict__, "concurrency": self.concurrency or 4}))
            total = new_report()
            for _ in range(1000):
                if self.closing or not worker.pending():
                    break
                r = await worker.run(64)
                for k, v in r.items():
                    if k != "paused" and isinstance(v, (int, float)):
                        total[k] += v
                if "paused" in r:
                    total["paused"] = r["paused"]
                    break
                if not r["windows"]:
                    break
            await self.index_words()
            return total

        return await self.exclusive(job)

    async def sleep(self) -> dict[str, Any]:
        async def job() -> dict[str, Any]:
            r = await Sleep(SleepDeps(**self.base().__dict__, usage=self.usage, state_path=os.path.join(self.dir, "sleep.json"))).run()
            await self.index_words()
            return r

        return await self.exclusive(job)

    async def rebuild(self) -> None:
        async def job() -> None:
            assert self.client is not None
            self.rebuilder.begin()  # before the drop: a crash, or a close, after it resumes at the next open
            await self.client.drop_collection()
            await self.client.ensure_collection()
            await self.rebuilder.run()

        await self.exclusive(job)

    async def close(self) -> None:
        self.closing = True
        if self._queue is not None:
            try:
                await asyncio.shield(self._queue)  # the job running stops at its next step; those waiting are refused
            except Exception:  # noqa: BLE001 - its caller was told
                pass
        await self.shut()

    async def shut(self) -> None:
        """Releases what an open started, whatever it got to: a Qdrant left running keeps the host alive."""
        if hasattr(self, "usage"):
            self.usage.close()
        if self.proc is not None:
            await self.proc.stop_and_wait()
        if self.client is not None:
            await self.client.aclose()
        if self.local is not None:
            await asyncio.to_thread(self.local.close)


def _span(text: str, a: int, b: int) -> str:
    """``text.slice(a, b)`` with a and b in UTF-16 units."""
    return u16slice(text, a, b)


# --- the public, synchronous memory --------------------------------------------------------------------------------


class _Loop:
    """An event loop on a thread of its own: what a memory runs on."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="memtrails", daemon=True)
        self.thread.start()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        if threading.current_thread() is self.thread:
            coro.close()
            raise RuntimeError("a memory's methods cannot be called from inside its own work (a reranker or a model it runs)")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


class Memory:
    """A memory, opened with ``open_memory``. Use it as a context manager, or call ``close()``."""

    def __init__(self, engine: _Engine, runner: _Loop, release: Callable[[], None]) -> None:
        self._engine = engine
        self._loop = runner
        self._release = release
        self._closed = False
        self._close_lock = threading.Lock()

    def _do(self, work: Coroutine[Any, Any, T]) -> T:
        if self._closed:
            work.close()
            raise RuntimeError("memory is closed")
        return self._loop.run(work)

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, kind: type[BaseException] | None, err: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    # --- adding ---

    def add(self, records: MemoryRecord | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """Add or update records. Unchanged ones cost nothing. Searchable when this returns: ``{added, unchanged}``.
        Raises ``TypeError`` for a record it cannot keep, saying why."""
        listed = [records] if isinstance(records, Mapping) else list(records)
        return self._do(self._engine.add(listed))

    def add_conversation(self, conversation: Conversation | Mapping[str, Any]) -> dict[str, int]:
        """Add or update a conversation: one record for each time the person speaks, with what was answered."""
        return self._do(self._engine.add_conversation(conversation))

    def import_(
        self,
        records: Iterable[Mapping[str, Any]] | AsyncIterable[Mapping[str, Any]],
        batch: int = 256,
        on_progress: Callable[[int], None] | None = None,
    ) -> dict[str, int]:
        """Add records from any source (an importer, a database cursor) in batches, streaming."""
        total = {"added": 0, "unchanged": 0}
        size = max(1, batch)

        def flush(part: list[Mapping[str, Any]]) -> None:
            if not part:
                return
            r = self.add(part)
            total["added"] += r["added"]
            total["unchanged"] += r["unchanged"]
            if on_progress:
                on_progress(total["added"] + total["unchanged"])

        part: list[Mapping[str, Any]] = []
        if isinstance(records, AsyncIterable):
            it = records.__aiter__()

            async def nxt() -> Any:
                try:
                    return await it.__anext__()
                except StopAsyncIteration:
                    return _END

            while (r := self._do(nxt())) is not _END:
                part.append(r)
                if len(part) >= size:
                    flush(part)
                    part = []
        else:
            for r in records:
                part.append(r)
                if len(part) >= size:
                    flush(part)
                    part = []
        flush(part)
        return total

    def remove(self, ids: Sequence[str | Mapping[str, Any]]) -> int:
        """Forget records (by id, or ``{id, source}``): gone from search. Returns how many were found."""
        return self._do(self._engine.remove(ids))

    # --- finding ---

    def search(self, q: SearchOptions | Mapping[str, Any]) -> list[RecalledItem]:
        """Best first: records and what consolidation learned, each with its key text, dates, and the passage that
        matched."""
        return self._do(memory_search(self._engine.deps, q))

    def search_text(self, q: SearchOptions | Mapping[str, Any]) -> str:
        """The same search as a reader is shown it: one line a memory, best first or oldest first (``order: time``)."""
        hits = self._do(memory_search(self._engine.deps, q))
        return search_text(hits, q.get("order") or "relevance")

    def room(self, q: SearchOptions | Mapping[str, Any]) -> str:
        """What a small room should hold for a question: the best sentences of what search found, oldest first, in
        ``chars`` (6,000). Needs the local model."""
        return self._do(self._engine.room(q))

    def open(self, id: str) -> dict[str, Any] | None:
        """One item whole, by the id search returned; for something learned, ``sources``: the records it was learned
        from."""
        return self._do(self._engine.open(id))

    def explore(self, q: Mapping[str, Any]) -> Explored:
        """Around a topic (``topic``), or around items you found (``from``: ids): what the concept graph reaches that
        search did not (``related``, with ``via``), and its ``concepts``."""
        return self._do(memory_explore(self._engine.deps, q))

    def gather(self, q: SearchOptions | Mapping[str, Any], notes: bool = False) -> Gathered:
        """A set for a question that counts, lists or totals: a wide search, then the language model keeps every
        instance (``found``: those first). ``limit`` 200 by default; with ``notes``, what the kept memories say."""
        e = self._engine
        return self._do(memory_gather(e.deps, q, ask=e.ask, limit=q.get("limit") or None, notes=notes))

    # --- knowledge ---

    def consolidate(self) -> dict[str, Any]:
        """Read what was added since last time and keep typed knowledge, dated observations and kept episodes. Needs a
        language model; ``paused`` says why it did not read, when it did not."""
        return self._do(self._engine.consolidate())

    def sleep(self) -> dict[str, Any]:
        """The offline pass: merge near-duplicates, retire what was superseded, link, write profiles. Once a day."""
        return self._do(self._engine.sleep())

    # --- upkeep ---

    def index_words(self) -> int:
        """Encode what word by word and the room do not hold yet (done after every add, consolidate and sleep)."""
        return self._do(self._engine.exclusive(self._engine.index_words))

    def rebuild(self) -> None:
        """Write the index again from the logs."""
        self._do(self._engine.rebuild())

    def tools(self) -> list[AgentTool]:
        """Tool definitions an agent can be handed: memory_search, memory_open, memory_explore."""
        return build_agent_tools(_Provider(self))

    def stats(self) -> dict[str, int]:
        e = self._engine
        return {"records": e.stm.size(), "items": sum(1 for _, m in e.ltm.entries() if m.get("status") == "active"), "words": e.words.size}

    def close(self) -> None:
        """Stops a consolidate, sleep, rebuild or index_words at its next step (a rebuild resumes at the next open) and
        waits for it, then releases Qdrant, the model and the home."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._loop.run(self._engine.close())  # closing: not refused
        finally:
            self._release()
            self._loop.stop()


_END = object()


class _Provider:
    def __init__(self, m: Memory) -> None:
        self._m = m

    def search(self, q: Mapping[str, Any]) -> list[dict[str, Any]]:
        return list(self._m.search(q))  # type: ignore[arg-type]

    def open(self, id: str) -> dict[str, Any] | None:
        return self._m.open(id)

    def explore(self, q: Mapping[str, Any]) -> Explored:
        return self._m.explore(q)


def open_memory(
    home: str | os.PathLike[str],
    *,
    embedding: EmbeddingOptions | None = None,
    rerank: Literal["words"] | RerankRoute | Callable[..., Any] | Literal[False] | None = None,
    llm: Ask | Mapping[str, Any] | None = None,
    qdrant: QdrantOptions | None = None,
    concurrency: int | None = None,
    log: Callable[[str], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Memory:
    """Open (or make) the memory in ``home``.

    ``embedding``: EmbeddingOptions. ``rerank``: ``"words"`` (default with a local model), a rerank server
    ``{url, api_key?, model?, gzip?}``, your own function ``(query, documents, timeout_ms) -> scores | None`` (plain or
    async), or ``False``; not given, MEMTRAILS_RERANK_URL names a server when set. ``llm``: your own ``Ask``, or the
    OpenAI-compatible one's options (``model``, ``api_key``, ``base_url``, ``reasoning_effort``, ``on_usage``,
    ``retries``); default the environment's (MEMTRAILS_MODEL and a key). ``qdrant``: a server of your own, or a binary;
    default, a process of its own (the pinned binary, fetched on first use). ``concurrency``: windows the language model
    reads at once (4). ``transport``: replaces the network for the servers you name (the embedding, rerank and Qdrant servers; tests)."""
    directory = os.path.abspath(os.fspath(home))
    say = log or (lambda _line: None)
    # Private: the logs, the index and the word vectors are everything that was added (a shared machine).
    os.makedirs(directory, mode=0o700, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError as err:
        say(f"[memtrails] {directory} could not be made private: {err}")
    release = _lock_home(directory)
    runner = _Loop()
    engine = _Engine(directory, say)
    try:
        runner.run(engine.start(embedding, rerank, llm, qdrant, concurrency, transport))
    except BaseException:
        try:
            runner.run(engine.shut())
        finally:
            release()
            runner.stop()
        raise
    return Memory(engine, runner, release)
