"""
memtrails: open a memory, add anything with text, ask it things.

    async with await open_memory(home="./memory") as memory:
        await memory.add({"id": "mail-1", "kind": "email", "title": "Gateway certificate", "author": "Ana",
                          "text": "It expired on the 1st; the renewal job had been removed.", "at": "2026-04-02"})
        await memory.search({"query": "why did the certificate stop renewing?"})

What is added is searchable at once, with no language model and no server: the embedding model runs here (fetched on
first use) and orders results word by word (words.py). A language model is needed only for ``consolidate``, ``sleep``
and ``gather``, which turn records into typed knowledge, dated observations and profiles — and memory works without
them.

The JSONL logs in ``home`` are the truth. The Qdrant index and the word vectors are derived: lost or out of date, they
are written again from the logs. A home is the reference library's too: either opens what the other wrote.

Everything is asyncio: the HTTP calls (one shared client), Qdrant's process, the language model's asks, several at
once; the model's inference, word-by-word scoring and the reading or rewriting of whole files run on threads, so the
event loop never waits on them.
"""

from __future__ import annotations

import asyncio
import atexit
import inspect
import os
import threading
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, TypedDict, TypeVar

import httpx

from ._fs import write_all, write_file
from ._http import new_client
from ._js import date_parse, iso_day, now_ms, parse, stringify, u16len, u16slice
from .core.consolidate import ConsolidateDeps, Consolidator, new_report
from .core.embedder import Embed, EmbedRoute, remote_embedder, resolve_embed_route
from .core.episodes import episode_body
from .core.gather import Gathered, memory_gather
from .core.local_embedder import LocalEmbedder
from .core.model import ensure_model
from .core.qdrant_binary import ensure_qdrant
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
from .llm import Ask, OpenAiAsk
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
            write_all(fd, stringify({"pid": os.getpid(), "boot": boot_time()}).encode("utf-8"))
        except BaseException:
            # A lock that says no pid would turn the next opener away for ten seconds, then be taken over anyway.
            os.close(fd)
            try:
                os.remove(path)
            except OSError:
                pass  # gone
            raise
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
    write_file(f"{path}.tmp", stringify({"layout": INDEX_LAYOUT}).encode("utf-8"))
    os.replace(f"{path}.tmp", path)


def _as_rerank(f: Callable[..., Any]) -> Rerank:
    """Your own reranker, ``async`` (or a plain function, run on a thread)."""

    async def rerank(query: str, documents: list[str], timeout_ms: float) -> list[float] | None:
        if inspect.iscoroutinefunction(f):
            got = await f(query, documents, timeout_ms)
        else:
            got = await asyncio.to_thread(f, query, documents, timeout_ms)
            if inspect.isawaitable(got):
                got = await got
        return list(got) if got is not None else None

    return rerank


def _span(text: str, a: int, b: int) -> str:
    """``text.slice(a, b)`` with a and b in UTF-16 units."""
    return u16slice(text, a, b)


class Memory:
    """A memory, opened with ``await open_memory(...)``. Use it as an async context manager, or ``await close()``."""

    def __init__(
        self, directory: str, log: Callable[[str], None], http: httpx.AsyncClient, own_http: bool, release: Callable[[], None]
    ) -> None:
        self.dir = directory
        self._log = log
        self._http = http
        self._own_http = own_http
        #: Memory's own Qdrant, on loopback with its key, is called through a client of its own that never takes a
        #: proxy (nor .netrc credentials) from the environment — yours may (httpx's trust_env), and would hand the key
        #: and every document to the proxy. The reference (Node's fetch) reads no proxy variables either.
        self._loopback: httpx.AsyncClient | None = None
        self._release = release
        self._proc: QdrantProcess | None = None
        self._local: LocalEmbedder | None = None
        self._client: QdrantClient | None = None
        self._usage: UsageBook | None = None
        self._book: RecallBook | None = None
        #: consolidate, sleep, rebuild, index_words and remove one at a time: each reads the logs and writes the index
        #: from them — two consolidations pay the model twice and write stale copies over each other, a rebuild clears
        #: what another is writing, a remove during a consolidation would see its record's episode written back after it.
        self._jobs = asyncio.Lock()
        self._closing = False
        self._closed = False

    async def __aenter__(self) -> Memory:
        return self

    async def __aexit__(self, kind: type[BaseException] | None, err: BaseException | None, tb: TracebackType | None) -> None:
        await self.close()

    # --- opening ---

    async def _start(
        self, embedding: EmbeddingOptions | None, rerank: Any, llm: Any, qdrant: QdrantOptions | None, concurrency: int | None
    ) -> None:
        e = embedding or {}
        # Embedding: here, or on a server — the options' own, else the one the environment names.
        server = e.get("server") or (None if e.get("model_dir") else resolve_embed_route())
        if not server or e.get("model_dir"):
            model_dir = e.get("model_dir") or await ensure_model(log=self._log, http=self._http)
            self._local = await asyncio.to_thread(LocalEmbedder, model_dir, e.get("threads", 2))
        if server:
            fixed = server
            self._embed: Embed = remote_embedder(lambda: fixed, http=self._http)
        else:
            assert self._local is not None
            self._embed = self._local.embed
        self._encode: WordEncoder | None = self._local.words if self._local else None
        self._words = WordStore(os.path.join(self.dir, "words.bin"))
        await asyncio.to_thread(self._words.load)

        # The index: a Qdrant of its own, or yours.
        q = qdrant or {}
        endpoint: Callable[[], QdrantEndpoint | None]
        if q.get("url"):
            fixed_endpoint = QdrantEndpoint(url=q["url"].rstrip("/"), api_key=q.get("api_key", ""))

            def endpoint() -> QdrantEndpoint | None:
                return fixed_endpoint

            collection = q.get("collection") or COLLECTION
        else:
            binary = q.get("binary") or await ensure_qdrant(log=self._log, http=self._http)
            self._loopback = new_client()
            self._proc = QdrantProcess(binary, os.path.join(self.dir, "qdrant"), self._log, http=self._loopback)
            await self._proc.start()
            endpoint = self._proc.current
            collection = COLLECTION
        self._client = QdrantClient(endpoint, 10_000, collection, http=self._loopback or self._http)
        await self._client.ensure_collection()

        self._stm = stm_log(os.path.join(self.dir, "records.jsonl"))
        self._ltm = ltm_log(os.path.join(self.dir, "knowledge.jsonl"))
        await asyncio.to_thread(self._stm.load)
        await asyncio.to_thread(self._ltm.load)
        self._usage = UsageBook(os.path.join(self.dir, "usage.json"))
        book = self._book = RecallBook(os.path.join(self.dir, "recall.jsonl"))
        await asyncio.to_thread(book.trim)  # the documented cap: once an open, and every so many events while open
        if callable(llm):
            self._ask: Ask = llm
        else:
            options = dict(llm) if isinstance(llm, Mapping) else {}
            options.setdefault("http", self._http)
            self._ask = OpenAiAsk(**options)
        self._concurrency = concurrency

        rr: Rerank | None
        if rerank is False:
            rr = None
        elif callable(rerank):
            rr = _as_rerank(rerank)
        elif isinstance(rerank, Mapping):
            route: RerankRoute = dict(rerank)  # type: ignore[assignment]
            rr = remote_reranker(lambda: route, http=self._http)
        else:
            env = resolve_rerank_route() if rerank is None else None
            rr = (
                remote_reranker(lambda: env, http=self._http) if env else word_reranker(self._encode, self._words) if self._encode else None
            )
        usage = self._usage
        self._deps = RecallDeps(
            client=self._client, embed=self._embed, rerank=rr, used=lambda ids, what: usage.note(ids, what), record=book.record
        )

        # The index is derived: written again from the logs when it is empty while they are not (a new Qdrant, a deleted
        # folder), or was written in another layout — or in one not written down (an index from before the layouts).
        self._rebuilder = MemoryRebuild(
            RebuildDeps(
                path=os.path.join(self.dir, "rebuild.json"),
                stm=self._stm,
                ltm=self._ltm,
                write_stm=lambda xs: write_stm(self._base(), xs),
                write_ltm=lambda xs: write_ltm(self._base(), xs),
                aborted=lambda: self._closing,
            )
        )
        indexed = await self._client.count()
        layout = _index_layout(self.dir)
        if layout != INDEX_LAYOUT:
            if indexed:
                shown = "unknown" if layout is None else f"{layout:g}"
                self._log(f"[memtrails] the index was written in layout {shown}, not {INDEX_LAYOUT}: writing it again from the logs")
                self._rebuilder.begin()  # before the drop: a crash after it resumes
                await self._client.drop_collection()
                await self._client.ensure_collection()
            _save_index_layout(self.dir)
        if self._rebuilder.pending() or (not indexed and (self._stm.size() or self._ltm.size())):
            self._log("[memtrails] writing the index from the logs")
            if not self._rebuilder.pending():
                self._rebuilder.begin()
            await self._rebuilder.run()

    def _base(self) -> ConsolidateDeps:
        return ConsolidateDeps(
            client=self._client,
            embed=self._embed,
            stm=self._stm,
            ltm=self._ltm,
            log=self._log,
            aborted=lambda: self._closing,
            ask=self._ask,
        )

    def _check(self) -> None:
        if self._closing:
            raise RuntimeError("memory is closed")

    async def _exclusive(self, job: Callable[[], Awaitable[T]]) -> T:
        async with self._jobs:
            self._check()
            return await job()

    # --- adding ---

    def _words_of_entries(self, es: Sequence[StmEntry]) -> list[str]:
        out: list[str] = []
        for x in es:
            room = u16slice(f"{x['request']}\n{x['reply']}", 0, _ROOM_TEXT)
            out += [f"{stm_title(x)}\n{raw_key(x)}", *(_span(room, a, b) for a, b in sentence_spans(room))]
        return out

    async def _encode_missing(self, texts: Sequence[str]) -> int:
        if not self._encode:
            return 0
        todo = [t for t in dict.fromkeys(texts) if t.strip() and not self._words.get(t)]
        for i in range(0, len(todo), _BATCH):
            if self._closing:
                break
            part = todo[i : i + _BATCH]
            encoded = await self._encode(part, "document")
            self._words.put([(t, pack_words(encoded[n])) for n, t in enumerate(part)])
        return len(todo)

    async def _write(self, entries: Sequence[StmEntry], total: int) -> dict[str, int]:
        fresh = [x for x in entries if (self._stm.meta(x["uid"]) or {}).get("fp") != x["fingerprint"]]
        for i in range(0, len(fresh), _BATCH):
            part = fresh[i : i + _BATCH]
            await write_stm(self._base(), part)
            self._stm.put(part)
            await self._encode_missing(self._words_of_entries(part))
        await asyncio.to_thread(self._stm.compact)
        return {"added": len(fresh), "unchanged": total - len(fresh)}

    async def add(self, records: MemoryRecord | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """Add or update records. Unchanged ones cost nothing. Searchable when this returns: ``{added, unchanged}``.
        Raises ``TypeError`` for a record it cannot keep, saying why."""
        self._check()
        listed = [records] if isinstance(records, Mapping) else list(records)
        now = now_ms()
        for r in listed:
            bad = invalid_record(r)
            if bad:
                raise TypeError(bad)
            if u16len(r["text"]) > _RECORD_LIMIT:
                raise TypeError(f"record {r['id']}: longer than {_RECORD_LIMIT} characters; split it")
        return await self._write([entry_of(r, now) for r in listed], len(listed))

    async def add_conversation(self, conversation: Conversation | Mapping[str, Any]) -> dict[str, int]:
        """Add or update a conversation: one record for each time the person speaks, with what was answered."""
        self._check()
        c = conversation
        if not isinstance(c, Mapping) or not c.get("id") or not isinstance(c.get("turns"), list):
            raise TypeError("a conversation needs an id and turns")
        entries = conversation_entries(c, now_ms())
        return await self._write(entries, len(entries))

    async def import_(
        self,
        records: AsyncIterable[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
        batch: int = 256,
        on_progress: Callable[[int], None] | None = None,
    ) -> dict[str, int]:
        """Add records from any source (an importer, a database cursor), async or not, in batches, streaming."""
        total = {"added": 0, "unchanged": 0}
        size = max(1, batch)
        part: list[Mapping[str, Any]] = []

        async def flush() -> None:
            if not part:
                return
            r = await self.add(part)
            part.clear()
            total["added"] += r["added"]
            total["unchanged"] += r["unchanged"]
            if on_progress:
                on_progress(total["added"] + total["unchanged"])

        if isinstance(records, AsyncIterable):
            async for r in records:
                part.append(r)
                if len(part) >= size:
                    await flush()
        else:
            for r in records:
                part.append(r)
                if len(part) >= size:
                    await flush()
        await flush()
        return total

    async def remove(self, ids: Sequence[str | Mapping[str, Any]]) -> int:
        """Forget records (by id, or ``{id, source}``): gone from search. Returns how many were found."""

        async def job() -> int:
            uids = [u for u in (record_uid({"id": x} if isinstance(x, str) else x) for x in ids) if self._stm.meta(u)]
            if not uids:
                return 0
            episodes = [f"ep:{u}" for u in uids]
            assert self._client is not None
            await self._client.delete_items([*(stm_item(u) for u in uids), *episodes])
            self._stm.remove(uids)
            # A kept episode is the record's own text: gone from the long-term log too, or a rebuild indexes it again and
            # the profiles go on reading it.
            self._ltm.remove(episodes)
            return len(uids)

        return await self._exclusive(job)

    # --- finding ---

    async def search(self, q: SearchOptions | Mapping[str, Any]) -> list[RecalledItem]:
        """Best first: records and what consolidation learned, each with its key text, dates, and the passage that
        matched."""
        self._check()
        return await memory_search(self._deps, q)

    async def search_text(self, q: SearchOptions | Mapping[str, Any]) -> str:
        """The same search as a reader is shown it: one line a memory, best first or oldest first (``order: time``)."""
        return search_text(await self.search(q), q.get("order") or "relevance")

    def _room_item(self, h: RecalledItem) -> RoomItem:
        """A found item's whole text for the room: a record's own, or what consolidation wrote."""
        r = self._stm.get(h["id"][4:]) if h["id"].startswith("stm:") else None
        it = None if r else self._ltm.get(h["id"])
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

    async def room(self, q: SearchOptions | Mapping[str, Any]) -> str:
        """What a small room should hold for a question: the best sentences of what search found, oldest first, in
        ``chars`` (6,000). Needs the local model."""
        self._check()
        if not self._encode:
            raise RuntimeError("the room needs the local embedding model (embedding model_dir, or no embedding server)")
        found = await memory_search(self._deps, {**q, "limit": q.get("limit") or 20})
        return await sentence_room(
            q["query"], [self._room_item(h) for h in found], self._encode, self._words, q.get("chars") or _ROOM_CHARS
        )

    async def open(self, id: str) -> dict[str, Any] | None:
        """One item whole, by the id search returned; for something learned, ``sources``: the records it was learned
        from."""
        self._check()
        p = await memory_open(self._deps, id)
        if not p:
            return None
        # A record: its kind as given. Something learned: the records it was learned from, whole.
        if p.get("tier") == "stm":
            return {**p, "kind": p.get("engine")}
        it = self._ltm.get(id)
        sources = []
        for u in (it or {}).get("provenance") or []:
            x = self._stm.get(u)
            if x:
                src = {"id": stm_item(x["uid"]), "when": x["ts"], "title": x["title"], "body": episode_body(x)}
                if x.get("record"):
                    src["record"] = x["record"]
                sources.append(src)
        return {**p, "sources": sources} if sources else p

    async def explore(self, q: Mapping[str, Any]) -> Explored:
        """Around a topic (``topic``), or around items you found (``from``: ids): what the concept graph reaches that
        search did not (``related``, with ``via``), and its ``concepts``."""
        self._check()
        return await memory_explore(self._deps, q)

    async def gather(self, q: SearchOptions | Mapping[str, Any], notes: bool = False) -> Gathered:
        """A set for a question that counts, lists or totals: a wide search, then the language model keeps every
        instance (``found``: those first). ``limit`` 200 by default; with ``notes``, what the kept memories say."""
        self._check()
        return await memory_gather(self._deps, q, ask=self._ask, limit=q.get("limit") or None, notes=notes)

    # --- knowledge ---

    async def _index_words(self) -> int:
        """Unlocked: consolidate and sleep run it inside their own turn."""
        if not self._encode:
            return 0
        texts = self._words_of_entries(await asyncio.to_thread(lambda: list(self._stm.all())))
        for uid, m in self._ltm.entries():
            if m.get("status") != "active":
                continue
            it = self._ltm.get(uid)
            if not it:
                continue
            room = u16slice(f"{it['title']}\n{it['summary']}\n{it['body']}", 0, _ROOM_TEXT)
            texts += [f"{it['title']}\n{it['summary']}", *(_span(room, a, b) for a, b in sentence_spans(room))]
        return await self._encode_missing(texts)

    async def consolidate(self) -> dict[str, Any]:
        """Read what was added since last time and keep typed knowledge, dated observations and kept episodes. Needs a
        language model; ``paused`` says why it did not read, when it did not."""

        async def job() -> dict[str, Any]:
            worker = Consolidator(ConsolidateDeps(**{**self._base().__dict__, "concurrency": self._concurrency or 4}))
            total = new_report()
            for _ in range(1000):
                if self._closing or not worker.pending():
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
            await self._index_words()
            return total

        return await self._exclusive(job)

    async def sleep(self) -> dict[str, Any]:
        """The offline pass: merge near-duplicates, retire what was superseded, link, write profiles. Once a day."""

        async def job() -> dict[str, Any]:
            r = await Sleep(SleepDeps(**self._base().__dict__, usage=self._usage, state_path=os.path.join(self.dir, "sleep.json"))).run()
            await self._index_words()
            return r

        return await self._exclusive(job)

    # --- upkeep ---

    async def index_words(self) -> int:
        """Encode what word by word and the room do not hold yet (done after every add, consolidate and sleep)."""
        return await self._exclusive(self._index_words)

    async def rebuild(self) -> None:
        """Write the index again from the logs."""

        async def job() -> None:
            assert self._client is not None
            self._rebuilder.begin()  # before the drop: a crash, or a close, after it resumes at the next open
            await self._client.drop_collection()
            await self._client.ensure_collection()
            await self._rebuilder.run()

        await self._exclusive(job)

    def tools(self) -> list[AgentTool]:
        """Tool definitions an agent can be handed: memory_search, memory_open, memory_explore (async handlers)."""
        return build_agent_tools(self)

    def stats(self) -> dict[str, int]:
        return {
            "records": self._stm.size(),
            "items": sum(1 for _, m in self._ltm.entries() if m.get("status") == "active"),
            "words": self._words.size,
        }

    async def close(self) -> None:
        """Stops a consolidate, sleep, rebuild or index_words at its next step (a rebuild resumes at the next open) and
        waits for it, then releases Qdrant, the model and the home. Jobs waiting their turn are refused."""
        if self._closed:
            return
        self._closing = True
        async with self._jobs:  # the job running stops at its next step; those waiting before this are refused
            if self._closed:
                return
            self._closed = True
            try:
                await self._shut()
            finally:
                self._release()

    async def _shut(self) -> None:
        """Releases what an open started, whatever it got to: a Qdrant left running keeps the host alive."""
        try:
            if self._book is not None:
                await self._book.settled()  # a trim left running would rename its file into a home that is let go
            if self._usage is not None:
                self._usage.close()
            if self._proc is not None:
                await self._proc.stop_and_wait()
            if self._local is not None:
                await self._local.close()
        finally:
            if self._loopback is not None:
                await self._loopback.aclose()
            if self._own_http:
                await self._http.aclose()


async def open_memory(
    home: str | os.PathLike[str],
    *,
    embedding: EmbeddingOptions | None = None,
    rerank: Literal["words"] | RerankRoute | Callable[..., Any] | Literal[False] | None = None,
    llm: Ask | Mapping[str, Any] | None = None,
    qdrant: QdrantOptions | None = None,
    concurrency: int | None = None,
    log: Callable[[str], None] | None = None,
    http: httpx.AsyncClient | None = None,
) -> Memory:
    """Open (or make) the memory in ``home``.

    ``embedding``: EmbeddingOptions. ``rerank``: ``"words"`` (default with a local model), a rerank server
    ``{url, api_key?, model?, gzip?}``, your own ``async`` function ``(query, documents, timeout_ms) -> scores | None``,
    or ``False``; not given, MEMTRAILS_RERANK_URL names a server when set. ``llm``: your own ``async`` Ask, or the
    OpenAI-compatible one's options (``model``, ``api_key``, ``base_url``, ``reasoning_effort``, ``on_usage``,
    ``retries``); default the environment's (MEMTRAILS_MODEL and a key). ``qdrant``: a server of your own, or a binary;
    default, a process of its own (the pinned binary, fetched on first use). ``concurrency``: windows the language model
    reads at once (4). ``http``: the ``httpx.AsyncClient`` every call goes through (yours stays open when the memory
    closes); default, one of its own. Memory's own Qdrant is called through a client of its own that takes no proxy from
    the environment."""
    directory = os.path.abspath(os.fspath(home))
    say = log or (lambda _line: None)
    # Private: the logs, the index and the word vectors are everything that was added (a shared machine).
    os.makedirs(directory, mode=0o700, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError as err:
        say(f"[memtrails] {directory} could not be made private: {err}")
    release = _lock_home(directory)
    memory = Memory(directory, say, http or new_client(), http is None, release)
    try:
        await memory._start(embedding, rerank, llm, qdrant, concurrency)
    except BaseException:
        try:
            await memory._shut()
        finally:
            release()
        raise
    return memory
