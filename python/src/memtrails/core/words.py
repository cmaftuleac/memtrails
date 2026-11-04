"""
Word by word: memory_search's order with no server. A memory's one vector averages it, so one exact detail in a long
memory is drowned out. The embedder has a vector per word before it averages them; a memory scores the sum, over the
question's words, of its closest word (MaxSim, as ColBERT does). Over the search's best 30, fused by rank as the
reranker is (reranker.py ``reranked_order``). No training, no word list, any language.

Measured (docs/BENCHMARKS.md): all 2,011 questions, answer shown at 5 50.3 → 53.1 (+80/−24), at 10 57.1 → 58.4
(+67/−40); no refinement of the score beat this plain sum; other languages against English memories gain more
(Romanian at 5 43.0 → 48.2). One bit a number keeps nearly all of it, and 128 words are enough where 64 are not.

A memory's words are kept small, the question's whole: its cosine with a word stored as signs is the sum of its
numbers with those signs, over √dims. Kept by the text's hash, so an unchanged text is never encoded twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import struct
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._js import utf8

#: Words kept of a memory (64 lost most of the gain).
WORD_LIMIT = 128
_KEY_BYTES = 8


@dataclass
class WordVectors:
    """Unit vectors, one a word, flat: ``words × dims`` numbers."""

    words: int
    dims: int
    data: npt.NDArray[np.float32]


@dataclass
class PackedWords:
    """A memory's words as signs, one bit a number, a word to a whole number of bytes."""

    words: int
    dims: int
    bits: bytes

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PackedWords) and (self.words, self.dims, bytes(self.bits)) == (
            other.words,
            other.dims,
            bytes(other.bits),
        )

    __hash__ = None  # type: ignore[assignment]


WordEncoder = Callable[[list[str], Literal["query", "document"]], Awaitable[list[WordVectors]]]


def pack_words(v: WordVectors, limit: int = WORD_LIMIT) -> PackedWords:
    words = min(v.words, limit)
    if not words or not v.dims:
        return PackedWords(words, v.dims, b"")
    m = np.asarray(v.data, dtype=np.float32)[: words * v.dims].reshape(words, v.dims)
    bits = np.packbits(m >= 0, axis=1, bitorder="little")
    return PackedWords(words, v.dims, bits.tobytes())


def _signs(p: PackedWords) -> npt.NDArray[np.float64]:
    stride = (p.dims + 7) // 8
    raw = np.frombuffer(p.bits, dtype=np.uint8)[: p.words * stride].reshape(p.words, stride)
    on = np.unpackbits(raw, axis=1, bitorder="little")[:, : p.dims]
    return np.where(on == 1, 1.0, -1.0)


def max_sim(query: WordVectors, memory: PackedWords) -> float:
    """The sum, over the question's words, of the closest word of the memory."""
    if not query.words or not memory.words or query.dims != memory.dims:
        return -math.inf
    q = np.asarray(query.data, dtype=np.float32)[: query.words * query.dims].reshape(query.words, query.dims)
    scores = q.astype(np.float64) @ _signs(memory).T
    return float(np.sum(scores.max(axis=1) / math.sqrt(query.dims)))


def _key_of(text: str) -> str:
    return hashlib.sha256(utf8(text)).hexdigest()[: _KEY_BYTES * 2]


class WordStore:
    """The packed words by text, in memory and appended to a file: key, word count, numbers a word, the bits. Derived,
    like the index: losing it only means encoding again."""

    def __init__(self, file: str | None = None) -> None:
        self._file = file
        self._by_key: dict[str, PackedWords] = {}

    @property
    def size(self) -> int:
        return len(self._by_key)

    def get(self, text: str) -> PackedWords | None:
        return self._by_key.get(_key_of(text))

    def load(self) -> None:
        if not self._file:
            return
        try:
            with open(self._file, "rb") as f:
                b = f.read()
        except OSError:
            return
        at = 0
        while at + _KEY_BYTES + 4 <= len(b):
            key = b[at : at + _KEY_BYTES].hex()
            words, dims = struct.unpack_from("<HH", b, at + _KEY_BYTES)
            size = words * ((dims + 7) // 8)
            start = at + _KEY_BYTES + 4
            if start + size > len(b):
                break  # cut short by a crash: what came before is whole
            self._by_key[key] = PackedWords(words, dims, b[start : start + size])
            at = start + size
        # The cut-off entry goes, or every entry appended after it would be read as part of it and never load again.
        if at < len(b):
            try:
                os.truncate(self._file, at)
            except OSError:
                pass  # read-only: encoded again when needed

    def put(self, entries: Sequence[tuple[str, PackedWords]]) -> None:
        parts: list[bytes] = []
        for text, p in entries:
            key = _key_of(text)
            self._by_key[key] = p
            parts.append(bytes.fromhex(key) + struct.pack("<HH", p.words, p.dims) + bytes(p.bits))
        if not self._file or not parts:
            return
        d = os.path.dirname(self._file)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self._file, "ab") as f:
            f.write(b"".join(parts))


Rerank = Callable[[str, list[str], float], Awaitable["list[float] | None"]]


def word_reranker(encode: WordEncoder, store: WordStore) -> Rerank:
    """A reranker for memory_search (RecallDeps.rerank) that needs no server: what is not stored yet is encoded here,
    once. None — the search keeps its own order — when the encoder fails or takes longer than it was given."""

    async def rerank(query: str, documents: list[str], timeout_ms: float) -> list[float] | None:
        async def work() -> list[float]:
            missing = list(dict.fromkeys(t for t in documents if not store.get(t)))
            if missing:
                encoded = await encode(missing, "document")
                store.put([(t, pack_words(encoded[i])) for i, t in enumerate(missing)])
            q = (await encode([query], "query"))[0]
            packed = [store.get(t) or PackedWords(0, q.dims, b"") for t in documents]
            return await asyncio.to_thread(lambda: [max_sim(q, p) for p in packed])

        try:
            return await asyncio.wait_for(work(), timeout_ms / 1000)
        except Exception:  # noqa: BLE001 - a failed or late encoder leaves the search its own order
            return None

    return rerank
