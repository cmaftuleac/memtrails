"""
EmbeddingGemma on this machine, for a memory that needs no server: the same quantised model and the same prompts as
the reference, so vectors and an index move between the two. One run gives both what the search keeps (the memory's
vector, ``sentence_embedding``) and what word by word keeps (a vector per word, the step before it is averaged:
``last_hidden_state``).

The model and its tokenizer are a directory you give it (onnx/model_q4.onnx, tokenizer.json). One inference at a time,
on a thread of its own, so the event loop stays free.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from .embedder import Vector, doc_text, query_text
from .words import WORD_LIMIT, WordVectors

_BOS_EOS = 2


class Tokens(Protocol):
    """What a tokenizer gives for a text: its token ids, the start and end tokens included."""

    def __call__(self, text: str) -> list[int]: ...


class Session(Protocol):
    """The slice of an onnxruntime InferenceSession used here."""

    def get_outputs(self) -> Sequence[Any]: ...

    def run(self, names: Sequence[str] | None, feeds: dict[str, Any]) -> Sequence[npt.NDArray[np.float32]]: ...


def _unit(v: npt.NDArray[Any]) -> Vector:
    """Divided by its length (a sum in double precision), kept as float32."""
    x = np.asarray(v, dtype=np.float64)
    n = float(np.sqrt(np.sum(x * x))) or 1.0
    return (x / n).astype(np.float32)


class LocalEmbedder:
    """``embed(texts)`` for prompted texts → unit vectors, in order; ``words(texts, kind)`` → a unit vector a word.
    ``session``/``tokenizer``: stand-ins for onnxruntime and the tokenizer (tests)."""

    def __init__(
        self,
        model_dir: str,
        threads: int = 1,
        batch: int = 8,
        max_tokens: int = 512,
        session: Session | None = None,
        tokenizer: Tokens | None = None,
    ) -> None:
        if tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as err:  # pragma: no cover - a dependency
                raise RuntimeError(f"The local embedder needs the package tokenizers: {err}") from err
            tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

            def encode(text: str) -> list[int]:
                return list(tok.encode(text).ids)

            tokenizer = encode
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as err:  # pragma: no cover - a dependency
                raise RuntimeError(f"The local embedder needs the package onnxruntime: {err}") from err
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, threads)
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                os.path.join(model_dir, "onnx", "model_q4.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        self._session = session
        self._tokens = tokenizer
        self._batch = max(1, batch)
        self._max_tokens = max_tokens
        self._names = [o.name for o in session.get_outputs()]
        # One inference at a time, like lengths in one run: each text is padded to the run's longest.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memtrails-embed")

    def _ids(self, text: str, limit: int) -> tuple[list[int], bool]:
        ids = self._tokens(text)
        return ids[:limit], len(ids) > limit

    def _run(self, encoded: Sequence[Sequence[int]]) -> dict[str, npt.NDArray[np.float32]]:
        width = max(1, *(len(e) for e in encoded))
        ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for b, e in enumerate(encoded):
            ids[b, : len(e)] = e
            mask[b, : len(e)] = 1
        out = self._session.run(None, {"input_ids": ids, "attention_mask": mask})
        return dict(zip(self._names, out, strict=True))

    def _batches(
        self, items: list[tuple[list[int], bool]], read: Callable[[dict[str, Any], int, tuple[list[int], bool]], Any]
    ) -> list[Any]:
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        results: list[Any] = [None] * len(items)
        for i in range(0, len(order), self._batch):
            part = order[i : i + self._batch]
            out = self._run([items[j][0] for j in part])
            for b, j in enumerate(part):
                results[j] = read(out, b, items[j])
        return results

    def _embed(self, texts: Sequence[str]) -> list[Vector]:
        def read(out: dict[str, Any], b: int, _it: tuple[list[int], bool]) -> Vector:
            t = out.get("sentence_embedding")
            if t is None:
                t = out[self._names[-1]]
            return _unit(t[b])

        return self._batches([self._ids(t, self._max_tokens) for t in texts], read)

    def _words(self, texts: Sequence[str], kind: Literal["query", "document"]) -> list[WordVectors]:
        prompted = [query_text(t) if kind == "query" else doc_text("", t) for t in texts]

        def read(out: dict[str, Any], b: int, it: tuple[list[int], bool]) -> WordVectors:
            h = out.get("last_hidden_state")
            if h is None:
                raise RuntimeError("This model gives no vector per word (last_hidden_state).")
            dims = int(h.shape[2])
            # Without its first token, and its last unless the text was cut before it.
            words = max(0, len(it[0]) - (1 if it[1] else _BOS_EOS))
            data = np.zeros(words * dims, dtype=np.float32)
            for w in range(words):
                data[w * dims : (w + 1) * dims] = _unit(h[b, w + 1])
            return WordVectors(words, dims, data)

        return self._batches([self._ids(t, WORD_LIMIT + _BOS_EOS) for t in prompted], read)

    async def embed(self, texts: Sequence[str], timeout_ms: float | None = None) -> list[Vector]:
        if not texts:
            return []
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._embed, list(texts))

    async def words(self, texts: list[str], kind: Literal["query", "document"]) -> list[WordVectors]:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._words, texts, kind)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
