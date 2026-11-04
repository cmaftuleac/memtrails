"""The local embedder over stand-ins for onnxruntime and the tokenizer (local-embedder.test.ts, ported)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fakes import run
from memtrails.core.local_embedder import LocalEmbedder

DIMS = 4


class Output:
    def __init__(self, name: str) -> None:
        self.name = name


class Session:
    """Each token id becomes a known vector, so what comes out shows which tokens were kept."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def get_outputs(self) -> list[Output]:
        return [Output("last_hidden_state"), Output("sentence_embedding")]

    def run(self, _names: Any, feeds: dict[str, Any]) -> list[np.ndarray]:
        ids = feeds["input_ids"]
        n, width = ids.shape
        self.seen.append((n, width))
        h = np.zeros((n, width, DIMS), dtype=np.float32)
        h[:, :, 0] = np.where(ids != 0, 3 * ids, 1)
        p = np.zeros((n, DIMS), dtype=np.float32)
        p[:, 1] = 5
        return [h, p]


def tokenizer(t: str) -> list[int]:
    return [1, *(len(w) + 10 for w in t.split(" ")), 2]


def test_a_unit_vector_per_text_in_order() -> None:
    s = Session()
    e = LocalEmbedder("/nowhere", session=s, tokenizer=tokenizer, batch=2)
    out = run(e.embed(["one two three", "a", "b c"]))
    assert len(out) == 3
    assert all(abs(float(np.linalg.norm(v)) - 1) < 1e-6 for v in out)
    assert s.seen == [(2, 4), (1, 5)]
    e.close()


def test_a_vector_per_word_without_start_and_end() -> None:
    encoded: list[str] = []

    def tok(t: str) -> list[int]:
        encoded.append(t)
        return tokenizer(t)

    e = LocalEmbedder("/nowhere", session=Session(), tokenizer=tok)
    q = run(e.words(["hello there"], "query"))[0]
    assert encoded[0] == "task: search result | query: hello there"
    assert q.words == len(encoded[0].split(" "))
    assert q.dims == DIMS
    assert len(q.data) == q.words * DIMS
    assert abs(float(np.linalg.norm(q.data[:DIMS])) - 1) < 1e-6
    run(e.words(["hello"], "document"))
    assert encoded[1] == "title: none | text: hello"
    e.close()


def test_the_last_word_of_a_cut_text_is_a_word() -> None:
    e = LocalEmbedder("/nowhere", session=Session(), tokenizer=tokenizer)
    d = run(e.words([" ".join(["word"] * 200)], "document"))[0]
    assert d.words == 129
    e.close()
