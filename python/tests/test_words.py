"""Word by word (words.py) and the sentence room (room.py): words.test.ts, words-recovery.test.ts, room.test.ts."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import numpy as np
import regex

from fakes import run
from memtrails.core.reranker import reranked_order
from memtrails.core.room import RoomItem, sentence_room, sentence_spans
from memtrails.core.words import WORD_LIMIT, PackedWords, WordStore, WordVectors, max_sim, pack_words, word_reranker

DIMS = 16


def word_vector(word: str, lower: bool = False) -> np.ndarray:
    h = 2166136261
    for c in word.lower() if lower else word:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    v = np.zeros(DIMS, dtype=np.float32)
    for d in range(DIMS):
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        v[d] = 1 if h & 1 else -1
    return (v / math.sqrt(DIMS)).astype(np.float32)


def vectors(text: str) -> WordVectors:
    words = text.split()
    data = np.concatenate([word_vector(w) for w in words]) if words else np.zeros(0, dtype=np.float32)
    return WordVectors(len(words), DIMS, data)


def test_scores_by_the_closest_word() -> None:
    long = pack_words(vectors("deploy failed because the ssl certificate on staging expired last week after renewal stopped"))
    other = pack_words(vectors("we talked about the weather and lunch and a long walk in the park on sunday"))
    assert max_sim(vectors("certificate expired"), long) > max_sim(vectors("certificate expired"), other)


def test_an_exact_word_scores_fully() -> None:
    q = vectors("certificate")
    assert abs(max_sim(q, pack_words(q)) - 1) < 1e-5


def test_keeps_at_most_the_first_words() -> None:
    p = pack_words(vectors(" ".join(f"w{i}" for i in range(WORD_LIMIT + 20))))
    assert p.words == WORD_LIMIT
    assert len(p.bits) == WORD_LIMIT * math.ceil(DIMS / 8)
    assert pack_words(vectors("a b"), 1).words == 1


def test_cannot_score_vectors_of_different_sizes() -> None:
    assert max_sim(vectors("a"), PackedWords(1, 8, bytes(1))) == -math.inf
    assert max_sim(vectors("a"), PackedWords(0, DIMS, b"")) == -math.inf


def test_kept_across_restarts_and_a_file_cut_short(tmp_path: Path) -> None:
    file = tmp_path / "nested" / "words.bin"
    first = WordStore(str(file))
    first.put([("alpha beta", pack_words(vectors("alpha beta"))), ("gamma", pack_words(vectors("gamma")))])
    again = WordStore(str(file))
    again.load()
    assert again.size == 2
    assert again.get("gamma") == first.get("gamma")
    cut = tmp_path / "cut.bin"
    cut.write_bytes(file.read_bytes()[:-1])
    part = WordStore(str(cut))
    part.load()
    assert part.size == 1
    assert part.get("alpha beta") is not None


def test_a_cut_off_entry_goes_so_what_follows_loads(tmp_path: Path) -> None:
    def packed(seed: int) -> PackedWords:
        return PackedWords(2, 16, bytes([seed, seed + 1, seed + 2, seed + 3]))

    file = tmp_path / "words.bin"
    WordStore(str(file)).put([("first", packed(1))])
    head = bytearray(12)
    head[8:10] = (2).to_bytes(2, "little")
    head[10:12] = (16).to_bytes(2, "little")
    with open(file, "ab") as f:
        f.write(bytes(head) + bytes([9, 9]))
    s = WordStore(str(file))
    s.load()
    assert s.size == 1
    s.put([("second", packed(5))])
    again = WordStore(str(file))
    again.load()
    assert again.size == 2
    assert again.get("first") == packed(1)
    assert again.get("second") == packed(5)


def test_orders_a_search_encoding_each_text_once(tmp_path: Path) -> None:
    asked: list[list[str]] = []

    async def encode(texts: list[str], _kind: str) -> list[WordVectors]:
        asked.append(texts)
        return [vectors(t) for t in texts]

    rerank = word_reranker(encode, WordStore(str(tmp_path / "w.bin")))
    docs = ["lunch in the park", "the ssl certificate on staging expired", "lunch in the park"]
    scores = run(rerank("certificate expired", docs, 1000))
    assert scores is not None and len(scores) == 3
    assert reranked_order(docs, scores)[0] == docs[1]
    assert asked[0] == ["lunch in the park", "the ssl certificate on staging expired"]
    asked.clear()
    run(rerank("another question", docs, 1000))
    assert asked == [["another question"]]


def test_keeps_the_search_order_when_the_encoder_fails_or_is_late() -> None:
    async def failing(*_: Any) -> list[WordVectors]:
        raise RuntimeError("no model")

    async def slow(*_: Any) -> list[WordVectors]:
        await asyncio.sleep(10)
        return []

    assert run(word_reranker(failing, WordStore())("q", ["a"], 1000)) is None
    assert run(word_reranker(slow, WordStore())("q", ["a"], 10)) is None


async def room_encode(texts: list[str], _kind: str) -> list[WordVectors]:
    out = []
    for t in texts:
        words = [w for w in regex.split(r"[^\p{L}\p{N}]+", t) if w]
        data = np.concatenate([word_vector(w, lower=True) for w in words]) if words else np.zeros(0, dtype=np.float32)
        out.append(WordVectors(len(words), DIMS, data))
    return out


def test_sentences_end_at_any_scripts_marks() -> None:
    text = "Version 2.5 shipped today. Did it work? Yes!\nNext line here。下一句在这里。"
    assert [text[a:b] for a, b in sentence_spans(text)] == ["Version 2.5 shipped today.", "Did it work?", "Next line here。下一句在这里。"]
    assert len(sentence_spans("word " * 300)) > 2


def test_the_room_keeps_the_sentence_that_answers() -> None:
    filler = "We talked about many different and unrelated things for a long while that afternoon. "
    items = [
        RoomItem("a", "(1 Mar)", f"{filler * 6}The proxy certificate expired because the renewal cron was removed.", 2),
        RoomItem("b", "(2 Mar)", filler * 8, 1),
    ]
    room = run(sentence_room("why did the proxy certificate expire", items, room_encode, WordStore(), 160))
    assert "certificate expired" in room
    assert len(room) <= 160
    assert room.startswith("(1 Mar)")


def test_the_room_shows_each_memory_once_oldest_first() -> None:
    items = [
        RoomItem("new", "(new)", "Ana likes the Italian place. Ana goes there on Fridays.", 9),
        RoomItem("old", "(old)", "Ana once mentioned the Italian place near the office.", 1),
    ]
    room = run(sentence_room("where does Ana eat on Fridays", items, room_encode, WordStore(), 1000))
    assert room.split("\n") == [
        "(old) Ana once mentioned the Italian place near the office.",
        "(new) Ana likes the Italian place. … Ana goes there on Fridays.",
    ]


async def test_a_late_encoder_still_stores_what_it_encoded(tmp_path: Path) -> None:
    asked: list[list[str]] = []

    async def slow(texts: list[str], _kind: str) -> list[WordVectors]:
        asked.append(texts)
        await asyncio.sleep(0.05)
        return [vectors(t) for t in texts]

    store = WordStore(str(tmp_path / "w.bin"))
    rerank = word_reranker(slow, store)
    assert await rerank("certificate expired", ["the certificate expired"], 10) is None
    await asyncio.sleep(0.2)
    # Kept by the search that was too late for it: the next one encodes only its question.
    assert store.get("the certificate expired") is not None
    asked.clear()
    assert await rerank("certificate", ["the certificate expired"], 1000) is not None
    assert asked == [["certificate"]]
