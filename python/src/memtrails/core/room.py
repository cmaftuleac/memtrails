"""
A small room filled with sentences, not memories. An agent reads a few thousand characters of what memory found; a
memory's first 500 of them are rarely where its answer is. Each found memory's text is cut into sentences (at any
script's end marks, no word list), each sentence scored word by word against the question, and the room filled with
the best sentences, a memory's place in the search counting too (docs/RESEARCH.md: held out, the answer's words in
6,000 characters 66.0 against 57.2 for Mem0's room on raw memory).

Sentences are kept packed beside the memories' own words (words.py WordStore), by their text, so each is encoded once.
Offsets are UTF-16 code units, as the reference counts them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import regex

from .._js import JS_SPACE, from16, js_is_space, to16
from .words import WordEncoder, WordStore, max_sim, pack_words

#: Sentences shorter than this say too little to be worth a place.
_MIN_SENTENCE = 8
#: A sentence longer than this is cut into pieces of it.
_MAX_SENTENCE = 400
#: A memory's date and kind, once, before its sentences.
_HEAD_CHARS = 20

_END = regex.compile(f"[.!?。！？](?=[{JS_SPACE}]|\\Z)|\n")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """[start, end) of each sentence, in UTF-16 units: after ``.``, ``!``, ``?``, their full-width forms, ``。`` or a
    line break, then a space."""
    u = to16(text)
    out: list[tuple[int, int]] = []

    def push(a: int, b: int) -> None:
        while a < b and js_is_space(u[a]):
            a += 1
        while b > a and js_is_space(u[b - 1]):
            b -= 1
        for s in range(a, b, _MAX_SENTENCE):
            out.append((s, min(b, s + _MAX_SENTENCE)))

    start = 0
    for m in _END.finditer(u):
        push(start, m.end())
        start = m.end()
    push(start, len(u))
    return [(a, b) for a, b in out if b - a >= _MIN_SENTENCE]


@dataclass(eq=False)
class RoomItem:
    id: str
    head: str
    text: str
    ts: float


async def sentence_room(query: str, items: list[RoomItem], encode: WordEncoder, store: WordStore, chars: int, top: int = 30) -> str:
    """The room: the best sentences of the first ``top`` memories, fitted in ``chars``, each memory's head once, oldest
    first."""
    listed = [(it, rank, to16(it.text), sentence_spans(it.text)) for rank, it in enumerate(items[:top])]
    texts = list(dict.fromkeys(from16(u[a:b]) for _, _, u, spans in listed for a, b in spans))
    missing = [t for t in texts if not store.get(t)]
    if missing:
        encoded = await encode(missing, "document")
        store.put([(t, pack_words(encoded[i])) for i, t in enumerate(missing)])
    q = (await encode([query], "query"))[0]
    scored = [(it, rank, u, a, b, store.get(from16(u[a:b]))) for it, rank, u, spans in listed for a, b in spans]

    def score_all() -> list[tuple[float, RoomItem, str, int, int]]:
        return [
            ((max_sim(q, packed) if packed else float("-inf")) * (1 + 2 / (1 + rank / 10)), it, u, a, b)
            for it, rank, u, a, b, packed in scored
        ]

    # Every sentence of up to 30 memories against the question: off the event loop.
    candidates = await asyncio.to_thread(score_all)
    candidates.sort(key=lambda c: -c[0])
    picked: dict[int, tuple[RoomItem, str, list[tuple[int, int]]]] = {}
    used = 0
    for _, it, u, a, b in candidates:
        head = 0 if id(it) in picked else _HEAD_CHARS + len(to16(it.head))
        extra = b - a + 3 + head
        if used + extra > chars:
            continue
        picked.setdefault(id(it), (it, u, []))[2].append((a, b))
        used += extra
    rows = sorted(picked.values(), key=lambda p: p[0].ts)
    return "\n".join(f"{it.head} {' … '.join(from16(u[a:b]) for a, b in sorted(spans, key=lambda s: s[0]))}" for it, u, spans in rows)
