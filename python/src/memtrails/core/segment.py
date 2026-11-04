"""
Words as the ICU word segmenter finds them — what ``Intl.Segmenter(undefined, { granularity: 'word' })`` gives the
reference library: UAX #29 boundaries everywhere, and dictionary words in scripts written without spaces (Chinese,
Japanese, Thai, Lao, Khmer, Myanmar). ``icu4py`` carries ICU itself, so no system library is needed.

ICU marks each segment with a rule status; JavaScript's ``isWordLike`` is "not none". ``icu4py`` gives the segments
but not their status, so it is read off the segment as ICU's rules assign it: a letter, a number, kana or an ideograph,
a letter of a script the dictionary reads, or two or more connector characters (``__``). Fuzzed against Node on
adversarial mixtures of every script, this agrees but for a combining mark left alone after a Hangul syllable or a
``‿`` joiner — text no one writes (python/README.md, known differences).
"""

from __future__ import annotations

from functools import lru_cache

import regex
from icu4py.breakers import WordBreaker

_WORDY = regex.compile(
    r"[\p{WB=ALetter}\p{WB=Hebrew_Letter}\p{WB=Numeric}\p{WB=Katakana}\p{Script=Han}\p{Script=Hiragana}]"
    r"|(?=\p{Line_Break=SA})\p{L}"
)
_CONNECTOR = regex.compile(r"\p{WB=ExtendNumLet}")


def _word_like(seg: str) -> bool:
    if _WORDY.search(seg):
        return True
    return len(_CONNECTOR.findall(seg)) >= 2


@lru_cache(maxsize=4096)
def _segments(text: str) -> tuple[tuple[int, int, bool], ...]:
    out: list[tuple[int, int, bool]] = []
    at = 0
    for seg in WordBreaker(text, "en"):
        out.append((at, at + len(seg), _word_like(seg)))
        at += len(seg)
    return tuple(out)


def segments(text: str) -> tuple[tuple[int, int, bool], ...]:
    """Every segment of ``text``: its start and end (Python indices) and whether it is a word."""
    if not text:
        return ()
    return _segments(text)


def words(text: str) -> list[str]:
    """The word segments of ``text``, in order."""
    return [text[a:b] for a, b, w in segments(text) if w]


def word_spans(text: str) -> list[tuple[int, int]]:
    """Where each word of ``text`` starts and ends (Python indices)."""
    return [(a, b) for a, b, w in segments(text) if w]
