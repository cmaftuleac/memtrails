"""
What gets indexed from an item's text, besides its content key: the keyword vector (BM25 as a Qdrant sparse vector —
term frequencies here, the IDF on Qdrant's side with ``modifier: idf``) and the small-to-big children.

Measured on a private history of real work: the keyword channel beside the dense one put a needed item in the top 5
for +5–8 points; ~96-word children beat 192-word ones; adding raw text to the key itself hurt.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import TypedDict

import regex

from .._js import JS_SPACE, cut, flatten_ws, js_trim, u16len
from .segment import word_spans, words

K1 = 1.2
B = 0.75
#: Longer than this is not a word a person would type (a hash, a base64 blob).
MAX_TOKEN = 60


class SparseVector(TypedDict):
    indices: list[int]
    values: list[float]


#: Scripts written without spaces between words: there the ICU word segmenter finds the words.
_UNSPACED = regex.compile(
    r"[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]"
)
_MARKS = regex.compile(r"([\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}])\p{M}+")
_SPLIT = regex.compile(r"[^\p{L}\p{N}\p{M}_.\-/]+")
_EDGES = regex.compile(r"^[._\-/]+|[._\-/]+$")
_JOINERS = regex.compile(r"[_.\-/]")
_JOINER_RUN = regex.compile(r"[_.\-/]+")


def fold(text: str) -> str:
    """Lower case, compatibility forms unified, accents dropped from Latin, Greek and Cyrillic letters — "bază" and
    "baza" are the same word to a person typing either. Marks that make the word elsewhere (Devanagari, Thai) stay."""
    return unicodedata.normalize("NFKC", _MARKS.sub(r"\1", unicodedata.normalize("NFKD", text))).lower()


def tokens(text: str) -> list[str]:
    """Words for the keyword channel, in any language: folded (fold()); identifiers kept whole AND split at
    ``_ . - /``; in scripts without spaces (Chinese, Japanese, Thai…) the segmenter's words."""
    out: list[str] = []
    for raw in _SPLIT.split(fold(text)):
        w = _EDGES.sub("", raw)
        if not w:
            continue
        if _UNSPACED.search(w):
            out.extend(words(w))
            continue
        n = u16len(w)
        if n < 2:
            continue
        # The length cap is per word: a path too long to be one keyword still gives its parts.
        if n <= MAX_TOKEN:
            out.append(w)
        if _JOINERS.search(w):
            for p in _JOINER_RUN.split(w):
                if 2 <= u16len(p) <= MAX_TOKEN and p != w:
                    out.append(p)
    return out


def flat(s: str) -> str:
    return js_trim(flatten_ws(s))


def first_words(text: str, n: int) -> str:
    """A text's first ``n`` words, as written (whitespace flattened)."""
    at = word_spans(text)
    return flat(text[: at[n - 1][1]] if len(at) > n else text)


def term_id(term: str) -> int:
    """32-bit FNV-1a over UTF-16 code units: a term's index in the sparse vector."""
    h = 0x811C9DC5
    for unit in term.encode("ascii") if term.isascii() else _units(term):
        h ^= unit
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _units(s: str) -> list[int]:
    b = s.encode("utf-16-le", "surrogatepass")
    return [b[i] | (b[i + 1] << 8) for i in range(0, len(b), 2)]


def doc_sparse(text: str, avg_len: float = 220) -> SparseVector:
    """A document's BM25 term weights (Qdrant applies the IDF). ``avg_len``: the typical length of what is indexed."""
    ts = tokens(text)
    tf: dict[int, int] = {}
    for t in ts:
        i = term_id(t)
        tf[i] = tf.get(i, 0) + 1
    norm = K1 * (1 - B + (B * len(ts)) / avg_len)
    indices = sorted(tf)
    return {"indices": indices, "values": [(tf[i] * (K1 + 1)) / (tf[i] + norm) for i in indices]}


def query_sparse(texts: Sequence[str]) -> SparseVector:
    """A query's terms, each once."""
    ids = sorted({term_id(t) for x in texts for t in tokens(x)})
    return {"indices": ids, "values": [1.0] * len(ids)}


CHILD_WORDS = 96
CHILD_STRIDE = 72


def children(text: str, n_words: int = CHILD_WORDS, stride: int = CHILD_STRIDE) -> list[str]:
    """~96-word windows of a body, a quarter overlapping, all of it — searched, never shown: the item is what comes
    back. Words as the segmenter finds them, so a body in a script without spaces is windowed too."""
    at = word_spans(text)
    if not at:
        return [flat(text)] if flat(text) else []
    if len(at) <= n_words:
        return [flat(text)]
    out: list[str] = []
    end = 0
    i = 0
    while i < len(at) - n_words // 2:
        end = min(i + n_words, len(at))
        out.append(flat(text[at[i][0] : at[end - 1][1]]))
        i += stride
    # The last words, when the stride stopped short of them: a window ending at the text's end.
    if end < len(at):
        out.append(flat(text[at[max(0, len(at) - n_words)][0] : at[-1][1]]))
    return out


_EDGE = f"[{JS_SPACE}.,:;\"'`()\\[\\]]+"
_CONCEPT_EDGES = regex.compile(f"^{_EDGE}|{_EDGE}$")


def concept_key(name: str) -> str:
    """One spelling per concept, the graph's key: folded like a word (case, accents, compatibility forms), inner spaces
    collapsed, stray punctuation off the ends — "Proxy.conf " and "proxy.conf" are one node."""
    return _CONCEPT_EDGES.sub("", flatten_ws(fold(name)))


__all__ = [
    "CHILD_STRIDE",
    "CHILD_WORDS",
    "SparseVector",
    "children",
    "concept_key",
    "cut",
    "doc_sparse",
    "first_words",
    "fold",
    "query_sparse",
    "term_id",
    "tokens",
]
