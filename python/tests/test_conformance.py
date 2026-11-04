"""The answer key (conformance/fixtures/*.json): what the TypeScript library gives for fixed inputs, this port gives
too — strings exactly, numbers to 1e-6."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from conftest import fixture
from memtrails.core.episodes import calendar_of, episode_body, render_for_extraction
from memtrails.core.qdrant_client import point_id
from memtrails.core.redact import redact_string
from memtrails.core.room import sentence_spans
from memtrails.core.store import fingerprint
from memtrails.core.text import children, concept_key, cut, doc_sparse, first_words, fold, query_sparse, term_id, tokens
from memtrails.core.words import PackedWords, WordVectors, max_sim, pack_words
from memtrails.core.writer import date_span, raw_key, stm_title
from memtrails.records import collection_key, conversation_entries, entry_of, invalid_record, record_uid, time_of
from memtrails.tools.agent_tools import lasted_text, readable_date, search_text

TEXT = fixture("text.json")
DATES = fixture("dates.json")
IDS = fixture("ids.json")
RECORDS = fixture("records.json")
WORDS = fixture("words.json")
SEARCH = fixture("search-text.json")


def close(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= 1e-6 for x, y in zip(a, b, strict=False))


@pytest.mark.parametrize("case", TEXT["cases"], ids=lambda c: repr(c["text"][:24]))
class TestText:
    def test_fold(self, case: dict[str, Any]) -> None:
        assert fold(case["text"]) == case["fold"]

    def test_tokens(self, case: dict[str, Any]) -> None:
        assert tokens(case["text"]) == case["tokens"]

    def test_first_words(self, case: dict[str, Any]) -> None:
        assert first_words(case["text"], 5) == case["firstWords5"]

    def test_children(self, case: dict[str, Any]) -> None:
        assert children(case["text"], 12, 9) == case["children"]

    def test_doc_sparse(self, case: dict[str, Any]) -> None:
        got = doc_sparse(case["text"])
        assert got["indices"] == case["docSparse"]["indices"]
        assert close(got["values"], case["docSparse"]["values"])

    def test_cut(self, case: dict[str, Any]) -> None:
        assert cut(case["text"], 20) == case["cut20"]
        assert cut(case["text"], 1) == case["cut1"]

    def test_concept_key(self, case: dict[str, Any]) -> None:
        assert concept_key(case["text"]) == case["conceptKey"]


def test_term_ids() -> None:
    for t in TEXT["termIds"]:
        assert term_id(t["term"]) == t["id"], t["term"]


def test_query_sparse() -> None:
    got = query_sparse(TEXT["querySparse"]["texts"])
    assert got["indices"] == TEXT["querySparse"]["vector"]["indices"]
    assert got["values"] == TEXT["querySparse"]["vector"]["values"]


def test_concept_keys() -> None:
    for c in TEXT["conceptKeys"]:
        assert concept_key(c["name"]) == c["key"]


@pytest.mark.parametrize("case", TEXT["sentenceSpans"], ids=lambda c: repr(c["text"][:24]))
def test_sentence_spans(case: dict[str, Any]) -> None:
    assert [list(s) for s in sentence_spans(case["text"])] == case["spans"]


@pytest.mark.parametrize("case", fixture("redact.json"), ids=lambda c: repr(c["input"][:24]))
def test_redact(case: dict[str, Any]) -> None:
    assert redact_string(case["input"]) == case["output"]


def test_ids() -> None:
    for f in IDS["fingerprint"]:
        assert fingerprint(*f["parts"]) == f["fp"], f["parts"]
    for p in IDS["pointId"]:
        assert point_id(p["item"], p["n"]) == p["id"]
    for r in IDS["recordUid"]:
        assert record_uid(r["record"]) == r["uid"]
    for c in IDS["collectionKey"]:
        assert collection_key(c["collection"]) == c["key"]


def test_dates() -> None:
    for d in DATES["dateSpan"]:
        assert date_span(d["date"]) == d["span"], d["date"]
    for d in DATES["readableDate"]:
        assert readable_date(d["date"]) == d["text"], d["date"]
    for d in DATES["lastedText"]:
        assert lasted_text(d["hit"]) == d["text"]
    for d in DATES["timeOf"]:
        assert time_of(d["at"]) == d["ms"], d["at"]
    assert calendar_of(DATES["calendarOf"]["times"]) == DATES["calendarOf"]["text"]


def test_records() -> None:
    now = RECORDS["now"]
    for c in RECORDS["invalid"]:
        assert invalid_record(c["record"]) == c["error"]
    for c in RECORDS["entries"]:
        assert entry_of(c["record"], now, c["turn"]) == c["entry"]
    assert conversation_entries(RECORDS["conversation"]["input"], now) == RECORDS["conversation"]["entries"]
    entries = [c["entry"] for c in RECORDS["entries"]] + RECORDS["conversation"]["entries"]
    for n, (e, want) in enumerate(zip(entries, RECORDS["rendered"], strict=False)):
        assert want["uid"] == e["uid"]
        assert raw_key(e) == want["rawKey"]
        assert stm_title(e) == want["stmTitle"]
        assert episode_body(e) == want["episodeBody"]
        assert render_for_extraction(e, n + 1) == want["extraction"]


def test_words() -> None:
    q = WORDS["query"]
    query = WordVectors(q["words"], q["dims"], np.asarray(q["data"], dtype=np.float32))
    for d in WORDS["docs"]:
        v = d["vectors"]
        packed = pack_words(WordVectors(v["words"], v["dims"], np.asarray(v["data"], dtype=np.float32)))
        assert packed == PackedWords(d["packed"]["words"], d["packed"]["dims"], bytes(d["packed"]["bits"]))
        assert abs(max_sim(query, packed) - d["maxSim"]) <= 1e-6


def test_search_text() -> None:
    hits = SEARCH["hits"]
    assert search_text(hits, "relevance") == SEARCH["relevance"]
    assert search_text(hits, "time") == SEARCH["time"]
    many = [{**hits[i % len(hits)], "id": f"x:{i}"} for i in range(22)]
    assert search_text(many, "relevance") == SEARCH["compact"]
