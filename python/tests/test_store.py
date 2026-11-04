"""Memory's logs (store.py): the truth the Qdrant index is rebuilt from — a replaced record coming back, a record read at
the wrong offset after an append or a rewrite, a multi-byte character shifting offsets, a crash's torn line, usage
counts lost. (store.test.ts and store-recovery.test.ts, ported.)"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memtrails.core.store import JsonlLog, UsageBook, fingerprint


def log(path: Path) -> JsonlLog[dict[str, Any]]:
    return JsonlLog(str(path), lambda r: {"n": r["n"]})


def test_reads_each_record_back_by_offset(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    a = log(path)
    a.put([{"uid": "a", "text": "первый — ünïcödé", "n": 1}, {"uid": "b", "text": "second", "n": 2}])
    a.put([{"uid": "a", "text": "replaced", "n": 3}])
    assert a.get("a")["text"] == "replaced"
    assert a.get("b")["text"] == "second"
    assert a.meta("a") == {"n": 3}
    b = log(path)
    b.load()
    assert b.size() == 2
    assert b.get("a")["text"] == "replaced"
    assert [r["uid"] for r in b.all()] == ["b", "a"]


def test_rewrites_itself_without_replaced_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    lg = log(path)
    for i in range(1200):
        lg.put([{"uid": f"u{i % 3}", "text": f"v{i} é", "n": i}])
    lg.compact()
    assert len(path.read_text().strip().split("\n")) == 3
    assert lg.get("u0")["text"] == "v1197 é"
    lg.put([{"uid": "u3", "text": "after", "n": 0}])
    assert lg.get("u3")["text"] == "after"
    assert lg.get("u2")["text"] == "v1199 é"


def test_writes_what_node_writes(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    log(path).put([{"uid": "a", "text": 'ü \u2028 "q" \n', "n": 1.5, "m": 2.0, "z": None}])
    assert path.read_text() == '{"uid":"a","text":"ü \u2028 \\"q\\" \\n","n":1.5,"m":2,"z":null}\n'


def test_drops_a_cut_off_last_line_and_keeps_what_follows(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    log(path).put([{"uid": "a", "text": "ok é", "n": 1}])
    with open(path, "a") as f:
        f.write('{"uid":"b","text":"torn')
    lg = log(path)
    lg.load()
    lg.put([{"uid": "c", "text": "after the crash", "n": 2}, {"uid": "d", "text": "and this", "n": 3}])
    assert lg.get("c")["text"] == "after the crash"
    assert lg.get("d")["text"] == "and this"
    assert "torn" not in path.read_text()
    r = log(path)
    r.load()
    assert [x["uid"] for x in r.all()] == ["a", "c", "d"]
    for i in range(1100):
        r.put([{"uid": "a", "text": f"v{i}", "n": i}])
    r.compact()
    s = log(path)
    s.load()
    assert [(x["uid"], x["text"]) for x in s.all()] == [("c", "after the crash"), ("d", "and this"), ("a", "v1099")]


def test_keeps_a_last_line_whole_but_for_its_newline(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"uid": "a", "text": "one", "n": 1}) + "\n" + json.dumps({"uid": "b", "text": "two", "n": 2}))
    lg = log(path)
    lg.load()
    lg.put([{"uid": "c", "text": "three", "n": 3}])
    assert lg.get("b")["text"] == "two"
    assert lg.get("c")["text"] == "three"
    r = log(path)
    r.load()
    assert [x["uid"] for x in r.all()] == ["a", "b", "c"]


def test_refuses_to_compact_when_a_record_cannot_be_read(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    lg = log(path)
    lg.put([{"uid": "keep", "text": "precious", "n": 0}])
    for i in range(1100):
        lg.put([{"uid": "u", "text": f"v{i}", "n": i}])
    before = path.read_text().replace("precious", 'prec"ous')
    path.write_text(before)
    assert lg.get("keep") is None
    with pytest.raises(RuntimeError, match="keep"):
        lg.compact()
    assert path.read_text() == before
    assert lg.get("u")["text"] == "v1099"


def test_compacts_a_log_larger_than_one_write(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    lg = log(path)
    big = "ü" * 3000
    live = 800
    for rnd in range(4):
        lg.put([{"uid": f"u{i}", "text": f"{rnd}:{i}:{big}", "n": rnd} for i in range(live)])
    lg.compact()
    assert len(path.read_text().strip().split("\n")) == live
    assert lg.get("u799")["text"] == f"3:799:{big}"
    lg.put([{"uid": "new", "text": "after", "n": 9}])
    r = log(path)
    r.load()
    assert r.size() == live + 1
    assert all(r.get(f"u{i}")["text"] == f"3:{i}:{big}" for i in range(live))
    assert r.get("new")["text"] == "after"


def test_skips_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    log(path).put([{"uid": "a", "text": "ok", "n": 1}])
    with open(path, "a") as f:
        f.write('{"uid":"b","te')
    r = log(path)
    r.load()
    assert r.size() == 1
    assert r.get("a")["text"] == "ok"


def test_usage_counts_are_kept(tmp_path: Path) -> None:
    path = str(tmp_path / "usage.json")
    u = UsageBook(path)
    u.note(["m1", "m2"], "shown", 100)
    u.note(["m1"], "opened", 200)
    u.close()
    again = UsageBook(path)
    assert again.of("m1") == {"shown": 1, "opened": 1, "lastUsed": 200}
    assert again.of("m3") == {"shown": 0, "opened": 0, "lastUsed": 0}


def test_fingerprint_changes_with_any_part() -> None:
    assert fingerprint("a", ["b", "c"]) != fingerprint("a", "b,c")
    assert fingerprint("a", None) == fingerprint("a", "")
    assert fingerprint("a") != fingerprint("b")
