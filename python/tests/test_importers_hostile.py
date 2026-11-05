"""Importers read files someone else wrote: no input may make them take more than linear time, and the linear scans keep
what the regexes they replaced did (hostile.test.ts, ported)."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

import pytest

from memtrails._js import date_parse
from memtrails.importers import from_eml, from_files, from_jsonl, from_mbox, from_slack_export
from memtrails.importers.text import MAX_LINE_BYTES, LineTooLong, html_title, html_to_text, read_lines

N = 200_000


async def collect(it: AsyncIterable[Any]) -> list[Any]:
    return [x async for x in it]


@pytest.mark.parametrize(
    "opening",
    [
        "<script",
        "<style",
        "<iframe",
        "<head",
        "<pre",
        "<![CDATA[",
        "<div",
        "<a",
        "<br",
        "<li",
        "</td",
        "<!x",
        "<title>",
        "<h1>",
        "<blockquote",
        "<<",
    ],
)
def test_html_stays_linear_on_unclosed_tags(opening: str) -> None:
    s = opening * -(-N // len(opening))
    t = time.perf_counter()
    html_to_text(s, drop_quotes=True)
    html_title(s)
    assert time.perf_counter() - t < 1.0


def test_hidden_elements_pre_and_open_scripts() -> None:
    assert html_to_text('<p>a</p><SCRIPT type="x">b</script ><p>c</p>') == "a\n\nc"
    assert html_to_text("<p>a</p><iframe src=x>keep<p>b</p>") == "a\nkeep\nb"
    assert html_to_text("a <script>var x = 1; <p>never</p>") == "a"
    assert html_to_text("<head><title>t</title><script>s</script></head><p>x   y</p>") == "x y"
    assert html_to_text("<p>a   b</p><pre>  x\r\n   y</pre><p>c   d</p>") == "a b\n\nx\ny\n\nc d"
    assert html_to_text("<pre>  open   <p>x</p>") == "open\nx"
    assert html_to_text("<![CDATA[x < y]]> z") == "x < y z"
    assert html_to_text("a <![CDATA[ unclosed") == "a <![CDATA[ unclosed"
    assert html_to_text("<blockquote>q<blockquote>qq</blockquote></blockquote>reply", drop_quotes=True) == "reply"
    assert html_to_text("<p>a</p> 1 < 2") == "a\n1 < 2"


def test_the_title_else_the_first_h1() -> None:
    assert html_title("<html><title lang=ro>Caf&eacute; <b>bun</b></title ><h1>H</h1>") == "Café bun"
    assert html_title("<title> </title><h1 class=x>Head<br>line</h1>") == "Head line"
    assert html_title("<title>open") is None
    assert html_title(f"{' ' * 70_000}<title>late</title>") is None


SP = " " * N
HOSTILE_FILES = {
    "atx.md": f"# a{SP}b",
    "atx-close.md": f"# a{SP}#",
    "setext.md": f"a{SP}b\n===",
    "eq.md": f"{'=' * N}x\n{'=' * N}",
    "front.md": f"---\ntitle: a{SP}b\nk: v{SP}x\n---\nbody",
    "front.toml.md": f"+++\na{SP}\nb = c{SP}d\n+++\n",
    "title.rst": f"a{SP}b\n{'=' * N}x",
    "stars.org": f"* {'*' * N}",
    "eq.adoc": "=" * N,
    "tags.html": "<title>" * (N // 7),
    "hashes.md": f"{'#' * N}\n# {'# ' * (N // 2)}",
}


@pytest.mark.parametrize("name", list(HOSTILE_FILES))
async def test_files_in_linear_time(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text(HOSTILE_FILES[name])
    t = time.perf_counter()
    await collect(from_files(tmp_path))
    assert time.perf_counter() - t < 1.0


async def test_keeps_the_titles_the_regexes_found(tmp_path: Path) -> None:
    cases = {
        "atx.md": ("intro\n# Plan   ##  \nbody", "Plan"),
        "atx-hash.md": ("# C# tips", "C# tips"),
        "atx-only.md": ("# ###", "###"),
        "setext.md": ("Planul   \n=====\nbody", "Planul"),
        "over.rst": ("=====\n Titlu \n=====\n\ntext", "Titlu"),
        "under.rst": ("Titlu\n~~~~~\n", "Titlu"),
        "short.rst": ("Titlu\n==\nNot\n---\n", "Not"),
        "doc.adoc": ("== Sub\n= Document  \n", "Document"),
        "notes.org": ("#+TITLE:  Notițe\n* Head", "Notițe"),
        "head.org": ("text\n** Capitol 2\n", "Capitol 2"),
        "cr.md": ("intro\r# Old Mac\rbody", "Old Mac"),
    }
    for name, (text, _) in cases.items():
        (tmp_path / name).write_bytes(text.encode())
    titles = {r["id"]: r["title"] for r in await collect(from_files(tmp_path))}
    for name, (_, title) in cases.items():
        assert titles[name] == title, name


async def test_reads_front_matter_as_before(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("---\ntitle: T # a comment\nurl: http://x.org/a:b\nk:v\ncount: 3\n---\nbody")
    (tmp_path / "b.md").write_text('+++\ntitle = "TOML"\nn = 2 # c\n+++\nbody')
    a, b = await collect(from_files(tmp_path))
    assert a["title"] == "T"
    assert a["metadata"] == {"url": "http://x.org/a:b", "count": 3}
    assert b["title"] == "TOML"
    assert b["metadata"] == {"n": 2}


EMLS = {
    "from-at.eml": f"From: {'a@' * (N // 2)}(x\nSubject: s\n\nhi\n",
    "from-spaces.eml": f"From: a{SP}<b\nSubject: s\n\nhi\n",
    "qp.eml": f"From: a@b.c\nContent-Transfer-Encoding: quoted-printable\n\na{SP}b\n",
    "date.eml": f"From: a@b.c\nDate: {'(' * N}\nSubject: s\n\nhi\n",
}


@pytest.mark.parametrize("name", list(EMLS))
async def test_mail_in_linear_time(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text(EMLS[name])
    t = time.perf_counter()
    await collect(from_eml(tmp_path / name))
    assert time.perf_counter() - t < 1.0


async def test_names_the_sender_and_reads_the_date(tmp_path: Path) -> None:
    authors = []
    for i, sender in enumerate(['"Pop, Ana" <ana@x.org>', "<ana@x.org>", "ana@x.org (Ana Pop)", "ana@x.org ()", "Ana"]):
        (tmp_path / f"{i}.eml").write_text(f"From: {sender}\nDate: Fri, 1 Mar 2024 10:00:00 +0000 (UTC)\nSubject: s\n\nhi\n")
        (r,) = await collect(from_eml(tmp_path / f"{i}.eml"))
        authors.append(r["author"])
        assert r["at"] == date_parse("2024-03-01T10:00:00Z")
    assert authors == ["Pop, Ana", "ana@x.org", "Ana Pop", "ana@x.org ()", "Ana"]


DEEP = "[" * 100_000 + "]" * 100_000


async def test_a_line_nested_too_deeply_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(f'{{"id":"deep","text":"x","metadata":{DEEP}}}\n{{"id":"after","text":"read on"}}\n')
    skipped: list[str] = []
    got = await collect(from_jsonl(path, on_skip=lambda _line, why, _n: skipped.append(why)))
    assert [r["id"] for r in got] == ["after"]
    assert skipped == ["not JSON: nested too deeply"]


async def test_a_slack_day_nested_too_deeply_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "export"
    (root / "general").mkdir(parents=True)
    (root / "channels.json").write_text('[{"id": "C1", "name": "general"}]')
    (root / "general" / "2024-01-01.json").write_text(DEEP)
    (root / "general" / "2024-01-02.json").write_text('[{"type": "message", "user": "U1", "text": "read on", "ts": "1704200000.0001"}]')
    assert [r["text"] for r in await collect(from_slack_export(root))] == ["read on"]


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="no links to swap in")
async def test_a_note_swapped_for_a_link_after_the_folder_was_listed_is_not_followed(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\nfine")
    secret = tmp_path / "secret.md"
    secret.write_text("# Secret\nthe private key")

    # `exclude` is asked just before a file is read: the note becomes a link to what is outside the folder then.
    def swap(_path: str) -> bool:
        (notes / "a.md").unlink()
        (notes / "a.md").symlink_to(secret)
        return False

    skipped: list[str] = []
    got = await collect(from_files(notes, exclude=swap, on_skip=lambda p, why: skipped.append(f"{p}: {why}")))
    assert [r["text"] for r in got] == []
    assert skipped == ["a.md: not a file: it became a link after the folder was read"]


def with_long_line(path: Path, head: str, long: int, tail: str) -> None:
    """A file of ``head``, then a line of ``long`` bytes, then ``tail``, written a megabyte at a time."""
    mb = b"x" * (1 << 20)
    with open(path, "wb") as f:
        f.write(head.encode())
        left = long
        while left > 0:
            f.write(mb[: min(left, len(mb))])
            left -= len(mb)
        f.write(tail.encode())


async def test_lines_end_as_readline_ends_them_and_one_past_the_bound_is_passed_over(tmp_path: Path) -> None:
    async def lines(data: bytes, max_bytes: int) -> list[str]:
        path = tmp_path / "lines.txt"
        path.write_bytes(data)
        return [f"! {x.reason}" if isinstance(x, LineTooLong) else x async for x in read_lines(str(path), "utf-8", max_bytes)]

    assert await lines(b"a\r\nb\rc\n\nd", 100) == ["a", "b", "c", "", "d"]
    assert await lines(b"a\r\r\n", 100) == ["a", ""]
    assert await lines(b"", 100) == []
    assert await lines(b"short\nfar too long a line\nafter\n", 8) == ["short", "! a line longer than 8 bytes (19)", "after"]


async def test_a_json_line_past_64_mib_is_skipped_with_why_and_the_next_read(tmp_path: Path) -> None:
    path = tmp_path / "big.jsonl"
    with_long_line(path, '{"id":"a","text":"before"}\n', MAX_LINE_BYTES + 1, '\n{"id":"b","text":"after"}\n')
    skipped: list[str] = []
    stats: dict[str, int] = {}
    got = await collect(from_jsonl(path, on_skip=lambda line, why, n: skipped.append(f"{n} {line!r}: {why}"), stats=stats))
    assert [r["id"] for r in got] == ["a", "b"]
    assert skipped == [f"2 '': a line longer than {MAX_LINE_BYTES} bytes ({MAX_LINE_BYTES + 1})"]
    assert stats == {"lines": 3, "records": 2, "skipped": 1}


async def test_an_mbox_line_past_64_mib_ends_its_message_and_the_next_is_read(tmp_path: Path) -> None:
    def msg(n: int, body: str) -> str:
        return f"From a@b.c Mon Jan  1 00:00:0{n} 2024\nFrom: a@b.c\nSubject: m{n}\nMessage-ID: <m{n}@b.c>\n\n{body}"

    path = tmp_path / "big.mbox"
    second = msg(2, "second\n")
    with_long_line(path, msg(1, "kept\n"), MAX_LINE_BYTES + 1, f"\nnot kept\n\n{second}")
    got = await collect(from_mbox(path, max_message_bytes=4 * MAX_LINE_BYTES))
    assert [(r["title"], r["text"]) for r in got] == [("m1", "kept"), ("m2", "second")]
