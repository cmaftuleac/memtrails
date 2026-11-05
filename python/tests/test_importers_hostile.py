"""Importers read files someone else wrote: no input may make them take more than linear time, and the linear scans keep
what the regexes they replaced did (hostile.test.ts, ported)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

import pytest

from memtrails._js import date_parse
from memtrails.importers import from_eml, from_files, from_jsonl, from_slack_export
from memtrails.importers.text import html_title, html_to_text

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
