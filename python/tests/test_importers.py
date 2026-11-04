"""The importers (mbox.test.ts, slack.test.ts, git.test.ts, files.test.ts, jsonl.test.ts, ported): each streams, reads
any language and charset, and leaves out what says nothing."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pytest

from memtrails._js import date_parse
from memtrails.importers import (
    batches,
    decode_encoded_words,
    from_eml,
    from_files,
    from_git,
    from_jsonl,
    from_mbox,
    from_slack_export,
    split_text,
)
from memtrails.importers.text import html_to_text

T = TypeVar("T")


async def collect(it: AsyncIterable[T]) -> list[T]:
    return [x async for x in it]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def wrap(s: str, n: int = 76) -> str:
    return "\n".join(s[i : i + n] for i in range(0, len(s), n))


def qp(text: str) -> str:
    """Quoted-printable, as a mail program writes it."""
    out_lines = []
    for line in text.split("\n"):
        out = ""
        width = 0
        for byte in line.encode("utf-8"):
            piece = f"={byte:02X}" if byte > 126 or byte == 61 else chr(byte)
            if width + len(piece) > 72:
                out += "=\n"
                width = 0
            out += piece
            width += len(piece)
        if out.endswith((" ", "\t")):
            out = out[:-1] + f"={ord(out[-1]):02X}"
        out_lines.append(out)
    return "\n".join(out_lines)


SUBJECT_BYTES = "会議の議事録 — 2024年".encode()
SUBJECT = f"=?UTF-8?B?{b64(SUBJECT_BYTES[:4])}?=\n =?UTF-8?B?{b64(SUBJECT_BYTES[4:])}?="
PLAIN = (
    "Bună ziua,\n\nAtașez procesul-verbal al ședinței — vă rog să-l citiți până mâine. Mulțumesc frumos pentru tot ajutorul! 🙏\n\n"
    "> Ce s-a hotărât?\n> Nimic încă.\n\n-- \nAndré, București"
)
HTML = "<html><body><p>Bună ziua,</p><p>Atașez procesul-verbal.</p></body></html>"

FIRST = "\n".join(
    [
        "From andre@example.org Mon Jan  1 10:00:00 2024",
        "Return-Path: <andre@example.org>",
        "From: =?iso-8859-1?Q?Andr=E9_Lef=E8vre?= <andre@example.org>",
        'To: "Ioana Mărgărit" <ioana@example.org>,',
        "\tbob@example.org",
        "Cc: =?utf-8?q?=C5=81ukasz?= <lukasz@example.org>",
        f"Subject: {SUBJECT}",
        "Date: Mon, 1 Jan 2024 10:00:00 +0100 (CET)",
        "Message-ID: <m1@example.org>",
        "MIME-Version: 1.0",
        'Content-Type: multipart/mixed; boundary="outer"',
        "",
        "This is a multi-part message in MIME format.",
        "--outer",
        "Content-Type: multipart/alternative;",
        " boundary=inner",
        "",
        "--inner",
        "Content-Type: text/plain; charset=utf-8",
        "Content-Transfer-Encoding: quoted-printable",
        "",
        qp(PLAIN),
        "--inner",
        "Content-Type: text/html; charset=utf-8",
        "Content-Transfer-Encoding: base64",
        "",
        wrap(b64(HTML.encode())),
        "--inner--",
        "--outer",
        "Content-Type: application/pdf",
        "Content-Disposition: attachment; filename*=UTF-8''%E8%AD%B0%E4%BA%8B%E9%8C%B2.pdf",
        "Content-Transfer-Encoding: base64",
        "",
        b64(b"%PDF-1.4 not really"),
        "--outer--",
        "",
    ]
)
SECOND = "\n".join(
    [
        "From ioana@example.org Mon Jan  1 12:00:00 2024",
        "From: Ioana Mărgărit <ioana@example.org>",
        "To: andre@example.org",
        "Subject: Re: =?UTF-8?B?5Lya6K2w44Gu6K2w5LqL6Yyy?=",
        "Date: Mon, 1 Jan 2024 12:00:00 +0100",
        "Message-ID: <m2@example.org>",
        "In-Reply-To: <m1@example.org>",
        "References: <m1@example.org>",
        "Content-Type: text/plain",
        "Content-Transfer-Encoding: 8bit",
        "",
        "Mulțumesc, am citit.",
        ">From the start it was clear.",
        ">>From a quoted line, kept escaped once.",
        "",
        "André Lefèvre wrote:",
        "> Bună ziua,",
        "> Atașez procesul-verbal.",
        "",
    ]
)
CYRILLIC = bytes([0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2, 0x2C, 0x20, 0xEC, 0xE8, 0xF0])
THIRD = "\n".join(
    [
        "From sergey@example.org Tue Jan  2 09:00:00 2024",
        "From: sergey@example.org (Сергей)",
        "Subject: =?gb2312?B?" + b64(bytes([0xD6, 0xD0, 0xCE, 0xC4])) + "?= =?koi8-r?Q?=F0=D2=C9=D7=C5=D4?=",
        "Date: Tue, 02 Jan 2024 09:00:00 GMT",
        "Content-Type: text/html; charset=windows-1251",
        "Content-Transfer-Encoding: base64",
        "",
        b64(b"<p>" + CYRILLIC + b"</p><blockquote>old text</blockquote>"),
        "",
    ]
)


@pytest.fixture
def mbox(tmp_path: Path) -> Path:
    path = tmp_path / "Inbox.mbox"
    path.write_bytes((FIRST + "\n" + SECOND + "\n" + THIRD).encode("utf-8"))
    return path


async def test_reads_a_thread_of_mail(mbox: Path) -> None:
    a, b, c = await collect(from_mbox(mbox))
    assert {k: a[k] for k in ("id", "title", "author", "kind", "thread", "source", "at")} == {
        "id": "m1@example.org",
        "title": "会議の議事録 — 2024年",
        "author": "André Lefèvre",
        "kind": "email",
        "thread": "m1@example.org",
        "source": "Inbox",
        "at": date_parse("2024-01-01T09:00:00Z"),
    }
    assert a["text"] == (
        "Bună ziua,\n\nAtașez procesul-verbal al ședinței — vă rog să-l citiți până mâine. Mulțumesc frumos pentru tot ajutorul! 🙏"
        "\n\nAttachments: 議事録.pdf"
    )
    assert a["metadata"] == {
        "from": "André Lefèvre <andre@example.org>",
        "to": '"Ioana Mărgărit" <ioana@example.org>, bob@example.org',
        "cc": "Łukasz <lukasz@example.org>",
    }
    assert (b["id"], b["title"], b["author"], b["thread"]) == ("m2@example.org", "Re: 会議の議事録", "Ioana Mărgărit", "m1@example.org")
    assert b["text"] == "Mulțumesc, am citit.\nFrom the start it was clear.\n\nAndré Lefèvre wrote:"
    assert b["metadata"]["inReplyTo"] == "m1@example.org"
    assert c["title"] == "中文Привет"
    assert c["text"] == "Привет, мир"
    assert c["author"] == "Сергей"
    assert len(c["id"]) == 32 and int(c["id"], 16) >= 0
    assert c["thread"] == c["id"]


async def test_keeps_quoted_lines_and_the_signature_when_asked(mbox: Path) -> None:
    a, b, c = await collect(from_mbox(mbox, keep_quoted=True, keep_signature=True, collection="work"))
    assert "> Ce s-a hotărât?" in a["text"]
    assert "-- \nAndré, București" in a["text"]
    assert ">From a quoted line, kept escaped once." in b["text"]
    assert c["text"] == "Привет, мир\n\nold text"
    assert a["collection"] == "work"


def test_decodes_encoded_words() -> None:
    assert decode_encoded_words("=?utf-8?Q?Gr=C3=BC=C3=9Fe_aus?= =?utf-8?Q?_K=C3=B6ln?= und mehr") == "Grüße aus Köln und mehr"
    assert decode_encoded_words("=?unknown-x?Q?abc?=") == "abc"


async def test_one_eml_with_crlf(tmp_path: Path) -> None:
    path = tmp_path / "note.eml"
    path.write_bytes("\r\n".join(FIRST.split("\n")[1:]).encode("utf-8"))
    (r,) = await collect(from_eml(path))
    assert (r["id"], r["title"], r["source"]) == ("m1@example.org", "会議の議事録 — 2024年", "note")
    assert r["text"].startswith("Bună ziua,\n\nAtașez")


@pytest.fixture
def slack(tmp_path: Path) -> Path:
    def put(path: str, value: Any) -> None:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    put(
        "users.json",
        [
            {"id": "U1", "name": "ana", "real_name": "Ana Popescu", "profile": {"real_name": "Ana Popescu", "display_name": "ana"}},
            {"id": "U2", "name": "tanaka", "profile": {"real_name": "", "display_name": "田中"}},
            {"id": "U3", "name": "omar"},
        ],
    )
    put("channels.json", [{"id": "C1", "name": "general"}, {"id": "C2", "name": "proiecte"}])
    put(
        "general/2024-03-01.json",
        [
            {"type": "message", "subtype": "channel_join", "user": "U3", "text": "<@U3> has joined the channel", "ts": "1709290000.000100"},
            {
                "type": "message",
                "user": "U1",
                "text": "Planul: <@U2> preia <#C2|proiecte>, vezi <https://example.org/doc|documentul> și "
                "<https://example.org/raw>. 1 &lt; 2 &amp;&amp; ok",
                "ts": "1709290800.000200",
                "thread_ts": "1709290800.000200",
                "reply_count": 1,
            },
            {
                "type": "message",
                "user": "U2",
                "text": "了解しました <@U1|ana> 👍",
                "ts": "1709291000.000300",
                "thread_ts": "1709290800.000200",
            },
            {
                "type": "message",
                "user": "U3",
                "text": "",
                "ts": "1709291100.000400",
                "files": [{"name": "تقرير.pdf", "title": "التقرير الشهري"}],
            },
            {"type": "message", "user": "U3", "text": "", "ts": "1709291150.000450"},
            {
                "type": "message",
                "subtype": "bot_message",
                "bot_id": "B1",
                "username": "deploy-bot",
                "text": "Deployed <!here>",
                "attachments": [{"title": "Build 42", "title_link": "https://example.org/b/42"}],
                "ts": "1709291200.000500",
            },
        ],
    )
    put(
        "general/2024-03-02.json",
        [
            {"type": "message", "user": "U1", "text": "Mulțumesc! <#C1>", "ts": "1709377200.000100"},
            {"type": "message", "subtype": "channel_topic", "user": "U1", "text": "set the topic", "ts": "1709377300.000200"},
        ],
    )
    put("proiecte/2024-03-01.json", [{"type": "message", "user": "U3", "text": "مرحبا بالجميع", "ts": "1709290900.000100"}])
    return tmp_path


async def test_reads_every_slack_message_that_says_something(slack: Path) -> None:
    records = await collect(from_slack_export(slack, workspace_url="https://example.slack.com/"))
    assert [r["id"] for r in records] == [
        "C1:1709290800.000200",
        "C1:1709291000.000300",
        "C1:1709291100.000400",
        "C1:1709291200.000500",
        "C1:1709377200.000100",
        "C2:1709290900.000100",
    ]
    root, reply, file, bot, later, other = records
    assert (
        root["text"] == "Planul: @田中 preia #proiecte, vezi documentul (https://example.org/doc) și https://example.org/raw. 1 < 2 && ok"
    )
    assert {k: root[k] for k in ("author", "kind", "collection", "thread", "at", "url")} == {
        "author": "Ana Popescu",
        "kind": "message",
        "collection": "general",
        "thread": "general:1709290800.000200",
        "at": 1709290800000,
        "url": "https://example.slack.com/archives/C1/p1709290800000200",
    }
    assert root["metadata"]["replies"] == 1
    assert (reply["text"], reply["author"], reply["thread"]) == ("了解しました @Ana Popescu 👍", "田中", "general:1709290800.000200")
    assert (file["text"], file["author"], file["thread"]) == ("File: التقرير الشهري", "omar", "general:1709291100.000400")
    assert (bot["text"], bot["author"]) == ("Deployed @here\nAttachment: Build 42 (https://example.org/b/42)", "deploy-bot")
    assert later["text"] == "Mulțumesc! #general"
    assert (other["text"], other["collection"], other["source"]) == ("مرحبا بالجميع", "proiecte", records[0]["source"])


async def test_only_the_channels_asked_for(slack: Path) -> None:
    records = await collect(from_slack_export(slack, channels=["#general"], collection="team", skip_bots=True))
    assert len(records) == 4
    assert all(r["collection"] == "team" and r["id"].startswith("C1:") for r in records)


async def test_no_conversation_folder_outside_the_export(tmp_path: Path) -> None:
    root = tmp_path / "export"
    (root / "general").mkdir(parents=True)
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "2024-01-01.json").write_text(
        json.dumps([{"type": "message", "user": "U1", "text": "outside", "ts": "1704100000.0001"}])
    )
    (root / "general" / "2024-01-01.json").write_text(
        json.dumps([{"type": "message", "user": "U1", "text": "inside", "ts": "1704100000.0002"}])
    )
    (root / "channels.json").write_text(
        json.dumps(
            [
                {"id": "C1", "name": "general"},
                {"id": "C2", "name": "../secret"},
                {"id": "C3", "name": ".."},
                {"id": "C4", "name": str(tmp_path / "secret")},
                {"id": "C5", "name": "a/../../secret"},
                {"id": "C6", "name": "..\\secret"},
            ]
        )
    )
    (root / "dms.json").write_text(json.dumps([{"id": "../secret", "members": ["U1"]}]))
    assert [r["text"] for r in await collect(from_slack_export(root))] == ["inside"]


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = tmp_path_factory.mktemp("git")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)

    def commit(name: str, email: str, date: str, message: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                f"user.name={name}",
                "-c",
                f"user.email={email}",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-q",
                "--cleanup=verbatim",
                "-F",
                "-",
            ],
            cwd=repo,
            env={**env, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date},
            input=message.encode(),
            check=True,
        )

    git("init", "-q", "-b", "main")
    (repo / "notițe").mkdir()
    (repo / "notițe" / "日本語.md").write_text("こんにちは\n")
    (repo / "README.md").write_text("hello\n")
    git("add", "-A")
    commit(
        "Ștefan Müller-Łukasiewicz 王小明",
        "stefan@example.org",
        "2024-03-01T10:00:00+02:00",
        "Adaugă notițele în japoneză 🗾\n\nCorpul mesajului: café, naïve, 漢字.\n",
    )
    (repo / "README.md").write_text("hello again\n")
    git("add", "-A")
    commit(
        "Ana",
        "ana@example.org",
        "2024-03-02T11:30:00Z",
        "\n".join(
            [
                "Tricky message with %H and --- and \u001e\u001f in it",
                "",
                "\u001e0123456789abcdef0123456789abcdef",
                "commit 0000000000000000000000000000000000000000",
                "Author: Nobody <nobody@example.org>",
                "-less but \u001fstill odd",
                "",
                "Files: not really",
            ]
        ),
    )
    return repo


async def test_reads_every_commit(repo: Path) -> None:
    second, first = await collect(from_git(repo))
    assert second["title"] == "Tricky message with %H and --- and \u001e\u001f in it"
    assert "commit 0000000000000000000000000000000000000000" in second["text"]
    assert "\u001e0123456789abcdef" in second["text"]
    assert second["text"].endswith("Files: README.md")
    assert second["author"] == "Ana"
    assert second["at"] == "2024-03-02T11:30:00Z"
    assert len(first["id"]) == 40
    assert first["metadata"]["sha"] == first["id"] and first["metadata"]["email"] == "stefan@example.org"
    assert first["author"] == "Ștefan Müller-Łukasiewicz 王小明"
    assert first["title"] == "Adaugă notițele în japoneză 🗾"
    assert first["text"] == "Adaugă notițele în japoneză 🗾\n\nCorpul mesajului: café, naïve, 漢字.\n\nFiles: README.md, notițe/日本語.md"
    assert date_parse(str(first["at"])) == date_parse("2024-03-01T08:00:00Z")
    assert (first["kind"], first["thread"], first["source"]) == ("commit", repo.name, repo.name)


async def test_git_options(repo: Path) -> None:
    assert len(await collect(from_git(repo, max_count=1))) == 1
    notes = await collect(from_git(repo, paths=["notițe"], thread="history", url=lambda sha: f"https://example.org/c/{sha}"))
    assert [r["title"] for r in notes] == ["Adaugă notițele în japoneză 🗾"]
    assert notes[0]["thread"] == "history" and notes[0]["url"] == f"https://example.org/c/{notes[0]['id']}"
    assert len(await collect(from_git(repo, since=datetime(2024, 3, 2, tzinfo=timezone.utc)))) == 1
    assert len(await collect(from_git(repo, branch="main", max_files=1))) == 2
    capped = await collect(from_git(repo, max_files=1))
    assert capped[1]["text"].endswith("Files: README.md, … and 1 more")


async def test_stops_early_and_fails_on_a_bad_revision(repo: Path, tmp_path: Path) -> None:
    gen = from_git(repo)
    async for r in gen:
        assert len(r["id"]) == 40
        break
    await gen.aclose()
    with pytest.raises(RuntimeError, match="git log failed"):
        await collect(from_git(repo, rev="no-such-branch"))
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(
        ["git", "init", "-q"], cwd=empty, check=True, env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    )
    assert await collect(from_git(empty)) == []


def paragraph(n: int) -> str:
    return f"Paragraph {n}: {'lorem ipsum dolor sit amet ' * 12}— ação, naïve, 漢字."


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    d = tmp_path
    (d / "notițe" / "日本").mkdir(parents=True)
    (d / ".git").mkdir()
    (d / "node_modules" / "pkg").mkdir(parents=True)
    (d / "notițe" / "plan.md").write_text(
        "\n".join(
            [
                "---",
                'title: "Planul pe trimestrul trei: priorități"',
                "date: 2024-07-01",
                "author: Ioana Mărgărit",
                "tags: [buget, echipă]",
                "reviewers:",
                "  - Ana",
                "  - Mihai",
                "draft: true",
                "version: 3",
                "---",
                "# Not the title",
                "",
                "Bugetul pentru țară crește cu 5%.",
            ]
        )
    )
    (d / "notițe" / "日本" / "メモ.md").write_text("Intro line\n\n# 東京オフィスの移転\n\n来月、新しいオフィスに移ります。🚚\n")
    (d / "readme.txt").write_text("Ελληνικά: καλημέρα κόσμε.\n")
    t = datetime(2023, 2, 3, 4, 5, 6, tzinfo=timezone.utc).timestamp()
    os.utime(d / "readme.txt", (t, t))
    (d / "page.html").write_text("""<!doctype html><html><head><title>Caf&eacute; &amp; Cr&egrave;me</title>
    <style>body { color: red }</style><script>var secret = "<p>not text</p>";</script></head>
    <body><h1>Menu</h1><p>Prix&nbsp;: 3&nbsp;&euro; &mdash; <b>très</b>
    bon</p><ul><li>Thé</li><li>&#x1F375; matcha &#8211; 抹茶</li></ul><!-- hidden --><p>Ünïcödé &lt;ok&gt;</p></body></html>""")
    (d / ".git" / "config.md").write_text("# hidden")
    (d / ".secret.md").write_text("# hidden")
    (d / "node_modules" / "pkg" / "README.md").write_text("# dependency")
    (d / "image.png").write_bytes(bytes([0x89, 0x50, 0x4E, 0x47, 0, 0, 0]))
    (d / "big.md").write_text("# Big\n\n" + "\n\n".join(paragraph(i) for i in range(30)))
    return d


async def test_reads_text_files(folder: Path) -> None:
    records = await collect(from_files(folder, collection="docs", max_chars=2000))
    assert [r["id"] for r in records if not r["id"].startswith("big.md")] == [
        "notițe/plan.md",
        "notițe/日本/メモ.md",
        "page.html",
        "readme.txt",
    ]
    assert all(r["collection"] == "docs" and r["kind"] == "note" for r in records)
    plan = next(r for r in records if r["id"] == "notițe/plan.md")
    assert plan["title"] == "Planul pe trimestrul trei: priorități"
    assert plan["author"] == "Ioana Mărgărit"
    assert plan["at"] == date_parse("2024-07-01")
    assert plan["text"].startswith("# Not the title")
    assert plan["metadata"] == {
        "date": "2024-07-01",
        "author": "Ioana Mărgărit",
        "tags": "buget, echipă",
        "reviewers": "Ana, Mihai",
        "draft": True,
        "version": 3,
    }
    assert next(r for r in records if r["id"] == "notițe/日本/メモ.md")["title"] == "東京オフィスの移転"
    readme = next(r for r in records if r["id"] == "readme.txt")
    assert readme["title"] == "readme"
    assert readme["at"] == date_parse("2023-02-03T04:05:06Z")
    assert readme["url"].startswith("file://")
    page = next(r for r in records if r["id"] == "page.html")
    assert page["title"] == "Café & Crème"
    for gone in ("secret", "color", "hidden", "<p>", "Café"):
        assert gone not in page["text"]
    assert "Prix : 3 € — très bon" in page["text"]
    assert "- Thé\n- 🍵 matcha – 抹茶" in page["text"]
    assert "Ünïcödé <ok>" in page["text"]


async def test_a_large_file_in_parts(folder: Path) -> None:
    parts = [r for r in await collect(from_files(folder, max_chars=2000)) if r["id"].startswith("big.md")]
    assert len(parts) > 3
    assert [p["id"] for p in parts] == [f"big.md#{i + 1}" for i in range(len(parts))]
    assert all(p["thread"] == "big.md" and len(p["text"]) <= 2000 for p in parts)
    assert parts[0]["title"] == f"Big (1/{len(parts)})"
    assert parts[1]["metadata"]["part"] == 2 and parts[1]["metadata"]["parts"] == len(parts)
    words = len(" ".join(p["text"] for p in parts).split())
    assert words == len(("# Big\n\n" + "\n\n".join(paragraph(i) for i in range(30))).split())


def test_split_text_without_spaces() -> None:
    cjk = "今日は晴れです。" * 40 + "😀" * 200
    parts = split_text(cjk, 150)
    assert "".join(parts) == cjk
    assert parts[0].endswith("。")


def test_drops_html_blockquotes() -> None:
    assert (
        html_to_text("<p>Yes.</p><blockquote>Old <blockquote>older</blockquote> text</blockquote><p>Bye</p>", drop_quotes=True)
        == "Yes.\n\nBye"
    )


async def test_jsonl_skips_and_counts(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "﻿"
        + "\r\n".join(
            [
                json.dumps(
                    {"id": "ja", "text": "会議は火曜日に延期されました 🎉", "title": "会議", "at": "2024-05-01T10:00:00Z"},
                    ensure_ascii=False,
                ),
                "{ not json",
                json.dumps({"text": "no id here"}),
                "",
                json.dumps({"id": 7, "text": "Întâlnirea de mâine e la București, în sala „Ștefan”."}, ensure_ascii=False),
                json.dumps({"id": "ar", "text": "تم تأجيل الاجتماع إلى يوم الثلاثاء"}, ensure_ascii=False),
                json.dumps({"id": "bad-time", "text": "x", "at": "whenever"}),
                "[1, 2]",
            ]
        ),
        encoding="utf-8",
    )
    skipped: list[str] = []
    stats: dict[str, int] = {}
    records = await collect(from_jsonl(path, on_skip=lambda _l, reason, _n: skipped.append(reason), stats=stats, source="notes"))
    assert [r["id"] for r in records] == ["ja", "7", "ar"]
    assert (records[0]["title"], records[0]["text"], records[0]["source"]) == ("会議", "会議は火曜日に延期されました 🎉", "notes")
    assert records[1]["text"] == "Întâlnirea de mâine e la București, în sala „Ștefan”."
    assert stats == {"lines": 7, "records": 3, "skipped": 4}
    assert skipped[0].startswith("not JSON")
    assert "needs an id" in skipped[1]


async def test_jsonl_map(tmp_path: Path) -> None:
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(o, ensure_ascii=False)
            for o in [
                {"msg_id": 1, "body": "Привет, как дела?", "who": "Олег", "ts": 1714557600000},
                {"msg_id": 2, "body": "", "who": "bot"},
                {"msg_id": 3, "body": "नमस्ते, सब ठीक है।", "who": "प्रिया", "ts": 1714557660000},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reasons: list[str] = []

    def to_record(o: Any, _n: int) -> dict[str, Any] | None:
        return {"id": str(o["msg_id"]), "text": o["body"], "author": o["who"], "at": o.get("ts"), "kind": "message"} if o["body"] else None

    records = await collect(from_jsonl(path, map=to_record, on_skip=lambda _l, r, _n: reasons.append(r)))
    assert [(r["id"], r["author"], r["text"]) for r in records] == [("1", "Олег", "Привет, как дела?"), ("3", "प्रिया", "नमस्ते, सब ठीक है।")]
    assert reasons == ["left out by map"]


async def test_batches() -> None:
    async def count() -> Any:
        for i in range(7):
            yield i

    assert await collect(batches(count(), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert await collect(batches(range(4), 2)) == [[0, 1], [2, 3]]
