"""The importers' answer key (conformance/fixtures/rust-importers.json, written by conformance/generate-rust.ts from the
TypeScript importers over conformance/inputs/importers): the same bytes in, the same records out."""

from __future__ import annotations

import os
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES, fixture

from memtrails.importers import decode_encoded_words, from_eml, from_files, from_jsonl, from_mbox, from_slack_export, front_matter, slack_text
from memtrails.importers.text import decode_bytes, decode_entities, html_title, html_to_text, split_text

KEY = fixture("rust-importers.json")
IN = FIXTURES.parent / "inputs" / "importers"
ROOT_URL = IN.resolve().as_uri()


async def collect(it: AsyncIterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [x async for x in it]


def stable(records: list[dict[str, Any]], directory: Path | None = None) -> list[dict[str, Any]]:
    """The machine's parts as the key writes them: the inputs' folder in a URL, a modification time."""
    out = []
    for r in records:
        r = dict(r)
        if isinstance(r.get("url"), str) and r["url"].startswith(ROOT_URL):
            r["url"] = "file://<root>" + r["url"][len(ROOT_URL) :]
        if directory is not None and isinstance(r.get("at"), (int, float)):
            path = directory / (r.get("thread") or r["id"])
            if path.exists() and int(os.stat(path).st_mtime * 1000 + 0.5) == r["at"]:
                r["at"] = "<mtime>"
        out.append(r)
    return out


async def test_jsonl() -> None:
    skipped: list[dict[str, Any]] = []

    def skip(_line: str, reason: str, n: int) -> None:
        skipped.append({"line": n, "reason": "not JSON" if reason.startswith("not JSON") else reason})

    assert await collect(from_jsonl(IN / "records.jsonl", source="notes", kind="record", on_skip=skip)) == KEY["jsonl"]["records"]
    assert skipped == KEY["jsonl"]["skipped"]


async def test_files() -> None:
    skipped: list[dict[str, str]] = []
    records = await collect(from_files(IN / "files", collection="docs", max_chars=1500, on_skip=lambda p, r: skipped.append({"path": p, "reason": r})))
    assert stable(records, IN / "files") == KEY["files"]["records"]
    assert skipped == KEY["files"]["skipped"]


async def test_slack() -> None:
    assert await collect(from_slack_export(IN / "slack", workspace_url="https://example.slack.com/")) == KEY["slack"]["records"]
    filtered = await collect(from_slack_export(IN / "slack", channels=["#general", "D1"], collection="team", skip_bots=True, source="ws"))
    assert filtered == KEY["slack"]["filtered"]


async def test_mbox_and_eml() -> None:
    skipped: list[int] = []
    assert await collect(from_mbox(IN / "mail" / "Inbox.mbox", on_skip=lambda _r, i: skipped.append(i))) == KEY["mbox"]["records"]
    assert skipped == KEY["mbox"]["skipped"]
    kept = await collect(from_mbox(IN / "mail" / "Inbox.mbox", keep_quoted=True, keep_signature=True, collection="work", source="box"))
    assert kept == KEY["mbox"]["kept"]
    assert [*await collect(from_eml(IN / "mail" / "note.eml")), *await collect(from_eml(IN / "mail" / "bare.eml"))] == KEY["eml"]


@pytest.mark.parametrize("case", KEY["html"], ids=lambda c: repr(c["html"][:20]))
def test_html(case: dict[str, Any]) -> None:
    assert html_to_text(case["html"]) == case["text"]
    assert html_to_text(case["html"], drop_quotes=True) == case["dropQuotes"]
    assert html_title(case["html"]) == case["title"]


def test_text_tools() -> None:
    for c in KEY["titles"]:
        assert html_title(c["html"]) == c["title"]
    for c in KEY["entities"]:
        assert decode_entities(c["s"]) == c["text"], c["s"]
    for c in KEY["bytes"]:
        assert decode_bytes(bytes(c["bytes"]), c["charset"]) == c["text"], c
    for c in KEY["split"]:
        assert split_text(c["text"], c["max"]) == c["parts"]
    for c in KEY["encodedWords"]:
        assert decode_encoded_words(c["s"]) == c["text"], c["s"]
    for c in KEY["slackText"]:
        assert slack_text(c["s"], {"U1": "Ana"}, {"C1": "general"}) == c["text"]
    for c in KEY["frontMatter"]:
        fields, body = front_matter(c["s"])
        assert (fields, body) == (c["fields"], c["body"]), c["s"]
