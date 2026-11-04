"""
Importers: records from what people already have — JSON Lines, a folder of text files, a Git history, a Slack export, an
mbox or ``.eml`` mailbox. Each one is an async generator that streams: records come one at a time as the source is read,
never the whole source in memory, so ``await memory.import_(from_mbox(path))`` works on a mailbox of any size.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TypeVar

from .files import TEXT_EXTENSIONS, from_files, front_matter
from .git import from_git
from .jsonl import JsonlMap, from_jsonl
from .mbox import decode_encoded_words, from_eml, from_mbox
from .slack import SLACK_SKIPPED_SUBTYPES, from_slack_export, slack_text
from .text import decode_bytes, decode_entities, html_to_text, split_text

T = TypeVar("T")


async def batches(items: AsyncIterable[T] | Iterable[T], size: int) -> AsyncIterator[list[T]]:
    """Items in lists of ``size`` (the last may be shorter), as they come."""
    n = max(1, int(size))
    batch: list[T] = []
    if isinstance(items, AsyncIterable):
        async for item in items:
            batch.append(item)
            if len(batch) >= n:
                yield batch
                batch = []
    else:
        for item in items:
            batch.append(item)
            if len(batch) >= n:
                yield batch
                batch = []
    if batch:
        yield batch


__all__ = [
    "SLACK_SKIPPED_SUBTYPES",
    "TEXT_EXTENSIONS",
    "JsonlMap",
    "batches",
    "decode_bytes",
    "decode_encoded_words",
    "decode_entities",
    "from_eml",
    "from_files",
    "from_git",
    "from_jsonl",
    "from_mbox",
    "from_slack_export",
    "front_matter",
    "html_to_text",
    "slack_text",
    "split_text",
]
