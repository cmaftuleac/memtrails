"""
A Slack workspace export, unzipped: ``users.json``, ``channels.json`` (and ``groups.json``, ``mpims.json``, ``dms.json``
when present), and a folder a conversation holding one ``YYYY-MM-DD.json`` a day. Read a day at a time, in order.

One record a message that says something. id: ``channel id:ts``. text: the message with ``<@U123>`` as ``@Name``,
``<#C1|name>`` as ``#name``, ``<url|label>`` as ``label (url)``, then the names of its files and attachments. author:
the person's real name, else display name, else user name (a bot: its name). at: the message's ts. thread:
``channel:ts`` of the thread's first message (a message outside a thread starts its own). collection: ``collection`` or
the channel's name. kind ``message``. Joins, leaves, topic changes and the like are left out.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import regex

from .._js import UNDEFINED, js_trim, parse
from .text import record, undef

#: Message subtypes left out by default: events about the channel, not things said in it.
SLACK_SKIPPED_SUBTYPES = [
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "channel_archive",
    "channel_unarchive",
    "channel_convert_to_private",
    "channel_convert_to_public",
    "group_join",
    "group_leave",
    "group_topic",
    "group_purpose",
    "group_name",
    "group_archive",
    "group_unarchive",
    "pinned_item",
    "unpinned_item",
    "bot_add",
    "bot_remove",
    "tombstone",
    "reminder_add",
    "joiner_notification",
    "joiner_notification_for_inviter",
    "message_deleted",
    "sh_room_created",
    "channel_canvas_updated",
]


def _read_json_sync(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    except OSError as err:
        raise RuntimeError(f"cannot read {path}: {err}") from err
    try:
        return parse(text.removeprefix("﻿"))
    except ValueError as err:
        raise RuntimeError(f"cannot read {path}: {err}") from err
    except RecursionError:
        # Nested deeper than Python's parser goes (JavaScript's has no such limit): the file is skipped and the export
        # read on, never a RecursionError out of the import.
        return None


async def _read_json(path: str) -> Any:
    return await asyncio.to_thread(_read_json_sync, path)


def _person_name(u: Mapping[str, Any]) -> str | None:
    found = u.get("profile")
    profile: Mapping[str, Any] = found if isinstance(found, Mapping) else {}
    for n in (profile.get("real_name"), u.get("real_name"), profile.get("display_name"), u.get("display_name"), u.get("name")):
        if isinstance(n, str) and js_trim(n):
            return js_trim(n)
    return None


_MARKUP = regex.compile(r"<([^<>\n]*)>")
_AFTER_CARET = regex.compile(r"\^.*$", regex.DOTALL)


def slack_text(text: str, users: Mapping[str, str], channels: Mapping[str, str]) -> str:
    """Slack's markup as plain text: mentions as names, channel links as ``#name``, links as ``label (url)``."""

    def one(m: regex.Match[str]) -> str:
        inner = m.group(1)
        bar = inner.find("|")
        target = inner[:bar] if bar >= 0 else inner
        label = inner[bar + 1 :] if bar >= 0 else ""
        if target.startswith("@"):
            return f"@{users.get(target[1:]) or (label.removeprefix('@') or target[1:])}"
        if target.startswith("#"):
            return f"#{label or channels.get(target[1:]) or target[1:]}"
        if target.startswith("!"):
            special = target[1:]
            if regex.fullmatch(r"here|channel|everyone", special):
                return f"@{special}"
            return label or "@" + _AFTER_CARET.sub("", special)
        if not target:
            return m.group(0)
        if target.startswith("mailto:"):
            return label or target[7:]
        return f"{label} ({target})" if label and label != target else target

    return _MARKUP.sub(one, text).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


_DAY_FILE = regex.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}\.json$")


def _list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


async def from_slack_export(
    directory: str | os.PathLike[str],
    channels: Sequence[str] | None = None,
    collection: str | None = None,
    source: str | None = None,
    skip_subtypes: Sequence[str] | None = None,
    skip_bots: bool = False,
    workspace_url: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Records from an unzipped Slack export folder. ``channels``: only these conversations (by name or id);
    ``collection``: default the channel's name; ``source``: default the export folder's name; ``skip_subtypes``: default
    SLACK_SKIPPED_SUBTYPES; ``skip_bots``: leave out messages posted by bots and integrations; ``workspace_url``
    (``https://example.slack.com``): a link to each message."""
    root = os.path.abspath(os.fspath(directory))
    src = source if source is not None else os.path.basename(root)
    skip = set(skip_subtypes if skip_subtypes is not None else SLACK_SKIPPED_SUBTYPES)
    users: dict[str, str] = {}
    for u in _list(await _read_json(os.path.join(root, "users.json"))):
        n = _person_name(u) if isinstance(u, Mapping) else None
        if isinstance(u, Mapping) and u.get("id") and n:
            users[u["id"]] = n
    # Each conversation: its id, its name, and the folder its days are in (a direct message's folder is its id).
    conversations: list[tuple[str, str, str]] = []
    for file, by_id in (("channels.json", False), ("groups.json", False), ("mpims.json", False), ("dms.json", True)):
        for c in _list(await _read_json(os.path.join(root, file))):
            if not isinstance(c, Mapping) or not c.get("id"):
                continue
            if by_id:
                name = ", ".join(users.get(m, m) for m in _list(c.get("members"))) or c["id"]
            else:
                name = c.get("name") or c["id"]
            conversations.append((c["id"], name, c["id"] if by_id else c.get("name") or c["id"]))
    if not conversations:
        # No lists: every folder is a conversation named after it.
        names = sorted(await asyncio.to_thread(os.listdir, root))
        for e in names:
            if not e.startswith(".") and os.path.isdir(os.path.join(root, e)):
                conversations.append((e, e, e))
    channel_names = {cid: name for cid, name, _ in conversations}
    wanted = {c.removeprefix("#") for c in channels} if channels is not None else None

    for cid, name, folder_name in conversations:
        if wanted is not None and name not in wanted and cid not in wanted:
            continue
        # The folder's name comes from the export's own lists: `../x`, `a/b` or an absolute path must not lead out of it.
        folder = os.path.abspath(os.path.join(root, folder_name))
        if not folder_name or os.path.dirname(folder) != root or not os.path.isdir(folder):
            continue
        days = sorted(f for f in await asyncio.to_thread(os.listdir, folder) if _DAY_FILE.match(f))
        for day in days:
            messages = await _read_json(os.path.join(folder, day))
            if not isinstance(messages, list):
                continue
            for m in messages:
                if not isinstance(m, Mapping) or not m.get("ts") or (m.get("type") and m["type"] != "message") or m.get("hidden"):
                    continue
                if m.get("subtype") and m["subtype"] in skip:
                    continue
                bot = m.get("subtype") == "bot_message" or (bool(m.get("bot_id")) and not m.get("user"))
                if bot and skip_bots:
                    continue
                extras = [
                    f"File: {t}"
                    for t in ((f or {}).get("title") or (f or {}).get("name") for f in _list(m.get("files")))
                    if isinstance(t, str) and js_trim(t)
                ]
                for a in _list(m.get("attachments")):
                    t = (a or {}).get("title") or (a or {}).get("fallback")
                    if not isinstance(t, str) or not js_trim(t):
                        continue
                    link = a.get("title_link") or a.get("from_url")
                    extras.append(f"Attachment: {slack_text(t, users, channel_names)}{f' ({link})' if link and link not in t else ''}")
                said = js_trim(slack_text(m.get("text") or "", users, channel_names))
                text = "\n".join(x for x in (said, *extras) if x)
                if not text:
                    continue
                profile = m.get("user_profile") if isinstance(m.get("user_profile"), Mapping) else None
                author = (
                    (m.get("user") and users.get(m["user"]))
                    or (profile and _person_name(profile))
                    or ((m.get("bot_profile") or {}).get("name"))
                    or m.get("username")
                    or m.get("user")
                    or None
                )
                ts = str(m["ts"])
                try:
                    ms: float | None = math.floor(float(ts) * 1000 + 0.5)  # Math.round
                except (ValueError, OverflowError):
                    ms = None
                if ms is not None and not math.isfinite(ms):
                    ms = None
                yield record(
                    {
                        "id": f"{cid}:{ts}",
                        "text": text,
                        "kind": "message",
                        "source": src,
                        "author": undef(author),
                        "collection": collection if collection is not None else name,
                        "thread": f"{name}:{m.get('thread_ts') or ts}",
                        "at": undef(ms),
                        "url": f"{workspace_url.rstrip('/')}/archives/{cid}/p{ts.replace('.', '', 1)}" if workspace_url else UNDEFINED,
                        "metadata": {
                            "channel": name,
                            "ts": ts,
                            "threadTs": m.get("thread_ts", UNDEFINED),
                            "subtype": m.get("subtype", UNDEFINED),
                            "replies": m.get("reply_count", UNDEFINED),
                        },
                    }
                )
