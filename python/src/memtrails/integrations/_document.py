"""A found item as a framework's document (a LangChain ``Document``, a LlamaIndex ``TextNode``): its id, the text a model
reads, and flat metadata a vector store or a prompt can carry. ``memory.open(id)`` gives the whole item."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, TypedDict

from ..core.recall import RecalledItem


class RetrievedDocument(TypedDict):
    id: str
    #: The item's summary, and the passage that matched when that was not its start.
    text: str
    #: id, kind, title, date (YYYY-MM-DD of ``ts``), ts, score; for an observation eventAt, startAt, endAt; for a record
    #: you added recordId, source, author, url. Only the ones it has.
    metadata: dict[str, str | int | float]


class Searches(Protocol):
    """An open memory (open_memory), or anything that searches like one."""

    async def search(self, q: Any) -> list[RecalledItem]: ...


def retrieved_document(h: RecalledItem) -> RetrievedDocument:
    record = h.get("record") or {}
    ts = h.get("ts") or 0
    m: dict[str, Any] = {
        "id": h["id"],
        "kind": h.get("kind"),
        "title": h.get("title"),
        "date": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if ts > 0 else None,
        "ts": h.get("ts"),
        "score": h.get("score"),
        "eventAt": h.get("eventAt"),
        "startAt": h.get("startAt"),
        "endAt": h.get("endAt"),
        "recordId": record.get("id"),
        "source": record.get("source"),
        "author": record.get("author"),
        "url": record.get("url"),
    }
    match = h.get("match")
    return {
        "id": h["id"],
        "text": f"{h['summary']} … {match}" if match else h["summary"],
        "metadata": {k: v for k, v in m.items() if v is not None and v != ""},
    }


SYNC_REFUSED = (
    "memtrails is async: its memory belongs to the event loop it was opened on. Use the async call "
    "(`await retriever.ainvoke(query)` in LangChain, `await retriever.aretrieve(query)` in LlamaIndex)."
)
