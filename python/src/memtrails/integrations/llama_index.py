"""A memory as a LlamaIndex retriever (``pip install memtrails[llama-index]``; llamaindex.ts, ported):

    from memtrails.integrations.llama_index import MemtrailsRetriever

    retriever = MemtrailsRetriever(memory, search={"limit": 8})
    nodes = await retriever.aretrieve("why did the certificate stop renewing?")

Each node is a found item with the search's score: its summary (and the passage that matched) as text, its id, and flat
metadata (``retrieved_document``). ``search`` is every search option but the query. The memory is async, so the
retriever is too: ``aretrieve`` (and async query engines), never ``retrieve``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from ._document import SYNC_REFUSED, Searches, retrieved_document


class MemtrailsRetriever(BaseRetriever):
    def __init__(self, memory: Searches, search: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memory = memory
        self._search = dict(search or {})

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        raise NotImplementedError(SYNC_REFUSED)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        hits = await self._memory.search({**self._search, "query": query_bundle.query_str})
        out = []
        for h in hits:
            d = retrieved_document(h)
            out.append(NodeWithScore(node=TextNode(id_=d["id"], text=d["text"], metadata=d["metadata"]), score=h.get("score")))
        return out
