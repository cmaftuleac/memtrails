"""A memory as a LangChain retriever (``pip install memtrails[langchain]``; langchain.ts, ported):

    from memtrails.integrations.langchain import MemtrailsRetriever

    retriever = MemtrailsRetriever(memory=memory, search={"limit": 8})
    docs = await retriever.ainvoke("why did the certificate stop renewing?")

Each Document is a found item: its summary (and the passage that matched) as ``page_content``, its id, and flat metadata
(``retrieved_document``). ``search`` is every search option but the query: collections, kinds, limit, from, to. The
memory is async, so the retriever is too: ``ainvoke`` (and async chains), never ``invoke``."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ._document import SYNC_REFUSED, retrieved_document


class MemtrailsRetriever(BaseRetriever):
    #: An open memory (open_memory), or anything with an async ``search`` like it.
    memory: Any
    #: Every search option but the query.
    search: dict[str, Any] = Field(default_factory=dict)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise NotImplementedError(SYNC_REFUSED)

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> list[Document]:
        hits = await self.memory.search({**self.search, "query": query})
        docs = [retrieved_document(h) for h in hits]
        return [Document(id=d["id"], page_content=d["text"], metadata=d["metadata"]) for d in docs]
