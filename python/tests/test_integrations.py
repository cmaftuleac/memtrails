"""The framework retrievers (integrations/; integrations.test.ts, ported) over a stand-in memory: each found item becomes
the reference's document (conformance/fixtures/retriever.json), the query and the search options reach the search, the
retrievers run inside the frameworks (a LangChain chain with a stand-in chat model, a LlamaIndex query engine with
LlamaIndex's mock LLM: no model is called), and the sync calls say to use the async ones."""

from __future__ import annotations

from typing import Any

import pytest

from conftest import fixture
from memtrails.integrations import retrieved_document

KEY = fixture("retriever.json")


class Stand:
    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    async def search(self, q: dict[str, Any]) -> list[Any]:
        self.asked.append(q)
        return [x["item"] for x in KEY[:2]]


def test_each_item_becomes_the_references_document() -> None:
    for x in KEY:
        assert retrieved_document(x["item"]) == x["document"], x["item"]["id"]


async def test_the_langchain_retriever_searches_and_gives_documents() -> None:
    pytest.importorskip("langchain_core")
    from memtrails.integrations.langchain import MemtrailsRetriever

    s = Stand()
    r = MemtrailsRetriever(memory=s, search={"limit": 5, "collections": ["ops"]})
    docs = await r.ainvoke("why did renewal stop?")
    assert s.asked == [{"limit": 5, "collections": ["ops"], "query": "why did renewal stop?"}]
    assert [(d.id, d.page_content, d.metadata) for d in docs] == [
        (x["document"]["id"], x["document"]["text"], x["document"]["metadata"]) for x in KEY[:2]
    ]


async def test_the_langchain_retriever_runs_in_a_chain_and_refuses_sync() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    from memtrails.integrations.langchain import MemtrailsRetriever

    retriever = MemtrailsRetriever(memory=Stand())
    prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
    seen: list[str] = []

    def context(docs: list[Any]) -> str:
        text = "\n".join(d.page_content for d in docs)
        seen.append(text)
        return text

    chain = (
        {"context": retriever | context, "question": RunnablePassthrough()}
        | prompt
        | FakeListChatModel(responses=["The cron job was removed."])
        | StrOutputParser()
    )
    assert await chain.ainvoke("why did renewal stop?") == "The cron job was removed."
    assert seen == ["\n".join(x["document"]["text"] for x in KEY[:2])]
    with pytest.raises(NotImplementedError, match="ainvoke"):
        retriever.invoke("why?")


async def test_the_llama_index_retriever_gives_scored_nodes_and_runs_in_a_query_engine() -> None:
    pytest.importorskip("llama_index.core")
    from llama_index.core.llms import MockLLM
    from llama_index.core.query_engine import RetrieverQueryEngine

    from memtrails.integrations.llama_index import MemtrailsRetriever

    s = Stand()
    r = MemtrailsRetriever(s, search={"kinds": ["email"]})
    nodes = await r.aretrieve("why did renewal stop?")
    assert s.asked == [{"kinds": ["email"], "query": "why did renewal stop?"}]
    assert [(n.node.node_id, n.score, n.node.get_content(), n.node.metadata) for n in nodes] == [
        (x["document"]["id"], x["item"]["score"], x["document"]["text"], x["document"]["metadata"]) for x in KEY[:2]
    ]
    engine = RetrieverQueryEngine.from_args(r, llm=MockLLM())
    response = await engine.aquery("why did renewal stop?")
    assert [n.node.node_id for n in response.source_nodes] == [x["item"]["id"] for x in KEY[:2]]
    with pytest.raises(NotImplementedError, match="aretrieve"):
        r.retrieve("why?")
