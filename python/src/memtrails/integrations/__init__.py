"""A memory in other frameworks (each imported on its own, so memtrails needs none of them):

- ``memtrails.integrations.langchain.MemtrailsRetriever`` — a LangChain retriever (``pip install memtrails[langchain]``);
- ``memtrails.integrations.llama_index.MemtrailsRetriever`` — a LlamaIndex retriever (``pip install memtrails[llama-index]``).

Each found item becomes the framework's document as ``retrieved_document`` shapes it (integrations/document.ts, ported:
held to conformance/fixtures/retriever.json)."""

from ._document import retrieved_document

__all__ = ["retrieved_document"]
