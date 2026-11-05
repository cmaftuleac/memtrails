"""
memtrails — retrieval and exploration for any text: e-mail, chat, notes, documents, news, commits, conversations.

  add          records, conversations, or a stream from an importer — searchable at once, no model asked
  search       dense + keyword, fused; reordered word by word on this machine; dates first when asked
  room         the best sentences of what was found, for a small context
  open         one item whole, with the records it was learned from
  explore      the concept graph from a topic or items to what search does not find
  consolidate  (with a language model) typed knowledge, dated observations, kept episodes
  sleep        (with a language model) merge, supersede, link, profiles
  gather       (with a language model) every instance a counting or listing question needs
  tools        memory_search / memory_open / memory_explore for an agent

A native port of the TypeScript library (npm ``memtrails``): the same algorithms, prompts and files.
"""

from .core.embedder import EmbedRoute
from .core.gather import Gathered
from .core.model import MODEL_REPO, MODEL_REVISION, default_model_dir, ensure_model
from .core.paths import cache_dir
from .core.qdrant_binary import QDRANT_VERSION, ensure_qdrant
from .core.recall import Explored, RecalledItem, RecordRef
from .core.reranker import RerankRoute
from .core.room import sentence_room, sentence_spans
from .core.words import WORD_LIMIT, PackedWords, WordStore, WordVectors, max_sim, pack_words
from .llm import Ask, LlmRequest, LlmUsage, OpenAiAsk, openai_ask
from .memory import EmbeddingOptions, Memory, QdrantOptions, SearchOptions, open_memory
from .records import DEFAULT_COLLECTION, Conversation, MemoryRecord, record_uid, time_of
from .tools.agent_tools import AgentTool, build_agent_tools, search_text

__version__ = "0.5.0"

__all__ = [
    "DEFAULT_COLLECTION",
    "MODEL_REPO",
    "MODEL_REVISION",
    "QDRANT_VERSION",
    "WORD_LIMIT",
    "AgentTool",
    "Ask",
    "Conversation",
    "EmbedRoute",
    "EmbeddingOptions",
    "Explored",
    "Gathered",
    "LlmRequest",
    "LlmUsage",
    "Memory",
    "MemoryRecord",
    "OpenAiAsk",
    "PackedWords",
    "QdrantOptions",
    "RecalledItem",
    "RecordRef",
    "RerankRoute",
    "SearchOptions",
    "WordStore",
    "WordVectors",
    "__version__",
    "build_agent_tools",
    "cache_dir",
    "default_model_dir",
    "ensure_model",
    "ensure_qdrant",
    "max_sim",
    "open_memory",
    "openai_ask",
    "pack_words",
    "record_uid",
    "search_text",
    "sentence_room",
    "sentence_spans",
    "time_of",
]
