/**
 * memtrails — retrieval and exploration for any text: e-mail, chat, notes, documents, news, commits, conversations.
 *
 *   add          records, conversations, or a stream from an importer — searchable at once, no model asked
 *   search       dense + keyword, fused; reordered word by word on this machine; dates first when asked
 *   room         the best sentences of what was found, for a small context
 *   open         one item whole, with the records it was learned from
 *   explore      the concept graph from a topic or items to what search does not find
 *   consolidate  (with a language model) typed knowledge, dated observations, kept episodes
 *   sleep        (with a language model) merge, supersede, link, profiles
 *   gather       (with a language model) every instance a counting or listing question needs
 *   tools        memory_search / memory_open / memory_explore for an agent
 */
export { openMemory, type Memory, type MemoryOptions, type SearchOptions } from './memory.js';
export { DEFAULT_COLLECTION, recordUid, timeOf, type Conversation, type MemoryRecord } from './records.js';
export type { RecalledItem, RecordRef, SearchQuery, Explored } from './core/recall.js';
export type { Gathered } from './core/gather.js';
export type { ConsolidationReport } from './core/consolidate.js';
export type { SleepReport } from './core/sleep.js';
export { openAiAsk, type Ask, type LlmRequest, type LlmAnswer, type LlmUsage, type OpenAiAskOptions } from './llm.js';
export { buildAgentTools, searchText, type AgentTool, type ToolProvider } from './tools/agent-tools.js';
export { ensureModel, defaultModelDir, MODEL_REPO, MODEL_REVISION } from './core/model.js';
export { ensureQdrant, QDRANT_VERSION } from './core/qdrant-binary.js';
export type { EmbedRoute } from './core/embedder.js';
export type { RerankRoute } from './core/reranker.js';
export { sentenceRoom, sentenceSpans } from './core/room.js';
export { maxSim, packWords, WordStore, WORD_LIMIT, type WordEncoder, type WordVectors, type PackedWords } from './core/words.js';
export { cacheDir } from './core/paths.js';
