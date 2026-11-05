/**
 * A memory as a LlamaIndex retriever (`@llamaindex/core`, installed beside memtrails):
 *
 *   import { MemtrailsRetriever } from 'memtrails/llamaindex';
 *   const retriever = new MemtrailsRetriever({ memory, search: { limit: 8 } });
 *   const nodes = await retriever.retrieve('why did the certificate stop renewing?');
 *
 * Each node is a found item with the search's score: its summary (and the passage that matched) as text, its id, and
 * flat metadata (integrations/document.ts). `search` is every search option but the query.
 */
import { BaseRetriever } from '@llamaindex/core/retriever';
import type { QueryBundle } from '@llamaindex/core/query-engine';
import { TextNode, type NodeWithScore } from '@llamaindex/core/schema';
import type { RecalledItem, SearchQuery } from './core/recall.js';
import { retrievedDocument } from './integrations/document.js';

export interface MemtrailsRetrieverInput {
  /** An open memory (openMemory), or anything that searches like one. */
  memory: { search(q: SearchQuery): Promise<RecalledItem[]> };
  search?: Omit<SearchQuery, 'query'>;
}

/** The words of a query: a string, or the text parts of a message. */
function queryText(q: QueryBundle['query']): string {
  if (typeof q === 'string') return q;
  return q.map((p) => (p.type === 'text' ? p.text : '')).filter(Boolean).join(' ');
}

export class MemtrailsRetriever extends BaseRetriever {
  private readonly memory: MemtrailsRetrieverInput['memory'];
  private readonly search: Omit<SearchQuery, 'query'>;

  constructor(fields: MemtrailsRetrieverInput) {
    super();
    this.memory = fields.memory;
    this.search = fields.search ?? {};
  }

  async _retrieve(params: QueryBundle): Promise<NodeWithScore[]> {
    const hits = await this.memory.search({ ...this.search, query: queryText(params.query) });
    return hits.map((h) => {
      const d = retrievedDocument(h);
      return { node: new TextNode({ id_: d.id, text: d.text, metadata: d.metadata }), score: h.score };
    });
  }
}
