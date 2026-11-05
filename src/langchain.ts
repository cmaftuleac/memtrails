/**
 * A memory as a LangChain retriever (`@langchain/core`, installed beside memtrails):
 *
 *   import { MemtrailsRetriever } from 'memtrails/langchain';
 *   const retriever = new MemtrailsRetriever({ memory, search: { limit: 8 } });
 *   const docs = await retriever.invoke('why did the certificate stop renewing?');
 *
 * Each Document is a found item: its summary (and the passage that matched) as `pageContent`, its id, and flat metadata
 * (integrations/document.ts). `search` is every search option but the query: collections, kinds, limit, from, to.
 */
import { Document } from '@langchain/core/documents';
import { BaseRetriever, type BaseRetrieverInput } from '@langchain/core/retrievers';
import type { RecalledItem, SearchQuery } from './core/recall.js';
import { retrievedDocument } from './integrations/document.js';

export interface MemtrailsRetrieverInput extends BaseRetrieverInput {
  /** An open memory (openMemory), or anything that searches like one. */
  memory: { search(q: SearchQuery): Promise<RecalledItem[]> };
  search?: Omit<SearchQuery, 'query'>;
}

export class MemtrailsRetriever extends BaseRetriever {
  lc_namespace = ['memtrails', 'retrievers'];
  private readonly memory: MemtrailsRetrieverInput['memory'];
  private readonly search: Omit<SearchQuery, 'query'>;

  constructor(fields: MemtrailsRetrieverInput) {
    super(fields);
    this.memory = fields.memory;
    this.search = fields.search ?? {};
  }

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const hits = await this.memory.search({ ...this.search, query });
    return hits.map((h) => {
      const d = retrievedDocument(h);
      return new Document({ id: d.id, pageContent: d.text, metadata: d.metadata });
    });
  }
}
