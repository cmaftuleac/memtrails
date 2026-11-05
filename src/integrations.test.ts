/**
 * The LangChain retriever (langchain.ts) over a stand-in memory: the query and the search options reach the search,
 * each item comes back as a Document with its id, text and flat metadata, and the retriever runs in a chain.
 */
import { RunnableLambda } from '@langchain/core/runnables';
import { describe, expect, it } from 'vitest';
import type { RecalledItem, SearchQuery } from './core/recall.js';
import { retrievedDocument } from './integrations/document.js';
import { MemtrailsRetriever as LangChainRetriever } from './langchain.js';

const EMAIL: RecalledItem = {
  id: 'stm:aa', title: 'Proxy certificate expired', summary: 'Proxy certificate expired Ana: Renewal stopped because the cron job was removed.',
  where: '', project: 'default', engine: 'email', ts: Date.UTC(2026, 3, 2), kind: 'email', score: 0.91,
  record: { id: 'mail-1', source: 'work-mailbox', author: 'Ana' },
};
const OBSERVATION: RecalledItem = {
  id: 'k:bb', title: 'Cron job removed', summary: 'The renewal cron job was removed during the March cleanup.', where: '', project: 'default',
  engine: 'conversation', ts: Date.UTC(2026, 3, 5), kind: 'observation', eventAt: '2026-03', score: 0.5,
  match: 'removed in March during the server cleanup',
};

function stand() {
  const asked: SearchQuery[] = [];
  return { asked, memory: { search: async (q: SearchQuery) => { asked.push(q); return [EMAIL, OBSERVATION]; } } };
}

describe('a found item as a document', () => {
  it('carries its id, the text a model reads, and only the metadata it has', () => {
    expect(retrievedDocument(EMAIL)).toEqual({
      id: 'stm:aa',
      text: EMAIL.summary,
      metadata: { id: 'stm:aa', kind: 'email', title: 'Proxy certificate expired', date: '2026-04-02', ts: EMAIL.ts, score: 0.91,
        recordId: 'mail-1', source: 'work-mailbox', author: 'Ana' },
    });
    const o = retrievedDocument(OBSERVATION);
    expect(o.text).toBe('The renewal cron job was removed during the March cleanup. … removed in March during the server cleanup');
    expect(o.metadata.eventAt).toBe('2026-03');
    expect('recordId' in o.metadata).toBe(false);
  });
});

describe('the LangChain retriever', () => {
  it('searches with the query and its options, and gives Documents', async () => {
    const s = stand();
    const r = new LangChainRetriever({ memory: s.memory, search: { limit: 5, collections: ['ops'] } });
    const docs = await r.invoke('why did renewal stop?');
    expect(s.asked).toEqual([{ limit: 5, collections: ['ops'], query: 'why did renewal stop?' }]);
    expect(docs.map((d) => [d.id, d.pageContent, d.metadata.kind])).toEqual([
      ['stm:aa', EMAIL.summary, 'email'],
      ['k:bb', retrievedDocument(OBSERVATION).text, 'observation'],
    ]);
  });

  it('runs in a chain', async () => {
    const s = stand();
    const chain = new LangChainRetriever({ memory: s.memory }).pipe(RunnableLambda.from((docs: Array<{ metadata: Record<string, unknown> }>) => docs.map((d) => d.metadata.date).join(',')));
    expect(await chain.invoke('renewal')).toBe('2026-04-02,2026-04-05');
  });
});
