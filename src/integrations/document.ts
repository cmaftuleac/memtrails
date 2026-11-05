/**
 * A found item as a framework's document (a LangChain `Document`; in Python also a LlamaIndex `TextNode`): its id,
 * the text a model reads, and flat metadata a vector store or a prompt can carry. `memory.open(id)` gives the whole item.
 */
import type { RecalledItem } from '../core/recall.js';

export interface RetrievedDocument {
  id: string;
  /** The item's summary, and the passage that matched when that was not its start. */
  text: string;
  /** id, kind, title, date (YYYY-MM-DD of `ts`), ts, score; for an observation eventAt, startAt, endAt; for a record you
   *  added recordId, source, author, url. Only the ones it has. */
  metadata: Record<string, string | number>;
}

export function retrievedDocument(h: RecalledItem): RetrievedDocument {
  const m: Record<string, string | number | undefined> = {
    id: h.id,
    kind: h.kind,
    title: h.title,
    date: h.ts > 0 ? new Date(h.ts).toISOString().slice(0, 10) : undefined,
    ts: h.ts,
    score: h.score,
    eventAt: h.eventAt,
    startAt: h.startAt,
    endAt: h.endAt,
    recordId: h.record?.id,
    source: h.record?.source,
    author: h.record?.author,
    url: h.record?.url,
  };
  const metadata = Object.fromEntries(Object.entries(m).filter((e): e is [string, string | number] => e[1] !== undefined && e[1] !== ''));
  return { id: h.id, text: h.match ? `${h.summary} … ${h.match}` : h.summary, metadata };
}
