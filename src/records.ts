/**
 * What memtrails stores: records. Anything with text is a record — an e-mail, a chat message, a note, a memo, a news
 * article, a commit message, a ticket, a page of a document. Records that belong together (an e-mail thread, a chat
 * thread, a meeting's notes) share a `thread`; a conversation between a person and an assistant is a thread of turns.
 *
 * Every record is searchable the moment it is added, with no model asked. Consolidation (optional, with a language
 * model) reads threads in windows and keeps typed knowledge from them; the records themselves are never lost.
 */
import { createHash } from 'node:crypto';
import type { StmEntry } from './core/store.js';

/** One piece of information to remember. */
export interface MemoryRecord {
  /** Stable and unique within its `source`: the same id added again replaces the record (unchanged: nothing is done). */
  id: string;
  /** The content. */
  text: string;
  /** A subject, headline or name; when absent, the first line of the text. */
  title?: string;
  /** What it is, in one word of your choosing: `email`, `message`, `note`, `memo`, `article`, `commit`, `ticket`, … */
  kind?: string;
  /** Where it came from: a mailbox, a workspace, a feed, a repository. Part of the record's identity. */
  source?: string;
  /** What it belongs to, for searches scoped to one collection: a project, a team, a customer, a person. */
  collection?: string;
  /** Records that belong together share a thread (an e-mail thread, a chat thread, a repository's history). */
  thread?: string;
  /** Who wrote it. */
  author?: string;
  /** When it was written: milliseconds since the epoch, an ISO date, or a Date. Default: when it was added. */
  at?: number | string | Date;
  /** Where to open it. */
  url?: string;
  /** Anything else, kept with the record and returned with it. Not searched. */
  metadata?: Record<string, string | number | boolean | null>;
}

/** A conversation: turns between a person and an assistant (or any two parties), in order. */
export interface Conversation {
  /** Stable: the same id added again is read again, not doubled. */
  id: string;
  collection?: string;
  source?: string;
  /** When it began (each turn may carry its own `at`). */
  at?: number | string | Date;
  turns: Array<{ role: 'user' | 'assistant'; text: string; author?: string; at?: number | string | Date }>;
}

/** The collection records without one belong to. */
export const DEFAULT_COLLECTION = 'default';

/** A time as milliseconds; undefined for nothing or something unreadable. */
export function timeOf(at: number | string | Date | undefined): number | undefined {
  if (at === undefined || at === null || at === '') return undefined;
  const t = at instanceof Date ? at.getTime() : typeof at === 'number' ? at : Date.parse(at);
  return Number.isFinite(t) ? t : undefined;
}

/** A collection as the index keys it: one path segment, any script kept. */
export function collectionKey(collection: string | undefined): string {
  const c = (collection ?? '').trim();
  return c ? c.replace(/[\s/\\:]+/g, '-') : DEFAULT_COLLECTION;
}

function hash(...parts: Array<string | undefined>): string {
  return createHash('sha256').update(parts.map((p) => p ?? '').join('\u0000')).digest('hex');
}

/** A record's id in memory: its source and id, hashed (any characters may be in either). */
export function recordUid(r: Pick<MemoryRecord, 'id' | 'source'>): string {
  return hash(r.source ?? '', r.id).slice(0, 32);
}

/** Checks a record and says what is wrong with it, or null. */
export function invalidRecord(r: MemoryRecord): string | null {
  if (!r || typeof r !== 'object') return 'a record must be an object';
  if (typeof r.id !== 'string' || !r.id.trim()) return 'a record needs an id';
  if (typeof r.text !== 'string') return `record ${r.id}: text must be a string`;
  if (!r.text.trim() && !r.title?.trim()) return `record ${r.id}: it has no text and no title`;
  if (r.at !== undefined && timeOf(r.at) === undefined) return `record ${r.id}: its time cannot be read (${String(r.at)})`;
  return null;
}

/**
 * A record as memory keeps it: the title is what it is about (the first line when there is none), the text after the
 * author is the content. Searchable as written: its start is its key, and every word of it is in a chunk.
 */
export function entryOf(r: MemoryRecord, now: number, turn = 0): StmEntry {
  const text = r.text.trim();
  const firstLine = text.split('\n').map((l) => l.trim()).find(Boolean) ?? '';
  const title = r.title?.trim() || firstLine;
  // Without a title its first line is the title, and not said twice.
  const body = r.title?.trim() ? text : text.slice(text.indexOf(firstLine) + firstLine.length).trim();
  const by = r.author?.trim() ? `${r.author.trim()}: ` : '';
  const uid = recordUid(r);
  const reply = `${by}${body}`;
  return {
    uid, fingerprint: hash(title, reply, r.url, r.kind, r.collection, r.thread, String(timeOf(r.at) ?? ''), JSON.stringify(r.metadata ?? {})),
    engine: r.kind?.trim() || 'record', session: r.thread?.trim() || uid, turn, cwd: null, project: collectionKey(r.collection),
    title, ts: timeOf(r.at) ?? now, synthetic: false, typed: 'human', request: title, reply, said: reply, steps: [],
    files: r.url ? [r.url] : [], done: false,
    record: { id: r.id, ...(r.source ? { source: r.source } : {}), ...(r.author ? { author: r.author } : {}), ...(r.url ? { url: r.url } : {}),
      ...(r.metadata ? { metadata: r.metadata } : {}) },
  };
}

/**
 * A conversation's turns as records: each time the person speaks, what they said is the request and what the other
 * side answered until they speak again is its reply — one record a request, so a question finds the exchange it was in.
 * Turns before the first request (an assistant that opened) are the first record's own.
 */
export function conversationEntries(c: Conversation, now: number): StmEntry[] {
  const began = timeOf(c.at);
  const out: StmEntry[] = [];
  let request: string[] = [];
  let reply: string[] = [];
  let at: number | undefined;
  const flush = (): void => {
    if (!request.length && !reply.length) return;
    const n = out.length;
    // A time only the caller gave is part of the fingerprint: one defaulted from when it is added would make the same
    // conversation added again a changed one — read, embedded and consolidated again.
    const given = at ?? (began === undefined ? undefined : began + n * 60_000);
    const e = entryOf({ id: `${c.id}#${n}`, source: c.source, collection: c.collection, thread: c.id, kind: 'conversation',
      title: request.join('\n').split('\n').find((l) => l.trim())?.trim() || reply.join('\n').split('\n').find((l) => l.trim())?.trim() || c.id,
      text: request.join('\n'), at: given }, now, n);
    out.push({ ...e, ts: given ?? now + n * 60_000, request: request.join('\n') || e.title, reply: reply.join('\n'), said: reply.join('\n') });
    request = [];
    reply = [];
    at = undefined;
  };
  for (const t of c.turns) {
    const text = `${t.author?.trim() ? `${t.author.trim()}: ` : ''}${t.text.trim()}`;
    if (!t.text.trim()) continue;
    if (t.role === 'user') {
      if (reply.length) flush();
      request.push(text);
      at ??= timeOf(t.at);
    } else {
      reply.push(text);
      at ??= timeOf(t.at);
    }
  }
  flush();
  return out.map((e) => ({ ...e, fingerprint: hash(e.fingerprint, e.request, e.reply) }));
}
