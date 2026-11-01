/**
 * How a record reads: to the index (its body, every word of which is searched in chunks) and to the consolidation model.
 *
 * A conversation's exchange keeps the format consolidation was measured on: what was asked, the steps, what was said.
 * Any other record reads as itself: its title, then its text.
 */

/** What the index and the consolidation model read of a record (store.ts StmEntry). */
export interface ReadableRequest {
  uid: string;
  /** The record's kind; `conversation` for an exchange of a conversation. */
  engine?: string;
  typed: 'human' | 'unknown';
  request: string;
  reply: string;
  /** Everything the other side said in an exchange, whole. */
  said: string;
  steps: string[];
  files: string[];
  ts: number | null;
}

export const isConversation = (e: { engine?: string }): boolean => e.engine === 'conversation';

/** A record's body: what its key summarises, its chunks are cut from, and opening it shows — whole, never shortened. */
export function episodeBody(e: Pick<ReadableRequest, 'request' | 'reply' | 'files'> & Partial<Pick<ReadableRequest, 'said' | 'steps' | 'engine'>>): string {
  if (e.engine !== undefined && !isConversation(e)) return e.reply.trim() ? `${e.request}\n${e.reply}` : e.request;
  const parts = [`Asked: ${e.request}`];
  if (e.steps?.length) parts.push(`Steps:\n${e.steps.join('\n')}`);
  parts.push(`Done: ${e.said?.trim() ? e.said : e.reply}`);
  if (e.files.length) parts.push(`Files: ${e.files.join(', ')}`);
  return parts.join('\n');
}

/** For each day a request was made, the days a time said relatively that day means — yesterday, the week before, the
 *  last weekend, each of the six days before by name, this week, last month, last year (UTC as the requests' times) —
 *  so the model reads "last week" or "on Friday" off a line instead of counting. "Last week" is the seven days before,
 *  as people mean it, not the Monday-to-Sunday week before this one: measured on LoCoMo, the calendar week put events
 *  said on a Sunday thirteen days back, a week early. */
export function calendarOf(tss: number[]): string {
  const DAY = 86_400_000;
  const iso = (t: number): string => new Date(t).toISOString().slice(0, 10);
  const wd = (t: number): string => new Date(t).toLocaleDateString('en-GB', { weekday: 'short', timeZone: 'UTC' });
  const days = [...new Set(tss.filter((t) => t > 0).map((t) => Math.floor(t / DAY) * DAY))].sort((a, b) => a - b);
  return days.map((d) => {
    const dt = new Date(d);
    const monday = d - ((dt.getUTCDay() + 6) % 7) * DAY;
    const sunday = d - (dt.getUTCDay() === 0 ? 7 : dt.getUTCDay()) * DAY;
    const lastMonth = new Date(Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth() - 1, 1)).toISOString().slice(0, 7);
    const before = [2, 3, 4, 5, 6, 7].map((n) => `${wd(d - n * DAY)} ${iso(d - n * DAY)}`).join(', ');
    return `- ${wd(d)} ${iso(d)}: yesterday ${wd(d - DAY)} ${iso(d - DAY)} · the days before ${before} · last week (the week before) ${iso(d - 7 * DAY)} to ${iso(d - DAY)} · last weekend ${iso(sunday - DAY)} to ${iso(sunday)} · this week ${iso(monday)} to ${iso(monday + 6 * DAY)} · last month ${lastMonth} · last year ${dt.getUTCFullYear() - 1}`;
  }).join('\n');
}

export function renderForExtraction(e: ReadableRequest, n: number): string {
  const when = e.ts ? new Date(e.ts).toISOString().slice(0, 16).replace('T', ' ') : '';
  // Whole: a record too long for one window is read in parts and its parts joined (extract.ts), never shortened.
  if (e.engine !== undefined && !isConversation(e)) return [`## Record ${n} · ${when} · ${e.engine}`, e.request, e.reply].filter(Boolean).join('\n');
  const who = e.typed === 'human' ? 'typed by the person' : 'sent to the agent';
  const parts = [`## Request ${n} · ${when} · ${who}`, e.request];
  if (e.steps.length) parts.push(`Steps:\n${e.steps.join('\n')}`);
  if (e.said) parts.push(`Agent said:\n${e.said}`);
  if (e.files.length) parts.push(`Files changed: ${e.files.join(', ')}`);
  return parts.join('\n');
}
