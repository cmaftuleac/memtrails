/**
 * memtrails: open a memory, add anything with text, ask it things.
 *
 *   const memory = await openMemory({ home: './memory' });
 *   await memory.add({ id: 'mail-1', kind: 'email', title: 'Gateway certificate', author: 'Ana',
 *                      text: 'It expired on the 1st; the renewal job had been removed.', at: '2026-04-02' });
 *   await memory.search({ query: 'why did the certificate stop renewing?' });
 *
 * What is added is searchable at once, with no language model and no server: the embedding model runs here (fetched
 * on first use) and orders results word by word (words.ts). A language model is needed only for `consolidate`,
 * `sleep` and `gather`, which turn records into typed knowledge, dated observations and profiles — and memory works
 * without them.
 *
 * The JSONL logs in `home` are the truth. The Qdrant index and the word vectors are derived: lost or out of date, they
 * are written again from the logs.
 */
import { join, resolve } from 'node:path';
import { chmodSync, closeSync, mkdirSync, openSync, readFileSync, renameSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { Consolidator, type ConsolidationReport } from './core/consolidate.js';
import { remoteEmbedder, resolveEmbedRoute, type EmbedRoute } from './core/embedder.js';
import { memoryGather, type Gathered } from './core/gather.js';
import { loadLocalEmbedder, type LocalEmbedder } from './core/local-embedder.js';
import { ensureModel } from './core/model.js';
import { ensureQdrant } from './core/qdrant-binary.js';
import { COLLECTION, INDEX_LAYOUT, QdrantClient } from './core/qdrant-client.js';
import { alive, bootTime, QdrantProcess, sameBoot, type QdrantEndpoint } from './core/qdrant-process.js';
import { memoryExplore, memoryOpen, memorySearch, type Explored, type RecallDeps, type RecalledItem, type SearchQuery } from './core/recall.js';
import { RecallBook } from './core/recall-log.js';
import { episodeBody } from './core/episodes.js';
import { MemoryRebuild } from './core/rebuild.js';
import { remoteReranker, resolveRerankRoute, type RerankRoute } from './core/reranker.js';
import { sentenceRoom, sentenceSpans, type RoomItem } from './core/room.js';
import { Sleep, type SleepReport } from './core/sleep.js';
import { ltmLog, stmLog, UsageBook, type StmEntry } from './core/store.js';
import { WordStore, packWords, wordReranker, type WordEncoder } from './core/words.js';
import { rawKey, stmItem, stmTitle, writeLtm, writeStm } from './core/writer.js';
import { defaultAsk, openAiAsk, type Ask, type OpenAiAskOptions } from './llm.js';
import { conversationEntries, entryOf, invalidRecord, recordUid, type Conversation, type MemoryRecord } from './records.js';
import { buildAgentTools, lastedText, readableDate, searchText, type AgentTool } from './tools/agent-tools.js';

export interface MemoryOptions {
  /** Where this memory lives: its logs, its index, its word vectors. */
  home: string;
  /** How texts are embedded (EmbeddingGemma-300m in every case: the index, prompts and measured thresholds are its):
   *  - default: on this machine, the model fetched on first use into the cache (218 MB);
   *  - `{ modelDir }`: on this machine, from a directory you provide;
   *  - `{ server }`: a server that serves it over the OpenAI embeddings API (text-embeddings-inference, Ollama, …).
   *    Word by word then needs `{ modelDir }` too, or it is off.
   *  With neither `server` nor `modelDir`, MEMTRAILS_EMBED_URL (+ _KEY, _MODEL) names a server when it is set. */
  embedding?: { modelDir?: string; server?: EmbedRoute; threads?: number };
  /** How the best 30 of a search are reordered: `'words'` (default with a local model: word by word, no server), a
   *  cross-encoder server (`RerankRoute`), your own function, or `false` for the search's own order. Not given:
   *  MEMTRAILS_RERANK_URL (+ _KEY, _MODEL) names a server when it is set. */
  rerank?: 'words' | RerankRoute | RecallDeps['rerank'] | false;
  /** The language model for consolidate, sleep and gather: your own `Ask`, or options for the OpenAI-compatible one.
   *  Default: the one the environment names (MEMTRAILS_MODEL and a key). */
  llm?: Ask | OpenAiAskOptions;
  /** Qdrant: default, a process of its own (the pinned binary, fetched on first use); or a server you run. */
  qdrant?: { url: string; apiKey?: string; collection?: string } | { binary: string };
  /** Windows the language model reads at once during consolidation (default 4). */
  concurrency?: number;
  log?: (line: string) => void;
}

export interface SearchOptions extends SearchQuery {
  /** `relevance` (default) or `time` (oldest first) — for searchText. */
  order?: 'relevance' | 'time';
}

export interface Memory {
  /** Add or update records. Unchanged ones cost nothing. Searchable when this resolves. */
  add(records: MemoryRecord | MemoryRecord[]): Promise<{ added: number; unchanged: number }>;
  /** Add or update a conversation: one record for each time the person speaks, with what was answered. */
  addConversation(conversation: Conversation): Promise<{ added: number; unchanged: number }>;
  /** Add records from any source (an importer, a database cursor) in batches, streaming. */
  import(records: AsyncIterable<MemoryRecord> | Iterable<MemoryRecord>, opts?: { batch?: number; onProgress?: (done: number) => void }): Promise<{ added: number; unchanged: number }>;
  /** Forget records: gone from search. What consolidation learned from them stays until sleep finds it unsupported. */
  remove(ids: Array<string | { id: string; source?: string }>): Promise<number>;
  /** Best first: records and what consolidation learned, each with its key text, dates, and the passage that matched. */
  search(q: SearchOptions): Promise<RecalledItem[]>;
  /** The same search as a reader is shown it: one line a memory, best first or oldest first. */
  searchText(q: SearchOptions): Promise<string>;
  /** What a small room should hold for a question: the best sentences of what search found, oldest first, in `chars`. */
  room(q: SearchQuery & { chars?: number }): Promise<string>;
  /** One item whole, by the id search returned. */
  open(id: string): Promise<Record<string, unknown> | null>;
  /** Around a topic, or around items you found: what the concept graph reaches that search did not, and its concepts. */
  explore(q: { topic?: string; from?: string[]; collections?: string[]; limit?: number }): Promise<Explored>;
  /** A set for a question that counts, lists or totals: a wide search, then the language model keeps every instance. */
  gather(q: SearchQuery, o?: { notes?: boolean }): Promise<Gathered>;
  /** Read what was added since last time and keep typed knowledge, dated observations and kept episodes. Needs a
   *  language model. Returns how many windows were read. */
  consolidate(): Promise<ConsolidationReport>;
  /** The offline pass: merge near-duplicates, retire what was superseded, link, write profiles. Needs a language model. */
  sleep(): Promise<SleepReport>;
  /** Encode what word by word and the room do not hold yet (done after every add, consolidate and sleep). */
  indexWords(): Promise<number>;
  /** Write the index again from the logs. */
  rebuild(): Promise<void>;
  /** Tool definitions an agent can be handed: memory_search, memory_open, memory_explore. */
  tools(): AgentTool[];
  stats(): { records: number; items: number; words: number };
  /** Stops a consolidate, sleep, rebuild or indexWords at its next step (a rebuild resumes at the next open) and waits
   *  for it and for the adds and searches in flight, then releases Qdrant, the model and the home. Every call after it
   *  is refused ("memory is closed"). */
  close(): Promise<void>;
}

const BATCH = 32;
const ROOM_TEXT = 1500;
const ROOM_CHARS = 6000;
const RECORD_LIMIT = 2_000_000;

/** In the home: who has it open (lockHome), and the index layout its Qdrant collection was written in. */
const LOCK_FILE = 'memory.lock';
const LAYOUT_FILE = 'index.json';

/** The homes this process holds, released if it exits without close(). */
const holding = new Map<string, () => void>();
let exitHook = false;

/** Who holds a lock: its live pid, 0 when it is being written this instant, null when nobody (stale or gone). */
function lockHolder(path: string): { text: string; pid: number | null } {
  let text: string;
  try { text = readFileSync(path, 'utf8'); } catch { return { text: '', pid: null }; }
  try {
    const l = JSON.parse(text) as { pid?: unknown; boot?: unknown };
    const pid = Number(l.pid);
    return { text, pid: sameBoot(l.boot) && alive(pid) ? pid : null };
  } catch {
    // Not written yet by the opener that just made it — or torn by a crash, and then old.
    try { return { text, pid: Date.now() - statSync(path).mtimeMs < 10_000 ? 0 : null }; } catch { return { text, pid: null }; }
  }
}

/** One process at a time in a home: two would run two Qdrants on one storage and append to the logs at offsets the
 *  other does not know. The lock is made exclusively (O_EXCL) with its pid and the boot; one whose process is gone, or
 *  ran before a reboot (its pid may be another's now), is taken over. Returns its release. */
function lockHome(dir: string): () => void {
  const path = join(dir, LOCK_FILE);
  for (let attempt = 0; ; attempt++) {
    let fd: number;
    try {
      fd = openSync(path, 'wx', 0o600);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err;
      const { text, pid } = lockHolder(path);
      if (pid !== null || attempt > 0) {
        throw new Error(`memory ${dir} is open${pid ? ` in process ${pid}` : ''}: close it there first (if no process has it, delete ${path})`);
      }
      // Stale: taken over — unless another opener took it over since it was read.
      try { if (readFileSync(path, 'utf8') === text) rmSync(path, { force: true }); } catch { /* gone already */ }
      continue;
    }
    try { writeSync(fd, JSON.stringify({ pid: process.pid, boot: bootTime() })); } finally { closeSync(fd); }
    break;
  }
  const release = (): void => {
    if (holding.get(path) !== release) return;
    holding.delete(path);
    try { if ((JSON.parse(readFileSync(path, 'utf8')) as { pid?: number }).pid === process.pid) rmSync(path, { force: true }); } catch { /* gone */ }
  };
  holding.set(path, release);
  if (!exitHook) {
    exitHook = true;
    process.on('exit', () => { for (const r of [...holding.values()]) r(); });
  }
  return release;
}

/** The layout the index was written in (qdrant-client.ts INDEX_LAYOUT); null when not written down. */
function indexLayout(dir: string): number | null {
  try { return Number((JSON.parse(readFileSync(join(dir, LAYOUT_FILE), 'utf8')) as { layout?: unknown }).layout); } catch { return null; }
}

function saveIndexLayout(dir: string): void {
  const path = join(dir, LAYOUT_FILE);
  writeFileSync(`${path}.tmp`, JSON.stringify({ layout: INDEX_LAYOUT }), { mode: 0o600 });
  renameSync(`${path}.tmp`, path);
}

/** Turns shared, and a turn alone: many share at once; one alone waits until no one shares, and keeps new sharers out
 *  while it waits, so a stream of them never holds it off. */
function turns(): { shared<T>(job: () => Promise<T>): Promise<T>; alone<T>(job: () => Promise<T>): Promise<T> } {
  let sharing = 0;
  let alone = false;
  let waiting = 0;
  let wakers: Array<() => void> = [];
  const wake = (): void => { const w = wakers; wakers = []; for (const f of w) f(); };
  const until = async (ok: () => boolean): Promise<void> => { while (!ok()) await new Promise<void>((r) => wakers.push(r)); };
  return {
    async shared(job) {
      await until(() => !alone && !waiting);
      sharing++;
      try { return await job(); } finally { sharing--; wake(); }
    },
    async alone(job) {
      waiting++;
      await until(() => !alone && !sharing);
      waiting--;
      alone = true;
      try { return await job(); } finally { alone = false; wake(); }
    },
  };
}

/** What an open has started, for an open that fails half-way to release: a Qdrant left running keeps the host alive. */
interface Started { proc: QdrantProcess | null; local: LocalEmbedder | null }

export async function openMemory(o: MemoryOptions): Promise<Memory> {
  const dir = resolve(o.home);
  const log = o.log ?? ((): void => {});
  // Private: the logs, the index and the word vectors are everything that was added (a shared machine).
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  try { chmodSync(dir, 0o700); } catch (err) { log(`[memtrails] ${dir} could not be made private: ${(err as Error).message}`); }
  const release = lockHome(dir);
  const started: Started = { proc: null, local: null };
  try {
    return await openIn(dir, o, log, started, release);
  } catch (err) {
    await started.proc?.stopAndWait();
    await started.local?.close().catch(() => undefined);
    release();
    throw err;
  }
}

async function openIn(dir: string, o: MemoryOptions, log: (line: string) => void, started: Started, release: () => void): Promise<Memory> {
  // Embedding: here, or on a server — the options' own, else the one the environment names.
  const e = o.embedding ?? {};
  const server = e.server ?? (e.modelDir ? null : resolveEmbedRoute());
  const local: LocalEmbedder | null = !server || e.modelDir
    ? await loadLocalEmbedder(e.modelDir ?? await ensureModel({ log }), { threads: e.threads ?? 2 })
    : null;
  started.local = local;
  const embed = server ? remoteEmbedder({ route: () => server }) : local!.embed;
  const encode: WordEncoder | null = local?.words ?? null;
  const words = new WordStore(join(dir, 'words.bin'));
  await words.load();

  // The index: a Qdrant of its own, or yours.
  const q = o.qdrant;
  let proc: QdrantProcess | null = null;
  let endpoint: () => QdrantEndpoint | null;
  if (q && 'url' in q) {
    const fixed = { url: q.url.replace(/\/+$/, ''), apiKey: q.apiKey ?? '' };
    endpoint = () => fixed;
  } else {
    proc = new QdrantProcess({ binary: q && 'binary' in q ? q.binary : await ensureQdrant({ log }), dir: join(dir, 'qdrant'), log });
    started.proc = proc;
    await proc.start();
    endpoint = () => proc!.current();
  }
  const client = new QdrantClient(endpoint, 10_000, { collection: q && 'url' in q ? q.collection ?? COLLECTION : COLLECTION });
  await client.ensureCollection();

  const stm = stmLog(join(dir, 'records.jsonl'));
  const ltm = ltmLog(join(dir, 'knowledge.jsonl'));
  await stm.load();
  await ltm.load();
  const usage = new UsageBook(join(dir, 'usage.json'));
  const book = new RecallBook(join(dir, 'recall.jsonl'));
  void book.trim(); // the documented cap: once an open, and every so many events while open (recall-log.ts)
  const ask: Ask = typeof o.llm === 'function' ? o.llm : o.llm ? openAiAsk(o.llm) : defaultAsk;
  /** Set by close(): what runs stops at its next step. */
  let closing = false;
  const aborted = (): boolean => closing;
  const base = { client, embed, stm, ltm, log, aborted, ask };

  const rerankEnv = o.rerank === undefined ? resolveRerankRoute() : null;
  const rerank: RecallDeps['rerank'] = o.rerank === false ? undefined
    : typeof o.rerank === 'function' ? o.rerank
    : o.rerank && typeof o.rerank === 'object' ? remoteReranker({ route: () => o.rerank as RerankRoute })
    : rerankEnv ? remoteReranker({ route: () => rerankEnv })
    : encode ? wordReranker(encode, words) : undefined;
  const deps: RecallDeps = { client, embed, rerank, used: (ids, what) => usage.note(ids, what), record: (ev) => book.record(ev) };

  // The index is derived: written again from the logs when it is empty while they are not (a new Qdrant, a deleted
  // folder), or was written in another layout — or in one not written down (an index from before the layouts were).
  const rebuild = new MemoryRebuild({
    path: join(dir, 'rebuild.json'), stm, ltm, aborted,
    writeStm: (xs) => writeStm(base, xs), writeLtm: (xs) => writeLtm(base, xs),
  });
  const indexed = await client.count();
  const layout = indexLayout(dir);
  if (layout !== INDEX_LAYOUT) {
    if (indexed) {
      log(`[memtrails] the index was written in layout ${layout ?? 'unknown'}, not ${INDEX_LAYOUT}: writing it again from the logs`);
      rebuild.begin(); // before the drop: a crash after it resumes
      await client.dropCollection();
      await client.ensureCollection();
    }
    saveIndexLayout(dir);
  }
  if (rebuild.pending() || (!indexed && (stm.size() || ltm.size()))) {
    log('[memtrails] writing the index from the logs');
    if (!rebuild.pending()) rebuild.begin();
    await rebuild.run();
  }

  // consolidate, sleep, rebuild, indexWords and remove one at a time: each reads the logs and writes the index from
  // them — two consolidations pay the model twice and write stale copies over each other, a rebuild clears what
  // another is writing, a remove during a consolidation would see its record's episode written back after it.
  let queue: Promise<unknown> = Promise.resolve();
  const exclusive = <T>(job: () => Promise<T>): Promise<T> => {
    const run = queue.then(() => {
      if (closing) throw new Error('memory is closed');
      return job();
    });
    queue = run.catch(() => undefined);
    return run;
  };
  let closed: Promise<void> | null = null;
  /** Every call (add, search, open, …) shares a turn; close() takes it alone, so it lets go of Qdrant, the model and the
   *  home only once the calls in flight are done. */
  const calls = turns();
  /** Adds share a turn; a rebuild takes it alone. An add beside a rebuild could write its points into the index about to
   *  be dropped and log its record after the rebuild read the log: logged, never indexed, and skipped as unchanged when
   *  added again. */
  const adding = turns();
  const check = (): void => { if (closing) throw new Error('memory is closed'); };
  /** A call's shared turn; refused once close() began, also when it waited for its turn meanwhile. */
  const call = async <T>(job: () => Promise<T>): Promise<T> => {
    check();
    return calls.shared(async () => { check(); return job(); });
  };

  /** The text word by word scores an item by: its key, as search returns it. */
  const keyText = (title: string, summary: string): string => `${title}\n${summary}`;

  /** A found item's whole text for the room: a record's own, or what consolidation wrote. */
  const roomItem = (h: RecalledItem): RoomItem => {
    const r = h.id.startsWith('stm:') ? stm.get(h.id.slice(4)) : null;
    const it = r ? null : ltm.get(h.id);
    const text = r ? `${r.request}\n${r.reply}` : it ? `${it.title}\n${it.summary}\n${it.body}` : `${h.title}\n${h.summary}`;
    const told = h.ts ? new Date(h.ts).toISOString().slice(0, 10) : '';
    const lasted = lastedText(h);
    const happened = [h.eventAt ? readableDate(h.eventAt) : '', lasted].filter(Boolean).join(', ');
    const when = happened ? `${happened}${h.eventAt !== told || lasted ? `; told ${readableDate(told)}` : ''}` : readableDate(told);
    return { id: h.id, head: `(${when} · ${h.kind})`, text: text.slice(0, ROOM_TEXT), ts: h.eventAt ? Date.parse(h.eventAt) : h.ts };
  };

  const encodeMissing = async (texts: string[]): Promise<number> => {
    if (!encode) return 0;
    const todo = [...new Set(texts)].filter((t) => t.trim() && !words.get(t));
    for (let i = 0; i < todo.length && !closing; i += BATCH) {
      const part = todo.slice(i, i + BATCH);
      const encoded = await encode(part, 'document');
      await words.put(part.map((t, n) => [t, packWords(encoded[n]!)]));
    }
    return todo.length;
  };
  const wordsOfEntries = (es: StmEntry[]): string[] => es.flatMap((x) => {
    const room = `${x.request}\n${x.reply}`.slice(0, ROOM_TEXT);
    return [keyText(stmTitle(x), rawKey(x)), ...sentenceSpans(room).map(([a, b]) => room.slice(a, b))];
  });

  const write = (entries: StmEntry[], total: number): Promise<{ added: number; unchanged: number }> => call(() => adding.shared(async () => {
    check(); // it may have waited for a rebuild that a close stopped
    const fresh = entries.filter((x) => stm.meta(x.uid)?.fp !== x.fingerprint);
    for (let i = 0; i < fresh.length; i += BATCH) {
      const part = fresh.slice(i, i + BATCH);
      await writeStm(base, part);
      stm.put(part);
      await encodeMissing(wordsOfEntries(part));
    }
    stm.compact();
    return { added: fresh.length, unchanged: total - fresh.length };
  }));

  /** Unlocked: consolidate and sleep run it inside their own turn. */
  const indexWords = async (): Promise<number> => {
    if (!encode) return 0;
    const texts = [...stm.all()].flatMap((x) => wordsOfEntries([x]));
    for (const [uid, m] of ltm.entries()) {
      if (m.status !== 'active') continue;
      const it = ltm.get(uid);
      if (!it) continue;
      const room = `${it.title}\n${it.summary}\n${it.body}`.slice(0, ROOM_TEXT);
      texts.push(keyText(it.title, it.summary), ...sentenceSpans(room).map(([a, b]) => room.slice(a, b)));
    }
    return encodeMissing(texts);
  };

  const memory: Memory = {
    async add(records) {
      const list = Array.isArray(records) ? records : [records];
      const now = Date.now();
      for (const r of list) {
        const bad = invalidRecord(r);
        if (bad) throw new TypeError(bad);
        if (r.text.length > RECORD_LIMIT) throw new TypeError(`record ${r.id}: longer than ${RECORD_LIMIT} characters; split it`);
      }
      return write(list.map((r) => entryOf(r, now)), list.length);
    },
    async addConversation(c) {
      if (!c?.id || !Array.isArray(c.turns)) throw new TypeError('a conversation needs an id and turns');
      const entries = conversationEntries(c, Date.now());
      return write(entries, entries.length);
    },
    async import(records, opts = {}) {
      const size = opts.batch ?? 256;
      let batch: MemoryRecord[] = [];
      const total = { added: 0, unchanged: 0 };
      const flush = async (): Promise<void> => {
        if (!batch.length) return;
        const r = await memory.add(batch);
        total.added += r.added;
        total.unchanged += r.unchanged;
        opts.onProgress?.(total.added + total.unchanged);
        batch = [];
      };
      for await (const r of records as AsyncIterable<MemoryRecord>) {
        batch.push(r);
        if (batch.length >= size) await flush();
      }
      await flush();
      return total;
    },
    remove: (ids) => exclusive(async () => {
      const uids = ids.map((x) => recordUid(typeof x === 'string' ? { id: x } : x)).filter((u) => stm.meta(u));
      if (!uids.length) return 0;
      const episodes = uids.map((u) => `ep:${u}`);
      await client.deleteItems([...uids.map(stmItem), ...episodes]);
      stm.remove(uids);
      // A kept episode is the record's own text: gone from the long-term log too, or a rebuild indexes it again and the
      // profiles go on reading it.
      ltm.remove(episodes);
      await Promise.all([stm.flush(), ltm.flush()]);
      return uids.length;
    }),
    search: (s) => call(() => memorySearch(deps, s)),
    searchText: (s) => call(async () => searchText(await memorySearch(deps, s), s.order ?? 'relevance')),
    async room(s) {
      check();
      if (!encode) throw new Error('the room needs the local embedding model (embedding.modelDir, or no embedding.server)');
      return call(async () => {
        const found = await memorySearch(deps, { ...s, limit: s.limit ?? 20 });
        return sentenceRoom(s.query, found.map(roomItem), encode, words, s.chars ?? ROOM_CHARS);
      });
    },
    async open(id) {
      const p = await call(() => memoryOpen(deps, id));
      if (!p) return null;
      // A record: its kind as given. Something learned: the records it was learned from, whole.
      if (p.tier === 'stm') return { ...p, kind: p.engine };
      const it = ltm.get(id);
      const sources = (it?.provenance ?? []).map((u) => stm.get(u)).filter((x): x is StmEntry => !!x)
        .map((x) => ({ id: stmItem(x.uid), when: x.ts, title: x.title, body: episodeBody(x), ...(x.record ? { record: x.record } : {}) }));
      return sources.length ? { ...p, sources } : p;
    },
    explore: (s) => call(() => memoryExplore(deps, s)),
    gather: (s, g) => call(() => memoryGather(deps, s, { ask, ...(s.limit ? { limit: s.limit } : {}), ...(g?.notes ? { notes: true } : {}) })),
    consolidate: () => exclusive(async () => {
      const worker = new Consolidator({ ...base, concurrency: o.concurrency ?? 4 });
      const total: ConsolidationReport = { windows: 0, added: 0, merged: 0, updated: 0, superseded: 0, linked: 0, dropped: 0, episodes: 0, observed: 0, cued: 0 };
      for (let round = 0; round < 1000 && !closing && worker.pending().length; round++) {
        const r = await worker.run(64);
        for (const k of Object.keys(total) as Array<keyof ConsolidationReport>) if (typeof r[k] === 'number') (total[k] as number) += r[k] as number;
        if (r.paused !== undefined) { total.paused = r.paused; break; }
        if (!r.windows) break;
      }
      await indexWords();
      return total;
    }),
    sleep: () => exclusive(async () => {
      const r = await new Sleep({ ...base, usage, statePath: join(dir, 'sleep.json') }).run();
      await indexWords();
      return r;
    }),
    indexWords: () => exclusive(indexWords),
    rebuild: () => exclusive(() => adding.alone(async () => { // the adds in flight end first; the next ones wait for it
      rebuild.begin(); // before the drop: a crash, or a close, after it resumes at the next open
      await client.dropCollection();
      await client.ensureCollection();
      await rebuild.run();
    })),
    tools: () => buildAgentTools({
      search: (s) => memory.search(s),
      open: (id) => memory.open(id),
      explore: (s) => memory.explore(s),
    }),
    stats: () => ({ records: stm.size(), items: [...ltm.entries()].filter(([, m]) => m.status === 'active').length, words: words.size }),
    close() {
      closed ??= (async () => {
        closing = true;
        await queue; // the job running stops at its next step; those waiting are refused
        // The adds and searches in flight end first; those waiting for their turn, and every call after, are refused.
        await calls.alone(async () => {
          try {
            await book.settled();
            usage.flush();
            await proc?.stopAndWait();
            await local?.close();
          } finally {
            release();
          }
        });
      })();
      return closed;
    },
  };
  return memory;
}
