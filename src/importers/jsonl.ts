/**
 * JSON Lines: one JSON object per line, read a line at a time. With no `map`, each line is a record as it is
 * (`{ "id": "…", "text": "…", "title": …, "at": … }`); with one, `map(obj)` turns each object into a record, or into
 * null to leave it out. Lines that are not JSON or not a record are skipped, counted and reported through `onSkip`; so is a
 * line longer than 64 MiB, which is never held whole.
 */
import { invalidRecord, type MemoryRecord } from '../records.js';
import { byteLines, LineTooLong } from './lines.js';
import { record } from './text.js';

/** Turns one parsed line into a record, or null to leave it out. */
export type JsonlMap = (obj: unknown, lineNumber: number) => MemoryRecord | null | undefined;

export interface JsonlOptions {
  map?: JsonlMap;
  /** Called for each line left out, with why. */
  onSkip?: (line: string, reason: string, lineNumber: number) => void;
  /** Filled in as the file is read: lines read, records given, lines skipped. */
  stats?: { lines: number; records: number; skipped: number };
  /** For records that have none. */
  source?: string;
  collection?: string;
  kind?: string;
}

/** Records from a JSON Lines file; `map` may be given here or in the options. */
export async function* fromJsonl(path: string, map?: JsonlMap | JsonlOptions, opts: JsonlOptions = {}): AsyncIterable<MemoryRecord> {
  const o: JsonlOptions = typeof map === 'function' ? { ...opts, map } : { ...map, ...opts };
  const stats = o.stats ?? { lines: 0, records: 0, skipped: 0 };
  const skip = (line: string, reason: string, n: number): void => {
    stats.skipped++;
    o.onSkip?.(line, reason, n);
  };
  let n = 0;
  for await (const bytes of byteLines(path)) {
    n++;
    if (bytes instanceof LineTooLong) {
      stats.lines++;
      skip('', bytes.reason, n);
      continue;
    }
    const raw = bytes.toString('utf8');
    const line = n === 1 ? raw.replace(/^﻿/, '') : raw;
    if (!line.trim()) continue;
    stats.lines++;
    let obj: unknown;
    try {
      obj = JSON.parse(line);
    } catch (e) {
      skip(line, `not JSON: ${(e as Error).message}`, n);
      continue;
    }
    let r: MemoryRecord | null | undefined;
    if (o.map) {
      try {
        r = o.map(obj, n);
      } catch (e) {
        skip(line, `map failed: ${(e as Error).message}`, n);
        continue;
      }
      if (!r) {
        skip(line, 'left out by map', n);
        continue;
      }
    } else {
      if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
        skip(line, 'not an object', n);
        continue;
      }
      const shaped = obj as MemoryRecord;
      r = typeof shaped.id === 'number' ? { ...shaped, id: String(shaped.id) } : shaped;
    }
    const filled = record({ ...r, source: r.source ?? o.source, collection: r.collection ?? o.collection, kind: r.kind ?? o.kind });
    const wrong = invalidRecord(filled);
    if (wrong) {
      skip(line, wrong, n);
      continue;
    }
    stats.records++;
    yield filled;
  }
}
