/**
 * A file read a line at a time, as bytes, with a bound on a line: Node's `readline` has none but V8's longest string
 * (and fails there), so one hostile line held the whole file in memory.
 */
import { createReadStream } from 'node:fs';

/** The longest line read: one past it is passed over, with a reason. */
export const MAX_LINE_BYTES = 64 * 1024 * 1024;

/** A line longer than the bound, in its place: its bytes are passed over, never held. */
export class LineTooLong {
  constructor(readonly reason: string) {}
}

/**
 * A file's lines as `readline` gives them — ended by `\n`, `\r\n` or a lone `\r`, without their ends, and no empty line
 * after the last end — as bytes, each decoded by the caller. One longer than `max` bytes comes as a `LineTooLong`, and
 * the lines after it follow.
 */
export async function* byteLines(path: string, max = MAX_LINE_BYTES): AsyncGenerator<Buffer | LineTooLong> {
  let parts: Buffer[] = [];
  let held = 0;
  let read = 0;
  let long = false;
  let afterCr = false;
  const line = (): Buffer | LineTooLong => {
    const out = long ? new LineTooLong(`a line longer than ${max} bytes (${read})`) : Buffer.concat(parts, held);
    parts = [];
    held = 0;
    read = 0;
    long = false;
    return out;
  };
  for await (const chunk of createReadStream(path) as AsyncIterable<Buffer>) {
    let at = 0;
    while (at < chunk.length) {
      // A `\n` right after a `\r` is the same end.
      if (afterCr) {
        afterCr = false;
        if (chunk[at] === 0x0a) {
          at++;
          continue;
        }
      }
      const lf = chunk.indexOf(0x0a, at);
      const cr = chunk.indexOf(0x0d, at);
      const end = lf < 0 ? cr : cr < 0 ? lf : Math.min(lf, cr);
      const piece = chunk.subarray(at, end < 0 ? chunk.length : end);
      if (!long) {
        if (held + piece.length > max) {
          long = true;
          parts = [];
          held = 0;
        } else if (piece.length) {
          parts.push(Buffer.from(piece)); // a copy: the stream's chunk is not this line's to keep
          held += piece.length;
        }
      }
      read += piece.length;
      if (end < 0) break;
      afterCr = chunk[end] === 0x0d;
      at = end + 1;
      yield line();
    }
  }
  if (read > 0) yield line();
}
