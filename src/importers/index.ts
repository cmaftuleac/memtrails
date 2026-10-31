/**
 * Importers: records from what people already have — JSON Lines, a folder of text files, a Git history, a Slack
 * export, an mbox or `.eml` mailbox. Each one streams: records come one at a time as the source is read, never the
 * whole source in memory, so `for await (const batch of batches(fromMbox(path), 200)) await memory.add(batch)` works
 * on a mailbox of any size.
 */
export { fromJsonl, type JsonlMap, type JsonlOptions } from './jsonl.js';
export { fromFiles, frontMatter, TEXT_EXTENSIONS, type FilesOptions } from './files.js';
export { fromGit, type GitOptions } from './git.js';
export { fromSlackExport, slackText, SLACK_SKIPPED_SUBTYPES, type SlackOptions } from './slack.js';
export { fromMbox, fromEml, decodeEncodedWords, type MboxOptions } from './mbox.js';
export { htmlToText, decodeEntities, decodeBytes, splitText } from './text.js';

/** Items in arrays of `size` (the last may be shorter), as they come. */
export async function* batches<T>(it: AsyncIterable<T> | Iterable<T>, size: number): AsyncIterable<T[]> {
  const n = Math.max(1, Math.floor(size));
  let batch: T[] = [];
  for await (const item of it) {
    batch.push(item);
    if (batch.length >= n) {
      yield batch;
      batch = [];
    }
  }
  if (batch.length) yield batch;
}
