/** Where memtrails keeps what it downloads (the Qdrant binary, the embedding model): $MEMTRAILS_CACHE, else
 *  $XDG_CACHE_HOME/memtrails, else ~/.cache/memtrails. Shared by every memory on the machine; safe to delete. */
import { homedir } from 'node:os';
import { join } from 'node:path';

export function cacheDir(): string {
  return process.env.MEMTRAILS_CACHE?.trim() || join(process.env.XDG_CACHE_HOME?.trim() || join(homedir(), '.cache'), 'memtrails');
}
