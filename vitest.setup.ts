// Every test run gets its own memtrails cache, so nothing on this machine is read or left behind (and a real Qdrant or
// model already downloaded here does not change what a test sees).
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

process.env.MEMTRAILS_CACHE = mkdtempSync(join(tmpdir(), 'memtrails-test-'));
