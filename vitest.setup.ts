// Every test run gets its own memtrails home, so nothing on this machine is read or left behind.
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

process.env.MEMTRAILS_HOME = mkdtempSync(join(tmpdir(), 'memtrails-test-'));
