/** Import a mailbox, a Slack export, a repository's history and a folder of notes, then search across all of them.
 *  npx tsx examples/import-sources.ts ./archive.mbox ./slack-export ./my-repo ./notes */
import { openMemory } from '../src/index.js';
import { fromFiles, fromGit, fromMbox, fromSlackExport } from '../src/importers/index.js';

const [mbox, slack, repo, notes] = process.argv.slice(2);
const memory = await openMemory({ home: './example-memory' });
const progress = (what: string) => (n: number) => process.stdout.write(`\r${what}: ${n}`);

if (mbox) console.log('\n', await memory.import(fromMbox(mbox, { collection: 'mail' }), { onProgress: progress('mail') }));
if (slack) console.log('\n', await memory.import(fromSlackExport(slack), { onProgress: progress('slack') }));
if (repo) console.log('\n', await memory.import(fromGit(repo, { maxCount: 5000 }), { onProgress: progress('commits') }));
if (notes) console.log('\n', await memory.import(fromFiles(notes, { collection: 'notes' }), { onProgress: progress('notes') }));

console.log(memory.stats());
console.log(await memory.searchText({ query: process.env.QUERY ?? 'what did we decide about the release?', limit: 10 }));
await memory.close();
