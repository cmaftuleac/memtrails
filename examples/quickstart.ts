/** Add a few records of different kinds and ask about them. No language model, no server.
 *  npx tsx examples/quickstart.ts */
import { openMemory } from '../src/index.js';

const memory = await openMemory({ home: './example-memory', log: console.log });

await memory.add([
  { id: 'mail-1', kind: 'email', author: 'Ana', title: 'Proxy certificate expired', at: '2026-04-02',
    text: 'The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.' },
  { id: 'c9f1e2', kind: 'commit', author: 'Ion', title: 'Restore certificate renewal cron', at: '2026-04-03', text: 'Files: deploy/cron/renew.sh' },
  { id: 'note-7', kind: 'note', title: 'Router', at: '2026-04-04', text: 'Parola routerului este în caietul albastru din sertarul de la birou.' },
]);
await memory.addConversation({ id: 'chat-1', at: '2026-04-05', turns: [
  { role: 'user', text: 'Where do we have lunch on Friday with Ana?' },
  { role: 'assistant', text: 'At the small Italian place near the office, at one.' },
] });

console.log(await memory.searchText({ query: 'why did the certificate stop renewing?', limit: 3 }));
console.log(await memory.searchText({ query: 'where is the router password written down?', limit: 1 }));
console.log(await memory.room({ query: 'what made the renewal stop?', chars: 400 }));

await memory.close();
