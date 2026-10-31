/** With a language model: consolidate what was added into typed knowledge and dated events, sleep, then gather every
 *  instance for a listing question. Any OpenAI-compatible endpoint works (baseUrl).
 *  OPENAI_API_KEY=… npx tsx examples/consolidate.ts */
import { openMemory } from '../src/index.js';

let tokens = 0;
const memory = await openMemory({
  home: './example-memory',
  llm: { model: process.env.MEMTRAILS_MODEL ?? 'gpt-6-luna', reasoningEffort: 'low', onUsage: (u) => { tokens += u.inputTokens + u.outputTokens; } },
});

console.log(await memory.consolidate());
console.log(await memory.sleep());
const set = await memory.gather({ query: 'every decision we made about the gateway' });
console.log(set.kept.length, 'kept of', set.found.length);
console.log(await memory.searchText({ query: 'gateway', kinds: ['decision', 'observation', 'profile'], limit: 10 }));
console.log('tokens used:', tokens);
await memory.close();
