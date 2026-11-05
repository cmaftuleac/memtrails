/** Memory as an agent's tools: an MCP server over stdio for Claude Desktop, Claude Code, Cursor and others (the same as
 *  `memtrails mcp`), or JSON Schemas for any function-calling API.
 *  npx tsx examples/agent-tools.ts               # serve MCP on stdin/stdout
 *  npx tsx examples/agent-tools.ts --schemas     # print the tools for function calling */
import { openMemory, serveMcp } from '../src/index.js';

if (process.argv.includes('--schemas')) {
  // For function calling: name, description and JSON Schema; call tool.handler(args) with the model's arguments.
  const memory = await openMemory({ home: './example-memory' });
  const tools = memory.tools().map((t) => ({ name: t.name, description: t.description, parameters: t.jsonSchema }));
  console.log(JSON.stringify(tools, null, 2));
  await memory.close();
} else {
  // The memory opens at the first tool call and closes after five idle minutes.
  await serveMcp({ open: () => openMemory({ home: './example-memory' }), version: '0.0.0-example' });
}
