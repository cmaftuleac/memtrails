/** Memory as an agent's tools: an MCP server over stdio (needs @modelcontextprotocol/sdk), or JSON Schemas for any
 *  function-calling API.
 *  npm install @modelcontextprotocol/sdk && npx tsx examples/agent-tools.ts */
import { openMemory } from '../src/index.js';

const memory = await openMemory({ home: './example-memory' });
const tools = memory.tools();

if (process.argv.includes('--schemas')) {
  // For function calling: name, description and JSON Schema; call tool.handler(args) with the model's arguments.
  console.log(JSON.stringify(tools.map((t) => ({ name: t.name, description: t.description, parameters: t.jsonSchema })), null, 2));
  await memory.close();
} else {
  const sdk = '@modelcontextprotocol/sdk';
  const { McpServer } = await import(`${sdk}/server/mcp.js`);
  const { StdioServerTransport } = await import(`${sdk}/server/stdio.js`);
  const server = new McpServer({ name: 'memtrails', version: '0.1.0' });
  for (const t of tools) server.tool(t.name, t.description, t.inputSchema, t.handler);
  await server.connect(new StdioServerTransport());
}
