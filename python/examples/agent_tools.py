"""Memory as an agent's tools: an MCP server over stdio for Claude Desktop, Claude Code, Cursor and others (the same as
``memtrails mcp``), or JSON Schemas for any function-calling API.

    python examples/agent_tools.py             # serve MCP on stdin/stdout
    python examples/agent_tools.py --schemas   # print the tools for function calling
"""

import asyncio
import json
import sys

from memtrails import __version__, open_memory
from memtrails.mcp import serve_mcp


async def main() -> None:
    if "--schemas" in sys.argv:
        # For function calling: name, description and JSON Schema; call `await tool.handler(args)` with the model's
        # arguments.
        async with await open_memory(home="./example-memory") as memory:
            tools = [{"name": t.name, "description": t.description, "parameters": t.json_schema} for t in memory.tools()]
            print(json.dumps(tools, indent=2, ensure_ascii=False))
    else:
        # The memory opens at the first tool call and closes after five idle minutes.
        await serve_mcp(lambda: open_memory(home="./example-memory"), version=__version__)


asyncio.run(main())
