"""With a language model: consolidate what was added into typed knowledge and dated events, sleep, then gather every
instance for a listing question. Any OpenAI-compatible endpoint works (``base_url``).

    OPENAI_API_KEY=… python examples/consolidate.py
"""

import asyncio
import os

from memtrails import LlmUsage, open_memory


async def main() -> None:
    tokens = 0

    def count(u: LlmUsage) -> None:
        nonlocal tokens
        tokens += u.input_tokens + u.output_tokens

    llm = {"model": os.environ.get("MEMTRAILS_MODEL", "gpt-6-luna"), "reasoning_effort": "low", "on_usage": count}
    async with await open_memory(home="./example-memory", llm=llm) as memory:
        print(await memory.consolidate())
        print(await memory.sleep())
        found = await memory.gather({"query": "every decision we made about the proxy"})
        print(len(found["kept"]), "kept of", len(found["found"]))
        print(await memory.search_text({"query": "proxy", "kinds": ["decision", "observation", "profile"], "limit": 10}))
        print("tokens used:", tokens)


asyncio.run(main())
