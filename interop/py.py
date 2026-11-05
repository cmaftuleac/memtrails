"""The Python side of the interop test (run.mjs runs it): write a memory, add one record to it, or read it.

python interop/py.py write <home> <corpus.json>
python interop/py.py add   <home> <corpus.json>
python interop/py.py read  <home> <queries.json>   -> JSON on stdout
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from memtrails import open_memory


def hit(h: Any) -> dict[str, Any]:
    record = h.get("record")
    return {
        "id": h["id"],
        "kind": h["kind"],
        "title": h["title"],
        "project": h["project"],
        "record": record["id"] if record else None,
        "summary": h["summary"],
        "match": h.get("match"),
        "score": h["score"],
    }


async def main(cmd: str, home: str, data: Any) -> None:
    memory = await open_memory(home, log=lambda line: print(f"[py] {line}", file=sys.stderr))
    try:
        if cmd == "write":
            await memory.add(data["records"])
            for c in data["conversations"]:
                await memory.add_conversation(c)
        elif cmd == "add":
            await memory.add(data["extra"])
        elif cmd == "read":
            results: dict[str, Any] = {}
            for query in data:
                name, op, q = query["name"], query["op"], query["q"]
                if op == "search":
                    results[name] = {"hits": [hit(h) for h in await memory.search(q)]}
                elif op == "searchText":
                    results[name] = {"text": await memory.search_text(q)}
                elif op == "room":
                    results[name] = {"text": await memory.room(q)}
                elif op == "explore":
                    e = await memory.explore(q)
                    results[name] = {
                        "found": [hit(h) for h in e["found"]],
                        "related": [{**hit(r), "via": r["via"]} for r in e["related"]],
                        "concepts": e["concepts"],
                    }
                elif op == "open":
                    found = await memory.search(q)
                    first = found[0]["id"] if found else None
                    results[name] = {"id": first, "item": await memory.open(first) if first else None}
                else:
                    raise ValueError(f"unknown op {op}")
            print(json.dumps({"lang": "py", "stats": memory.stats(), "results": results}, ensure_ascii=False))
        else:
            raise ValueError(f"unknown command {cmd}")
    finally:
        await memory.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: py.py write|add|read <home> <file>")
    with open(sys.argv[3], encoding="utf-8") as f:
        asyncio.run(main(sys.argv[1], sys.argv[2], json.load(f)))
