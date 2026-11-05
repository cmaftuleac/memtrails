"""Add a few records of different kinds and ask about them. No language model, no server.

python examples/quickstart.py
"""

import asyncio

from memtrails import open_memory


async def main() -> None:
    async with await open_memory(home="./example-memory", log=print) as memory:
        await memory.add(
            [
                {
                    "id": "mail-1",
                    "kind": "email",
                    "author": "Ana",
                    "title": "Proxy certificate expired",
                    "at": "2026-04-02",
                    "text": "The TLS certificate expired on the 1st. Renewal stopped because the cron job was removed in March.",
                },
                {
                    "id": "c9f1e2",
                    "kind": "commit",
                    "author": "Ion",
                    "title": "Restore certificate renewal cron",
                    "at": "2026-04-03",
                    "text": "Files: deploy/cron/renew.sh",
                },
                {
                    "id": "note-7",
                    "kind": "note",
                    "title": "Router",
                    "at": "2026-04-04",
                    "text": "Parola routerului este în caietul albastru din sertarul de la birou.",
                },
            ]
        )
        await memory.add_conversation(
            {
                "id": "chat-1",
                "at": "2026-04-05",
                "turns": [
                    {"role": "user", "text": "Where do we have lunch on Friday with Ana?"},
                    {"role": "assistant", "text": "At the small Italian place near the office, at one."},
                ],
            }
        )

        print(await memory.search_text({"query": "why did the certificate stop renewing?", "limit": 3}))
        print(await memory.search_text({"query": "where is the router password written down?", "limit": 1}))
        print(await memory.room({"query": "what made the renewal stop?", "chars": 400}))


asyncio.run(main())
