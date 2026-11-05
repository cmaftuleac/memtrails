"""Import a mailbox, a Slack export, a repository's history and a folder of notes, then search across all of them.

python examples/import_sources.py ./archive.mbox ./slack-export ./my-repo ./notes
"""

import asyncio
import os
import sys

from memtrails import open_memory
from memtrails.importers import from_files, from_git, from_mbox, from_slack_export


def progress(what: str):
    return lambda n: print(f"\r{what}: {n}", end="", flush=True)


async def main(mbox: str | None, slack: str | None, repo: str | None, notes: str | None) -> None:
    async with await open_memory(home="./example-memory") as memory:
        if mbox:
            print("\n", await memory.import_(from_mbox(mbox, collection="mail"), on_progress=progress("mail")))
        if slack:
            print("\n", await memory.import_(from_slack_export(slack), on_progress=progress("slack")))
        if repo:
            print("\n", await memory.import_(from_git(repo, max_count=5000), on_progress=progress("commits")))
        if notes:
            print("\n", await memory.import_(from_files(notes, collection="notes"), on_progress=progress("notes")))

        print(memory.stats())
        query = os.environ.get("QUERY", "what did we decide about the release?")
        print(await memory.search_text({"query": query, "limit": 10}))


args = sys.argv[1:] + [None] * 4
asyncio.run(main(*args[:4]))
