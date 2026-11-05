"""memtrails on the command line (cli.ts, ported): ``mcp`` serves a memory to an MCP client, ``add`` fills it from
files, mail, Slack or git, ``search`` asks it. The memory is MEMTRAILS_HOME, or ~/.memtrails, unless --home names
another."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from . import __version__
from .importers import from_eml, from_files, from_git, from_jsonl, from_mbox, from_slack_export
from .mcp import serve_mcp
from .memory import Memory, open_memory

_EPILOG = """MCP client configuration (Claude Desktop, Cursor, …):
  { "mcpServers": { "memory": { "command": "uvx", "args": ["memtrails", "mcp"] } } }
Claude Code:
  claude mcp add memory -- uvx memtrails mcp
"""


def _say(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _home(a: argparse.Namespace) -> str:
    return str(Path(a.home or os.environ.get("MEMTRAILS_HOME") or Path.home() / ".memtrails").expanduser().resolve())


def _importer_for(path: Path, collection: str | None) -> AsyncIterator[dict[str, Any]]:
    """What a path holds, as records: the importer its kind calls for."""
    if not path.exists():
        raise ValueError(f"{path}: not found")
    if path.is_dir():
        if (path / "channels.json").exists() and (path / "users.json").exists():
            return from_slack_export(path, collection=collection)
        return from_files(path, collection=collection, on_skip=lambda p, why: _say(f"  skipped {p}: {why}"))
    ext = path.suffix.lower()
    if ext == ".mbox":
        return from_mbox(path, collection=collection)
    if ext == ".eml":
        return from_eml(path, collection=collection)
    if ext == ".jsonl":
        return from_jsonl(path, collection=collection, on_skip=lambda _line, why, n: _say(f"  skipped line {n}: {why}"))
    raise ValueError(f"{path}: give a folder of notes or documents, an .mbox or .eml file, a .jsonl file, or a Slack export folder")


async def _with_memory(a: argparse.Namespace, f: Callable[[Memory], Awaitable[None]]) -> None:
    async with await open_memory(_home(a), log=_say) as memory:
        await f(memory)


async def _add(a: argparse.Namespace) -> None:
    sources = [(p, _importer_for(Path(p).resolve(), a.collection)) for p in a.paths]
    sources += [(f"{p} (git)", from_git(Path(p).resolve(), collection=a.collection)) for p in a.git or []]
    if not sources:
        raise ValueError("add what? give one or more paths, or --git REPO")

    async def run(m: Memory) -> None:
        for name, records in sources:
            _say(f"adding {name}")
            r = await m.import_(records, on_progress=lambda n: _say(f"  {n} records"))
            _say(f"  {r['added']} added, {r['unchanged']} unchanged")
        _say(f"{_home(a)}: {m.stats()['records']} records")

    await _with_memory(a, run)


async def _search(a: argparse.Namespace) -> None:
    query = " ".join(a.query).strip()
    if not query:
        raise ValueError("search for what? give the question as words")

    async def run(m: Memory) -> None:
        print(await m.search_text({"query": query, "limit": a.limit}), flush=True)

    await _with_memory(a, run)


async def _mcp(a: argparse.Namespace) -> None:
    home = _home(a)
    await serve_mcp(lambda: open_memory(home, log=_say), version=__version__, read_only=a.read_only, idle_s=a.idle * 60)


def _parser() -> argparse.ArgumentParser:
    def at_least(low: float, kind: Callable[[str], float]) -> Callable[[str], float]:
        def check(raw: str) -> float:
            n = kind(raw)
            if n < low:
                raise argparse.ArgumentTypeError(f"must be at least {low:g}")
            return n

        return check

    p = argparse.ArgumentParser(
        prog="memtrails", description="Local memory for agents.", epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", "-v", action="version", version=__version__)
    home = argparse.ArgumentParser(add_help=False)
    home.add_argument("--home", help="the memory (default: MEMTRAILS_HOME, or ~/.memtrails)")
    sub = p.add_subparsers(dest="command")
    m = sub.add_parser("mcp", parents=[home], help="serve the memory to an MCP client over stdio")
    m.add_argument("--read-only", action="store_true", help="without memory_save")
    m.add_argument(
        "--idle", type=at_least(0, float), default=5, metavar="MINUTES", help="close the memory after this long unused (0: never)"
    )
    ad = sub.add_parser("add", parents=[home], help="add folders, .mbox/.eml mail, .jsonl records, a Slack export, a git history")
    ad.add_argument("paths", nargs="*")
    ad.add_argument("--git", action="append", metavar="REPO")
    ad.add_argument("--collection")
    s = sub.add_parser("search", parents=[home], help="search, best first")
    s.add_argument("query", nargs="+")
    s.add_argument("--limit", type=at_least(1, int), default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    run = {"mcp": _mcp, "add": _add, "search": _search}.get(a.command or "")
    if run is None:
        p.print_help()
        return 0
    try:
        asyncio.run(run(a))
    except (ValueError, RuntimeError, OSError) as e:
        _say(f"memtrails: {e}")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
